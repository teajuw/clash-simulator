#!/usr/bin/env python3 -u
"""Behavioral cloning from smart bot demonstrations.

Records smart bot vs smart bot games, then pretrains the CNN policy
via supervised learning. The pretrained model goes straight to self-play.

Usage:
    python scripts/pretrain_bc.py --games 500 --epochs 20
    python scripts/pretrain_bc.py --games 1000 --epochs 30 --then-train 10000
"""

import argparse
import functools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

print = functools.partial(print, flush=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import clasher.config as _cfg
_cfg.VERBOSE = False

from clasher.env_v3 import ClashRoyaleEnvV3, DECK, CARD_TO_IDX, ELIXIR_COST, GRID_X, GRID_Y, N_CARD_CHOICES
from clasher.smart_bot import SmartBot
from clasher.network import CRFeatureExtractor
from stable_baselines3 import PPO


POLICY_KWARGS = dict(
    features_extractor_class=CRFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
)


def record_demonstrations(n_games: int = 500):
    """Record smart bot playing as P0. Returns (spatial_obs, scalar_obs, actions)."""
    print(f"Recording {n_games} demonstration games...")

    env = ClashRoyaleEnvV3(opponent="none", domain_randomization=False)
    bot_p0 = SmartBot(player_id=0)
    bot_p1 = SmartBot(player_id=1)
    env._opponent_fn = lambda b: bot_p1.act(b)

    all_spatial = []
    all_scalars = []
    all_actions = []
    wins = 0

    for game in range(n_games):
        obs, _ = env.reset(seed=game)

        while True:
            # Get bot's action for P0
            action = _bot_to_action(bot_p0, env.battle)

            all_spatial.append(obs["spatial"].copy())
            all_scalars.append(obs["scalars"].copy())
            all_actions.append(action.copy())

            obs, r, term, trunc, info = env.step(action)
            if term:
                if info["winner"] == 0:
                    wins += 1
                break

        if (game + 1) % 100 == 0:
            wr = wins / (game + 1) * 100
            print(f"  [{game+1}/{n_games}] WR={wr:.0f}% | frames={len(all_actions):,}")

    wr = wins / n_games * 100
    print(f"  Done. {len(all_actions):,} frames from {n_games} games. Bot WR={wr:.0f}%")

    return (
        np.array(all_spatial, dtype=np.float32),
        np.array(all_scalars, dtype=np.float32),
        np.array(all_actions, dtype=np.int64),
    )


def _bot_to_action(bot: SmartBot, battle) -> np.ndarray:
    """Convert smart bot's decision to a MultiDiscrete([5, 18, 15]) action."""
    player = battle.players[bot.pid]
    hand_before = list(player.hand)
    elixir_before = player.elixir

    # Let the bot act
    bot.act(battle)

    hand_after = list(player.hand)
    elixir_after = player.elixir

    # Detect what was played by comparing hand/elixir
    if hand_before == hand_after and abs(elixir_before - elixir_after) < 0.5:
        # Bot chose WAIT
        return np.array([0, 0, 0], dtype=np.int64)

    # Find which card was played
    for i, card in enumerate(hand_before):
        if card not in hand_after or (hand_before.count(card) > hand_after.count(card)):
            # Card i was played. But we need the position...
            # We can't easily get the position the bot chose, so we'll
            # detect it from new entities on the field
            break

    # Undo the bot's action — we need the env.step to execute it instead
    # Restore state
    player.hand = hand_before
    player.elixir = elixir_before

    # Re-execute to capture the deployment position
    # Track new entities
    entities_before = set(battle.entities.keys())
    bot.act(battle)
    entities_after = set(battle.entities.keys())
    new_entities = entities_after - entities_before

    if not new_entities:
        return np.array([0, 0, 0], dtype=np.int64)

    # Find the new entity's position
    new_id = min(new_entities)
    new_entity = battle.entities[new_id]
    deploy_x = int(new_entity.position.x)
    deploy_y = int(new_entity.position.y)

    # Find which hand slot was played
    hand_after2 = list(player.hand)
    card_slot = 0
    for i, card in enumerate(hand_before):
        if i < len(hand_after2) and hand_before[i] != hand_after2[i]:
            card_slot = i
            break

    # Clamp to valid ranges
    card_choice = min(4, card_slot + 1)  # 1-indexed
    tile_x = max(0, min(GRID_X - 1, deploy_x))
    tile_y = max(0, min(GRID_Y - 1, deploy_y))

    return np.array([card_choice, tile_x, tile_y], dtype=np.int64)


def behavioral_cloning(spatial, scalars, actions, env, epochs=20, lr=1e-3, batch_size=256):
    """Pretrain PPO policy via supervised learning on demonstrations."""
    print(f"\nBehavioral Cloning: {len(actions):,} frames, {epochs} epochs")

    model = PPO("MultiInputPolicy", env, verbose=0, device="cpu",
                policy_kwargs=POLICY_KWARGS, n_steps=512, batch_size=64,
                gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
                clip_range=0.2, ent_coef=0.025)

    policy = model.policy
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Network params: {total_params:,}")

    # Prepare tensors
    spatial_t = torch.FloatTensor(spatial)
    scalars_t = torch.FloatTensor(scalars)
    # Actions: MultiDiscrete([5, 18, 15]) — 3 separate classification heads
    actions_card = torch.LongTensor(actions[:, 0])
    actions_x = torch.LongTensor(actions[:, 1])
    actions_y = torch.LongTensor(actions[:, 2])

    dataset = TensorDataset(spatial_t, scalars_t, actions_card, actions_x, actions_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Weight card actions higher (most are WAIT=0)
    wait_count = (actions[:, 0] == 0).sum()
    play_count = len(actions) - wait_count
    card_weights = torch.ones(N_CARD_CHOICES)
    if wait_count > 0 and play_count > 0:
        card_weights[1:] = wait_count / max(1, play_count) * 2.0
    card_loss_fn = nn.CrossEntropyLoss(weight=card_weights)

    for epoch in range(epochs):
        total_loss = 0
        correct_card = 0
        total = 0

        for batch_sp, batch_sc, batch_ac, batch_ax, batch_ay in loader:
            obs = {"spatial": batch_sp, "scalars": batch_sc}

            # Forward through feature extractor
            features = policy.extract_features(obs, policy.features_extractor)
            latent_pi, _ = policy.mlp_extractor(features)
            action_logits = policy.action_net(latent_pi)

            # Split logits for MultiDiscrete: [5, 18, 15] = 38 total
            card_logits = action_logits[:, :N_CARD_CHOICES]
            x_logits = action_logits[:, N_CARD_CHOICES:N_CARD_CHOICES + GRID_X]
            y_logits = action_logits[:, N_CARD_CHOICES + GRID_X:]

            loss_card = card_loss_fn(card_logits, batch_ac)
            loss_x = loss_fn(x_logits, batch_ax)
            loss_y = loss_fn(y_logits, batch_ay)
            loss = loss_card + loss_x + loss_y

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_sp)
            correct_card += (card_logits.argmax(1) == batch_ac).sum().item()
            total += len(batch_sp)

        avg_loss = total_loss / total
        card_acc = correct_card / total * 100
        play_pct = play_count / len(actions) * 100
        print(f"  Epoch {epoch+1:2d}/{epochs}: loss={avg_loss:.4f} card_acc={card_acc:.1f}% (play={play_pct:.0f}%)")

    # Scale down action weights to prevent MaskablePPO Simplex issues
    with torch.no_grad():
        for param in policy.action_net.parameters():
            param.mul_(0.3)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--then-train", type=int, default=0,
                        help="Continue with PPO self-play for N episodes after BC")
    args = parser.parse_args()

    print("=" * 60)
    print("  BEHAVIORAL CLONING → SELF-PLAY PIPELINE")
    print("=" * 60)
    print(f"  Demo games: {args.games}")
    print(f"  BC epochs: {args.epochs}")
    print("=" * 60)
    print()

    # 1. Record demonstrations
    t0 = time.time()
    spatial, scalars, actions = record_demonstrations(args.games)
    print(f"  Recording time: {time.time()-t0:.0f}s")

    # Action distribution
    wait_pct = (actions[:, 0] == 0).sum() / len(actions) * 100
    print(f"  WAIT: {wait_pct:.0f}% | PLAY: {100-wait_pct:.0f}%")

    # 2. Behavioral cloning
    env = ClashRoyaleEnvV3(opponent="none", domain_randomization=False)
    model = behavioral_cloning(spatial, scalars, actions, env, epochs=args.epochs)

    # 3. Save
    os.makedirs("models", exist_ok=True)
    model.save("models/bc_pretrained")
    print(f"\nSaved: models/bc_pretrained")

    # 4. Quick benchmark
    from clasher.smart_bot import make_smart_bot_policy
    bench_env = ClashRoyaleEnvV3(opponent="none", domain_randomization=False)
    bench_env._opponent_fn = make_smart_bot_policy(player_id=1)
    wins = 0
    for i in range(20):
        obs, _ = bench_env.reset(seed=i + 2000)
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = bench_env.step(act)
            if term:
                if info["winner"] == 0:
                    wins += 1
                break
    print(f"  BC model vs smart bot: {wins}/20 ({wins*5}%)")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
