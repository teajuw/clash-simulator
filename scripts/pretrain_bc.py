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
        interceptor = DeployInterceptor(env.battle)

        while True:
            # Record observation BEFORE bot acts
            all_spatial.append(obs["spatial"].copy())
            all_scalars.append(obs["scalars"].copy())

            # Bot acts (deploy is captured by interceptor, happens once)
            action = _bot_to_action(bot_p0, env.battle, interceptor)
            all_actions.append(action.copy())

            # Step env (P1 bot acts inside, simulation advances)
            obs, r, term, trunc, info = env.step(np.array([0, 0, 0]))  # WAIT — bot already acted
            if term:
                if info["winner"] == 0:
                    wins += 1
                interceptor.restore()
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


class DeployInterceptor:
    """Monkey-patches battle.deploy_card to capture what the bot deploys."""

    def __init__(self, battle):
        self.battle = battle
        self.last_deploy = None  # (player_id, card_name, x, y)
        self._original_deploy = battle.deploy_card

        def intercepted_deploy(player_id, card_name, position):
            result = self._original_deploy(player_id, card_name, position)
            if result and player_id == 0:
                self.last_deploy = (player_id, card_name, position.x, position.y)
            return result

        battle.deploy_card = intercepted_deploy

    def reset(self):
        self.last_deploy = None

    def restore(self):
        self.battle.deploy_card = self._original_deploy


def _bot_to_action(bot: SmartBot, battle, interceptor: DeployInterceptor) -> np.ndarray:
    """Let the bot act once, capture what it deployed via interceptor."""
    interceptor.reset()
    player = battle.players[bot.pid]
    hand_before = list(player.hand)

    bot.act(battle)

    if interceptor.last_deploy is None:
        return np.array([0, 0, 0], dtype=np.int64)

    _, card_name, x, y = interceptor.last_deploy

    # Find which hand slot was played
    card_slot = 0
    for i, card in enumerate(hand_before):
        if card == card_name:
            card_slot = i
            break

    card_choice = min(4, card_slot + 1)
    tile_x = max(0, min(GRID_X - 1, int(x)))
    tile_y = max(0, min(GRID_Y - 1, int(y)))

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

    # Filter: keep ALL frames for card choice, but only PLAY frames for x/y
    # This teaches: "when to play" from full data, "where to play" from play-only data
    spatial_t = torch.FloatTensor(spatial)
    scalars_t = torch.FloatTensor(scalars)
    actions_card = torch.LongTensor(actions[:, 0])
    actions_x = torch.LongTensor(actions[:, 1])
    actions_y = torch.LongTensor(actions[:, 2])

    # Also create play-only dataset for x/y training
    play_mask = actions[:, 0] > 0
    play_spatial = torch.FloatTensor(spatial[play_mask])
    play_scalars = torch.FloatTensor(scalars[play_mask])
    play_x = torch.LongTensor(actions[play_mask, 1])
    play_y = torch.LongTensor(actions[play_mask, 2])
    play_card = torch.LongTensor(actions[play_mask, 0])

    print(f"  Play frames: {play_mask.sum():,} / {len(actions):,} ({play_mask.sum()/len(actions)*100:.0f}%)")

    # Full dataset for card choice
    full_dataset = TensorDataset(spatial_t, scalars_t, actions_card)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    # Play-only dataset for card + position
    play_dataset = TensorDataset(play_spatial, play_scalars, play_card, play_x, play_y)
    play_loader = DataLoader(play_dataset, batch_size=min(batch_size, len(play_dataset)), shuffle=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Strong weighting for play actions
    wait_count = (actions[:, 0] == 0).sum()
    play_count = len(actions) - wait_count
    card_weights = torch.ones(N_CARD_CHOICES)
    if wait_count > 0 and play_count > 0:
        card_weights[1:] = wait_count / max(1, play_count) * 5.0  # 5x weight
    card_loss_fn = nn.CrossEntropyLoss(weight=card_weights)

    for epoch in range(epochs):
        total_loss = 0
        correct_card = 0
        correct_play = 0
        total = 0
        total_play = 0

        # Phase 1: Train card choice on ALL data (when to play vs wait)
        for batch_sp, batch_sc, batch_ac in full_loader:
            obs = {"spatial": batch_sp, "scalars": batch_sc}
            features = policy.extract_features(obs, policy.features_extractor)
            latent_pi, _ = policy.mlp_extractor(features)
            action_logits = policy.action_net(latent_pi)

            card_logits = action_logits[:, :N_CARD_CHOICES]
            loss = card_loss_fn(card_logits, batch_ac)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_sp)
            correct_card += (card_logits.argmax(1) == batch_ac).sum().item()
            total += len(batch_sp)

        # Phase 2: Train position (x, y) on PLAY-ONLY data (where to place)
        for batch_sp, batch_sc, batch_ac, batch_ax, batch_ay in play_loader:
            obs = {"spatial": batch_sp, "scalars": batch_sc}
            features = policy.extract_features(obs, policy.features_extractor)
            latent_pi, _ = policy.mlp_extractor(features)
            action_logits = policy.action_net(latent_pi)

            card_logits = action_logits[:, :N_CARD_CHOICES]
            x_logits = action_logits[:, N_CARD_CHOICES:N_CARD_CHOICES + GRID_X]
            y_logits = action_logits[:, N_CARD_CHOICES + GRID_X:]

            loss = card_loss_fn(card_logits, batch_ac) + loss_fn(x_logits, batch_ax) + loss_fn(y_logits, batch_ay)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct_play += (card_logits.argmax(1) == batch_ac).sum().item()
            total_play += len(batch_sp)

        avg_loss = total_loss / max(1, total)
        card_acc = correct_card / max(1, total) * 100
        play_acc = correct_play / max(1, total_play) * 100
        print(f"  Epoch {epoch+1:2d}/{epochs}: loss={avg_loss:.4f} card_acc={card_acc:.1f}% play_acc={play_acc:.1f}%")

    # No weight scaling — vanilla PPO doesn't have the Simplex issue

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
