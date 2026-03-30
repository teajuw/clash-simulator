#!/usr/bin/env python3 -u
"""Behavioral cloning from expert demonstrations, then RL fine-tuning.

Usage:
    # Generate demos + BC pretrain + RL fine-tune (full pipeline):
    python scripts/train_bc.py --timesteps 200000

    # Just generate demos and pretrain (no RL):
    python scripts/train_bc.py --bc-only --demos 1000

    # Load existing BC model and fine-tune with RL:
    python scripts/train_bc.py --bc-model models/bc_expert --timesteps 200000
"""

import argparse
import functools
import os
import sys
import time

# Force unbuffered output so tail -f works
print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import clasher.config as _cfg
_cfg.VERBOSE = False

from clasher.env import ClashRoyaleEnv, N_ACTIONS, OBS_SIZE
from clasher.expert import ExpertPolicy, generate_demonstrations

from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO


class WinRateCallback(BaseCallback):
    """Same callback as train.py."""

    def __init__(self, eval_every: int = 20, verbose: int = 1):
        super().__init__(verbose)
        self.eval_every = eval_every
        self.episode_count = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_winners = []
        self._current_reward = 0.0
        self._current_length = 0
        self._start_time = time.time()

    def _on_step(self) -> bool:
        self._current_reward += self.locals.get("rewards", [0])[0]
        self._current_length += 1

        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_count += 1
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            self._current_reward = 0.0
            self._current_length = 0

            infos = self.locals.get("infos", [{}])
            winner = infos[0].get("winner", None)
            self.episode_winners.append(winner)
            if winner == 0:
                self.wins += 1
            elif winner == 1:
                self.losses += 1
            else:
                self.draws += 1

            if self.episode_count % self.eval_every == 0:
                total = self.wins + self.losses + self.draws
                cum_wr = self.wins / total * 100 if total > 0 else 0
                n = self.eval_every
                recent_wins = sum(1 for w in self.episode_winners[-n:] if w == 0)
                recent_wr = recent_wins / n * 100
                avg_r = np.mean(self.episode_rewards[-n:])
                avg_len = np.mean(self.episode_lengths[-n:])
                elapsed = time.time() - self._start_time
                steps = self.num_timesteps
                print(
                    f"[Ep {self.episode_count:4d} | {steps:7,} steps | {elapsed:5.0f}s] "
                    f"last{n}_WR={recent_wr:4.1f}%  cum_WR={cum_wr:4.1f}%  "
                    f"avg_R={avg_r:+7.2f}  avg_len={avg_len:3.0f}  "
                    f"({self.wins}W/{self.losses}L/{self.draws}D)"
                )
        return True


def behavioral_cloning(observations, actions, env, epochs=10, lr=1e-3, batch_size=256):
    """Train a MaskablePPO policy with supervised learning on expert data."""
    print(f"\n{'='*60}")
    print(f"Behavioral Cloning: {len(observations)} frames, {epochs} epochs")
    print(f"{'='*60}")

    # Create model
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
    )

    # Get the policy network
    policy = model.policy

    # Prepare data
    obs_tensor = torch.FloatTensor(np.array(observations))
    act_tensor = torch.LongTensor(np.array(actions))
    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Class weights: heavily upweight deploy actions vs WAIT
    # WAIT is ~83% of data, deploys are ~17% spread across 1080 classes
    act_array = np.array(actions)
    wait_count = (act_array == 0).sum()
    deploy_count = len(act_array) - wait_count
    # Weight deploys 10x more than WAIT
    weights = torch.ones(N_ACTIONS)
    if wait_count > 0 and deploy_count > 0:
        weights[0] = 1.0  # WAIT weight
        weights[1:] = wait_count / max(1, deploy_count) * 5.0  # deploy weight

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_obs, batch_acts in loader:
            # Get action logits from the policy
            features = policy.extract_features(batch_obs, policy.features_extractor)
            latent_pi, _ = policy.mlp_extractor(features)
            action_logits = policy.action_net(latent_pi)

            loss = loss_fn(action_logits, batch_acts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_obs)
            predicted = action_logits.argmax(dim=1)
            correct += (predicted == batch_acts).sum().item()
            total += len(batch_obs)

        avg_loss = total_loss / total
        accuracy = correct / total * 100
        print(f"  Epoch {epoch+1:2d}/{epochs}: loss={avg_loss:.4f}  accuracy={accuracy:.1f}%")

    # Stabilize policy for MaskablePPO: BC pushes logits to extremes
    # that cause Simplex violations in float32 after masking.
    # Scale down action_net weights so the policy is less certain,
    # giving RL room to explore while preserving the learned ranking.
    with torch.no_grad():
        for param in policy.action_net.parameters():
            param.mul_(0.1)

    return model


def evaluate_model(model, n_games=50):
    """Evaluate model win rate using manual action selection with masking."""
    env = ClashRoyaleEnv(opponent="rule_bot")
    wins = 0

    for i in range(n_games):
        obs, _ = env.reset()
        while True:
            mask = env.action_masks()
            # Manual forward pass to avoid MaskablePPO's strict validation
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                features = model.policy.extract_features(obs_t, model.policy.features_extractor)
                latent_pi, _ = model.policy.mlp_extractor(features)
                logits = model.policy.action_net(latent_pi).squeeze(0).numpy()

            # Apply mask: set invalid actions to -inf
            logits[~mask] = -1e8
            action = int(np.argmax(logits))

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                if info["winner"] == 0:
                    wins += 1
                break

    wr = wins / n_games * 100
    print(f"Eval: {wins}/{n_games} wins ({wr:.1f}%)")
    return wr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=int, default=500, help="Number of expert demo games")
    parser.add_argument("--bc-epochs", type=int, default=15, help="BC training epochs")
    parser.add_argument("--timesteps", type=int, default=200_000, help="RL fine-tuning steps")
    parser.add_argument("--bc-only", action="store_true", help="Only do BC, no RL")
    parser.add_argument("--bc-model", type=str, default=None, help="Load existing BC model")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = ClashRoyaleEnv(opponent="rule_bot")

    if args.bc_model:
        print(f"Loading BC model from {args.bc_model}")
        model = MaskablePPO.load(args.bc_model, env=env)
    else:
        # Step 1: Generate expert demonstrations
        print("Step 1: Generating expert demonstrations...")
        t0 = time.time()
        observations, actions = generate_demonstrations(n_games=args.demos)
        print(f"  Done in {time.time()-t0:.1f}s")

        # Check expert action distribution
        from collections import Counter
        act_counts = Counter(actions)
        wait_pct = act_counts[0] / len(actions) * 100
        print(f"  WAIT actions: {wait_pct:.1f}% | Deploy actions: {100-wait_pct:.1f}%")

        # Step 2: Behavioral cloning
        model = behavioral_cloning(
            observations, actions, env,
            epochs=args.bc_epochs,
        )

        # Save BC model
        os.makedirs("models", exist_ok=True)
        model.save("models/bc_expert")
        print("BC model saved to models/bc_expert")

    # Evaluate BC model
    print("\nEvaluating BC model (pre-RL)...")
    bc_wr = evaluate_model(model, n_games=50)

    if args.bc_only:
        return

    # Step 3: RL fine-tuning with MaskablePPO
    print(f"\n{'='*60}")
    print(f"RL Fine-tuning: {args.timesteps:,} steps with MaskablePPO")
    print(f"{'='*60}")

    # Re-attach env (needed after load)
    model.set_env(env)

    callback = WinRateCallback(eval_every=args.log_every)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    elapsed = time.time() - t0

    total = callback.wins + callback.losses + callback.draws
    wr = callback.wins / total * 100 if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Episodes: {total} | Win rate: {wr:.1f}%")
    print(f"BC WR (before RL): {bc_wr:.1f}% → Final WR: {wr:.1f}%")

    model.save(f"models/bc_ppo_{args.timesteps}")
    print(f"Model saved to models/bc_ppo_{args.timesteps}")


if __name__ == "__main__":
    main()
