#!/usr/bin/env python3 -u
"""Train with V2 environment: CNN observation + simplified reward.

Usage:
    python scripts/train_v2.py --timesteps 500000
    python scripts/train_v2.py --timesteps 500000 --device mps  # try GPU
    python scripts/train_v2.py --timesteps 500000 --lstm  # with memory
"""

import argparse
import functools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

print = functools.partial(print, flush=True)

import numpy as np
from datetime import datetime

import clasher.config as _cfg
_cfg.VERBOSE = False

from clasher.env_v2 import ClashRoyaleEnvV2
from clasher.network import CRFeatureExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help="cpu, mps, cuda, or auto")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lstm", action="store_true",
                        help="Use RecurrentPPO with LSTM")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("  CLASH ROYALE RL v2 — CNN + SIMPLIFIED REWARD")
    print("=" * 60)
    print(f"  Steps: {args.timesteps:,}")
    print(f"  Device: {device}")
    print(f"  LSTM: {args.lstm}")
    print(f"  Observation: spatial (3×32×18) + scalars (15)")
    print(f"  Reward: tower crowns + card penalty + win/loss")
    print("=" * 60)
    print()

    env = ClashRoyaleEnvV2(opponent="rule_bot")

    # Policy kwargs with our custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CRFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),  # bigger actor/critic
    )

    if args.lstm:
        try:
            from sb3_contrib import RecurrentPPO
            if args.resume:
                model = RecurrentPPO.load(args.resume, env=env, device=device)
            else:
                model = RecurrentPPO(
                    "MultiInputLstmPolicy",
                    env,
                    verbose=0,
                    seed=args.seed,
                    device=device,
                    n_steps=512,
                    batch_size=64,
                    n_epochs=4,
                    gamma=0.99,
                    gae_lambda=0.95,
                    learning_rate=3e-4,
                    clip_range=0.2,
                    ent_coef=0.02,
                    policy_kwargs=policy_kwargs,
                )
            print(f"Using RecurrentPPO (LSTM) on {device}")
        except ImportError:
            print("sb3-contrib RecurrentPPO not available, falling back to PPO")
            args.lstm = False

    if not args.lstm:
        from stable_baselines3 import PPO
        if args.resume:
            model = PPO.load(args.resume, env=env, device=device)
        else:
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=0,
                seed=args.seed,
                device=device,
                n_steps=512,
                batch_size=64,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=3e-4,
                clip_range=0.2,
                ent_coef=0.02,
                policy_kwargs=policy_kwargs,
            )
        print(f"Using PPO (no LSTM) on {device}")

    # Count parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Network parameters: {total_params:,}")
    print()

    # Training with inline logging
    episode_count = 0
    wins = 0
    losses = 0
    episode_rewards = []
    episode_winners = []
    current_reward = 0.0
    start_time = time.time()
    benchmark_history = []

    class V2Callback:
        def __init__(self):
            self.n_calls = 0

    cb = V2Callback()

    from stable_baselines3.common.callbacks import BaseCallback

    class LogCallback(BaseCallback):
        def __init__(self):
            super().__init__()

        def _on_step(self):
            nonlocal episode_count, wins, losses, current_reward
            nonlocal episode_rewards, episode_winners

            current_reward += self.locals.get("rewards", [0])[0]
            dones = self.locals.get("dones", [False])

            if dones[0]:
                episode_count += 1
                episode_rewards.append(current_reward)
                current_reward = 0.0

                infos = self.locals.get("infos", [{}])
                winner = infos[0].get("winner", None)
                episode_winners.append(winner)
                if winner == 0:
                    wins += 1
                elif winner == 1:
                    losses += 1

                if episode_count % args.log_every == 0:
                    # Benchmark
                    bench_env = ClashRoyaleEnvV2(opponent="rule_bot")
                    bench_wins = 0
                    for _ in range(10):
                        obs, _ = bench_env.reset()
                        while True:
                            act, _ = self.model.predict(obs, deterministic=True)
                            obs, r, term, trunc, info = bench_env.step(act)
                            if term:
                                if info["winner"] == 0:
                                    bench_wins += 1
                                break
                    bench_wr = bench_wins * 10
                    benchmark_history.append(bench_wr)
                    bench_trend = "→".join(
                        f"{w}" for w in benchmark_history[-6:]
                    )

                    total = wins + losses
                    cum_wr = wins / total * 100 if total > 0 else 0
                    n = min(args.log_every, len(episode_winners))
                    recent_wins = sum(1 for w in episode_winners[-n:] if w == 0)
                    recent_wr = recent_wins / n * 100
                    avg_r = np.mean(episode_rewards[-n:])
                    elapsed = time.time() - start_time
                    sps = self.num_timesteps / elapsed

                    if elapsed > 3600:
                        el_str = f"{elapsed/3600:.1f}h"
                    else:
                        el_str = f"{elapsed/60:.0f}m"

                    eta = (args.timesteps - self.num_timesteps) / sps if sps > 0 else 0
                    if eta > 3600:
                        eta_str = f"{eta/3600:.1f}h"
                    else:
                        eta_str = f"{eta/60:.0f}m"

                    ts = datetime.now().strftime("%H:%M:%S")
                    print(
                        f"[{ts}] "
                        f"ep={episode_count:<5d} steps={self.num_timesteps:>9,} | "
                        f"bench={bench_wr:3d}%[{bench_trend}] | "
                        f"WR={recent_wr:4.1f}% cum={cum_wr:4.1f}% | "
                        f"R={avg_r:+6.1f} | "
                        f"{sps:.0f} sps {el_str} / ETA {eta_str}"
                    )

            return True

    callback = LogCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback)

    elapsed = time.time() - start_time
    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0

    print()
    print("=" * 60)
    print(f"  DONE in {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"  Episodes: {total} | WR: {wr:.1f}%")
    print(f"  Params: {total_params:,} | Device: {device}")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/v2_{'lstm_' if args.lstm else ''}{args.timesteps}")
    print(f"Saved: models/v2_{'lstm_' if args.lstm else ''}{args.timesteps}")


if __name__ == "__main__":
    main()
