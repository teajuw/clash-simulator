#!/usr/bin/env python3 -u
"""Parallel self-play training using SubprocVecEnv.

Runs N environments in parallel on separate CPU cores.
Each env plays against the rule bot (no cross-process snapshot pool).
Snapshot saving and benchmarking happen in the callback on the main process.

Usage:
    python scripts/train_parallel.py --n-envs 4 --timesteps 10000000
    python scripts/train_parallel.py --resume models/selfplay_overnight_elo1670 --n-envs 4
"""

import argparse
import functools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

print = functools.partial(print, flush=True)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import clasher.config as _cfg
_cfg.VERBOSE = False

from clasher.env import ClashRoyaleEnv


def make_env(seed: int):
    """Factory function for SubprocVecEnv."""
    def _init():
        env = ClashRoyaleEnv(opponent="rule_bot")
        env.reset(seed=seed)
        return env
    return _init


class ParallelCallback(BaseCallback):
    """Tracks episodes across parallel envs, saves snapshots, runs benchmarks."""

    def __init__(self, log_every: int = 50, save_dir: str = "models"):
        super().__init__(verbose=0)
        self.log_every = log_every
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.episode_count = 0
        self.wins = 0
        self.losses = 0
        self.episode_rewards = []
        self.episode_winners = []
        self._ep_rewards = {}  # per-env running reward
        self._start_time = time.time()
        self.benchmark_history = []
        self._last_bench_wr = 0.0

    def _on_step(self) -> bool:
        # Track rewards per env
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i in range(len(dones)):
            env_id = i
            if env_id not in self._ep_rewards:
                self._ep_rewards[env_id] = 0.0
            self._ep_rewards[env_id] += rewards[i]

            if dones[i]:
                self.episode_count += 1
                self.episode_rewards.append(self._ep_rewards[env_id])
                self._ep_rewards[env_id] = 0.0

                winner = infos[i].get("winner", None)
                self.episode_winners.append(winner)
                if winner == 0:
                    self.wins += 1
                elif winner == 1:
                    self.losses += 1

                if self.episode_count % self.log_every == 0:
                    self._benchmark()
                    self._display()

                # Save every 500 episodes
                if self.episode_count % 500 == 0:
                    path = os.path.join(self.save_dir, f"parallel_ep{self.episode_count}")
                    self.model.save(path)

        return True

    def _benchmark(self, n_games: int = 10):
        bench_env = ClashRoyaleEnv(opponent="rule_bot")
        wins = 0
        for _ in range(n_games):
            obs, _ = bench_env.reset()
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = bench_env.step(action)
                if terminated:
                    if info["winner"] == 0:
                        wins += 1
                    break
        self._last_bench_wr = wins / n_games * 100
        self.benchmark_history.append((self.episode_count, self._last_bench_wr))

    def _display(self):
        from datetime import datetime
        total = self.wins + self.losses
        cum_wr = self.wins / total * 100 if total > 0 else 0
        n = min(self.log_every, len(self.episode_winners))
        recent_wins = sum(1 for w in self.episode_winners[-n:] if w == 0)
        recent_wr = recent_wins / n * 100 if n > 0 else 0
        avg_r = np.mean(self.episode_rewards[-n:]) if self.episode_rewards else 0
        elapsed = time.time() - self._start_time
        sps = self.num_timesteps / elapsed if elapsed > 0 else 0
        bench_trend = "→".join(f"{w:.0f}" for _, w in self.benchmark_history[-6:])
        ts = datetime.now().strftime("%H:%M:%S")

        print(
            f"[{ts}] "
            f"ep={self.episode_count:<5d} steps={self.num_timesteps:>9,} | "
            f"bench={self._last_bench_wr:3.0f}%[{bench_trend}] | "
            f"WR={recent_wr:4.1f}% cum={cum_wr:4.1f}% | "
            f"R={avg_r:+6.1f} | "
            f"{sps:.0f} sps"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=1024)
    args = parser.parse_args()

    print("=" * 60)
    print("  CLASH ROYALE RL — PARALLEL TRAINING")
    print("=" * 60)
    print(f"  Steps: {args.timesteps:,}  |  Envs: {args.n_envs} parallel")
    print(f"  n_steps: {args.n_steps}  |  batch: {args.n_steps * args.n_envs // 4}")
    print("=" * 60)
    print()

    # Create parallel environments
    env = SubprocVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])

    batch_size = min(256, args.n_steps * args.n_envs // 4)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=args.seed,
            device="cpu",
            n_steps=args.n_steps,
            batch_size=batch_size,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.02,
        )

    callback = ParallelCallback(log_every=args.log_every)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    elapsed = time.time() - t0

    total = callback.wins + callback.losses
    wr = callback.wins / total * 100 if total > 0 else 0
    print("=" * 60)
    print(f"  DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Episodes: {total} | WR: {wr:.1f}%")
    print(f"  Throughput: {args.timesteps/elapsed:.0f} steps/sec")
    print("=" * 60)

    model.save("models/parallel_final")
    print("Saved: models/parallel_final")


if __name__ == "__main__":
    main()
