#!/usr/bin/env python3 -u
"""Self-play training with snapshot pool.

Usage:
    python scripts/train_selfplay.py --timesteps 500000
    python scripts/train_selfplay.py --timesteps 1000000 --save-every 100
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
from stable_baselines3.common.callbacks import BaseCallback

import clasher.config as _cfg
_cfg.VERBOSE = False

from clasher.selfplay import SelfPlayEnv, render_pool_status


class SelfPlayCallback(BaseCallback):
    """Callback that saves snapshots and displays training progress."""

    def __init__(
        self,
        env: SelfPlayEnv,
        log_every: int = 20,
        save_every: int = 50,
    ):
        super().__init__(verbose=0)
        self.env = env
        self.log_every = log_every
        self.save_every = save_every

        self.episode_count = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.episode_rewards = []
        self.episode_winners = []
        self._current_reward = 0.0
        self._start_time = time.time()
        self.benchmark_history: list = []
        self._last_bench_wr: float = 0.0

    def _on_step(self) -> bool:
        self._current_reward += self.locals.get("rewards", [0])[0]

        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_count += 1
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0

            infos = self.locals.get("infos", [{}])
            winner = infos[0].get("winner", None)
            self.episode_winners.append(winner)
            if winner == 0:
                self.wins += 1
            elif winner == 1:
                self.losses += 1
            else:
                self.draws += 1

            # Save snapshot
            if self.episode_count % self.save_every == 0:
                self.env.save_snapshot(self.model)

            # Display progress with inline benchmark
            if self.episode_count % self.log_every == 0:
                self._benchmark(n_games=10)
                self._display()

        return True

    def _benchmark(self, n_games: int = 10):
        """Run fixed benchmark: current model vs rule bot."""
        from clasher.env import ClashRoyaleEnv
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
        total = self.wins + self.losses + self.draws
        cum_wr = self.wins / total * 100 if total > 0 else 0

        n = min(self.log_every, len(self.episode_winners))
        recent_wins = sum(1 for w in self.episode_winners[-n:] if w == 0)
        recent_wr = recent_wins / n * 100 if n > 0 else 0

        avg_r = np.mean(self.episode_rewards[-n:]) if self.episode_rewards else 0

        # Benchmark trend
        bench_trend = "→".join(f"{w:.0f}" for _, w in self.benchmark_history[-6:])

        output = render_pool_status(
            episode=self.episode_count,
            pool_size=self.env.pool.size,
            win_rate=cum_wr,
            recent_wr=recent_wr,
            wins=self.wins,
            losses=self.losses,
            avg_reward=avg_r,
            elapsed=time.time() - self._start_time,
            steps=self.num_timesteps,
            agent_elo=self.env.pool.agent_elo,
            bench_wr=self._last_bench_wr,
            bench_trend=bench_trend,
            opponent_dist=self.env.get_opponent_distribution(),
            pool_stats=self.env.pool.get_stats_summary(),
        )
        print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save snapshot every N episodes")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--snapshot-dir", type=str, default="snapshots")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model to resume from")
    parser.add_argument("--n-steps", type=int, default=1024,
                        help="Steps per PPO rollout (higher=faster but more memory)")
    args = parser.parse_args()

    print("=" * 55)
    print("  CLASH ROYALE RL — SELF-PLAY TRAINING")
    print("=" * 55)
    print(f"  Steps: {args.timesteps:,}")
    print(f"  Snapshot pool: {args.snapshot_dir}/")
    print(f"  Save every: {args.save_every} episodes")
    print(f"  Seed: {args.seed}")
    print("=" * 55)
    print()

    env = SelfPlayEnv(
        snapshot_dir=args.snapshot_dir,
        save_every=args.save_every,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=args.seed,
            n_steps=args.n_steps,
            batch_size=128,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.02,
        )

    callback = SelfPlayCallback(
        env=env,
        log_every=args.log_every,
        save_every=args.save_every,
    )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    elapsed = time.time() - t0

    # Final stats
    total = callback.wins + callback.losses + callback.draws
    wr = callback.wins / total * 100 if total > 0 else 0

    print("=" * 55)
    print(f"  TRAINING COMPLETE")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Episodes: {total}")
    print(f"  Win rate: {wr:.1f}% ({callback.wins}W/{callback.losses}L)")
    print(f"  Snapshot pool: {env.pool.size} models")
    print("=" * 55)

    # Save final model
    os.makedirs("models", exist_ok=True)
    model.save(f"models/selfplay_final_{args.timesteps}")
    print(f"Final model: models/selfplay_final_{args.timesteps}")


if __name__ == "__main__":
    main()
