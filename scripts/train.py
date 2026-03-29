"""Train DQN and PPO on the Clash Royale 2.6 Hog Cycle environment.

Usage:
    python scripts/train.py --algo dqn --timesteps 50000
    python scripts/train.py --algo ppo --timesteps 50000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from clasher.env import ClashRoyaleEnv


class WinRateCallback(BaseCallback):
    """Log win rate every N episodes."""

    def __init__(self, eval_every: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.eval_every = eval_every
        self.episode_count = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_reward = 0.0
        self._current_length = 0

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

            # Check winner
            infos = self.locals.get("infos", [{}])
            winner = infos[0].get("winner", None)
            if winner == 0:
                self.wins += 1
            elif winner == 1:
                self.losses += 1
            else:
                self.draws += 1

            if self.episode_count % self.eval_every == 0:
                total = self.wins + self.losses + self.draws
                wr = self.wins / total * 100 if total > 0 else 0
                avg_r = np.mean(self.episode_rewards[-self.eval_every:])
                avg_len = np.mean(self.episode_lengths[-self.eval_every:])
                print(
                    f"[Ep {self.episode_count:4d}] "
                    f"WR={wr:.1f}% ({self.wins}W/{self.losses}L/{self.draws}D) "
                    f"avg_reward={avg_r:.2f} avg_steps={avg_len:.0f}"
                )
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Training {args.algo.upper()} for {args.timesteps:,} timesteps")
    print(f"Opponent: rule_bot | Seed: {args.seed}")
    print("-" * 60)

    env = ClashRoyaleEnv(opponent="rule_bot")

    if args.algo == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            verbose=0,
            seed=args.seed,
            buffer_size=50_000,
            learning_starts=500,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            target_update_interval=500,
            learning_rate=1e-4,
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=args.seed,
            n_steps=256,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.01,
        )

    callback = WinRateCallback(eval_every=20)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    elapsed = time.time() - t0

    # Final stats
    total = callback.wins + callback.losses + callback.draws
    wr = callback.wins / total * 100 if total > 0 else 0
    print("-" * 60)
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Episodes: {total} | Win rate: {wr:.1f}%")
    print(f"Wins: {callback.wins} | Losses: {callback.losses} | Draws: {callback.draws}")

    # Save model
    os.makedirs("models", exist_ok=True)
    path = f"models/{args.algo}_hog26_{args.timesteps}"
    model.save(path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    main()
