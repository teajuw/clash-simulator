#!/usr/bin/env python3 -u
"""Single agent process for league training. Launched by train_league.py."""

import argparse
import functools
import os
import sys
import time
import random as _random
from glob import glob
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

print = functools.partial(print, flush=True)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import clasher.config as _cfg
_cfg.VERBOSE = False

from clasher.env_v2 import ClashRoyaleEnvV2 as ClashRoyaleEnv, DECK, tile_to_position, ELIXIR_COST
from clasher.arena import Position
from clasher.entities import Troop


# ── Opponent Functions ────────────────────────────────────────────────────────

def make_pool_opponent(snapshot_dir: str, role: str):
    """Create an opponent function based on the agent's role."""

    def _load_random_snapshot():
        """Load a random snapshot from the pool."""
        zips = sorted(glob(os.path.join(snapshot_dir, "*.zip")))
        if not zips:
            return None
        path = _random.choice(zips)
        try:
            model = PPO.load(path.replace(".zip", ""))
            return model
        except Exception:
            return None

    def _load_latest_main_snapshot():
        """Load the latest main agent snapshot."""
        zips = sorted(glob(os.path.join(snapshot_dir, "main_*.zip")))
        if not zips:
            # Fall back to any snapshot
            return _load_random_snapshot()
        path = zips[-1]
        try:
            model = PPO.load(path.replace(".zip", ""))
            return model
        except Exception:
            return None

    # Cache the opponent model — reload periodically
    state = {"model": None, "loaded_at": 0, "reload_every": 50}

    def opponent_fn(battle):
        """Act as player 1 using a loaded snapshot."""
        state["loaded_at"] += 1

        # Reload opponent model periodically
        if state["model"] is None or state["loaded_at"] % state["reload_every"] == 0:
            if role == "main_exploiter":
                state["model"] = _load_latest_main_snapshot()
            else:  # main or league_exploiter
                state["model"] = _load_random_snapshot()

        model = state["model"]
        if model is None:
            # No snapshots yet — play simple rule-based
            _simple_opponent(battle)
            return

        # Build mirrored observation for player 1
        obs = _mirror_obs(battle)
        try:
            action, _ = model.predict(obs, deterministic=False)
            card_choice = int(action[0])
            tile_x = int(action[1])
            tile_y = int(action[2])

            if card_choice > 0:
                slot = card_choice - 1
                player = battle.players[1]
                if slot < len(player.hand):
                    card_name = player.hand[slot]
                    # Mirror y for player 1
                    mirrored_y = 31 - tile_y
                    pos = Position(tile_x + 0.5, mirrored_y + 0.5)
                    battle.deploy_card(1, card_name, pos)
        except Exception:
            _simple_opponent(battle)

    return opponent_fn


def _simple_opponent(battle):
    """Minimal fallback when no snapshots exist yet."""
    player = battle.players[1]
    if player.elixir >= 4 and "HogRider" in player.hand:
        bridge_x = _random.choice([3.5, 14.5])
        battle.deploy_card(1, "HogRider", Position(bridge_x, 18.0))
    elif player.elixir >= 9.5:
        cheapest = min(player.hand, key=lambda c: ELIXIR_COST.get(c, 10))
        battle.deploy_card(1, cheapest, Position(9.0, 20.0))


def _mirror_obs(battle):
    """Build observation from player 1's perspective."""
    from clasher.env import OBS_SIZE, CARD_TO_IDX, NUM_CARD_IDS
    from clasher.entities import Building

    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    p0 = battle.players[1]  # p1 sees itself as p0
    p1 = battle.players[0]

    obs[0] = p0.elixir / 10.0
    obs[1] = p1.elixir / 10.0
    obs[2] = p0.king_tower_hp / 4824.0
    obs[3] = p0.left_tower_hp / 3052.0
    obs[4] = p0.right_tower_hp / 3052.0
    obs[5] = p1.king_tower_hp / 4824.0
    obs[6] = p1.left_tower_hp / 3052.0
    obs[7] = p1.right_tower_hp / 3052.0
    obs[8] = min(battle.time / 360.0, 1.0)
    obs[9] = float(battle.double_elixir)
    obs[10] = float(battle.triple_elixir)
    obs[11] = float(battle.overtime)

    offset = 12
    for i in range(min(4, len(p0.hand))):
        obs[offset + i] = CARD_TO_IDX.get(p0.hand[i], 0) / (NUM_CARD_IDS - 1)
    if p0.cycle_queue:
        obs[offset + 4] = CARD_TO_IDX.get(p0.cycle_queue[0], 0) / (NUM_CARD_IDS - 1)
    for i in range(min(4, len(p0.hand))):
        cost = ELIXIR_COST.get(p0.hand[i], 10)
        obs[offset + 5 + i] = 1.0 if p0.elixir >= cost else 0.0

    # Units — swap owner, flip y
    offset = 21
    unit_idx = 0
    for entity in battle.entities.values():
        if unit_idx >= 30 or not entity.is_alive:
            continue
        if not isinstance(entity, (Troop, Building)):
            continue

        if isinstance(entity, Building) and entity.position.y in (2.5, 29.5):
            card_id = 9
        elif isinstance(entity, Building) and entity.position.y in (6.5, 25.5):
            card_id = 8
        elif entity.card_stats:
            card_id = CARD_TO_IDX.get(entity.card_stats.name, 0)
        else:
            card_id = 0

        base = offset + unit_idx * 5
        obs[base] = card_id / (NUM_CARD_IDS - 1)
        obs[base + 1] = np.clip(entity.position.x / 18.0, 0.0, 1.0)
        obs[base + 2] = np.clip((32.0 - entity.position.y) / 32.0, 0.0, 1.0)  # flip y
        obs[base + 3] = entity.hitpoints / entity.max_hitpoints if entity.max_hitpoints > 0 else 0.0
        obs[base + 4] = 0.0 if entity.player_id == 1 else 1.0  # swap owner
        unit_idx += 1

    return obs


# ── Callback ──────────────────────────────────────────────────────────────────

class LeagueCallback(BaseCallback):
    def __init__(self, role, log_prefix, log_every, save_every,
                 snapshot_dir, ent_coef, reset_every=10000):
        super().__init__(verbose=0)
        self.role = role
        self.log_prefix = log_prefix
        self.log_every = log_every
        self.save_every = save_every
        self.snapshot_dir = snapshot_dir
        self.ent_coef = ent_coef
        self.reset_every = reset_every

        self.episode_count = 0
        self.wins = 0
        self.losses = 0
        self.episode_rewards = []
        self.episode_winners = []
        self._current_reward = 0.0
        self._start_time = time.time()

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

            # Save snapshot with role prefix
            if self.episode_count % self.save_every == 0:
                path = os.path.join(self.snapshot_dir,
                                    f"{self.role}_{self.episode_count}")
                self.model.save(path)

            # Main exploiter: reset weights periodically to keep exploring
            if self.role == "main_exploiter" and self.episode_count % self.reset_every == 0:
                # Partial reset: re-randomize last layer only
                import torch
                with torch.no_grad():
                    for param in self.model.policy.action_net.parameters():
                        param.normal_(0, 0.1)

            # Display
            if self.episode_count % self.log_every == 0:
                self._display()

        return True

    def _display(self):
        from datetime import datetime
        total = self.wins + self.losses
        wr = self.wins / total * 100 if total > 0 else 0
        n = min(self.log_every, len(self.episode_winners))
        recent_wins = sum(1 for w in self.episode_winners[-n:] if w == 0)
        recent_wr = recent_wins / n * 100 if n > 0 else 0
        avg_r = np.mean(self.episode_rewards[-n:]) if self.episode_rewards else 0
        elapsed = time.time() - self._start_time
        sps = self.num_timesteps / elapsed if elapsed > 0 else 0
        ts = datetime.now().strftime("%H:%M:%S")

        pool_size = len(glob(os.path.join(self.snapshot_dir, "*.zip")))

        if elapsed > 3600:
            elapsed_str = f"{elapsed/3600:.1f}h"
        else:
            elapsed_str = f"{elapsed/60:.0f}m"

        print(
            f"{self.log_prefix} [{ts}] "
            f"ep={self.episode_count:<5d} steps={self.num_timesteps:>9,} | "
            f"pool={pool_size:2d} | "
            f"WR={recent_wr:4.1f}% cum={wr:4.1f}% | "
            f"R={avg_r:+6.1f} | "
            f"{sps:.0f} sps {elapsed_str}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True,
                        choices=["main", "main_exploiter", "league_exploiter"])
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=50_000_000)
    parser.add_argument("--snapshot-dir", type=str, default="snapshots")
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--log-prefix", type=str, default="")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Create opponent function based on role
    opponent_fn = make_pool_opponent(args.snapshot_dir, args.role)

    # Create env with custom opponent
    env = ClashRoyaleEnv(opponent="none")
    env._opponent_fn = opponent_fn

    from clasher.network import CRFeatureExtractor
    policy_kwargs = dict(
        features_extractor_class=CRFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu")
        model.ent_coef = args.ent_coef
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=0,
            seed=args.seed,
            device="cpu",
            policy_kwargs=policy_kwargs,
            n_steps=512,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=args.ent_coef,
        )

    callback = LeagueCallback(
        role=args.role,
        log_prefix=args.log_prefix,
        log_every=args.log_every,
        save_every=args.save_every,
        snapshot_dir=args.snapshot_dir,
        ent_coef=args.ent_coef,
    )

    print(f"{args.log_prefix} Starting {args.role} (ent={args.ent_coef}, seed={args.seed})")

    model.learn(total_timesteps=args.timesteps, callback=callback)

    # Save final model
    final_path = os.path.join("models", f"league_{args.role}_final")
    os.makedirs("models", exist_ok=True)
    model.save(final_path)
    print(f"{args.log_prefix} Saved: {final_path}")


if __name__ == "__main__":
    main()
