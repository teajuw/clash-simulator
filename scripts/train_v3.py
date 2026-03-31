#!/usr/bin/env python3 -u
"""V3 Training: MAIN + Minimax Exploiter, alternating rounds.

Single process, synchronized co-evolution:
  Phase 1 (warmup):  MAIN trains vs training bot for 1000 eps
  Phase 2 (league):  Alternating 500-ep rounds:
                     MMAX trains vs main_latest → saves exploit
                     MAIN trains vs pool (50% latest_mmax, 25% main_best, 25% history)

Usage:
    python scripts/train_v3.py --total-episodes 10000
    python scripts/train_v3.py --total-episodes 50000 --warmup 2000
"""

import argparse
import functools
import os
import sys
import time
from glob import glob
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

print = functools.partial(print, flush=True)

import numpy as np
import torch
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import clasher.config as _cfg
_cfg.VERBOSE = False

from clasher.env_v3 import ClashRoyaleEnvV3, DECK, ELIXIR_COST, CARD_TO_IDX
from clasher.network import CRFeatureExtractor
from clasher.training_bot import training_bot_policy


SNAPSHOT_DIR = "snapshots_v3"
POLICY_KWARGS = dict(
    features_extractor_class=CRFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
)


# ── Opponent Functions ────────────────────────────────────────────────────────

def make_main_opponent(snapshot_dir: str, rng: np.random.Generator):
    """MAIN's opponent: 50% latest mmax, 25% main_best, 25% random history."""

    def opponent_fn(battle):
        roll = rng.random()

        if roll < 0.50:
            # Latest minimax snapshot
            mmax_zips = sorted(glob(os.path.join(snapshot_dir, "mmax_*.zip")),
                               key=lambda f: os.path.getmtime(f))
            if mmax_zips:
                path = mmax_zips[-1].replace(".zip", "")
            else:
                training_bot_policy(battle)
                return
        elif roll < 0.75:
            # main_best
            best_path = os.path.join(snapshot_dir, "main_best.zip")
            if os.path.exists(best_path):
                path = best_path.replace(".zip", "")
            else:
                training_bot_policy(battle)
                return
        else:
            # Random from all history
            all_zips = glob(os.path.join(snapshot_dir, "*.zip"))
            # Exclude main_best and main_latest from random (they have dedicated slots)
            history = [z for z in all_zips
                       if "main_best" not in z and "main_latest" not in z]
            if history:
                path = rng.choice(history).replace(".zip", "")
            else:
                training_bot_policy(battle)
                return

        _play_as_opponent(battle, path)

    return opponent_fn


def make_mmax_opponent(snapshot_dir: str):
    """MMAX's opponent: always main_latest."""
    state = {"model": None, "path": None}

    def opponent_fn(battle):
        latest = os.path.join(snapshot_dir, "main_latest.zip")
        if not os.path.exists(latest):
            training_bot_policy(battle)
            return
        # Reload if changed
        if state["path"] != latest or state["model"] is None:
            try:
                state["model"] = PPO.load(latest.replace(".zip", ""))
                state["path"] = latest
            except Exception:
                training_bot_policy(battle)
                return
        _play_as_opponent(battle, None, model=state["model"])

    return opponent_fn


def _play_as_opponent(battle, path: str = None, model=None):
    """Play as player 1 using a loaded model."""
    if model is None and path is not None:
        try:
            model = PPO.load(path)
        except Exception:
            training_bot_policy(battle)
            return

    if model is None:
        training_bot_policy(battle)
        return

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
                mirrored_y = 31 - tile_y
                from clasher.arena import Position
                pos = Position(tile_x + 0.5, mirrored_y + 0.5)
                battle.deploy_card(1, card_name, pos)
    except Exception:
        training_bot_policy(battle)


def _mirror_obs(battle):
    """Build observation from player 1's perspective."""
    from clasher.env_v3 import N_SCALARS, ARENA_W, ARENA_H, NUM_CARD_IDS
    from clasher.env_v3 import PRINCESS_TOWER_ID, KING_TOWER_ID, MAX_KING_HP, MAX_PRINCESS_HP, MAX_GAME_TIME
    from clasher.entities import Troop, Building

    obs = {
        "spatial": np.zeros((3, ARENA_H, ARENA_W), dtype=np.float32),
        "scalars": np.zeros(N_SCALARS, dtype=np.float32),
    }

    p0 = battle.players[1]  # P1 sees itself as P0
    p1 = battle.players[0]

    # Spatial — flip y, swap owner
    for entity in battle.entities.values():
        if not entity.is_alive or not isinstance(entity, (Troop, Building)):
            continue
        gx = max(0, min(ARENA_W - 1, int(entity.position.x)))
        gy = max(0, min(ARENA_H - 1, ARENA_H - 1 - int(entity.position.y)))

        if isinstance(entity, Building) and entity.position.y in (2.5, 29.5):
            card_id = KING_TOWER_ID
        elif isinstance(entity, Building) and entity.position.y in (6.5, 25.5):
            card_id = PRINCESS_TOWER_ID
        elif entity.card_stats:
            card_id = CARD_TO_IDX.get(entity.card_stats.name, 0) + 1
        else:
            card_id = 0

        obs["spatial"][0, gy, gx] = card_id / NUM_CARD_IDS
        obs["spatial"][1, gy, gx] = entity.hitpoints / entity.max_hitpoints if entity.max_hitpoints > 0 else 0.0
        obs["spatial"][2, gy, gx] = 0.0 if entity.player_id == 1 else 1.0  # swap

    # Scalars
    s = obs["scalars"]
    s[0] = p0.elixir / 10.0
    s[1] = p1.elixir / 10.0
    s[2] = max(0, p0.king_tower_hp) / MAX_KING_HP
    s[3] = max(0, p0.left_tower_hp) / MAX_PRINCESS_HP
    s[4] = max(0, p0.right_tower_hp) / MAX_PRINCESS_HP
    s[5] = max(0, p1.king_tower_hp) / MAX_KING_HP
    s[6] = max(0, p1.left_tower_hp) / MAX_PRINCESS_HP
    s[7] = max(0, p1.right_tower_hp) / MAX_PRINCESS_HP
    s[8] = min(battle.time / MAX_GAME_TIME, 1.0)
    s[9] = (0.0 if not battle.double_elixir
            else 0.5 if not battle.triple_elixir else 1.0)
    # Card tracker not mirrored — opponent doesn't get perfect tracking
    # Fill with 0.5 (unknown) for simplicity
    s[10:18] = 0.5
    s[18:26] = 0.5

    return obs


# ── Minimax Reward Wrapper ────────────────────────────────────────────────────

class MinimaxRewardWrapper:
    """Wraps env to add minimax term: R = R_normal - α * V_main(next_state)."""

    def __init__(self, env, alpha: float = 0.01):
        self.env = env
        self.alpha = alpha
        self.main_critic = None  # loaded externally

    def load_critic(self, main_model_path: str):
        try:
            model = PPO.load(main_model_path)
            self.main_critic = model.policy
        except Exception:
            pass

    def compute_minimax_bonus(self, obs: dict) -> float:
        if self.main_critic is None:
            return 0.0
        try:
            with torch.no_grad():
                spatial_t = torch.FloatTensor(obs["spatial"]).unsqueeze(0)
                scalar_t = torch.FloatTensor(obs["scalars"]).unsqueeze(0)
                obs_t = {"spatial": spatial_t, "scalars": scalar_t}
                features = self.main_critic.extract_features(obs_t, self.main_critic.features_extractor)
                value = self.main_critic.value_net(self.main_critic.mlp_extractor.forward_critic(features))
                v = value.item()
            return min(0.0, -self.alpha * v)
        except Exception:
            return 0.0


# ── Training Loop ─────────────────────────────────────────────────────────────

class RoundCallback(BaseCallback):
    """Tracks wins/losses per round with split stats."""

    def __init__(self, agent_name: str, log_every: int = 100):
        super().__init__(verbose=0)
        self.agent_name = agent_name
        self.log_every = log_every
        self.episode_count = 0
        self.wins = 0
        self.losses = 0
        self.episode_rewards = []
        self._current_reward = 0.0
        self._start_time = time.time()

    def _on_step(self):
        self._current_reward += self.locals.get("rewards", [0])[0]
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_count += 1
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
            infos = self.locals.get("infos", [{}])
            winner = infos[0].get("winner")
            if winner == 0:
                self.wins += 1
            elif winner == 1:
                self.losses += 1

            if self.episode_count % self.log_every == 0:
                self._display()
        return True

    def _display(self):
        ts = datetime.now().strftime("%H:%M:%S")
        total = self.wins + self.losses
        wr = self.wins / total * 100 if total > 0 else 0
        n = min(self.log_every, len(self.episode_rewards))
        avg_r = np.mean(self.episode_rewards[-n:]) if self.episode_rewards else 0
        elapsed = time.time() - self._start_time
        sps = self.num_timesteps / elapsed if elapsed > 0 else 0
        el = f"{elapsed/60:.0f}m" if elapsed < 3600 else f"{elapsed/3600:.1f}h"
        print(f"[{self.agent_name}] [{ts}] ep={self.episode_count:<5d} "
              f"steps={self.num_timesteps:>9,} | WR={wr:4.1f}% | "
              f"R={avg_r:+6.1f} | {sps:.0f} sps {el}")


def benchmark(model, n_games: int = 10) -> int:
    """Benchmark MAIN vs training bot. Returns win percentage."""
    env = ClashRoyaleEnvV3(opponent="training_bot", domain_randomization=False)
    wins = 0
    for _ in range(n_games):
        obs, _ = env.reset()
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            if term:
                if info["winner"] == 0:
                    wins += 1
                break
    return wins * 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-episodes", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=1000,
                        help="Episodes of MAIN-only warmup vs training bot")
    parser.add_argument("--round-size", type=int, default=500,
                        help="Episodes per alternating round")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 65)
    print("  CLASH ROYALE RL V3 — MAIN + MINIMAX (Alternating Rounds)")
    print("=" * 65)
    print(f"  Total episodes: {args.total_episodes:,}")
    print(f"  Warmup: {args.warmup} eps vs training bot")
    print(f"  Round size: {args.round_size} eps")
    print(f"  MAIN: ent=0.025, 50% latest_mmax / 25% main_best / 25% history")
    print(f"  MMAX: ent=0.05, minimax reward, vs main_latest only")
    print("=" * 65)
    print()

    # ── Create models ─────────────────────────────────────────────────────

    # MAIN
    main_env = ClashRoyaleEnvV3(opponent="training_bot")
    if args.resume:
        main_model = PPO.load(args.resume, env=main_env, device="cpu")
        print(f"Resumed MAIN from {args.resume}")
    else:
        main_model = PPO("MultiInputPolicy", main_env, verbose=0, seed=args.seed,
                         device="cpu", policy_kwargs=POLICY_KWARGS,
                         n_steps=512, batch_size=64, n_epochs=4,
                         gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
                         clip_range=0.2, ent_coef=0.025)

    # MMAX
    mmax_env = ClashRoyaleEnvV3(opponent="none")
    mmax_model = PPO("MultiInputPolicy", mmax_env, verbose=0, seed=args.seed + 100,
                     device="cpu", policy_kwargs=POLICY_KWARGS,
                     n_steps=512, batch_size=64, n_epochs=4,
                     gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
                     clip_range=0.2, ent_coef=0.05)

    minimax_wrapper = MinimaxRewardWrapper(mmax_env, alpha=0.01)
    bench_history = []
    best_bench = 0
    total_main_eps = 0
    total_mmax_eps = 0

    t0 = time.time()

    # ── Phase 1: Warmup ───────────────────────────────────────────────────

    print(f"── PHASE 1: Warmup ({args.warmup} eps vs training bot) ──")
    warmup_steps = args.warmup * 400  # ~400 steps/ep average
    cb = RoundCallback("MAIN warmup", log_every=args.log_every)
    main_model.learn(total_timesteps=warmup_steps, callback=cb, reset_num_timesteps=False)
    total_main_eps += cb.episode_count

    # Save first snapshots
    main_model.save(os.path.join(SNAPSHOT_DIR, "main_latest"))
    main_model.save(os.path.join(SNAPSHOT_DIR, "main_0"))
    b = benchmark(main_model)
    bench_history.append(b)
    if b > best_bench:
        best_bench = b
        main_model.save(os.path.join(SNAPSHOT_DIR, "main_best"))
    print(f"── Warmup done. Bench: {b}% ──")
    print()

    # ── Phase 2: Alternating Rounds ───────────────────────────────────────

    round_num = 0
    eps_so_far = args.warmup

    while eps_so_far < args.total_episodes:
        round_num += 1
        round_steps = args.round_size * 400  # ~400 steps/ep average, overshoot to ensure full rounds

        # ── MMAX round ────────────────────────────────────────────────
        print(f"── Round {round_num}A: MMAX vs main_latest ({args.round_size} eps) ──")

        # Load main's critic for minimax reward
        minimax_wrapper.load_critic(os.path.join(SNAPSHOT_DIR, "main_latest"))

        # Copy main's weights into MMAX (fresh start each round)
        mmax_model.policy.load_state_dict(main_model.policy.state_dict())

        # Set MMAX opponent to main_latest
        mmax_opponent = make_mmax_opponent(SNAPSHOT_DIR)
        mmax_env._opponent_fn = mmax_opponent

        # Custom step wrapper for minimax reward
        original_step = mmax_env.step
        def minimax_step(action):
            obs, reward, terminated, truncated, info = original_step(action)
            bonus = minimax_wrapper.compute_minimax_bonus(obs)
            return obs, reward + bonus, terminated, truncated, info
        mmax_env.step = minimax_step

        cb_mmax = RoundCallback("MMAX", log_every=args.log_every)
        mmax_model.learn(total_timesteps=round_steps, callback=cb_mmax, reset_num_timesteps=False)
        total_mmax_eps += cb_mmax.episode_count

        # Restore original step
        mmax_env.step = original_step

        # Always save MMAX — even 50% WR contains useful minimax-targeted strategies
        mmax_wr = cb_mmax.wins / max(1, cb_mmax.wins + cb_mmax.losses) * 100
        mmax_model.save(os.path.join(SNAPSHOT_DIR, f"mmax_{round_num}"))
        print(f"  MMAX saved (WR={mmax_wr:.0f}%)")

        # ── MAIN round ────────────────────────────────────────────────
        print(f"── Round {round_num}B: MAIN vs pool ({args.round_size} eps) ──")

        # Set MAIN opponent to pool mix
        main_opponent = make_main_opponent(SNAPSHOT_DIR, rng)
        main_env._opponent_fn = main_opponent

        cb_main = RoundCallback("MAIN", log_every=args.log_every)
        main_model.learn(total_timesteps=round_steps, callback=cb_main, reset_num_timesteps=False)
        total_main_eps += cb_main.episode_count

        # Save snapshots
        main_model.save(os.path.join(SNAPSHOT_DIR, "main_latest"))
        main_model.save(os.path.join(SNAPSHOT_DIR, f"main_{round_num}"))

        # Benchmark
        b = benchmark(main_model)
        bench_history.append(b)
        bench_trend = "→".join(f"{x}" for x in bench_history[-6:])

        if b > best_bench:
            best_bench = b
            main_model.save(os.path.join(SNAPSHOT_DIR, "main_best"))
            print(f"  ★ New best! bench={b}%")

        elapsed = time.time() - t0
        el = f"{elapsed/60:.0f}m" if elapsed < 3600 else f"{elapsed/3600:.1f}h"
        print(f"  Bench: {b}% [{bench_trend}] | MMAX WR: {mmax_wr:.0f}% | "
              f"Pool: {len(glob(os.path.join(SNAPSHOT_DIR, '*.zip')))} | {el}")
        print()

        eps_so_far += args.round_size * 2  # both rounds

    # ── Final ─────────────────────────────────────────────────────────────

    elapsed = time.time() - t0
    print("=" * 65)
    print(f"  DONE in {elapsed/60:.1f} min")
    print(f"  MAIN episodes: {total_main_eps}")
    print(f"  MMAX episodes: {total_mmax_eps}")
    print(f"  Best bench: {best_bench}%")
    print(f"  Final bench: {bench_history[-1] if bench_history else '?'}%")
    print("=" * 65)

    main_model.save("models/v3_main_final")
    print("Saved: models/v3_main_final")


if __name__ == "__main__":
    main()
