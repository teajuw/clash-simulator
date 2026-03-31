#!/usr/bin/env python3 -u
"""V3 Training with Curriculum Learning + Minimax.

4-level curriculum:
  Level 0: No opponent (learn to attack undefended towers)
  Level 1: Predictable bot (learn basic defense)
  Level 2: Self-play (learn to beat adaptive opponents)
  Level 3: Self-play + Minimax (break plateaus)

Usage:
    python scripts/train_v3.py --total-episodes 10000
    python scripts/train_v3.py --total-episodes 50000 --start-level 2
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
from clasher.curriculum_bot import make_curriculum_bot_policy
from clasher.training_bot import training_bot_policy
from clasher.entities import Troop, Building

SNAPSHOT_DIR = "snapshots_v3"
POLICY_KWARGS = dict(
    features_extractor_class=CRFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
)
STEPS_PER_EP = 400  # approximate


# ── Callbacks ─────────────────────────────────────────────────────────────────

class TrainCallback(BaseCallback):
    def __init__(self, name: str, log_every: int = 100):
        super().__init__(verbose=0)
        self.name = name
        self.log_every = log_every
        self.ep = 0
        self.wins = 0
        self.losses = 0
        self.rewards = []
        self._r = 0.0
        self._t0 = time.time()
        self._start_steps = None

    def _on_training_start(self):
        self._start_steps = self.num_timesteps

    def _on_step(self):
        self._r += self.locals.get("rewards", [0])[0]
        if self.locals.get("dones", [False])[0]:
            self.ep += 1
            self.rewards.append(self._r)
            self._r = 0.0
            w = self.locals.get("infos", [{}])[0].get("winner")
            if w == 0: self.wins += 1
            elif w == 1: self.losses += 1

            if self.ep % self.log_every == 0:
                total = self.wins + self.losses
                wr = self.wins / total * 100 if total > 0 else 0
                n = min(self.log_every, len(self.rewards))
                avg_r = np.mean(self.rewards[-n:])
                elapsed = time.time() - self._t0
                steps = self.num_timesteps - (self._start_steps or 0)
                sps = steps / elapsed if elapsed > 0 else 0
                ts = datetime.now().strftime("%H:%M:%S")
                el = f"{elapsed/60:.0f}m" if elapsed < 3600 else f"{elapsed/3600:.1f}h"
                print(f"  [{self.name}] [{ts}] ep={self.ep:<5d} | "
                      f"WR={wr:4.1f}% | R={avg_r:+6.1f} | {sps:.0f} sps {el}")
        return True


def benchmark(model, opponent="training_bot", n=10) -> int:
    """Returns win percentage."""
    env = ClashRoyaleEnvV3(opponent=opponent if opponent != "none" else "none",
                           domain_randomization=False)
    wins = 0
    for i in range(n):
        obs, _ = env.reset(seed=i + 1000)
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            if term:
                if info["winner"] == 0: wins += 1
                break
    return wins * (100 // n)


# ── Opponent for self-play ────────────────────────────────────────────────────

def load_opponent_model(path):
    try:
        return PPO.load(path.replace(".zip", ""))
    except Exception:
        return None


def make_selfplay_opponent(snapshot_dir, rng):
    """50% latest mmax, 25% main_best, 25% random history."""
    state = {"model": None, "reload_at": 0}

    def policy(battle):
        state["reload_at"] += 1
        if state["model"] is None or state["reload_at"] % 10 == 0:
            roll = rng.random()
            if roll < 0.50:
                mmax_zips = sorted(glob(os.path.join(snapshot_dir, "mmax_*.zip")),
                                   key=os.path.getmtime)
                path = mmax_zips[-1] if mmax_zips else None
            elif roll < 0.75:
                best = os.path.join(snapshot_dir, "main_best.zip")
                path = best if os.path.exists(best) else None
            else:
                history = [z for z in glob(os.path.join(snapshot_dir, "*.zip"))
                          if "main_best" not in z and "main_latest" not in z]
                path = rng.choice(history) if history else None

            if path:
                state["model"] = load_opponent_model(path)

        if state["model"] is None:
            return

        obs = _mirror_obs(battle)
        try:
            action, _ = state["model"].predict(obs, deterministic=False)
            c, tx, ty = int(action[0]), int(action[1]), int(action[2])
            if c > 0:
                slot = c - 1
                player = battle.players[1]
                if slot < len(player.hand):
                    from clasher.arena import Position
                    pos = Position(tx + 0.5, 31 - ty + 0.5)
                    battle.deploy_card(1, player.hand[slot], pos)
        except Exception:
            pass

    return policy


def make_mmax_opponent(snapshot_dir):
    """Always plays main_latest."""
    state = {"model": None, "mtime": 0}

    def policy(battle):
        latest = os.path.join(snapshot_dir, "main_latest.zip")
        if not os.path.exists(latest):
            return
        mtime = os.path.getmtime(latest)
        if state["model"] is None or mtime > state["mtime"]:
            state["model"] = load_opponent_model(latest)
            state["mtime"] = mtime

        if state["model"] is None:
            return

        obs = _mirror_obs(battle)
        try:
            action, _ = state["model"].predict(obs, deterministic=False)
            c, tx, ty = int(action[0]), int(action[1]), int(action[2])
            if c > 0:
                slot = c - 1
                player = battle.players[1]
                if slot < len(player.hand):
                    from clasher.arena import Position
                    pos = Position(tx + 0.5, 31 - ty + 0.5)
                    battle.deploy_card(1, player.hand[slot], pos)
        except Exception:
            pass

    return policy


def _mirror_obs(battle):
    from clasher.env_v3 import N_SCALARS, ARENA_W, ARENA_H, NUM_CARD_IDS
    from clasher.env_v3 import PRINCESS_TOWER_ID, KING_TOWER_ID
    from clasher.env_v3 import MAX_KING_HP, MAX_PRINCESS_HP, MAX_GAME_TIME

    obs = {
        "spatial": np.zeros((3, ARENA_H, ARENA_W), dtype=np.float32),
        "scalars": np.zeros(N_SCALARS, dtype=np.float32),
    }
    p0 = battle.players[1]
    p1 = battle.players[0]

    for entity in battle.entities.values():
        if not entity.is_alive or not isinstance(entity, (Troop, Building)):
            continue
        gx = max(0, min(ARENA_W - 1, int(entity.position.x)))
        gy = max(0, min(ARENA_H - 1, ARENA_H - 1 - int(entity.position.y)))

        if isinstance(entity, Building) and entity.position.y in (2.5, 29.5):
            cid = KING_TOWER_ID
        elif isinstance(entity, Building) and entity.position.y in (6.5, 25.5):
            cid = PRINCESS_TOWER_ID
        elif entity.card_stats:
            cid = CARD_TO_IDX.get(entity.card_stats.name, 0) + 1
        else:
            cid = 0
        obs["spatial"][0, gy, gx] = cid / NUM_CARD_IDS
        obs["spatial"][1, gy, gx] = entity.hitpoints / entity.max_hitpoints if entity.max_hitpoints > 0 else 0.0
        obs["spatial"][2, gy, gx] = 0.0 if entity.player_id == 1 else 1.0

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
    s[9] = 0.0 if not battle.double_elixir else 0.5 if not battle.triple_elixir else 1.0
    s[10:18] = 0.5
    s[18:26] = 0.5

    return obs


# ── Minimax ───────────────────────────────────────────────────────────────────

class MinimaxReward:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.critic = None

    def load(self, path):
        try:
            m = PPO.load(path.replace(".zip", ""))
            self.critic = m.policy
        except Exception:
            pass

    def bonus(self, obs):
        if self.critic is None:
            return 0.0
        try:
            with torch.no_grad():
                st = torch.FloatTensor(obs["spatial"]).unsqueeze(0)
                sc = torch.FloatTensor(obs["scalars"]).unsqueeze(0)
                feats = self.critic.extract_features({"spatial": st, "scalars": sc},
                                                     self.critic.features_extractor)
                v = self.critic.value_net(
                    self.critic.mlp_extractor.forward_critic(feats)).item()
            return min(0.0, -self.alpha * v)
        except Exception:
            return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-episodes", type=int, default=10000)
    parser.add_argument("--round-size", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-level", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    LEVELS = [
        {"name": "NO OPPONENT",      "bench_target": 90, "bench_opponent": "none"},
        {"name": "PREDICTABLE BOT",  "bench_target": 70, "bench_opponent": "training_bot"},
        {"name": "SELF-PLAY",        "bench_target": None, "bench_opponent": "training_bot"},
        {"name": "SELF-PLAY+MINIMAX","bench_target": None, "bench_opponent": "training_bot"},
    ]

    print("=" * 65)
    print("  CLASH ROYALE RL V3 — CURRICULUM + MINIMAX")
    print("=" * 65)
    print("  Level 0: No opponent         → advance at bench ≥ 90%")
    print("  Level 1: Predictable bot     → advance at bench ≥ 70%")
    print("  Level 2: Self-play           → advance when Elo plateaus")
    print("  Level 3: Self-play + Minimax → final stage")
    print("=" * 65)
    print()

    # Create env and model
    env = ClashRoyaleEnvV3(opponent="none", domain_randomization=False)

    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu")
        print(f"Resumed from {args.resume}")
    else:
        model = PPO("MultiInputPolicy", env, verbose=0, seed=args.seed,
                     device="cpu", policy_kwargs=POLICY_KWARGS,
                     n_steps=512, batch_size=64, n_epochs=4,
                     gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
                     clip_range=0.2, ent_coef=0.025)

    # MMAX model (created when needed)
    mmax_model = None
    minimax = MinimaxReward(alpha=0.01)

    level = args.start_level
    total_eps = 0
    bench_history = []
    best_bench = 0
    round_num = 0
    stagnation_counter = 0
    prev_bench = 0

    t0 = time.time()

    while total_eps < args.total_episodes:
        lvl = LEVELS[level]
        round_num += 1

        # ── Set opponent based on level ───────────────────────────────
        if level == 0:
            env._opponent_fn = lambda b: None
            env._domain_rand.enabled = False
        elif level == 1:
            curriculum_bot = make_curriculum_bot_policy()
            env._opponent_fn = curriculum_bot
            env._domain_rand.enabled = False
        elif level == 2:
            env._opponent_fn = make_selfplay_opponent(SNAPSHOT_DIR, rng)
            env._domain_rand.enabled = True
        elif level == 3:
            env._opponent_fn = make_selfplay_opponent(SNAPSHOT_DIR, rng)
            env._domain_rand.enabled = True

        # ── MMAX round (level 3 only) ────────────────────────────────
        if level == 3:
            print(f"── L{level} Round {round_num}A: MMAX vs main_latest ──")

            if mmax_model is None:
                mmax_env = ClashRoyaleEnvV3(opponent="none", domain_randomization=False)
                mmax_model = PPO("MultiInputPolicy", mmax_env, verbose=0,
                                 seed=args.seed + 100, device="cpu",
                                 policy_kwargs=POLICY_KWARGS,
                                 n_steps=512, batch_size=64, n_epochs=4,
                                 gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
                                 clip_range=0.2, ent_coef=0.05)

            # Copy main weights, load main critic
            mmax_model.policy.load_state_dict(model.policy.state_dict())
            minimax.load(os.path.join(SNAPSHOT_DIR, "main_latest"))

            # Set MMAX opponent
            mmax_opp = make_mmax_opponent(SNAPSHOT_DIR)
            mmax_env = mmax_model.get_env().envs[0]
            while hasattr(mmax_env, 'env'):
                mmax_env = mmax_env.env
            mmax_env._opponent_fn = mmax_opp

            # Minimax step wrapper
            real_env = mmax_model.get_env().envs[0]
            while hasattr(real_env, 'env'):
                real_env = real_env.env
            orig_step = real_env.step
            def mm_step(action):
                obs, r, t, tr, info = orig_step(action)
                return obs, r + minimax.bonus(obs), t, tr, info
            real_env.step = mm_step

            cb = TrainCallback("MMAX", log_every=args.log_every)
            mmax_model.learn(total_timesteps=args.round_size * STEPS_PER_EP,
                            callback=cb, reset_num_timesteps=False)

            real_env.step = orig_step  # restore

            mmax_wr = cb.wins / max(1, cb.wins + cb.losses) * 100
            mmax_model.save(os.path.join(SNAPSHOT_DIR, f"mmax_{round_num}"))
            print(f"  MMAX saved (WR={mmax_wr:.0f}%)")

        # ── MAIN round ────────────────────────────────────────────────
        print(f"── L{level} Round {round_num}: MAIN — {lvl['name']} ──")

        cb = TrainCallback("MAIN", log_every=args.log_every)
        model.learn(total_timesteps=args.round_size * STEPS_PER_EP,
                    callback=cb, reset_num_timesteps=False)
        total_eps += cb.ep

        # Save snapshots
        model.save(os.path.join(SNAPSHOT_DIR, "main_latest"))
        model.save(os.path.join(SNAPSHOT_DIR, f"main_{round_num}"))

        # Benchmark
        b = benchmark(model, opponent=lvl["bench_opponent"])
        bench_history.append(b)
        bench_trend = "→".join(str(x) for x in bench_history[-6:])

        if b > best_bench:
            best_bench = b
            model.save(os.path.join(SNAPSHOT_DIR, "main_best"))
            print(f"  ★ New best! bench={b}%")

        elapsed = time.time() - t0
        el = f"{elapsed/60:.0f}m" if elapsed < 3600 else f"{elapsed/3600:.1f}h"
        pool = len(glob(os.path.join(SNAPSHOT_DIR, "*.zip")))
        print(f"  Bench: {b}% [{bench_trend}] | Pool: {pool} | "
              f"Total eps: {total_eps} | {el}")

        # ── Level advancement ─────────────────────────────────────────
        target = lvl["bench_target"]
        if target is not None and b >= target:
            # Check 3 consecutive passes
            recent = bench_history[-3:]
            if len(recent) >= 3 and all(x >= target for x in recent):
                level = min(level + 1, len(LEVELS) - 1)
                stagnation_counter = 0
                print(f"\n  ▲ ADVANCING TO LEVEL {level}: {LEVELS[level]['name']}\n")
        elif level == 2:
            # Self-play: advance to level 3 on stagnation
            if b <= prev_bench:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            if stagnation_counter >= 5:
                level = 3
                stagnation_counter = 0
                print(f"\n  ▲ ADVANCING TO LEVEL 3: SELF-PLAY + MINIMAX\n")

        prev_bench = b
        print()

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=" * 65)
    print(f"  DONE in {elapsed/60:.1f} min | Total eps: {total_eps}")
    print(f"  Best bench: {best_bench}% | Final bench: {bench_history[-1]}%")
    print(f"  Final level: {level} — {LEVELS[level]['name']}")
    print("=" * 65)
    model.save("models/v3_final")
    print("Saved: models/v3_final")


if __name__ == "__main__":
    main()
