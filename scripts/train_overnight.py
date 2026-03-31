#!/usr/bin/env python3 -u
"""Overnight training pipeline: kickstart → self-play → minimax.

Phase 1: Expert kickstart (1000 eps)
  - Smart bot controls 100% of actions for 500 eps
  - Linear decay to 0% over next 500 eps
  - Agent learns from expert's rewards

Phase 2: Self-play (until stagnation)
  - Agent vs own snapshots (50% best, 50% history)
  - Domain randomization ON
  - Benchmark vs smart bot every round

Phase 3: Self-play + Minimax (final)
  - Alternating MMAX rounds
  - Minimax reward using main's critic

Usage:
    python scripts/train_overnight.py --total-episodes 20000
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

from clasher.env_v3 import ClashRoyaleEnvV3
from clasher.network import CRFeatureExtractor
from clasher.kickstart import KickstartWrapper
from clasher.smart_bot import SmartBot, make_smart_bot_policy

SNAPSHOT_DIR = "snapshots_v3"
POLICY_KWARGS = dict(
    features_extractor_class=CRFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
)
STEPS_PER_EP = 400


class LogCallback(BaseCallback):
    def __init__(self, name, log_every=100):
        super().__init__(verbose=0)
        self.name = name
        self.log_every = log_every
        self.ep = 0
        self.wins = 0
        self.losses = 0
        self.winners = []
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
            self.winners.append(w)
            if w == 0: self.wins += 1
            elif w == 1: self.losses += 1

            if self.ep % self.log_every == 0:
                recent = self.winners[-100:]
                wr = sum(1 for w in recent if w == 0) / len(recent) * 100 if recent else 0
                avg_r = np.mean(self.rewards[-self.log_every:])
                elapsed = time.time() - self._t0
                steps = self.num_timesteps - (self._start_steps or 0)
                sps = steps / elapsed if elapsed > 0 else 0
                ts = datetime.now().strftime("%H:%M:%S")
                el = f"{elapsed/60:.0f}m" if elapsed < 3600 else f"{elapsed/3600:.1f}h"
                print(f"  [{self.name}] [{ts}] ep={self.ep:<5d} | "
                      f"WR={wr:4.1f}% | R={avg_r:+6.1f} | {sps:.0f} sps {el}")
        return True

    def last100_wr(self):
        recent = self.winners[-100:]
        return sum(1 for w in recent if w == 0) / len(recent) * 100 if recent else 0


def benchmark_vs_smart_bot(model, n=20):
    env = ClashRoyaleEnvV3(opponent="none", domain_randomization=False)
    env._opponent_fn = make_smart_bot_policy(player_id=1)
    wins = 0
    for i in range(n):
        obs, _ = env.reset(seed=i + 5000)
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            if term:
                if info["winner"] == 0: wins += 1
                break
    return wins * (100 // n)


def make_selfplay_opponent(snapshot_dir, rng):
    """50% main_best, 50% random history."""
    state = {"model": None, "reload": 0}

    def policy(battle):
        state["reload"] += 1
        if state["model"] is None or state["reload"] % 10 == 0:
            zips = glob(os.path.join(snapshot_dir, "*.zip"))
            if not zips:
                return
            if rng.random() < 0.5:
                best = os.path.join(snapshot_dir, "main_best.zip")
                path = best if os.path.exists(best) else rng.choice(zips)
            else:
                path = rng.choice(zips)
            try:
                state["model"] = PPO.load(path.replace(".zip", ""))
            except Exception:
                return

        if state["model"] is None:
            return

        from clasher.entities import Troop, Building
        obs = _quick_mirror_obs(battle)
        try:
            action, _ = state["model"].predict(obs, deterministic=False)
            c, tx, ty = int(action[0]), int(action[1]), int(action[2])
            if c > 0:
                slot = c - 1
                player = battle.players[1]
                if slot < len(player.hand):
                    from clasher.arena import Position
                    battle.deploy_card(1, player.hand[slot],
                                      Position(tx + 0.5, 31 - ty + 0.5))
        except Exception:
            pass

    return policy


def _quick_mirror_obs(battle):
    """Minimal mirrored observation for opponent snapshots."""
    from clasher.env_v3 import N_SCALARS, ARENA_W, ARENA_H, NUM_CARD_IDS
    from clasher.env_v3 import KING_TOWER_ID, PRINCESS_TOWER_ID
    from clasher.env_v3 import MAX_KING_HP, MAX_PRINCESS_HP, MAX_GAME_TIME
    from clasher.entities import Troop, Building

    obs = {
        "spatial": np.zeros((3, ARENA_H, ARENA_W), dtype=np.float32),
        "scalars": np.zeros(N_SCALARS, dtype=np.float32),
    }
    p0, p1 = battle.players[1], battle.players[0]

    for e in battle.entities.values():
        if not e.is_alive or not isinstance(e, (Troop, Building)):
            continue
        gx = max(0, min(ARENA_W-1, int(e.position.x)))
        gy = max(0, min(ARENA_H-1, ARENA_H-1-int(e.position.y)))
        if isinstance(e, Building) and e.position.y in (2.5,29.5):
            cid = KING_TOWER_ID
        elif isinstance(e, Building) and e.position.y in (6.5,25.5):
            cid = PRINCESS_TOWER_ID
        elif e.card_stats:
            from clasher.env_v3 import CARD_TO_IDX
            cid = CARD_TO_IDX.get(e.card_stats.name, 0) + 1
        else:
            cid = 0
        obs["spatial"][0,gy,gx] = cid / NUM_CARD_IDS
        obs["spatial"][1,gy,gx] = e.hitpoints / e.max_hitpoints if e.max_hitpoints > 0 else 0
        obs["spatial"][2,gy,gx] = 0.0 if e.player_id == 1 else 1.0

    s = obs["scalars"]
    s[0] = p0.elixir / 10.0
    s[1] = p1.elixir / 10.0
    s[2:8] = [max(0,p0.king_tower_hp)/MAX_KING_HP, max(0,p0.left_tower_hp)/MAX_PRINCESS_HP,
              max(0,p0.right_tower_hp)/MAX_PRINCESS_HP, max(0,p1.king_tower_hp)/MAX_KING_HP,
              max(0,p1.left_tower_hp)/MAX_PRINCESS_HP, max(0,p1.right_tower_hp)/MAX_PRINCESS_HP]
    s[8] = min(battle.time / MAX_GAME_TIME, 1.0)
    s[9] = 0.0 if not battle.double_elixir else 0.5 if not battle.triple_elixir else 1.0
    s[10:26] = 0.5
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-episodes", type=int, default=20000)
    parser.add_argument("--kickstart-eps", type=int, default=500)
    parser.add_argument("--decay-eps", type=int, default=2000)
    parser.add_argument("--round-size", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 65)
    print("  OVERNIGHT PIPELINE: KICKSTART → SELF-PLAY → MINIMAX")
    print("=" * 65)
    print(f"  Total: {args.total_episodes:,} episodes")
    print(f"  Phase 1: Expert kickstart ({args.kickstart_eps}+{args.decay_eps} eps)")
    print(f"  Phase 2: Self-play (until stagnation)")
    print(f"  Phase 3: Self-play + Minimax (final)")
    print("=" * 65)
    print()

    # ── Phase 1: Kickstart ────────────────────────────────────────────
    print("═══ PHASE 1: EXPERT KICKSTART ═══")

    base_env = ClashRoyaleEnvV3(opponent="none", domain_randomization=False)
    base_env._opponent_fn = make_smart_bot_policy(player_id=1)

    env = KickstartWrapper(base_env,
                           expert_episodes=args.kickstart_eps,
                           decay_episodes=args.decay_eps)

    model = PPO("MultiInputPolicy", env, verbose=0, seed=args.seed,
                device="cpu", policy_kwargs=POLICY_KWARGS,
                n_steps=512, batch_size=64, n_epochs=4,
                gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
                clip_range=0.2, ent_coef=0.025)

    kickstart_steps = (args.kickstart_eps + args.decay_eps) * STEPS_PER_EP
    cb = LogCallback("KICK", log_every=args.log_every)
    model.learn(total_timesteps=kickstart_steps, callback=cb)

    total_eps = cb.ep

    # Save initial snapshots
    model.save(os.path.join(SNAPSHOT_DIR, "main_latest"))
    model.save(os.path.join(SNAPSHOT_DIR, "main_best"))
    model.save(os.path.join(SNAPSHOT_DIR, "main_0"))

    b = benchmark_vs_smart_bot(model)
    best_bench = b
    bench_history = [b]
    print(f"\n  Kickstart done. Bench vs smart bot: {b}%")
    print(f"  Episodes: {total_eps}")
    print()

    # ── Phase 2: Self-play ────────────────────────────────────────────
    print("═══ PHASE 2: SELF-PLAY ═══")

    # Unwrap kickstart, switch to self-play opponent
    selfplay_env = ClashRoyaleEnvV3(opponent="none", domain_randomization=True)
    selfplay_env._opponent_fn = make_selfplay_opponent(SNAPSHOT_DIR, rng)
    model.set_env(selfplay_env)

    stagnation = 0
    round_num = 0

    while total_eps < args.total_episodes:
        round_num += 1
        round_steps = args.round_size * STEPS_PER_EP

        print(f"── Self-play Round {round_num} ──")
        cb = LogCallback("SELF", log_every=args.log_every)
        model.learn(total_timesteps=round_steps, callback=cb, reset_num_timesteps=False)
        total_eps += cb.ep

        # Save
        model.save(os.path.join(SNAPSHOT_DIR, "main_latest"))
        model.save(os.path.join(SNAPSHOT_DIR, f"main_{round_num}"))

        # Benchmark
        b = benchmark_vs_smart_bot(model)
        bench_history.append(b)
        trend = "→".join(str(x) for x in bench_history[-6:])

        if b > best_bench:
            best_bench = b
            model.save(os.path.join(SNAPSHOT_DIR, "main_best"))
            print(f"  ★ New best! bench={b}%")

        pool = len(glob(os.path.join(SNAPSHOT_DIR, "*.zip")))
        elapsed_total = time.time()
        print(f"  Bench: {b}% [{trend}] | Pool: {pool} | Total eps: {total_eps}")

        # Stagnation check → advance to minimax
        if b <= (bench_history[-2] if len(bench_history) > 1 else 0):
            stagnation += 1
        else:
            stagnation = 0

        if stagnation >= 5:
            print(f"\n  ▲ Stagnated. Advancing to MINIMAX phase.\n")
            break

        print()

    # ── Phase 3: Self-play + Minimax ──────────────────────────────────
    if total_eps < args.total_episodes:
        print("═══ PHASE 3: SELF-PLAY + MINIMAX ═══")

        from clasher.env_v3 import ClashRoyaleEnvV3 as V3

        mmax_env = V3(opponent="none", domain_randomization=False)
        mmax_model = PPO("MultiInputPolicy", mmax_env, verbose=0,
                         seed=args.seed + 100, device="cpu",
                         policy_kwargs=POLICY_KWARGS,
                         n_steps=512, batch_size=64, n_epochs=4,
                         gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
                         clip_range=0.2, ent_coef=0.05)

        while total_eps < args.total_episodes:
            round_num += 1

            # MMAX round: copy main, play vs main_latest
            print(f"── Minimax Round {round_num}A: MMAX ──")
            mmax_model.policy.load_state_dict(model.policy.state_dict())

            # MMAX opponent = main_latest
            mmax_opp_state = {"model": None}
            def mmax_opp(battle):
                latest = os.path.join(SNAPSHOT_DIR, "main_latest.zip")
                if os.path.exists(latest) and mmax_opp_state["model"] is None:
                    try:
                        mmax_opp_state["model"] = PPO.load(latest.replace(".zip",""))
                    except: pass
                if mmax_opp_state["model"]:
                    obs = _quick_mirror_obs(battle)
                    try:
                        act, _ = mmax_opp_state["model"].predict(obs, deterministic=False)
                        c = int(act[0])
                        if c > 0:
                            p = battle.players[1]
                            if c-1 < len(p.hand):
                                from clasher.arena import Position
                                battle.deploy_card(1, p.hand[c-1],
                                    Position(int(act[1])+0.5, 31-int(act[2])+0.5))
                    except: pass

            real_mmax_env = mmax_model.get_env().envs[0]
            while hasattr(real_mmax_env, 'env'):
                real_mmax_env = real_mmax_env.env
            real_mmax_env._opponent_fn = mmax_opp

            cb = LogCallback("MMAX", log_every=args.log_every)
            mmax_model.learn(total_timesteps=round_steps, callback=cb, reset_num_timesteps=False)

            mmax_wr = cb.last100_wr()
            mmax_model.save(os.path.join(SNAPSHOT_DIR, f"mmax_{round_num}"))
            print(f"  MMAX saved (WR={mmax_wr:.0f}%)")

            # MAIN round
            print(f"── Minimax Round {round_num}B: MAIN ──")
            selfplay_env._opponent_fn = make_selfplay_opponent(SNAPSHOT_DIR, rng)
            cb = LogCallback("MAIN", log_every=args.log_every)
            model.learn(total_timesteps=round_steps, callback=cb, reset_num_timesteps=False)
            total_eps += cb.ep

            model.save(os.path.join(SNAPSHOT_DIR, "main_latest"))
            model.save(os.path.join(SNAPSHOT_DIR, f"main_{round_num}"))

            b = benchmark_vs_smart_bot(model)
            bench_history.append(b)
            trend = "→".join(str(x) for x in bench_history[-6:])
            if b > best_bench:
                best_bench = b
                model.save(os.path.join(SNAPSHOT_DIR, "main_best"))
                print(f"  ★ New best! bench={b}%")

            pool = len(glob(os.path.join(SNAPSHOT_DIR, "*.zip")))
            print(f"  Bench: {b}% [{trend}] | Pool: {pool} | Total eps: {total_eps}")
            print()

    # ── Done ──────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"  DONE | Total eps: {total_eps} | Best bench: {best_bench}%")
    print(f"  Final bench: {bench_history[-1]}%")
    print("=" * 65)
    model.save("models/overnight_final")
    print("Saved: models/overnight_final")


if __name__ == "__main__":
    main()
