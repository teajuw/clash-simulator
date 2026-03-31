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

class FilePFSP:
    """File-based PFSP tracker for cross-process opponent weighting.

    Writes win/loss records to a JSON file. Each process reads it
    to compute sampling weights. No shared memory needed.
    """

    def __init__(self, snapshot_dir: str, decay: float = 0.95):
        self._path = os.path.join(snapshot_dir, "_pfsp_records.json")
        self._decay = decay
        self._cache = {}  # {snapshot_path: ema_lose_rate}
        self._games = {}  # {snapshot_path: n_games}

    def record(self, snapshot_path: str, won: bool):
        """Record a result and update EMA. Flushes to disk every 100 games."""
        n = self._games.get(snapshot_path, 0)
        lost = 0.0 if won else 1.0
        if n < 3:
            w = self._cache.get(snapshot_path, 0.5)
            self._cache[snapshot_path] = (w * n + lost) / (n + 1)
        else:
            old = self._cache.get(snapshot_path, 0.5)
            self._cache[snapshot_path] = self._decay * old + (1 - self._decay) * lost
        self._games[snapshot_path] = n + 1
        self._total_games = getattr(self, '_total_games', 0) + 1

        # Flush every 100 games (not every game)
        if self._total_games % 100 == 0:
            self._flush()

    def _flush(self):
        """Write EMA state to disk (best effort)."""
        try:
            import json
            with open(self._path, "w") as f:
                json.dump({"ema": self._cache, "games": self._games}, f)
        except Exception:
            pass

    def sample(self, snapshots: list) -> str:
        """PFSP-weighted sample: harder opponents sampled more."""
        if not snapshots:
            return None

        weights = []
        for s in snapshots:
            n = self._games.get(s, 0)
            if n < 3:
                weights.append(0.3)  # explore
            else:
                lr = self._cache.get(s, 0.5)
                weights.append(max(0.05, lr ** 2))

        # Cap at 30%
        total = sum(weights)
        probs = [w / total for w in weights]
        for i in range(len(probs)):
            if probs[i] > 0.30:
                probs[i] = 0.30
        total = sum(probs)
        probs = [p / total for p in probs]

        return _random.choices(snapshots, weights=probs, k=1)[0]


def make_pool_opponent(snapshot_dir: str, role: str):
    """Create an opponent function based on the agent's role."""

    pfsp = FilePFSP(snapshot_dir) if role == "main" else None

    def _load_snapshot(path: str):
        try:
            return PPO.load(path.replace(".zip", ""))
        except Exception:
            return None

    def _get_all_snapshots():
        return sorted(glob(os.path.join(snapshot_dir, "*.zip")))

    def _get_main_snapshots():
        return sorted(glob(os.path.join(snapshot_dir, "main_*.zip")))

    # Cache: opponent model + which snapshot it came from
    state = {"model": None, "snapshot": None, "loaded_at": 0, "reload_every": 10}

    def opponent_fn(battle):
        """Act as player 1 using a loaded snapshot."""
        state["loaded_at"] += 1

        # Reload opponent model periodically
        if state["model"] is None or state["loaded_at"] % state["reload_every"] == 0:
            zips = _get_all_snapshots()
            if not zips:
                _simple_opponent(battle)
                return

            if role in ("minimax_exploiter", "entropy_explorer"):
                main_zips = _get_main_snapshots()
                path = main_zips[-1] if main_zips else zips[-1]
            elif role == "main" and pfsp:
                # PFSP weighted selection
                path = pfsp.sample(zips)
            else:
                # League exploiter: uniform random
                path = _random.choice(zips)

            state["snapshot"] = path
            state["model"] = _load_snapshot(path)

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

    # Attach pfsp and state so callback can access them
    opponent_fn._pfsp = pfsp
    opponent_fn._state = state

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
                 snapshot_dir, ent_coef, pfsp=None, opponent_state=None):
        super().__init__(verbose=0)
        self.role = role
        self.log_prefix = log_prefix
        self.log_every = log_every
        self.save_every = save_every
        self.snapshot_dir = snapshot_dir
        self.ent_coef = ent_coef
        self.pfsp = pfsp
        self.opponent_state = opponent_state

        self.episode_count = 0
        self.wins = 0
        self.losses = 0
        self.episode_rewards = []
        self.episode_winners = []
        self._current_reward = 0.0
        self._start_time = time.time()
        self._post_reset_lr_steps = 0
        # Explorer entropy schedule state
        self._explorer_phase = "explore"  # "explore" or "exploit"
        self._explorer_phase_start = 0
        # Rolling WR for performance-based decisions
        self._recent_window = 100
        # Stagnation detection for exploiters
        self._best_wr = 0.0
        self._best_wr_ep = 0
        self._stagnation_patience = 500  # reset after 500 eps without improvement
        # Benchmark (main agent only)
        self._bench_history = []
        self._last_bench = 0

    def _recent_wr(self) -> float:
        """Win rate over last N episodes."""
        recent = self.episode_winners[-self._recent_window:]
        if not recent:
            return 0.5
        return sum(1 for w in recent if w == 0) / len(recent)

    def _reset_to_main(self):
        """Copy latest main agent's weights into this agent.
        For minimax exploiter, also load main's critic for reward shaping."""
        main_zips = sorted(glob(os.path.join(self.snapshot_dir, "main_*.zip")))
        if not main_zips:
            return
        try:
            latest_main = main_zips[-1].replace(".zip", "")
            main_model = PPO.load(latest_main)
            self.model.policy.load_state_dict(main_model.policy.state_dict())
            self._post_reset_lr_steps = 300
            # For minimax: update the critic reference in the env
            if self.role == "minimax_exploiter":
                self.model.env.envs[0].unwrapped._minimax_critic = main_model.policy.predict_values
            print(f"{self.log_prefix} Reset to {Path(latest_main).stem}")
        except Exception:
            pass

    def _quality_save(self, min_wr: float = 0.6):
        """Save snapshot only if recent WR exceeds threshold."""
        if self._recent_wr() >= min_wr:
            path = os.path.join(self.snapshot_dir,
                                f"{self.role}_{self.episode_count}")
            self.model.save(path)
            print(f"{self.log_prefix} Saved exploit (WR={self._recent_wr():.0%})")
            return True
        return False

    def _on_step(self) -> bool:
        self._current_reward += self.locals.get("rewards", [0])[0]
        dones = self.locals.get("dones", [False])

        if dones[0]:
            self.episode_count += 1
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0

            # LR decay after reset
            if self._post_reset_lr_steps > 0:
                self._post_reset_lr_steps -= 1
                progress = self._post_reset_lr_steps / 300.0
                lr = 3e-4 + (7e-4) * progress  # 1e-3 → 3e-4
                for pg in self.model.policy.optimizer.param_groups:
                    pg['lr'] = lr

            infos = self.locals.get("infos", [{}])
            winner = infos[0].get("winner", None)
            self.episode_winners.append(winner)
            if winner == 0:
                self.wins += 1
            elif winner == 1:
                self.losses += 1

            # PFSP recording (main agent only)
            if self.pfsp and self.opponent_state:
                snap = self.opponent_state.get("snapshot")
                if snap:
                    self.pfsp.record(snap, winner == 0)

            # ── Role-specific logic ──────────────────────────────────

            if self.role == "main":
                # Save on schedule
                if self.episode_count % self.save_every == 0:
                    path = os.path.join(self.snapshot_dir,
                                        f"main_{self.episode_count}")
                    self.model.save(path)

            elif self.role == "minimax_exploiter":
                # Reset best_wr when pool first appears (warmup WR is meaningless)
                pool_size = len(glob(os.path.join(self.snapshot_dir, "*.zip")))
                if pool_size > 0 and self._best_wr > 0.75:
                    self._best_wr = 0.0
                    self._best_wr_ep = self.episode_count

                # Load main's critic if not yet loaded
                if (self.model.env.envs[0].unwrapped._minimax_critic is None
                        and self.episode_count % 50 == 0):
                    main_zips = sorted(glob(os.path.join(self.snapshot_dir, "main_*.zip")))
                    if main_zips:
                        try:
                            m = PPO.load(main_zips[-1].replace(".zip", ""))
                            self.model.env.envs[0].unwrapped._minimax_critic = m.policy.predict_values
                            print(f"{self.log_prefix} Loaded main critic")
                        except Exception:
                            pass
                # Track best WR for stagnation detection
                wr = self._recent_wr()
                if wr > self._best_wr:
                    self._best_wr = wr
                    self._best_wr_ep = self.episode_count

                # Save quality exploits
                if self.episode_count % self.save_every == 0:
                    self._quality_save(min_wr=0.6)

                # Reset when stagnated — no WR improvement for 500 eps
                eps_since_improvement = self.episode_count - self._best_wr_ep
                if (self.episode_count > 500
                        and eps_since_improvement >= self._stagnation_patience):
                    print(f"{self.log_prefix} Stagnated at WR={wr:.0%} (best={self._best_wr:.0%}), resetting")
                    self._quality_save(min_wr=0.5)
                    self._reset_to_main()
                    self._best_wr = 0.0
                    self._best_wr_ep = self.episode_count

            elif self.role == "entropy_explorer":
                # Don't start explore/exploit schedule until main has a snapshot
                main_zips = sorted(glob(os.path.join(self.snapshot_dir, "main_*.zip")))
                if not main_zips:
                    pass  # still warmup, keep exploring against fallback
                else:
                    if self._explorer_phase_start == 0:
                        # First main snapshot just appeared — start the clock
                        self._explorer_phase_start = self.episode_count
                        self._explorer_phase = "explore"
                        self.model.ent_coef = 0.10
                        print(f"{self.log_prefix} Main snapshot found, starting explore phase")

                # Entropy schedule: explore (0.10) for 200 eps, exploit (0.02) after
                eps_in_phase = self.episode_count - self._explorer_phase_start if self._explorer_phase_start > 0 else 0
                if self._explorer_phase == "explore" and self._explorer_phase_start > 0 and eps_in_phase >= 200:
                    self._explorer_phase = "exploit"
                    self._explorer_phase_start = self.episode_count
                    self.model.ent_coef = 0.02
                    print(f"{self.log_prefix} Phase: exploit (ent→0.02)")

                # Save quality exploits during exploit phase
                if self._explorer_phase == "exploit" and self.episode_count % self.save_every == 0:
                    self._quality_save(min_wr=0.6)

                # Track stagnation during exploit phase
                if self._explorer_phase == "exploit":
                    wr = self._recent_wr()
                    if wr > self._best_wr:
                        self._best_wr = wr
                        self._best_wr_ep = self.episode_count

                    eps_since_improvement = self.episode_count - self._best_wr_ep
                    if eps_since_improvement >= self._stagnation_patience:
                        self._explorer_phase = "explore"
                        self._explorer_phase_start = self.episode_count
                        self.model.ent_coef = 0.10
                        self._reset_to_main()
                        self._best_wr = 0.0
                        self._best_wr_ep = self.episode_count
                        print(f"{self.log_prefix} Stagnated at WR={wr:.0%}, resetting (ent→0.10)")

            elif self.role == "league_exploiter":
                # Save on schedule (no quality gate — diverse strategies welcome)
                if self.episode_count % self.save_every == 0:
                    path = os.path.join(self.snapshot_dir,
                                        f"league_exploiter_{self.episode_count}")
                    self.model.save(path)

            # Benchmark (main agent only)
            if self.role == "main" and self.episode_count % self.log_every == 0:
                self._run_benchmark()

            # Display
            if self.episode_count % self.log_every == 0:
                self._display()

        return True

    def _run_benchmark(self, n_games=10):
        """Run MAIN against rule bot to measure absolute skill."""
        bench_env = ClashRoyaleEnv(opponent="rule_bot")
        wins = 0
        for _ in range(n_games):
            obs, _ = bench_env.reset()
            while True:
                act, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = bench_env.step(act)
                if term:
                    if info["winner"] == 0:
                        wins += 1
                    break
        self._last_bench = wins * 10  # percentage
        self._bench_history.append(self._last_bench)

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

        # Benchmark trend (main only)
        bench_str = ""
        if self.role == "main" and self._bench_history:
            trend = "→".join(f"{b}" for b in self._bench_history[-6:])
            bench_str = f" bench={self._last_bench:3d}%[{trend}]"

        # Current opponent
        opp_str = ""
        if self.opponent_state:
            snap = self.opponent_state.get("snapshot", "")
            if snap:
                opp_str = f" vs={Path(snap).stem}"
            else:
                opp_str = " vs=fallback"

        print(
            f"{self.log_prefix} [{ts}] "
            f"ep={self.episode_count:<5d} steps={self.num_timesteps:>9,} | "
            f"pool={pool_size:2d} |{bench_str} "
            f"WR={recent_wr:4.1f}% cum={wr:4.1f}% | "
            f"R={avg_r:+6.1f} | "
            f"{sps:.0f} sps {elapsed_str}{opp_str}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True,
                        choices=["main", "minimax_exploiter", "entropy_explorer", "league_exploiter"])
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

    # Get PFSP tracker and state dict (attached to the function by make_pool_opponent)
    pfsp_tracker = getattr(opponent_fn, '_pfsp', None)
    opp_state = getattr(opponent_fn, '_state', None)

    # Create env with custom opponent
    is_adversarial = args.role == "minimax_exploiter"
    env = ClashRoyaleEnv(opponent="none", adversarial=is_adversarial)
    env._opponent_fn = opponent_fn

    # Minimax exploiter: load main's critic for reward shaping
    main_critic = None
    if args.role == "minimax_exploiter":
        env._minimax_alpha = 0.01  # weight of minimax term

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
        # Same LR for all — exploiter gets diversity from entropy + adversarial reward
        lr = 3e-4

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
            learning_rate=lr,
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
        pfsp=pfsp_tracker,
        opponent_state=opp_state,
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
