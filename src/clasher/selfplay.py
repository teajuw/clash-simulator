"""Self-play with snapshot pool for Clash Royale RL training.

Architecture:
  - Save agent checkpoints every N episodes
  - Opponent sampled from pool: 70% random past snapshot, 30% latest
  - Rule bot stays in pool permanently as baseline
  - Observation is mirrored for opponent (swap player 0/1, flip y)

Usage:
    from clasher.selfplay import SelfPlayEnv

    env = SelfPlayEnv(snapshot_dir="snapshots")
    model = PPO("MlpPolicy", env)
    model.learn(total_timesteps=500_000, callback=env.get_callback())
"""

from __future__ import annotations

import os
import random as _random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .battle import BattleState
from .entities import Building, Troop
from .env import (
    ClashRoyaleEnv, DECK, ELIXIR_COST, CARD_TO_IDX,
    N_REGIONS, OBS_SIZE, region_to_position, rule_bot_policy,
)


# ── Name Generator ────────────────────────────────────────────────────────────

_ADJECTIVES = [
    "swift", "bold", "keen", "dark", "iron", "cold", "wild", "grim",
    "pale", "rust", "neon", "void", "pure", "raw", "deep", "torn",
    "stark", "haze", "flux", "zero", "null", "core", "edge", "volt",
    "apex", "nova", "warp", "drift", "blaze", "frost", "ghost", "stone",
]
_NOUNS = [
    "hog", "cannon", "musk", "tower", "log", "spirit", "golem", "skel",
    "lance", "spark", "blade", "storm", "viper", "pulse", "fang", "bolt",
    "rook", "pawn", "knight", "bishop", "sage", "forge", "anvil", "hex",
    "arc", "claw", "pike", "axe", "helm", "ward", "sigil", "rune",
]


def _generate_name(episode: int) -> str:
    """Generate a unique readable name like 'swift-hog-200'."""
    adj = _ADJECTIVES[episode % len(_ADJECTIVES)]
    noun = _NOUNS[(episode // len(_ADJECTIVES)) % len(_NOUNS)]
    return f"{adj}-{noun}-{episode}"


def _update_elo(rating_a: float, rating_b: float, a_won: bool, k: float = 32.0) -> Tuple[float, float]:
    """Standard Elo update. Returns (new_a, new_b)."""
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    score_a = 1.0 if a_won else 0.0
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1.0 - score_a) - (1.0 - expected_a))
    return new_a, new_b


class SnapshotPool:
    """Snapshot pool with Prioritized Fictitious Self-Play (PFSP) sampling.

    Opponents you lose to get sampled more often. Opponents you've mastered
    fade out naturally. Rule bot is always in the pool as a baseline anchor.
    Pool is capped at max_size — oldest non-rule-bot snapshots are dropped.
    """

    def __init__(self, snapshot_dir: str = "snapshots", max_size: int = 20):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._rule_bot_id = "__rule_bot__"

        self.snapshots: List[str] = [self._rule_bot_id]
        # Track RECENT win/loss per opponent (rolling window of last N results)
        self._max_record_history = 50  # only consider last 50 games per opponent
        self.records: Dict[str, List[int]] = {self._rule_bot_id: [0, 0]}
        self._result_history: Dict[str, List[bool]] = {self._rule_bot_id: []}
        # Elo ratings: {opponent_id: rating}
        self.elo: Dict[str, float] = {self._rule_bot_id: 1000.0}
        # Readable names: {path: "swift-hog-200"}
        self.names: Dict[str, str] = {self._rule_bot_id: "bot"}
        # Current agent's Elo (the model being trained)
        self.agent_elo: float = 1000.0

        # Load any existing snapshots from disk
        for f in sorted(self.snapshot_dir.glob("*.zip")):
            path = str(f).replace(".zip", "")
            if path not in self.snapshots:
                self.snapshots.append(path)
                self.records[path] = [0, 0]
                self.elo[path] = 1000.0
                # Derive name from filename
                self.names[path] = Path(path).stem

    def save_snapshot(self, model, episode: int) -> str:
        """Save a model checkpoint with a readable name. Prunes oldest if full."""
        name = _generate_name(episode)
        path = str(self.snapshot_dir / name)
        model.save(path)
        self.snapshots.append(path)
        self.records[path] = [0, 0]
        self.elo[path] = self.agent_elo
        self.names[path] = name

        # Prune oldest (not rule bot, not latest 3) if over max
        while len(self.snapshots) > self.max_size:
            # Find oldest non-rule-bot that isn't in the newest 3
            for i, s in enumerate(self.snapshots):
                if s != self._rule_bot_id and i < len(self.snapshots) - 3:
                    self.snapshots.pop(i)
                    self.records.pop(s, None)
                    break
            else:
                break

        return path

    def record_result(self, opponent_id: str, won: bool) -> None:
        """Record a win or loss and update Elo ratings."""
        if opponent_id not in self.records:
            self.records[opponent_id] = [0, 0]
            self._result_history[opponent_id] = []
        if won:
            self.records[opponent_id][0] += 1
        else:
            self.records[opponent_id][1] += 1

        # Rolling window: track recent results for PFSP
        history = self._result_history.setdefault(opponent_id, [])
        history.append(won)
        if len(history) > self._max_record_history:
            history.pop(0)

        # Update agent Elo only — opponent ratings are frozen at save time
        if opponent_id not in self.elo:
            self.elo[opponent_id] = 1000.0
        opp_elo = self.elo[opponent_id]
        self.agent_elo, _ = _update_elo(self.agent_elo, opp_elo, won)

    def sample_opponent(self, **kwargs) -> str:
        """PFSP sampling using rolling window of recent results.

        weight = (lose_rate_recent)^2 — harder opponents sampled more.
        No single opponent can exceed 30% of the sampling probability.
        """
        if len(self.snapshots) <= 1:
            return self.snapshots[0]

        weights = []
        for s in self.snapshots:
            history = self._result_history.get(s, [])
            if len(history) < 3:
                weight = 1.0  # explore new opponents
            else:
                recent_losses = sum(1 for r in history if not r)
                lose_rate = recent_losses / len(history)
                weight = max(0.05, lose_rate ** 2)
            weights.append(weight)

        # Normalize then cap any single opponent at 30%
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        max_prob = 0.30
        capped = False
        for i in range(len(probs)):
            if probs[i] > max_prob:
                probs[i] = max_prob
                capped = True
        if capped:
            total_p = sum(probs)
            probs = [p / total_p for p in probs]

        return _random.choices(self.snapshots, weights=probs, k=1)[0]

    def get_win_rate(self, opponent_id: str) -> Optional[float]:
        """Get win rate against a specific opponent."""
        w, l = self.records.get(opponent_id, [0, 0])
        total = w + l
        return w / total if total > 0 else None

    def get_stats_summary(self) -> List[Tuple[str, int, int, float]]:
        """Return (name, recent_wins, recent_losses, weight%) for each snapshot."""
        stats = []
        weights = []
        for s in self.snapshots:
            history = self._result_history.get(s, [])
            if len(history) < 3:
                weight = 1.0
            else:
                recent_losses = sum(1 for r in history if not r)
                lose_rate = recent_losses / len(history)
                weight = max(0.05, lose_rate ** 2)
            weights.append(weight)

        total_w = sum(weights) or 1.0
        probs = [w / total_w for w in weights]
        # Apply same cap as sampling
        for i in range(len(probs)):
            if probs[i] > 0.30:
                probs[i] = 0.30
        total_p = sum(probs) or 1.0
        probs = [p / total_p for p in probs]

        for i, s in enumerate(self.snapshots):
            history = self._result_history.get(s, [])
            rw = sum(1 for r in history if r)
            rl = sum(1 for r in history if not r)
            name = self.names.get(s, Path(s).stem)
            stats.append((name, rw, rl, probs[i] * 100))
        return stats

    @property
    def size(self) -> int:
        return len(self.snapshots)

    def is_rule_bot(self, opponent_id: str) -> bool:
        return opponent_id == self._rule_bot_id


class SnapshotOpponent:
    """Plays as opponent using a loaded model snapshot."""

    def __init__(self):
        self._model = None
        self._opponent_id: str = "__rule_bot__"

    def load(self, opponent_id: str) -> None:
        """Load a snapshot as the current opponent. Falls back to rule bot if missing."""
        from stable_baselines3 import PPO

        self._opponent_id = opponent_id
        if opponent_id == "__rule_bot__":
            self._model = None
        else:
            try:
                self._model = PPO.load(opponent_id)
            except (FileNotFoundError, ValueError):
                # Snapshot file missing or corrupt — fall back to rule bot
                self._model = None
                self._opponent_id = "__rule_bot__"

    def act(self, battle: BattleState) -> None:
        """Take an action as player 1."""
        if self._model is None:
            rule_bot_policy(battle)
            return

        # Build mirrored observation (opponent sees itself as player 0)
        obs = self._mirror_obs(battle)

        # Get action from model
        action, _ = self._model.predict(obs, deterministic=False)
        card_choice = int(action[0])
        region = int(action[1])

        if card_choice == 0:
            return  # WAIT

        slot = card_choice - 1
        player = battle.players[1]

        if slot >= len(player.hand):
            return

        card_name = player.hand[slot]
        cost = ELIXIR_COST.get(card_name, 10)
        if player.elixir < cost:
            return

        # Mirror the region: flip y for player 1
        # Player 0's y=12-14 (bridge) → Player 1's y=17-19 (their bridge)
        pos = region_to_position(region)
        mirrored_pos = _mirror_position(pos)
        battle.deploy_card(1, card_name, mirrored_pos)

    def _mirror_obs(self, battle: BattleState) -> np.ndarray:
        """Build observation from player 1's perspective (as if they're player 0)."""
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        # Swap players: p1 sees itself as p0
        p0 = battle.players[1]  # opponent is "self" from their view
        p1 = battle.players[0]  # we are "opponent" from their view

        from .env import MAX_KING_HP, MAX_PRINCESS_HP, MAX_GAME_TIME, N_SCALARS, N_HAND
        from .env import NUM_CARD_IDS, PRINCESS_TOWER_ID, KING_TOWER_ID, MAX_UNITS, UNIT_FEATURES

        obs[0] = p0.elixir / 10.0
        obs[1] = p1.elixir / 10.0
        obs[2] = p0.king_tower_hp / MAX_KING_HP
        obs[3] = p0.left_tower_hp / MAX_PRINCESS_HP
        obs[4] = p0.right_tower_hp / MAX_PRINCESS_HP
        obs[5] = p1.king_tower_hp / MAX_KING_HP
        obs[6] = p1.left_tower_hp / MAX_PRINCESS_HP
        obs[7] = p1.right_tower_hp / MAX_PRINCESS_HP
        obs[8] = min(battle.time / MAX_GAME_TIME, 1.0)
        obs[9] = float(battle.double_elixir)
        obs[10] = float(battle.triple_elixir)
        obs[11] = float(battle.overtime)

        offset = N_SCALARS
        for i in range(min(4, len(p0.hand))):
            card = p0.hand[i]
            obs[offset + i] = CARD_TO_IDX.get(card, 0) / (NUM_CARD_IDS - 1)
        if p0.cycle_queue:
            obs[offset + 4] = CARD_TO_IDX.get(p0.cycle_queue[0], 0) / (NUM_CARD_IDS - 1)
        for i in range(min(4, len(p0.hand))):
            card = p0.hand[i]
            cost = ELIXIR_COST.get(card, 10)
            obs[offset + 5 + i] = 1.0 if p0.elixir >= cost else 0.0

        offset = N_SCALARS + N_HAND
        unit_idx = 0
        for entity in battle.entities.values():
            if unit_idx >= MAX_UNITS:
                break
            if not entity.is_alive:
                continue
            if not isinstance(entity, (Troop, Building)):
                continue

            if isinstance(entity, Building) and entity.position.y in (2.5, 29.5):
                card_id = KING_TOWER_ID
            elif isinstance(entity, Building) and entity.position.y in (6.5, 25.5):
                card_id = PRINCESS_TOWER_ID
            elif entity.card_stats:
                card_id = CARD_TO_IDX.get(entity.card_stats.name, 0)
            else:
                card_id = 0

            # Flip y and swap owner for mirrored perspective
            mirrored_y = 32.0 - entity.position.y
            owner = 1.0 if entity.player_id == 1 else 0.0  # swap: p1 is "self"

            base = offset + unit_idx * UNIT_FEATURES
            obs[base + 0] = card_id / (NUM_CARD_IDS - 1)
            obs[base + 1] = np.clip(entity.position.x / 18.0, 0.0, 1.0)
            obs[base + 2] = np.clip(mirrored_y / 32.0, 0.0, 1.0)
            obs[base + 3] = (
                entity.hitpoints / entity.max_hitpoints
                if entity.max_hitpoints > 0 else 0.0
            )
            obs[base + 4] = owner
            unit_idx += 1

        return obs

    @property
    def name(self) -> str:
        if self._opponent_id == "__rule_bot__":
            return "bot"
        return Path(self._opponent_id).stem


def _mirror_position(pos) -> "Position":
    """Mirror a position for player 1 (flip y across center)."""
    from .arena import Position
    return Position(pos.x, 32.0 - pos.y)


# ── Self-Play Environment ────────────────────────────────────────────────────

class SelfPlayEnv(ClashRoyaleEnv):
    """ClashRoyaleEnv with PFSP self-play opponent from snapshot pool."""

    def __init__(
        self,
        snapshot_dir: str = "snapshots",
        save_every: int = 50,
        max_pool: int = 20,
        **kwargs,
    ):
        super().__init__(opponent="none", **kwargs)
        self.pool = SnapshotPool(snapshot_dir, max_size=max_pool)
        self.opponent = SnapshotOpponent()
        self.save_every = save_every
        self._episode_count = 0
        self._current_opponent_id: str = "__rule_bot__"

        # Override opponent function
        self._opponent_fn = self.opponent.act

        # Track opponent distribution for display
        self.opponent_name = "rule_bot"
        self._recent_opponents: List[str] = []

    def reset(self, **kwargs):
        # Record result of previous episode (if there was one)
        if self._episode_count > 0 and self.battle is not None:
            won = self.battle.winner == 0
            self.pool.record_result(self._current_opponent_id, won)

        self._episode_count += 1

        # Sample new opponent (PFSP weighted)
        self._current_opponent_id = self.pool.sample_opponent()
        self.opponent.load(self._current_opponent_id)
        self.opponent_name = self.opponent.name
        self._recent_opponents.append(self.opponent_name)
        if len(self._recent_opponents) > 20:
            self._recent_opponents.pop(0)

        return super().reset(**kwargs)

    def save_snapshot(self, model) -> str:
        """Save current model to snapshot pool."""
        return self.pool.save_snapshot(model, self._episode_count)

    def get_opponent_distribution(self) -> Dict[str, int]:
        """How many of the last 20 games were against each opponent."""
        dist: Dict[str, int] = {}
        for name in self._recent_opponents:
            dist[name] = dist.get(name, 0) + 1
        return dist

    @property
    def episode_count(self) -> int:
        return self._episode_count


# ── Visual Display ────────────────────────────────────────────────────────────

def _short_name(n: str) -> str:
    """Shorten snapshot names for display."""
    if n == "rule_bot" or n == "bot":
        return "bot"
    # If it's already a readable name like "swift-hog-200", use it
    stem = Path(n).stem if "/" in n else n
    # Strip old-style "snapshot_ep" prefix
    if stem.startswith("snapshot_ep"):
        return "ep" + stem.replace("snapshot_ep", "").lstrip("0") or "ep0"
    return stem


def render_pool_status(
    episode: int,
    pool_size: int,
    win_rate: float,
    recent_wr: float,
    wins: int,
    losses: int,
    avg_reward: float,
    elapsed: float,
    steps: int,
    agent_elo: float = 1000.0,
    bench_wr: float = 0.0,
    bench_trend: str = "",
    opponent_dist: Optional[Dict[str, int]] = None,
    pool_stats: Optional[List[Tuple[str, int, int, float]]] = None,
) -> str:
    """Render compact training status line."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%H:%M:%S")

    if pool_size <= 1:
        phase = "BOT"
    elif pool_size <= 5:
        phase = "EARLY"
    else:
        phase = "SELF"

    # Opponent mix summary
    opp_str = ""
    if opponent_dist:
        top = sorted(opponent_dist.items(), key=lambda x: -x[1])[:3]
        opp_str = " ".join(f"{_short_name(n)}:{c}" for n, c in top)

    # Hardest opponent
    hard_str = ""
    if pool_stats and len(pool_stats) > 1:
        played = [(n, w, l, p) for n, w, l, p in pool_stats if w + l > 0]
        if played:
            h = sorted(played, key=lambda x: -x[3])[0]
            hard_str = f"{_short_name(h[0])}({h[1]}W/{h[2]}L)"

    return (
        f"[{timestamp}] "
        f"ep={episode:<5d} steps={steps:>9,} | "
        f"{phase:<5s} pool={pool_size:2d} | "
        f"elo={agent_elo:5.0f} | "
        f"bench={bench_wr:3.0f}%[{bench_trend}] | "
        f"R={avg_reward:+6.1f} | "
        f"vs=[{opp_str}] hard={hard_str}"
    )
