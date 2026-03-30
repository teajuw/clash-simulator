"""V2 Gymnasium environment — CNN observation, simplified reward, LSTM-ready.

Changes from v1:
  1. CNN observation: 18×32×3 spatial grid (unit_type, hp_fraction, owner)
     + 15-element scalar vector (elixir, tower HP, hand, time)
  2. Simplified reward: tower damage + card penalty + win/loss (per SEAT paper)
  3. LSTM support: RecurrentPPO from sb3-contrib
  4. Larger network: CnnPolicy with custom feature extractor

Action space unchanged: MultiDiscrete([5, 18, 15])

Usage:
    from clasher.env_v2 import ClashRoyaleEnvV2

    env = ClashRoyaleEnvV2()
    obs, info = env.reset()
    # obs is a dict: {"spatial": (3, 32, 18), "scalars": (15,)}
"""

from __future__ import annotations

import random as _random
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .arena import Position
from .battle import BattleState
from .entities import Building, Troop
from .tournament_standard import apply_tournament_overrides

try:
    from .rust_backend import RustBattle, RUST_AVAILABLE
except ImportError:
    RUST_AVAILABLE = False

# ── Deck ──────────────────────────────────────────────────────────────────────

DECK = [
    "HogRider", "Musketeer", "IceGolem", "IceSpirit",
    "Cannon", "Fireball", "Skeletons", "TheLog",
]
CARD_TO_IDX = {name: i for i, name in enumerate(DECK)}
ELIXIR_COST = {
    "HogRider": 4, "Musketeer": 4, "IceGolem": 2, "IceSpirit": 1,
    "Cannon": 3, "Fireball": 4, "Skeletons": 1, "TheLog": 2,
}

# Card type IDs for the spatial grid (0=empty, 1-8=deck cards, 9=princess, 10=king)
NUM_CARD_IDS = 11
PRINCESS_TOWER_ID = 9
KING_TOWER_ID = 10

# Arena dimensions
ARENA_W = 18
ARENA_H = 32

# Action space
N_CARD_CHOICES = 5
GRID_X = 18
GRID_Y = 15
TICKS_PER_STEP = 20

# Tower HP for normalization
MAX_KING_HP = 4824.0
MAX_PRINCESS_HP = 3052.0
MAX_GAME_TIME = 360.0

# Scalar observation size: 2 elixir + 6 tower HP + 4 hand + 1 next + 1 time + 1 phase = 15
N_SCALARS = 15


def tile_to_position(tx: int, ty: int) -> Position:
    return Position(tx + 0.5, ty + 0.5)


# ── Rule Bot (fallback) ──────────────────────────────────────────────────────

def rule_bot_policy(battle: BattleState) -> None:
    player = battle.players[1]
    if player.elixir >= 4 and "HogRider" in player.hand:
        battle.deploy_card(1, "HogRider", Position(_random.choice([3.5, 14.5]), 18.0))
        return
    has_threat = any(
        isinstance(e, Troop) and e.player_id == 0 and e.position.y > 16
        for e in battle.entities.values()
    )
    if has_threat and player.elixir >= 3 and "Cannon" in player.hand:
        battle.deploy_card(1, "Cannon", Position(9.0, 22.0))
        return
    if player.elixir >= 9.5:
        cheapest = min(player.hand, key=lambda c: ELIXIR_COST.get(c, 10))
        battle.deploy_card(1, cheapest, Position(9.0, 20.0))


# ── Environment ───────────────────────────────────────────────────────────────

class ClashRoyaleEnvV2(gym.Env):
    """V2 environment with CNN spatial observation and simplified reward.

    Observation is a Dict:
      "spatial": Box(0, 1, shape=(3, 32, 18)) — channels: unit_type, hp, owner
      "scalars": Box(0, 1, shape=(15,)) — elixir, tower HP, hand, time

    Action: MultiDiscrete([5, 18, 15])
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent: str = "rule_bot",
        ticks_per_step: int = TICKS_PER_STEP,
        backend: str = "auto",
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.ticks_per_step = ticks_per_step
        self.render_mode = render_mode

        if backend == "auto":
            self.use_rust = RUST_AVAILABLE
        elif backend == "rust":
            self.use_rust = True
        else:
            self.use_rust = False

        if opponent == "rule_bot":
            self._opponent_fn: Callable = rule_bot_policy
        elif opponent == "none":
            self._opponent_fn = lambda b: None
        else:
            raise ValueError(f"Unknown opponent: {opponent}")

        # Action space: card × tile_x × tile_y
        self.action_space = spaces.MultiDiscrete([N_CARD_CHOICES, GRID_X, GRID_Y])

        # Observation space: Dict with spatial grid + scalar features
        self.observation_space = spaces.Dict({
            "spatial": spaces.Box(
                low=0.0, high=1.0,
                shape=(3, ARENA_H, ARENA_W),  # CHW format for CNN
                dtype=np.float32,
            ),
            "scalars": spaces.Box(
                low=0.0, high=1.0,
                shape=(N_SCALARS,),
                dtype=np.float32,
            ),
        })

        # State
        self.battle: Optional[BattleState] = None
        self._prev_my_tower_hp = 0.0
        self._prev_opp_tower_hp = 0.0
        self._prev_my_crowns = 0
        self._prev_opp_crowns = 0
        self._cards_played_this_step = 0

    # ── Gym API ───────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.use_rust:
            self.battle = RustBattle()
            for pid in range(2):
                shuffled = list(DECK)
                self.np_random.shuffle(shuffled)
                self.battle.set_deck(pid, shuffled[:4], shuffled[4:])
        else:
            self.battle = BattleState()
            apply_tournament_overrides(self.battle)
            for p in self.battle.players:
                shuffled = list(DECK)
                self.np_random.shuffle(shuffled)
                p.deck = list(DECK)
                p.hand = shuffled[:4]
                p.cycle_queue = deque(shuffled[4:])
                p.elixir = 5.0

        self._prev_my_tower_hp = self._total_tower_hp(0)
        self._prev_opp_tower_hp = self._total_tower_hp(1)
        self._prev_my_crowns = 0
        self._prev_opp_crowns = 0

        return self._get_obs(), {"battle_time": 0.0}

    def step(self, action):
        assert self.battle is not None

        card_choice = int(action[0])
        tile_x = int(action[1])
        tile_y = int(action[2])

        # 1. Execute action
        self._cards_played_this_step = 0
        if card_choice > 0:
            slot = card_choice - 1
            player = self.battle.players[0]
            if slot < len(player.hand):
                card_name = player.hand[slot]
                pos = tile_to_position(tile_x, tile_y)
                if self.battle.deploy_card(0, card_name, pos):
                    self._cards_played_this_step = 1

        # 2. Opponent acts
        self._opponent_fn(self.battle)

        # 3. Advance simulation
        if self.use_rust:
            self.battle.step_n(self.ticks_per_step)
        else:
            for _ in range(self.ticks_per_step):
                self.battle.step(speed_factor=1.0)
                if self.battle.game_over:
                    break

        # 4. Simplified reward (SEAT-inspired)
        reward = self._compute_reward()

        # 5. Observation
        obs = self._get_obs()

        # 6. Terminal
        terminated = self.battle.game_over
        truncated = False

        info = {
            "battle_time": self.battle.time,
            "winner": self.battle.winner,
            "my_crowns": self.battle.players[1].get_crown_count(),
            "opp_crowns": self.battle.players[0].get_crown_count(),
        }

        return obs, reward, terminated, truncated, info

    # ── CNN Observation ───────────────────────────────────────────────────

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Build spatial + scalar observation."""
        spatial = np.zeros((3, ARENA_H, ARENA_W), dtype=np.float32)
        scalars = np.zeros(N_SCALARS, dtype=np.float32)

        p0 = self.battle.players[0]
        p1 = self.battle.players[1]

        # ── Spatial grid: 3 channels × 32 × 18 ──
        # Channel 0: unit type (normalized card ID)
        # Channel 1: HP fraction (0-1)
        # Channel 2: ownership (0=mine, 0.5=neutral, 1=enemy)

        if not self.use_rust:
            for entity in self.battle.entities.values():
                if not entity.is_alive:
                    continue
                if not isinstance(entity, (Troop, Building)):
                    continue

                # Grid position (clamp to bounds)
                gx = max(0, min(ARENA_W - 1, int(entity.position.x)))
                gy = max(0, min(ARENA_H - 1, int(entity.position.y)))

                # Card type
                if isinstance(entity, Building) and entity.position.y in (2.5, 29.5):
                    card_id = KING_TOWER_ID
                elif isinstance(entity, Building) and entity.position.y in (6.5, 25.5):
                    card_id = PRINCESS_TOWER_ID
                elif entity.card_stats:
                    card_id = CARD_TO_IDX.get(entity.card_stats.name, 0) + 1  # 1-indexed
                else:
                    card_id = 0

                spatial[0, gy, gx] = card_id / NUM_CARD_IDS
                spatial[1, gy, gx] = (
                    entity.hitpoints / entity.max_hitpoints
                    if entity.max_hitpoints > 0 else 0.0
                )
                spatial[2, gy, gx] = 0.0 if entity.player_id == 0 else 1.0

        # ── Scalar features (15) ──
        scalars[0] = p0.elixir / 10.0
        scalars[1] = p1.elixir / 10.0
        scalars[2] = max(0, p0.king_tower_hp) / MAX_KING_HP
        scalars[3] = max(0, p0.left_tower_hp) / MAX_PRINCESS_HP
        scalars[4] = max(0, p0.right_tower_hp) / MAX_PRINCESS_HP
        scalars[5] = max(0, p1.king_tower_hp) / MAX_KING_HP
        scalars[6] = max(0, p1.left_tower_hp) / MAX_PRINCESS_HP
        scalars[7] = max(0, p1.right_tower_hp) / MAX_PRINCESS_HP

        # Hand (4 cards, normalized)
        for i in range(min(4, len(p0.hand))):
            scalars[8 + i] = (CARD_TO_IDX.get(p0.hand[i], 0) + 1) / NUM_CARD_IDS

        # Next card
        if p0.cycle_queue:
            scalars[12] = (CARD_TO_IDX.get(p0.cycle_queue[0], 0) + 1) / NUM_CARD_IDS

        # Time + phase
        scalars[13] = min(self.battle.time / MAX_GAME_TIME, 1.0)
        scalars[14] = (
            0.0 if not self.battle.double_elixir
            else 0.5 if not self.battle.triple_elixir
            else 1.0
        )

        return {"spatial": spatial, "scalars": scalars}

    # ── Simplified Reward (SEAT-inspired) ─────────────────────────────────

    def _compute_reward(self) -> float:
        """Three-term reward: tower damage + card penalty + win/loss.

        Per SEAT paper (IJCAI 2019):
          r_tower = +20 per enemy tower destroyed, -30 per own tower lost
          r_card  = -3 per card played
          r_win   = +50 / -50 terminal
        """
        my_hp = self._total_tower_hp(0)
        opp_hp = self._total_tower_hp(1)

        # Crown changes
        my_crowns = self.battle.players[1].get_crown_count()  # towers I destroyed
        opp_crowns = self.battle.players[0].get_crown_count()  # my towers destroyed
        crowns_scored = my_crowns - self._prev_my_crowns
        crowns_lost = opp_crowns - self._prev_opp_crowns

        # Tower reward: +20 per crown scored, -30 per crown lost
        r_tower = crowns_scored * 20.0 - crowns_lost * 30.0

        # Card penalty: -3 per card played (encourages elixir efficiency)
        r_card = -3.0 * self._cards_played_this_step

        # Win/loss terminal
        r_win = 0.0
        if self.battle.game_over:
            if self.battle.winner == 0:
                r_win = 50.0
            elif self.battle.winner == 1:
                r_win = -50.0

        # Small dense signal: tower HP change (normalized)
        tower_dmg_dealt = (self._prev_opp_tower_hp - opp_hp) / (MAX_KING_HP + 2 * MAX_PRINCESS_HP)
        tower_dmg_taken = (self._prev_my_tower_hp - my_hp) / (MAX_KING_HP + 2 * MAX_PRINCESS_HP)
        r_hp = tower_dmg_dealt * 5.0 - tower_dmg_taken * 5.0

        # Update state
        self._prev_my_tower_hp = my_hp
        self._prev_opp_tower_hp = opp_hp
        self._prev_my_crowns = my_crowns
        self._prev_opp_crowns = opp_crowns

        return r_tower + r_card + r_win + r_hp

    def _total_tower_hp(self, player_id: int) -> float:
        p = self.battle.players[player_id]
        return float(max(0, p.king_tower_hp) + max(0, p.left_tower_hp) + max(0, p.right_tower_hp))
