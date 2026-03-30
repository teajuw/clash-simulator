"""Gymnasium environment for Clash Royale 2.6 Hog Cycle mirror matches.

Two-step action space:
  action[0] = card choice: 0=WAIT, 1-4=hand slot
  action[1] = region: 0-29 (6x5 grid of 3-tile zones)

Usage:
    from clasher.env import ClashRoyaleEnv

    env = ClashRoyaleEnv(opponent="rule_bot")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step([0, 0])  # WAIT
    obs, reward, terminated, truncated, info = env.step([1, 25]) # card slot 0 at region 25
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

# ── Deck ──────────────────────────────────────────────────────────────────────

DECK = [
    "HogRider", "Musketeer", "IceGolem", "IceSpirit",
    "Cannon", "Fireball", "Skeletons", "TheLog",
]
CARD_TO_IDX = {name: i for i, name in enumerate(DECK)}
IDX_TO_CARD = {i: name for i, name in enumerate(DECK)}

ELIXIR_COST = {
    "HogRider": 4, "Musketeer": 4, "IceGolem": 2, "IceSpirit": 1,
    "Cannon": 3, "Fireball": 4, "Skeletons": 1, "TheLog": 2,
}

# ── Region Grid (6×5 = 30 regions) ────────────────────────────────────────────
# Each region is a 3-tile-wide × 3-tile-tall zone.
# Deploy at the center of the region.
#
#  y=12-14 | R24  R25  R26  R27  R28  R29 |  ← bridge approach
#  y= 9-11 | R18  R19  R20  R21  R22  R23 |  ← mid field
#  y= 6- 8 | R12  R13  R14  R15  R16  R17 |  ← behind towers
#  y= 3- 5 | R06  R07  R08  R09  R10  R11 |  ← king area
#  y= 0- 2 | R00  R01  R02  R03  R04  R05 |  ← deep back
#            x0-2  x3-5  x6-8  x9-11 x12-14 x15-17

X_BINS = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 17)]
Y_BINS = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14)]
N_REGIONS = len(X_BINS) * len(Y_BINS)  # 30

# Pre-compute region centers
REGION_CENTERS: List[Tuple[float, float]] = []
for y0, y1 in Y_BINS:
    for x0, x1 in X_BINS:
        cx = (x0 + x1) / 2.0 + 0.5
        cy = (y0 + y1) / 2.0 + 0.5
        REGION_CENTERS.append((cx, cy))


def region_to_position(region_idx: int) -> Position:
    """Convert region index (0-29) to arena Position."""
    cx, cy = REGION_CENTERS[region_idx]
    return Position(cx, cy)


def position_to_region(x: float, y: float) -> int:
    """Convert arena coordinates to region index."""
    xi = min(5, max(0, int(x) // 3))
    yi = min(4, max(0, int(y) // 3))
    return yi * 6 + xi


# ── Observation / Action Constants ────────────────────────────────────────────

PRINCESS_TOWER_ID = 8
KING_TOWER_ID = 9
NUM_CARD_IDS = 10

MAX_UNITS = 30
UNIT_FEATURES = 5

N_SCALARS = 12
N_HAND = 9
N_UNITS = MAX_UNITS * UNIT_FEATURES
OBS_SIZE = N_SCALARS + N_HAND + N_UNITS  # 171

N_CARD_CHOICES = 5   # 0=WAIT, 1-4=hand slots
TICKS_PER_STEP = 30

MAX_KING_HP = 4824.0
MAX_PRINCESS_HP = 3052.0
MAX_TOTAL_HP = MAX_KING_HP + 2 * MAX_PRINCESS_HP
MAX_GAME_TIME = 360.0


# ── Rule Bot ──────────────────────────────────────────────────────────────────

def rule_bot_policy(battle: BattleState) -> None:
    """Simple opponent: rush Hog at bridge, Cannon on defense, dump at 10."""
    player = battle.players[1]

    if player.elixir >= 4 and "HogRider" in player.hand:
        bridge_x = _random.choice([3.5, 14.5])
        battle.deploy_card(1, "HogRider", Position(bridge_x, 18.0))
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

class ClashRoyaleEnv(gym.Env):
    """Gymnasium environment for 2.6 Hog Cycle mirror match.

    Two-step action: MultiDiscrete([5, 30])
      action[0] = card choice (0=WAIT, 1-4=hand slot)
      action[1] = region (0-29, ignored if WAIT)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent: str = "rule_bot",
        ticks_per_step: int = TICKS_PER_STEP,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.ticks_per_step = ticks_per_step
        self.render_mode = render_mode

        if opponent == "rule_bot":
            self._opponent_fn: Callable = rule_bot_policy
        elif opponent == "none":
            self._opponent_fn = lambda b: None
        else:
            raise ValueError(f"Unknown opponent: {opponent}")

        # Two-step action space
        self.action_space = spaces.MultiDiscrete([N_CARD_CHOICES, N_REGIONS])

        # Observation
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32,
        )

        # State
        self.battle: Optional[BattleState] = None
        self._prev_my_tower_hp = 0.0
        self._prev_opp_tower_hp = 0.0
        self._prev_my_crowns = 0
        self._prev_opp_crowns = 0
        self._prev_opp_entity_hp = 0.0

    # ── Gym API ───────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

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
        self._prev_opp_entity_hp = self._total_entity_hp(self.battle, 1)

        return self._get_obs(), {"battle_time": 0.0}

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.battle is not None, "Call reset() first"

        card_choice = int(action[0])
        region = int(action[1])

        # 1. Execute action
        deployed = False
        if card_choice > 0:
            slot = card_choice - 1
            player = self.battle.players[0]
            if slot < len(player.hand):
                card_name = player.hand[slot]
                pos = region_to_position(region)
                deployed = self.battle.deploy_card(0, card_name, pos)

        # 2. Opponent acts
        self._opponent_fn(self.battle)

        # 3. Advance simulation
        for _ in range(self.ticks_per_step):
            self.battle.step(speed_factor=1.0)
            if self.battle.game_over:
                break

        # 4. Reward
        reward = self._compute_reward(deployed)

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

    def action_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return masks for both action dimensions.

        Returns:
            card_mask: shape (5,) — which card choices are valid
            region_mask: shape (30,) — which regions are valid for deployment
        """
        card_mask = np.zeros(N_CARD_CHOICES, dtype=bool)
        card_mask[0] = True  # WAIT always valid

        region_mask = np.ones(N_REGIONS, dtype=bool)  # all regions valid by default

        if self.battle is None or self.battle.game_over:
            return card_mask, region_mask

        player = self.battle.players[0]

        for slot in range(min(4, len(player.hand))):
            card_name = player.hand[slot]
            cost = ELIXIR_COST.get(card_name, 10)
            if player.elixir >= cost:
                card_mask[slot + 1] = True

        # Region masking: check which regions have valid deploy tiles
        for r in range(N_REGIONS):
            pos = region_to_position(r)
            if not self.battle.arena.can_deploy_at(pos, 0, self.battle, False, None):
                # Check if it's valid for spells (any card in hand is a spell)
                from .spells import SPELL_REGISTRY
                from .card_aliases import resolve_card_name
                any_spell_in_hand = any(
                    resolve_card_name(c, self.battle.card_loader.load_card_definitions())
                    in SPELL_REGISTRY
                    for c in player.hand
                )
                if not any_spell_in_hand:
                    region_mask[r] = False

        return card_mask, region_mask

    # ── Observation ───────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build flat observation vector of shape (171,)."""
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        p0 = self.battle.players[0]
        p1 = self.battle.players[1]

        # Scalars [0..11]
        obs[0] = p0.elixir / 10.0
        obs[1] = p1.elixir / 10.0
        obs[2] = p0.king_tower_hp / MAX_KING_HP
        obs[3] = p0.left_tower_hp / MAX_PRINCESS_HP
        obs[4] = p0.right_tower_hp / MAX_PRINCESS_HP
        obs[5] = p1.king_tower_hp / MAX_KING_HP
        obs[6] = p1.left_tower_hp / MAX_PRINCESS_HP
        obs[7] = p1.right_tower_hp / MAX_PRINCESS_HP
        obs[8] = min(self.battle.time / MAX_GAME_TIME, 1.0)
        obs[9] = float(self.battle.double_elixir)
        obs[10] = float(self.battle.triple_elixir)
        obs[11] = float(self.battle.overtime)

        # Hand [12..20]
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

        # Units [21..170]
        offset = N_SCALARS + N_HAND
        unit_idx = 0
        for entity in self.battle.entities.values():
            if unit_idx >= MAX_UNITS:
                break
            if not entity.is_alive:
                continue

            is_troop = isinstance(entity, Troop)
            is_building = isinstance(entity, Building)
            if not (is_troop or is_building):
                continue

            if is_building and entity.position.y in (2.5, 29.5):
                card_id = KING_TOWER_ID
            elif is_building and entity.position.y in (6.5, 25.5):
                card_id = PRINCESS_TOWER_ID
            elif entity.card_stats:
                card_id = CARD_TO_IDX.get(entity.card_stats.name, 0)
            else:
                card_id = 0

            base = offset + unit_idx * UNIT_FEATURES
            obs[base + 0] = card_id / (NUM_CARD_IDS - 1)
            obs[base + 1] = np.clip(entity.position.x / 18.0, 0.0, 1.0)
            obs[base + 2] = np.clip(entity.position.y / 32.0, 0.0, 1.0)
            obs[base + 3] = (
                entity.hitpoints / entity.max_hitpoints
                if entity.max_hitpoints > 0 else 0.0
            )
            obs[base + 4] = 0.0 if entity.player_id == 0 else 1.0
            unit_idx += 1

        return obs

    # ── Reward ────────────────────────────────────────────────────────────

    def _compute_reward(self, deployed: bool = False) -> float:
        """Dense reward from tower HP, entity damage, crowns, and win bonus."""
        my_hp = self._total_tower_hp(0)
        opp_hp = self._total_tower_hp(1)

        tower_damage_dealt = (self._prev_opp_tower_hp - opp_hp) / MAX_TOTAL_HP
        tower_damage_taken = (self._prev_my_tower_hp - my_hp) / MAX_TOTAL_HP

        opp_entity_hp = self._total_entity_hp(self.battle, 1)
        entity_damage = max(0, self._prev_opp_entity_hp - opp_entity_hp)
        entity_damage_norm = entity_damage / 5000.0

        my_crowns = self.battle.players[1].get_crown_count()
        opp_crowns = self.battle.players[0].get_crown_count()
        crown_delta = (
            (my_crowns - self._prev_my_crowns)
            - (opp_crowns - self._prev_opp_crowns)
        )

        win_bonus = 0.0
        if self.battle.game_over:
            if self.battle.winner == 0:
                win_bonus = 50.0
            elif self.battle.winner == 1:
                win_bonus = -50.0

        elixir = self.battle.players[0].elixir
        leak_penalty = -0.005 * max(0, elixir - 8.0)

        self._prev_my_tower_hp = my_hp
        self._prev_opp_tower_hp = opp_hp
        self._prev_my_crowns = my_crowns
        self._prev_opp_crowns = opp_crowns
        self._prev_opp_entity_hp = opp_entity_hp

        return (
            tower_damage_dealt * 1.0
            - tower_damage_taken * 0.5
            + entity_damage_norm * 0.1
            + crown_delta * 10.0
            + win_bonus
            + leak_penalty
        )

    def _total_tower_hp(self, player_id: int) -> float:
        p = self.battle.players[player_id]
        return float(
            max(0, p.king_tower_hp)
            + max(0, p.left_tower_hp)
            + max(0, p.right_tower_hp)
        )

    @staticmethod
    def _total_entity_hp(battle: BattleState, player_id: int) -> float:
        total = 0.0
        for e in battle.entities.values():
            if not e.is_alive or e.player_id != player_id:
                continue
            if isinstance(e, Building) and e.position.y in (2.5, 6.5, 25.5, 29.5):
                continue
            if isinstance(e, (Troop, Building)):
                total += e.hitpoints
        return total
