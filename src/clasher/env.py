"""Gymnasium environment for Clash Royale 2.6 Hog Cycle mirror matches.

Usage:
    import gymnasium as gym
    from clasher.env import ClashRoyaleEnv

    env = ClashRoyaleEnv(opponent="rule_bot")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # WAIT
"""

from __future__ import annotations

import random as _random
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .arena import Position
from .battle import BattleState
from .entities import Building, Troop
from .tournament_standard import apply_tournament_overrides

# ── Constants ─────────────────────────────────────────────────────────────────

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

# Unit IDs for towers (beyond the 8 deck cards)
PRINCESS_TOWER_ID = 8
KING_TOWER_ID = 9
NUM_CARD_IDS = 10  # 0-7 deck cards, 8 princess tower, 9 king tower

MAX_UNITS = 30
UNIT_FEATURES = 5  # card_id_norm, x_norm, y_norm, hp_pct, owner

GRID_X = 18
GRID_Y = 15  # player 0 deploy zone: y=0..14
TICKS_PER_STEP = 30  # 1 second of game time

# Observation sizes
N_SCALARS = 12  # 2 elixir + 6 tower hp + time + double + triple + overtime
N_HAND = 9     # 4 hand + 1 next + 4 can_afford
N_UNITS = MAX_UNITS * UNIT_FEATURES
OBS_SIZE = N_SCALARS + N_HAND + N_UNITS  # 171

# Action space: 0=WAIT, 1..1080 = (card_slot, tile_x, tile_y)
N_ACTIONS = 1 + 4 * GRID_X * GRID_Y  # 1081

# Tower max HPs for normalization
MAX_KING_HP = 4824.0
MAX_PRINCESS_HP = 3052.0
MAX_TOTAL_HP = MAX_KING_HP + 2 * MAX_PRINCESS_HP  # 10928

MAX_GAME_TIME = 360.0  # tiebreaker time in seconds


def decode_action(action: int) -> Tuple[int, int, int]:
    """Decode action index to (card_slot, tile_x, tile_y). Action 0 is WAIT."""
    a = action - 1
    card_slot = a // (GRID_X * GRID_Y)
    remainder = a % (GRID_X * GRID_Y)
    tile_x = remainder // GRID_Y
    tile_y = remainder % GRID_Y
    return card_slot, tile_x, tile_y


def encode_action(card_slot: int, tile_x: int, tile_y: int) -> int:
    """Encode (card_slot, tile_x, tile_y) to action index."""
    return 1 + card_slot * (GRID_X * GRID_Y) + tile_x * GRID_Y + tile_y


# ── Rule Bot ──────────────────────────────────────────────────────────────────

def rule_bot_policy(battle: BattleState) -> None:
    """Simple opponent: rush Hog at bridge, Cannon on defense, dump at 10 elixir."""
    player = battle.players[1]

    # Rush Hog if affordable
    if player.elixir >= 4 and "HogRider" in player.hand:
        bridge_x = _random.choice([3.5, 14.5])
        battle.deploy_card(1, "HogRider", Position(bridge_x, 18.0))
        return

    # Defensive Cannon if enemy troops on our side
    has_threat = any(
        isinstance(e, Troop) and e.player_id == 0 and e.position.y > 16
        for e in battle.entities.values()
    )
    if has_threat and player.elixir >= 3 and "Cannon" in player.hand:
        battle.deploy_card(1, "Cannon", Position(9.0, 22.0))
        return

    # Leak prevention: dump cheapest card at 10 elixir
    if player.elixir >= 9.5:
        cheapest = min(player.hand, key=lambda c: ELIXIR_COST.get(c, 10))
        battle.deploy_card(1, cheapest, Position(9.0, 20.0))


# ── Environment ───────────────────────────────────────────────────────────────

class ClashRoyaleEnv(gym.Env):
    """Gymnasium environment for 2.6 Hog Cycle mirror match."""

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

        # Opponent policy
        if opponent == "rule_bot":
            self._opponent_fn: Callable = rule_bot_policy
        elif opponent == "none":
            self._opponent_fn = lambda b: None
        else:
            raise ValueError(f"Unknown opponent: {opponent}")

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # State (initialized in reset)
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

        # Set decks with random opening hands
        for p in self.battle.players:
            shuffled = list(DECK)
            self.np_random.shuffle(shuffled)
            p.deck = list(DECK)
            p.hand = shuffled[:4]
            p.cycle_queue = deque(shuffled[4:])
            p.elixir = 5.0  # starting elixir in real CR

        # Clear deploy grid cache
        self._deploy_grid = None

        # Record initial state for reward
        self._prev_my_tower_hp = self._total_tower_hp(0)
        self._prev_opp_tower_hp = self._total_tower_hp(1)
        self._prev_my_crowns = 0
        self._prev_opp_crowns = 0
        self._prev_opp_entity_hp = self._total_entity_hp(self.battle, 1)

        obs = self._get_obs()
        info = {"battle_time": 0.0}
        return obs, info

    def step(
        self, action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.battle is not None, "Call reset() first"

        # 1. Decode and execute player action
        deployed = False
        if action != 0:
            card_slot, tile_x, tile_y = decode_action(action)
            player = self.battle.players[0]
            if 0 <= card_slot < len(player.hand):
                card_name = player.hand[card_slot]
                deployed = self.battle.deploy_card(
                    0, card_name, Position(tile_x + 0.5, tile_y + 0.5),
                )

        # 2. Opponent acts
        self._opponent_fn(self.battle)

        # 3. Advance simulation
        for _ in range(self.ticks_per_step):
            self.battle.step(speed_factor=1.0)
            if self.battle.game_over:
                break

        # 3b. Invalidate deploy grid if tower state changed
        if self._deploy_grid is not None:
            current_tower_state = (
                self.battle.players[0].left_tower_hp > 0,
                self.battle.players[0].right_tower_hp > 0,
                self.battle.players[1].left_tower_hp > 0,
                self.battle.players[1].right_tower_hp > 0,
            )
            if current_tower_state != self._deploy_grid_tower_state:
                self._deploy_grid = None

        # 4. Compute reward
        reward = self._compute_reward(deployed=deployed)

        # 5. Build observation
        obs = self._get_obs()

        # 6. Terminal check
        terminated = self.battle.game_over
        truncated = False

        info = {
            "battle_time": self.battle.time,
            "winner": self.battle.winner,
            "my_crowns": self.battle.players[1].get_crown_count(),
            "opp_crowns": self.battle.players[0].get_crown_count(),
        }

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """MaskablePPO-compatible action mask. Shape (1081,).

        Guarantees at least one action is always valid (WAIT).
        """
        mask = self.valid_action_mask()
        # Safety: ensure mask is never all-False (would crash MaskablePPO)
        if not mask.any():
            mask[0] = True
        return mask

    def valid_action_mask(self) -> np.ndarray:
        """Return boolean mask of valid actions. Shape (1081,)."""
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[0] = True  # WAIT is always valid

        if self.battle is None or self.battle.game_over:
            return mask

        player = self.battle.players[0]

        # Pre-compute valid deploy grid once (changes only when towers die)
        if not hasattr(self, "_deploy_grid") or self._deploy_grid is None:
            self._recompute_deploy_grid()

        for slot in range(min(4, len(player.hand))):
            card_name = player.hand[slot]
            cost = ELIXIR_COST.get(card_name, 10)
            if player.elixir < cost:
                continue

            # Spells can target anywhere on the 18x15 grid
            from .spells import SPELL_REGISTRY
            from .card_aliases import resolve_card_name
            resolved = resolve_card_name(
                card_name,
                self.battle.card_loader.load_card_definitions(),
            )
            is_spell = resolved in SPELL_REGISTRY

            if is_spell:
                # Enable all grid positions for this card slot
                base = 1 + slot * (GRID_X * GRID_Y)
                mask[base:base + GRID_X * GRID_Y] = True
            else:
                # Use cached deploy grid
                base = 1 + slot * (GRID_X * GRID_Y)
                mask[base:base + GRID_X * GRID_Y] = self._deploy_grid

        return mask

    def _recompute_deploy_grid(self) -> None:
        """Cache which tiles are valid for troop deployment."""
        grid = np.zeros(GRID_X * GRID_Y, dtype=bool)
        for tx in range(GRID_X):
            for ty in range(GRID_Y):
                pos = Position(tx + 0.5, ty + 0.5)
                if self.battle.arena.can_deploy_at(pos, 0, self.battle, False, None):
                    grid[tx * GRID_Y + ty] = True
        self._deploy_grid = grid
        # Track tower state to know when to recompute
        self._deploy_grid_tower_state = (
            self.battle.players[0].left_tower_hp > 0,
            self.battle.players[0].right_tower_hp > 0,
            self.battle.players[1].left_tower_hp > 0,
            self.battle.players[1].right_tower_hp > 0,
        )

    # ── Internals ─────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build flat observation vector of shape (171,)."""
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        p0 = self.battle.players[0]
        p1 = self.battle.players[1]

        # Scalars [0..11]
        obs[0] = p0.elixir / 10.0
        obs[1] = p1.elixir / 10.0  # perfect info in sim
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
        # Next card
        if p0.cycle_queue:
            obs[offset + 4] = CARD_TO_IDX.get(p0.cycle_queue[0], 0) / (NUM_CARD_IDS - 1)
        # Can afford
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

            # Card ID
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

    def _compute_reward(self, deployed: bool = False) -> float:
        """Dense reward from tower HP, entity damage, crowns, and win bonus.

        Reward components:
        - Tower damage dealt/taken (primary objective)
        - Entity damage dealt (dense signal — rewards combat engagement)
        - Crown bonus (milestone reward)
        - Win/loss terminal bonus
        - Elixir leak penalty (continuous, proportional above 8)
        """
        my_hp = self._total_tower_hp(0)
        opp_hp = self._total_tower_hp(1)

        # Tower damage deltas (normalized)
        tower_damage_dealt = (self._prev_opp_tower_hp - opp_hp) / MAX_TOTAL_HP
        tower_damage_taken = (self._prev_my_tower_hp - my_hp) / MAX_TOTAL_HP

        # Entity damage dealt — any damage to enemy troops/buildings
        # This gives dense signal even before reaching towers
        opp_entity_hp = self._total_entity_hp(self.battle, 1)
        entity_damage = max(0, self._prev_opp_entity_hp - opp_entity_hp)
        # Normalize: a typical troop has ~1000 HP, normalize by 5000
        entity_damage_norm = entity_damage / 5000.0

        # Crown deltas
        my_crowns = self.battle.players[1].get_crown_count()
        opp_crowns = self.battle.players[0].get_crown_count()
        crown_delta = (
            (my_crowns - self._prev_my_crowns)
            - (opp_crowns - self._prev_opp_crowns)
        )

        # Win bonus
        win_bonus = 0.0
        if self.battle.game_over:
            if self.battle.winner == 0:
                win_bonus = 50.0
            elif self.battle.winner == 1:
                win_bonus = -50.0

        # Elixir leak penalty — continuous above 8 elixir
        elixir = self.battle.players[0].elixir
        leak_penalty = -0.005 * max(0, elixir - 8.0)  # -0.01 at 10 elixir per step

        # Update previous state
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

    @staticmethod
    def _total_entity_hp(battle: BattleState, player_id: int) -> float:
        """Sum HP of all living non-tower entities for a player."""
        total = 0.0
        for e in battle.entities.values():
            if not e.is_alive or e.player_id != player_id:
                continue
            if isinstance(e, Building) and e.position.y in (2.5, 6.5, 25.5, 29.5):
                continue  # skip towers
            if isinstance(e, (Troop, Building)):
                total += e.hitpoints
        return total

    def _total_tower_hp(self, player_id: int) -> float:
        """Sum of king + left + right tower HP for a player."""
        p = self.battle.players[player_id]
        return float(
            max(0, p.king_tower_hp)
            + max(0, p.left_tower_hp)
            + max(0, p.right_tower_hp)
        )
