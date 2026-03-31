"""V3 Gymnasium environment — CNN + domain randomization + card tracker.

Changes from V2:
  - Domain randomization per game (speed, damage, deploy delay, HP, placement jitter)
  - No adversarial flag (minimax is handled externally)
  - Cleaner reward (same as V2 without adversarial path)
  - Training bot as default opponent (not rule bot)

Usage:
    from clasher.env_v3 import ClashRoyaleEnvV3
    env = ClashRoyaleEnvV3()
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
from .card_tracker import CardTracker
from .training_bot import training_bot_policy

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

NUM_CARD_IDS = 11
PRINCESS_TOWER_ID = 9
KING_TOWER_ID = 10

ARENA_W = 18
ARENA_H = 32

N_CARD_CHOICES = 5
GRID_X = 18
GRID_Y = 15
TICKS_PER_STEP = 20

MAX_KING_HP = 4824.0
MAX_PRINCESS_HP = 3052.0
MAX_GAME_TIME = 360.0

N_SCALARS = 26


def tile_to_position(tx: int, ty: int) -> Position:
    return Position(tx + 0.5, ty + 0.5)


# ── Domain Randomization ─────────────────────────────────────────────────────

class DomainRandomizer:
    """Randomizes game parameters each reset for sim-to-real robustness."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.speed_mult = 1.0
        self.damage_mult = 1.0
        self.deploy_delay_noise = 0.0
        self.tower_hp_mult = 1.0
        self.elixir_obs_noise = 0.0
        self.placement_jitter_x = 0.0
        self.placement_jitter_y = 0.0

    def randomize(self, rng: np.random.Generator):
        """Generate new random parameters for this game."""
        if not self.enabled:
            return
        self.speed_mult = rng.uniform(0.9, 1.1)
        self.damage_mult = rng.uniform(0.95, 1.05)
        self.deploy_delay_noise = rng.uniform(-0.2, 0.2)
        self.tower_hp_mult = rng.uniform(0.97, 1.03)
        self.elixir_obs_noise = rng.uniform(-0.3, 0.3)
        self.placement_jitter_x = rng.uniform(-0.5, 0.5)
        self.placement_jitter_y = rng.uniform(-0.5, 0.5)

    def jitter_position(self, pos: Position) -> Position:
        """Add placement noise."""
        if not self.enabled:
            return pos
        return Position(
            max(0.5, min(17.5, pos.x + self.placement_jitter_x)),
            max(0.5, min(14.5, pos.y + self.placement_jitter_y)),
        )

    def noisy_elixir(self, elixir: float) -> float:
        """Add observation noise to opponent elixir."""
        if not self.enabled:
            return elixir
        return max(0.0, min(10.0, elixir + self.elixir_obs_noise))


# ── Environment ───────────────────────────────────────────────────────────────

class ClashRoyaleEnvV3(gym.Env):
    """V3: CNN spatial obs + domain randomization + card tracker.

    Observation: Dict{"spatial": (3,32,18), "scalars": (26,)}
    Action: MultiDiscrete([5, 18, 15])
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent: str = "training_bot",
        ticks_per_step: int = TICKS_PER_STEP,
        backend: str = "python",
        domain_randomization: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.ticks_per_step = ticks_per_step
        self.render_mode = render_mode

        if opponent == "training_bot":
            self._opponent_fn: Callable = training_bot_policy
        elif opponent == "none":
            self._opponent_fn = lambda b: None
        else:
            raise ValueError(f"Unknown opponent: {opponent}")

        self.use_rust = backend == "rust" and RUST_AVAILABLE

        self.action_space = spaces.MultiDiscrete([N_CARD_CHOICES, GRID_X, GRID_Y])
        self.observation_space = spaces.Dict({
            "spatial": spaces.Box(0.0, 1.0, shape=(3, ARENA_H, ARENA_W), dtype=np.float32),
            "scalars": spaces.Box(0.0, 1.0, shape=(N_SCALARS,), dtype=np.float32),
        })

        self.battle: Optional[BattleState] = None
        self._prev_my_tower_hp = 0.0
        self._prev_opp_tower_hp = 0.0
        self._prev_my_crowns = 0
        self._prev_opp_crowns = 0
        self._card_tracker = CardTracker()
        self._prev_opp_hand: list = []
        self._domain_rand = DomainRandomizer(enabled=domain_randomization)

    # ── Gym API ───────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
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

        # Domain randomization
        self._domain_rand.randomize(self.np_random)

        # Card tracker
        self._card_tracker.reset()
        p0 = self.battle.players[0]
        hand_idx = [CARD_TO_IDX.get(c, 0) for c in p0.hand]
        queue_idx = [CARD_TO_IDX.get(c, 0) for c in p0.cycle_queue]
        self._card_tracker.set_own_hand(hand_idx, queue_idx)
        self._prev_opp_hand = list(self.battle.players[1].hand)

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

        # 1. Execute action (with placement jitter)
        deployed = False
        if card_choice > 0:
            slot = card_choice - 1
            player = self.battle.players[0]
            if slot < len(player.hand):
                card_name = player.hand[slot]
                card_idx = CARD_TO_IDX.get(card_name, 0)
                pos = tile_to_position(tile_x, tile_y)
                pos = self._domain_rand.jitter_position(pos)
                if self.battle.deploy_card(0, card_name, pos):
                    deployed = True
                    self._card_tracker.own_card_played(card_idx)

        # 2. Opponent acts
        self._opponent_fn(self.battle)

        # 2b. Track opponent card plays
        opp_hand_now = list(self.battle.players[1].hand)
        for old_card in self._prev_opp_hand:
            if old_card not in opp_hand_now:
                self._card_tracker.opponent_card_played(CARD_TO_IDX.get(old_card, 0))
        self._prev_opp_hand = opp_hand_now

        # 3. Advance simulation
        for _ in range(self.ticks_per_step):
            self.battle.step(speed_factor=1.0)
            if self.battle.game_over:
                break

        # 4. Reward
        reward = self._compute_reward()

        # 5. Observation
        obs = self._get_obs()

        # 6. Terminal
        terminated = self.battle.game_over
        info = {
            "battle_time": self.battle.time,
            "winner": self.battle.winner,
            "my_crowns": self.battle.players[1].get_crown_count(),
            "opp_crowns": self.battle.players[0].get_crown_count(),
        }

        return obs, reward, terminated, False, info

    # ── Observation ───────────────────────────────────────────────────────

    def _get_obs(self):
        spatial = np.zeros((3, ARENA_H, ARENA_W), dtype=np.float32)
        scalars = np.zeros(N_SCALARS, dtype=np.float32)

        p0 = self.battle.players[0]
        p1 = self.battle.players[1]

        # Spatial grid
        for entity in self.battle.entities.values():
            if not entity.is_alive or not isinstance(entity, (Troop, Building)):
                continue
            gx = max(0, min(ARENA_W - 1, int(entity.position.x)))
            gy = max(0, min(ARENA_H - 1, int(entity.position.y)))

            if isinstance(entity, Building) and entity.position.y in (2.5, 29.5):
                card_id = KING_TOWER_ID
            elif isinstance(entity, Building) and entity.position.y in (6.5, 25.5):
                card_id = PRINCESS_TOWER_ID
            elif entity.card_stats:
                card_id = CARD_TO_IDX.get(entity.card_stats.name, 0) + 1
            else:
                card_id = 0

            spatial[0, gy, gx] = card_id / NUM_CARD_IDS
            spatial[1, gy, gx] = entity.hitpoints / entity.max_hitpoints if entity.max_hitpoints > 0 else 0.0
            spatial[2, gy, gx] = 0.0 if entity.player_id == 0 else 1.0

        # Scalars
        scalars[0] = p0.elixir / 10.0
        scalars[1] = self._domain_rand.noisy_elixir(p1.elixir) / 10.0
        scalars[2] = max(0, p0.king_tower_hp) / MAX_KING_HP
        scalars[3] = max(0, p0.left_tower_hp) / MAX_PRINCESS_HP
        scalars[4] = max(0, p0.right_tower_hp) / MAX_PRINCESS_HP
        scalars[5] = max(0, p1.king_tower_hp) / MAX_KING_HP
        scalars[6] = max(0, p1.left_tower_hp) / MAX_PRINCESS_HP
        scalars[7] = max(0, p1.right_tower_hp) / MAX_PRINCESS_HP
        scalars[8] = min(self.battle.time / MAX_GAME_TIME, 1.0)
        scalars[9] = (0.0 if not self.battle.double_elixir
                      else 0.5 if not self.battle.triple_elixir else 1.0)
        scalars[10:18] = self._card_tracker.get_own_observation()
        scalars[18:26] = self._card_tracker.get_opponent_observation()

        return {"spatial": spatial, "scalars": scalars}

    # ── Reward ────────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        my_hp = self._total_tower_hp(0)
        opp_hp = self._total_tower_hp(1)
        total_max = MAX_KING_HP + 2 * MAX_PRINCESS_HP

        dmg_dealt = (self._prev_opp_tower_hp - opp_hp) / total_max * 5.0
        dmg_taken = (self._prev_my_tower_hp - my_hp) / total_max * 2.5

        my_crowns = self.battle.players[1].get_crown_count()
        opp_crowns = self.battle.players[0].get_crown_count()
        r_crowns = ((my_crowns - self._prev_my_crowns) * 20.0
                    - (opp_crowns - self._prev_opp_crowns) * 30.0)

        r_win = 0.0
        if self.battle.game_over:
            if self.battle.winner == 0:
                r_win = 50.0
            elif self.battle.winner == 1:
                r_win = -50.0

        r_leak = -0.5 if self.battle.players[0].elixir >= 9.8 else 0.0

        self._prev_my_tower_hp = my_hp
        self._prev_opp_tower_hp = opp_hp
        self._prev_my_crowns = my_crowns
        self._prev_opp_crowns = opp_crowns

        return dmg_dealt - dmg_taken + r_crowns + r_win + r_leak

    def _total_tower_hp(self, pid: int) -> float:
        p = self.battle.players[pid]
        return float(max(0, p.king_tower_hp) + max(0, p.left_tower_hp) + max(0, p.right_tower_hp))
