"""Rust backend wrapper — makes PyBattle look like the Python BattleState.

This provides a compatibility layer so the Gymnasium env can use either
backend with minimal code changes.

Usage:
    from clasher.rust_backend import RustBattle
    battle = RustBattle()
    battle.deploy_card(0, "HogRider", Position(14, 14))
    battle.step()
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

try:
    from clash_sim import PyBattle
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

from .arena import Position

# Deck order matches cards.rs DECK array
DECK_NAMES = [
    "HogRider", "Musketeer", "IceGolem", "IceSpirit",
    "Cannon", "Fireball", "Skeletons", "TheLog",
]
NAME_TO_IDX = {name: i for i, name in enumerate(DECK_NAMES)}
ELIXIR_COST = {
    "HogRider": 4, "Musketeer": 4, "IceGolem": 2, "IceSpirit": 1,
    "Cannon": 3, "Fireball": 4, "Skeletons": 1, "TheLog": 2,
}


class RustPlayerView:
    """Read-only view of a player's state from the Rust backend."""

    def __init__(self, state: dict, player_id: int):
        prefix = f"p{player_id}"
        self.player_id = player_id
        self.elixir = state[f"{prefix}_elixir"]
        self.king_tower_hp = state[f"{prefix}_king_hp"]
        self.left_tower_hp = state[f"{prefix}_left_hp"]
        self.right_tower_hp = state[f"{prefix}_right_hp"]
        # Hand as card names
        hand_indices = state[f"{prefix}_hand"]
        self.hand = [DECK_NAMES[i] for i in hand_indices]
        self.cycle_queue = deque()  # TODO: expose from Rust

    def get_crown_count(self) -> int:
        if self.king_tower_hp <= 0:
            return 3
        crowns = 0
        if self.left_tower_hp <= 0:
            crowns += 1
        if self.right_tower_hp <= 0:
            crowns += 1
        return crowns

    def is_alive(self) -> bool:
        return self.king_tower_hp > 0


class RustBattle:
    """Wrapper around PyBattle that provides a BattleState-like interface."""

    # Ticks per step in Rust (20/sec, 1 second = 20 ticks)
    TICKS_PER_GAME_SECOND = 20

    def __init__(self):
        if not RUST_AVAILABLE:
            raise ImportError(
                "Rust backend not available. Install with: "
                "cd clash-simulator-rs && python3 -m maturin develop --release"
            )
        self._inner = PyBattle()
        self._state = self._inner.get_state()

        # Player hand/deck state (managed on Python side for card name tracking)
        self._hands: List[List[str]] = [
            list(DECK_NAMES[:4]),
            list(DECK_NAMES[:4]),
        ]
        self._queues: List[deque] = [
            deque(DECK_NAMES[4:]),
            deque(DECK_NAMES[4:]),
        ]

    def _sync_state(self):
        """Pull latest state from Rust."""
        self._state = self._inner.get_state()

    @property
    def time(self) -> float:
        return self._state["time"]

    @property
    def tick(self) -> int:
        return self._state["tick"]

    @property
    def game_over(self) -> bool:
        return self._state["game_over"]

    @property
    def winner(self) -> Optional[int]:
        return self._state["winner"]

    @property
    def double_elixir(self) -> bool:
        return self._state["double_elixir"]

    @property
    def triple_elixir(self) -> bool:
        return self._state["triple_elixir"]

    @property
    def overtime(self) -> bool:
        return self._state["overtime"]

    @property
    def players(self) -> List[RustPlayerView]:
        """Return player views (refreshed from Rust state)."""
        self._sync_state()
        views = []
        for i in range(2):
            view = RustPlayerView(self._state, i)
            # Override hand with our tracked names
            view.hand = list(self._hands[i])
            view.cycle_queue = deque(self._queues[i])
            views.append(view)
        return views

    @property
    def entities(self) -> dict:
        """Minimal entities dict for compatibility. Returns empty — use get_observation() instead."""
        return {}

    def set_deck(self, player_id: int, hand: List[str], queue: List[str]):
        """Set hand and cycle queue for a player."""
        self._hands[player_id] = list(hand)
        self._queues[player_id] = deque(queue)
        hand_idx = [NAME_TO_IDX[c] for c in hand]
        queue_idx = [NAME_TO_IDX[c] for c in queue]
        self._inner.set_deck_order(
            player_id,
            [hand_idx[0], hand_idx[1], hand_idx[2], hand_idx[3]],
            [queue_idx[0], queue_idx[1], queue_idx[2], queue_idx[3]],
        )
        self._inner.set_elixir(player_id, 5.0)

    def deploy_card(self, player_id: int, card_name: str, position: Position) -> bool:
        """Deploy a card. Handles name→slot resolution."""
        hand = self._hands[player_id]
        if card_name not in hand:
            return False

        slot = hand.index(card_name)
        cost = ELIXIR_COST.get(card_name, 10)

        # Check elixir (read from Rust state)
        self._sync_state()
        prefix = f"p{player_id}"
        elixir = self._state[f"{prefix}_elixir"]
        if elixir < cost:
            return False

        ok = self._inner.deploy_card(player_id, slot, position.x, position.y)
        if ok:
            # Cycle hand on Python side too
            played = hand[slot]
            if self._queues[player_id]:
                next_card = self._queues[player_id].popleft()
                hand[slot] = next_card
                self._queues[player_id].append(played)
        return ok

    def step(self, speed_factor: float = 1.0):
        """Advance one tick. speed_factor is ignored (Rust uses fixed 20/sec)."""
        self._inner.step()
        self._sync_state()

    def step_n(self, n: int):
        """Advance N ticks."""
        self._inner.step_n(n)
        self._sync_state()

    def get_observation(self) -> np.ndarray:
        """Get the 171-element observation vector directly from Rust."""
        return np.array(self._inner.get_observation(), dtype=np.float32)
