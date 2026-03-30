"""Card cycle tracker for Hog 2.6 mirror match.

Tracks what we KNOW and can DEDUCE about both players' hands,
mimicking real Clash Royale information rules:

OWN HAND (perfect info):
  - Know all 4 cards in hand
  - Know the 5th card (next in queue)
  - Can deduce 6th, 7th, 8th from deck knowledge
  → 8 values: position in cycle (1.0=in hand, 0.0=8th in queue)

OPPONENT HAND (imperfect info — deduction only):
  - Start of game: know NOTHING about opponent's hand order
  - When opponent plays a card: we see it, know it's cycling back
  - After seeing all 8 cards played: we know the full cycle order
  - Card returns to hand after 4 more cards are played
  → 8 values: estimated probability card is in hand (0.0 to 1.0)

IMPORTANT: This assumes Hog 2.6 mirror (both players same 8 cards).
The opponent tracking would need rework for non-mirror matches.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np


DECK_SIZE = 8


class CardTracker:
    """Tracks card cycle state for both players in a mirror match."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracking state for a new game."""
        # ── Own hand: perfect information ──
        # Positions 0-3 = in hand, 4 = next card, 5-7 = further in queue
        # We track cycle position per card index (0-7 into DECK)
        self._own_cycle: List[int] = list(range(DECK_SIZE))  # overwritten on game start
        self._own_hand_set: set = set()

        # ── Opponent: imperfect information ──
        # Cards we've SEEN the opponent play (in order)
        self._opp_played_order: List[int] = []  # card indices in order played
        self._opp_cards_seen: set = set()  # which card indices we've ever seen
        self._opp_total_plays: int = 0  # total cards played by opponent
        # Per-card: how many opponent plays ago was this card last seen?
        # None = never seen
        self._opp_last_seen_at: List[Optional[int]] = [None] * DECK_SIZE

    def set_own_hand(self, hand_indices: List[int], queue_indices: List[int]):
        """Set own hand at game start. hand=[0,3,5,2], queue=[1,4,6,7]."""
        self._own_cycle = list(hand_indices) + list(queue_indices)
        self._own_hand_set = set(hand_indices)

    def own_card_played(self, card_idx: int):
        """Called when we play a card. Updates own cycle."""
        if card_idx in self._own_hand_set:
            self._own_hand_set.discard(card_idx)
            # Card moves to back of cycle
            if card_idx in self._own_cycle:
                self._own_cycle.remove(card_idx)
                self._own_cycle.append(card_idx)
            # New hand = first 4 of cycle
            self._own_hand_set = set(self._own_cycle[:4])

    def opponent_card_played(self, card_idx: int):
        """Called when we observe the opponent play a card."""
        self._opp_total_plays += 1
        self._opp_cards_seen.add(card_idx)
        self._opp_last_seen_at[card_idx] = self._opp_total_plays
        self._opp_played_order.append(card_idx)

    def get_own_observation(self) -> np.ndarray:
        """Own hand knowledge — 8 floats.

        Simple linear encoding of cycle position:
          pos 0-3 (in hand):  1.0
          pos 4 (next card):  0.75
          pos 5:              0.50
          pos 6:              0.25
          pos 7 (furthest):   0.00
        """
        obs = np.zeros(DECK_SIZE, dtype=np.float32)
        for pos, card_idx in enumerate(self._own_cycle):
            if pos < 4:
                obs[card_idx] = 1.0
            else:
                # pos 4→0.75, pos 5→0.50, pos 6→0.25, pos 7→0.00
                obs[card_idx] = 1.0 - (pos - 3) * 0.25
        return obs

    def get_opponent_observation(self) -> np.ndarray:
        """Opponent hand estimate — 8 floats.

        Rules mimicking real CR deduction:
        1. Before opponent plays any card: all 0.5 (unknown)
        2. When opponent plays card X: X is NOT in hand (cycling back)
        3. After opponent plays 4 more cards: X is back in hand
        4. Cards never seen: probability based on how many unknown slots remain

        Each value = estimated probability card is currently in opponent's hand.
        """
        obs = np.zeros(DECK_SIZE, dtype=np.float32)

        if self._opp_total_plays == 0:
            # Game just started — we know nothing. 4/8 = 0.5 chance each card is in hand
            obs[:] = 0.5
            return obs

        for card_idx in range(DECK_SIZE):
            last_seen = self._opp_last_seen_at[card_idx]

            if last_seen is None:
                # Never seen this card played
                # It could be in their opening hand or in their queue
                # More cards we've seen without seeing this one = more likely it's deep in queue
                # But we can't be sure
                n_unseen = DECK_SIZE - len(self._opp_cards_seen)
                if n_unseen > 0:
                    # Of the unseen cards, 4 - (seen cards in hand) are in hand
                    seen_in_hand = sum(
                        1 for c in range(DECK_SIZE)
                        if self._opp_last_seen_at[c] is not None
                        and self._opp_total_plays - self._opp_last_seen_at[c] >= 4
                    )
                    unknown_hand_slots = max(0, 4 - seen_in_hand)
                    obs[card_idx] = min(1.0, unknown_hand_slots / n_unseen)
                else:
                    obs[card_idx] = 0.5
            else:
                # We've seen this card. How many plays ago?
                plays_since = self._opp_total_plays - last_seen

                if plays_since < 4:
                    # Card was played recently — it's cycling, NOT in hand
                    # Closer to 4 = closer to being back
                    obs[card_idx] = plays_since / 4.0 * 0.5  # 0.0 → 0.375
                else:
                    # Card has cycled back into hand (4+ plays since we saw it)
                    obs[card_idx] = 1.0

        return obs

    def get_full_observation(self) -> np.ndarray:
        """Combined observation: own cycle (8) + opponent estimate (8) = 16 floats."""
        return np.concatenate([
            self.get_own_observation(),
            self.get_opponent_observation(),
        ])
