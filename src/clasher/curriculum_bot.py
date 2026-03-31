"""Level 1 curriculum bot: plays fixed cards on elixir cooldown.

Bypasses the deck/hand system entirely. Directly deploys cards
when it can afford them. No cycling, no hand management.

This teaches the agent:
- Hog is coming, learn to defend
- Musketeer provides ranged pressure
- Cards come at realistic elixir pacing
"""

from __future__ import annotations

import random as _random

from .arena import Position
from .battle import BattleState

ELIXIR_COST = {
    "HogRider": 4, "Musketeer": 4, "Cannon": 3, "IceGolem": 2, "Skeletons": 1,
}

# Play order and positions for player 1
PLAY_ORDER = [
    ("HogRider",  None),          # bridge position set per game
    ("Musketeer", Position(13.0, 27.0)),  # behind right tower
    ("Skeletons", Position(9.0, 20.0)),   # mid field
    ("HogRider",  None),          # bridge again
    ("IceGolem",  Position(9.0, 28.0)),   # behind king
]


class CurriculumBot:
    """Plays fixed cards directly, bypassing hand/deck system."""

    def __init__(self):
        self.play_idx = 0
        self.hog_x = _random.choice([3.5, 14.5])

    def reset(self):
        self.play_idx = 0
        self.hog_x = _random.choice([3.5, 14.5])

    def act(self, battle: BattleState) -> None:
        player = battle.players[1]

        card_name, pos = PLAY_ORDER[self.play_idx % len(PLAY_ORDER)]
        cost = ELIXIR_COST[card_name]

        if player.elixir < cost:
            return

        if card_name == "HogRider":
            pos = Position(self.hog_x, 18.0)

        # Spend elixir manually
        player.elixir -= cost

        # Spawn the troop/building directly (bypass hand system)
        card_stats = battle.card_loader.get_card(card_name)
        if card_stats is None:
            # Try alias
            from .card_aliases import resolve_card_name
            resolved = resolve_card_name(card_name, battle.card_loader.load_card_definitions())
            card_stats = battle.card_loader.get_card(resolved)

        if card_stats is not None:
            card_type = str(getattr(card_stats, "card_type", "")).lower()
            if card_type == "building":
                battle._spawn_entity(
                    type(list(battle.entities.values())[0]),  # Building class
                    pos, 1, card_stats
                )
            else:
                battle._spawn_troop(pos, 1, card_stats)
            self.play_idx += 1


def make_curriculum_bot_policy():
    bot = CurriculumBot()

    def policy(battle: BattleState):
        if battle.time < 0.1:
            bot.reset()
        bot.act(battle)

    return policy
