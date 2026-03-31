"""Curriculum bots for progressive training.

Each level teaches ONE concept. Bots bypass the deck system
and spawn troops directly on elixir cooldown.

Level 0: No opponent
Level 1: Hog only (~28s cycle)        → learn to defend
Level 2: Hog + Musketeer (same side)  → learn to kill support
Level 3: Hog + Cannon (off-center)    → learn lane switching
Level 4: Hog + Musketeer + Cannon     → offense AND defense
"""

from __future__ import annotations

import random as _random
from typing import List, Tuple, Optional

from .arena import Position
from .battle import BattleState

ELIXIR_COST = {
    "HogRider": 4, "Musketeer": 4, "Cannon": 3,
    "IceGolem": 2, "Skeletons": 1, "IceSpirit": 1,
}

# Hog cycle time: dump 4 cheapest (1+1+2+2=6) + Hog (4) = 10 elixir
# At 2.8s/elixir = 28 seconds between Hogs
HOG_CYCLE_ELIXIR = 10


def _make_play_order(level: int, hog_x: float) -> List[Tuple[str, Position]]:
    """Build the play order for a given curriculum level."""
    # Musketeer placed same side as Hog, behind tower
    musk_x = 5.0 if hog_x < 9 else 13.0
    # Cannon placed same side as Hog (off-center, not a center pull)
    cannon_x = 5.0 if hog_x < 9 else 13.0

    if level <= 0:
        return []  # no opponent
    elif level == 1:
        # Hog only
        return [
            ("HogRider", Position(hog_x, 18.0)),
        ]
    elif level == 2:
        # Hog + Musketeer (same side push)
        return [
            ("HogRider", Position(hog_x, 18.0)),
            ("Musketeer", Position(musk_x, 27.0)),
        ]
    elif level == 3:
        # Hog + Cannon (teaches lane switching)
        return [
            ("HogRider", Position(hog_x, 18.0)),
            ("Cannon", Position(cannon_x, 22.0)),
        ]
    else:
        # Level 4+: Hog + Musketeer + Cannon (full offense + defense)
        return [
            ("HogRider", Position(hog_x, 18.0)),
            ("Musketeer", Position(musk_x, 27.0)),
            ("Cannon", Position(cannon_x, 22.0)),
        ]


class CurriculumBot:
    """Spawns troops directly on elixir cooldown. Bypasses deck system."""

    def __init__(self, level: int = 1):
        self.level = level
        self.play_idx = 0
        self.hog_x = _random.choice([3.5, 14.5])
        self.play_order = _make_play_order(level, self.hog_x)

    def reset(self):
        self.play_idx = 0
        self.hog_x = _random.choice([3.5, 14.5])
        self.play_order = _make_play_order(self.level, self.hog_x)

    def act(self, battle: BattleState) -> None:
        if not self.play_order:
            return

        player = battle.players[1]
        card_name, pos = self.play_order[self.play_idx % len(self.play_order)]
        cost = ELIXIR_COST.get(card_name, 4)

        if player.elixir < cost:
            return

        # Spend elixir
        player.elixir -= cost

        # Spawn directly
        card_stats = battle.card_loader.get_card(card_name)
        if card_stats is None:
            from .card_aliases import resolve_card_name
            resolved = resolve_card_name(card_name, battle.card_loader.load_card_definitions())
            card_stats = battle.card_loader.get_card(resolved)

        if card_stats is not None:
            card_type = str(getattr(card_stats, "card_type", "")).lower()
            if card_type == "building":
                from .entities import Building
                battle._spawn_entity(Building, pos, 1, card_stats)
            else:
                battle._spawn_troop(pos, 1, card_stats)
            self.play_idx += 1


def make_curriculum_bot_policy(level: int = 1):
    """Create a curriculum bot for the given level."""
    bot = CurriculumBot(level=level)

    def policy(battle: BattleState):
        if battle.time < 0.1:
            bot.reset()
        bot.act(battle)

    policy._bot = bot  # expose for level changes
    return policy
