"""Level 1 curriculum bot: predictable 5-card loop on a timer.

Plays cards on a fixed schedule regardless of what the agent does.
No reaction, no defense logic. Just a metronome of card plays
that the agent must learn to defend against.

Slight randomization: Hog goes left or right bridge (50/50 per game).
"""

from __future__ import annotations

import random as _random

from .arena import Position
from .battle import BattleState

# Fixed cycle: Hog → Musketeer → Cannon → IceGolem → Skeletons → repeat
CYCLE = ["HogRider", "Musketeer", "Cannon", "IceGolem", "Skeletons"]

# Placement positions for player 1 (top side)
PLACEMENTS = {
    "HogRider": None,      # set per game (left or right bridge)
    "Musketeer": Position(13.0, 27.0),   # behind right tower
    "Cannon": Position(9.0, 22.0),        # center pull
    "IceGolem": Position(9.0, 28.0),      # behind king tower
    "Skeletons": Position(9.0, 20.0),     # mid field
}


class CurriculumBot:
    """Predictable bot that plays a fixed cycle on a timer."""

    def __init__(self):
        self.cycle_idx = 0
        self.last_play_time = -5.0  # play immediately at start
        self.play_interval = 5.0   # seconds between plays
        self.hog_bridge_x = _random.choice([3.5, 14.5])  # fixed per game

    def reset(self):
        """Call on game reset."""
        self.cycle_idx = 0
        self.last_play_time = -5.0
        self.hog_bridge_x = _random.choice([3.5, 14.5])

    def act(self, battle: BattleState) -> None:
        """Play next card in cycle if timer elapsed."""
        player = battle.players[1]
        game_time = battle.time

        if game_time - self.last_play_time < self.play_interval:
            return

        # Find next card in cycle that's in hand
        for _ in range(len(CYCLE)):
            card_name = CYCLE[self.cycle_idx % len(CYCLE)]
            self.cycle_idx += 1

            if card_name not in player.hand:
                continue

            from .env_v3 import ELIXIR_COST
            cost = ELIXIR_COST.get(card_name, 10)
            if player.elixir < cost:
                return  # can't afford, wait

            # Get position
            if card_name == "HogRider":
                pos = Position(self.hog_bridge_x, 18.0)
            else:
                pos = PLACEMENTS[card_name]

            if battle.deploy_card(1, card_name, pos):
                self.last_play_time = game_time
                return


def make_curriculum_bot_policy():
    """Create a curriculum bot with per-game state."""
    bot = CurriculumBot()

    def policy(battle: BattleState):
        # Reset bot on new game (detect by time near 0)
        if battle.time < 0.1:
            bot.reset()
        bot.act(battle)

    return policy
