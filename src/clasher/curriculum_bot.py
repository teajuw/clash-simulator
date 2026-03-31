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
    """Predictable bot that plays a fixed cycle on elixir cooldown.

    Plays the next card in the cycle as soon as it can afford it.
    No timer — purely elixir-gated. This matches real CR pacing
    where you can only play as fast as elixir regenerates.
    """

    def __init__(self):
        self.cycle_idx = 0
        self.hog_bridge_x = _random.choice([3.5, 14.5])

    def reset(self):
        self.cycle_idx = 0
        self.hog_bridge_x = _random.choice([3.5, 14.5])

    def act(self, battle: BattleState) -> None:
        """Play next card in cycle when affordable."""
        player = battle.players[1]

        card_name = CYCLE[self.cycle_idx % len(CYCLE)]

        # Wait until this card is in hand AND affordable
        if card_name not in player.hand:
            return

        from .env_v3 import ELIXIR_COST
        cost = ELIXIR_COST.get(card_name, 10)
        if player.elixir < cost:
            return

        # Deploy
        if card_name == "HogRider":
            pos = Position(self.hog_bridge_x, 18.0)
        else:
            pos = PLACEMENTS[card_name]

        if battle.deploy_card(1, card_name, pos):
            self.cycle_idx += 1


def make_curriculum_bot_policy():
    """Create a curriculum bot with per-game state."""
    bot = CurriculumBot()

    def policy(battle: BattleState):
        # Reset bot on new game (detect by time near 0)
        if battle.time < 0.1:
            bot.reset()
        bot.act(battle)

    return policy
