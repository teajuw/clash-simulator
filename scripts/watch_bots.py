#!/usr/bin/env python3
"""Watch smart bot vs smart bot in pygame.

Usage:
    python scripts/watch_bots.py
"""

import sys
import os
import random
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import clasher.config as cfg
cfg.VERBOSE = False

from clasher.battle import BattleState
from clasher.arena import Position
from clasher.tournament_standard import apply_tournament_overrides
from clasher.smart_bot import SmartBot
from clasher.env_v3 import DECK, ELIXIR_COST

from visualize_battle import BattleVisualizer, pygame, SCREEN_WIDTH, SCREEN_HEIGHT


class BotVsBotVisualizer(BattleVisualizer):
    """Watch two smart bots play each other."""

    def __init__(self):
        super().__init__()
        self.bot_p0 = SmartBot(player_id=0)
        self.bot_p1 = SmartBot(player_id=1)
        self.tick_counter = 0
        self.decision_interval = 20  # bots decide every 20 ticks (1 second)

        # Set up 2.6 deck with tournament standard
        apply_tournament_overrides(self.battle)
        for p in self.battle.players:
            shuffled = list(DECK)
            random.shuffle(shuffled)
            p.deck = list(DECK)
            p.hand = shuffled[:4]
            p.cycle_queue = deque(shuffled[4:])
            p.elixir = 5.0

        self.paused = False
        self.last_action_p0 = "WAIT"
        self.last_action_p1 = "WAIT"

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.__init__()

            if self.paused or self.battle.game_over:
                self.render_frame()
                self.clock.tick(30)
                continue

            # Step simulation
            self.battle.step(speed_factor=1.0)
            self.tick_counter += 1

            # Bots decide every N ticks
            if self.tick_counter % self.decision_interval == 0:
                # Track actions
                hand_p0 = list(self.battle.players[0].hand)
                hand_p1 = list(self.battle.players[1].hand)

                self.bot_p0.act(self.battle)
                self.bot_p1.act(self.battle)

                hand_p0_after = list(self.battle.players[0].hand)
                hand_p1_after = list(self.battle.players[1].hand)

                if hand_p0 != hand_p0_after:
                    played = [c for c in hand_p0 if c not in hand_p0_after]
                    self.last_action_p0 = played[0] if played else "?"
                else:
                    self.last_action_p0 = "WAIT"

                if hand_p1 != hand_p1_after:
                    played = [c for c in hand_p1 if c not in hand_p1_after]
                    self.last_action_p1 = played[0] if played else "?"
                else:
                    self.last_action_p1 = "WAIT"

            self.render_frame()
            self.clock.tick(30)

        pygame.quit()

    def render_frame(self):
        self.draw_arena()
        self.draw_entities()
        self.draw_ui()

        font = self.large_font
        y = SCREEN_HEIGHT - 100

        # P0 info
        p0 = self.battle.players[0]
        p0_text = font.render(
            f"P0 (Blue): {self.last_action_p0}  Elixir: {p0.elixir:.0f}  Hand: {', '.join(p0.hand)}",
            True, (100, 100, 255))
        self.screen.blit(p0_text, (20, y))

        # P1 info
        p1 = self.battle.players[1]
        p1_text = font.render(
            f"P1 (Red):  {self.last_action_p1}  Elixir: {p1.elixir:.0f}  Hand: {', '.join(p1.hand)}",
            True, (255, 100, 100))
        self.screen.blit(p1_text, (20, y + 25))

        # Game state
        time_text = self.small_font.render(
            f"Time: {self.battle.time:.0f}s  |  SPACE=pause  R=reset  ESC=quit",
            True, (200, 200, 200))
        self.screen.blit(time_text, (20, y + 55))

        if self.battle.game_over:
            winner = "P0 (Blue)" if self.battle.winner == 0 else "P1 (Red)" if self.battle.winner == 1 else "DRAW"
            win_text = font.render(f"GAME OVER — {winner} wins!", True, (255, 255, 0))
            self.screen.blit(win_text, (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2))

        pygame.display.flip()


if __name__ == "__main__":
    viz = BotVsBotVisualizer()
    viz.run()
