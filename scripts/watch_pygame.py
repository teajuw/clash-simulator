#!/usr/bin/env python3
"""Watch a trained agent play with the pygame visualizer.

Usage:
    python scripts/watch_pygame.py --model models/selfplay_overnight_elo1670
    python scripts/watch_pygame.py  # random play
"""

import argparse
import sys
import os
import random
import time
from collections import deque

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)  # for visualize_battle.py
os.chdir(REPO_ROOT)  # for gamedata.json + hitboxes.json

import numpy as np

import clasher.config as cfg
cfg.VERBOSE = False

from clasher.env import (
    DECK, ELIXIR_COST, CARD_TO_IDX,
    N_REGIONS, N_CARD_CHOICES, OBS_SIZE,
    region_to_position,
)
from clasher.battle import BattleState
from clasher.entities import Building, Troop
from clasher.arena import Position
from clasher.tournament_standard import apply_tournament_overrides

# Import the existing visualizer
from visualize_battle import BattleVisualizer, pygame, SCREEN_WIDTH, SCREEN_HEIGHT


def build_observation(battle: BattleState) -> np.ndarray:
    """Build the same obs vector the env produces, from a Python BattleState."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    p0 = battle.players[0]
    p1 = battle.players[1]

    obs[0] = p0.elixir / 10.0
    obs[1] = p1.elixir / 10.0
    obs[2] = p0.king_tower_hp / 4824.0
    obs[3] = p0.left_tower_hp / 3052.0
    obs[4] = p0.right_tower_hp / 3052.0
    obs[5] = p1.king_tower_hp / 4824.0
    obs[6] = p1.left_tower_hp / 3052.0
    obs[7] = p1.right_tower_hp / 3052.0
    obs[8] = min(battle.time / 360.0, 1.0)
    obs[9] = float(battle.double_elixir)
    obs[10] = float(battle.triple_elixir)
    obs[11] = float(battle.overtime)

    offset = 12
    for i in range(min(4, len(p0.hand))):
        obs[offset + i] = CARD_TO_IDX.get(p0.hand[i], 0) / 9.0
    if p0.cycle_queue:
        obs[offset + 4] = CARD_TO_IDX.get(p0.cycle_queue[0], 0) / 9.0
    for i in range(min(4, len(p0.hand))):
        cost = ELIXIR_COST.get(p0.hand[i], 10)
        obs[offset + 5 + i] = 1.0 if p0.elixir >= cost else 0.0

    offset = 21
    unit_idx = 0
    for entity in battle.entities.values():
        if unit_idx >= 30 or not entity.is_alive:
            continue
        if not isinstance(entity, (Troop, Building)):
            continue

        if isinstance(entity, Building) and entity.position.y in (2.5, 29.5):
            card_id = 9
        elif isinstance(entity, Building) and entity.position.y in (6.5, 25.5):
            card_id = 8
        elif entity.card_stats:
            card_id = CARD_TO_IDX.get(entity.card_stats.name, 0)
        else:
            card_id = 0

        base = offset + unit_idx * 5
        obs[base] = card_id / 9.0
        obs[base + 1] = np.clip(entity.position.x / 18.0, 0.0, 1.0)
        obs[base + 2] = np.clip(entity.position.y / 32.0, 0.0, 1.0)
        obs[base + 3] = entity.hitpoints / entity.max_hitpoints if entity.max_hitpoints > 0 else 0.0
        obs[base + 4] = 0.0 if entity.player_id == 0 else 1.0
        unit_idx += 1

    return obs


def rule_bot_action(battle: BattleState):
    """Simple rule bot for player 1."""
    player = battle.players[1]
    if player.elixir >= 4 and "HogRider" in player.hand:
        bridge_x = random.choice([3.5, 14.5])
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


class PolicyVisualizer(BattleVisualizer):
    """Extends the existing visualizer with trained policy control."""

    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.step_counter = 0
        self.ticks_per_decision = 20  # 1 decision per game-second

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
        self.last_action_str = "WAIT"

    def run(self):
        """Main loop: render + step + agent decision."""
        running = True
        decision_timer = 0

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
                        self.__init__(self.model)
                        decision_timer = 0

            if self.paused or self.battle.game_over:
                self.render_frame()
                self.clock.tick(30)
                continue

            # Step simulation
            self.battle.step(speed_factor=1.0)
            decision_timer += 1

            # Agent decision every N ticks
            if decision_timer >= self.ticks_per_decision:
                decision_timer = 0
                self._agent_act()
                rule_bot_action(self.battle)

            # Render
            self.render_frame()
            self.clock.tick(30)  # 30 FPS display

        pygame.quit()

    def _agent_act(self):
        """Get action from policy and execute."""
        obs = build_observation(self.battle)

        if self.model:
            action, _ = self.model.predict(obs, deterministic=True)
            card_choice = int(action[0])
            region = int(action[1])
        else:
            # Random with bias toward playing cards
            if random.random() < 0.3 and self.battle.players[0].elixir >= 2:
                card_choice = random.randint(1, 4)
                region = random.randint(0, N_REGIONS - 1)
            else:
                card_choice = 0
                region = 0

        if card_choice > 0:
            slot = card_choice - 1
            player = self.battle.players[0]
            if slot < len(player.hand):
                card_name = player.hand[slot]
                pos = region_to_position(region)
                ok = self.battle.deploy_card(0, card_name, pos)
                if ok:
                    self.last_action_str = f"{card_name} → ({pos.x:.0f},{pos.y:.0f})"
                else:
                    self.last_action_str = f"{card_name} FAILED"
            else:
                self.last_action_str = "WAIT"
        else:
            self.last_action_str = "WAIT"

    def render_frame(self):
        """Render + overlay policy info."""
        # Use parent's drawing
        self.draw_arena()
        self.draw_entities()
        self.draw_ui()

        # Overlay: policy action + info
        font = self.large_font
        y_offset = SCREEN_HEIGHT - 80

        # Action display
        action_text = font.render(f"Agent: {self.last_action_str}", True, (255, 255, 0))
        self.screen.blit(action_text, (500, y_offset))

        # Hand display
        hand = self.battle.players[0].hand
        elixir = self.battle.players[0].elixir
        hand_str = "  ".join(f"{c}({ELIXIR_COST.get(c,0)})" for c in hand)
        hand_text = self.small_font.render(f"Hand: {hand_str}  Elixir: {elixir:.1f}", True, (200, 200, 200))
        self.screen.blit(hand_text, (500, y_offset + 30))

        # Model info
        model_str = "Model" if self.model else "Random"
        model_text = self.small_font.render(f"[{model_str}] SPACE=pause  R=reset  ESC=quit", True, (150, 150, 150))
        self.screen.blit(model_text, (500, y_offset + 50))

        pygame.display.flip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    model = None
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        print(f"Loaded: {args.model}")

    viz = PolicyVisualizer(model=model)
    viz.run()


if __name__ == "__main__":
    main()
