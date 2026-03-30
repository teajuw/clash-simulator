"""Scripted expert policy for generating behavioral cloning demonstrations.

This expert plays 2.6 Hog Cycle at a basic-competent level:
- Rushes Hog at the bridge when affordable
- Defends with Cannon, Musketeer, or Skeletons
- Uses Fireball/Log on clustered enemy troops
- Cycles cheap cards to avoid elixir leak
- Plays Ice Golem as a tank in front of Hog

Usage:
    from clasher.expert import ExpertPolicy, generate_demonstrations
    demos = generate_demonstrations(n_games=500)
"""

from __future__ import annotations

import random as _random
from typing import List, Optional, Tuple

from .arena import Position
from .battle import BattleState
from .entities import Building, Troop
from .env import (
    DECK, CARD_TO_IDX, ELIXIR_COST, GRID_X, GRID_Y,
    N_ACTIONS, OBS_SIZE, encode_action,
)


class ExpertPolicy:
    """Scripted expert that plays 2.6 Hog Cycle competently."""

    def __init__(self, player_id: int = 0):
        self.player_id = player_id
        self.opp_id = 1 - player_id

    def act(self, battle: BattleState) -> int:
        """Return an action index (0=WAIT, 1-1080=deploy)."""
        player = battle.players[self.player_id]
        hand = player.hand
        elixir = player.elixir

        # ── Priority 1: Defend against threats ────────────────────────────
        threats = self._get_threats(battle)

        if threats:
            action = self._defend(battle, threats, hand, elixir)
            if action is not None:
                return action

        # ── Priority 2: Attack with Hog ───────────────────────────────────
        if elixir >= 4 and "HogRider" in hand:
            action = self._attack_hog(battle, hand, elixir)
            if action is not None:
                return action

        # ── Priority 3: Ice Golem + Hog push ─────────────────────────────
        if elixir >= 6 and "IceGolem" in hand and "HogRider" in hand:
            action = self._ice_golem_hog(battle, hand)
            if action is not None:
                return action

        # ── Priority 4: Prevent elixir leak ──────────────────────────────
        if elixir >= 9.0:
            action = self._cycle_cheapest(battle, hand, elixir)
            if action is not None:
                return action

        # ── Default: WAIT ─────────────────────────────────────────────────
        return 0

    def _get_threats(self, battle: BattleState) -> List[Troop]:
        """Find enemy troops on our side of the map."""
        threats = []
        if self.player_id == 0:
            threat_y_max = 16.0  # enemy crossed river onto our side
        else:
            threat_y_min = 16.0

        for e in battle.entities.values():
            if not isinstance(e, Troop) or not e.is_alive:
                continue
            if e.player_id == self.opp_id:
                if self.player_id == 0 and e.position.y < 16.0:
                    threats.append(e)
                elif self.player_id == 1 and e.position.y > 16.0:
                    threats.append(e)
        return threats

    def _defend(self, battle, threats, hand, elixir) -> Optional[int]:
        """Deploy defensive cards against threats."""
        # Find the most dangerous threat (closest to our king tower)
        if self.player_id == 0:
            threats.sort(key=lambda t: t.position.y)  # lowest y = closest to king
        else:
            threats.sort(key=lambda t: -t.position.y)

        threat = threats[0]
        tx = max(0, min(17, int(threat.position.x)))

        # Cannon to pull building-targeting troops
        if elixir >= 3 and "Cannon" in hand:
            if threat.card_stats and getattr(threat.card_stats, 'targets_only_buildings', False):
                # Place Cannon to pull: center of arena, slightly behind threat
                cx = 9
                if self.player_id == 0:
                    cy = max(0, min(14, int(threat.position.y) - 3))
                else:
                    cy = min(14, int(32 - threat.position.y) - 3)
                slot = hand.index("Cannon")
                return encode_action(slot, cx, cy)

        # Musketeer for ranged defense
        if elixir >= 4 and "Musketeer" in hand:
            if self.player_id == 0:
                cy = max(0, min(14, int(threat.position.y) - 4))
            else:
                cy = min(14, int(32 - threat.position.y) - 4)
            slot = hand.index("Musketeer")
            return encode_action(slot, tx, cy)

        # Skeletons for cheap distraction
        if elixir >= 1 and "Skeletons" in hand:
            if self.player_id == 0:
                cy = max(0, min(14, int(threat.position.y) - 1))
            else:
                cy = min(14, int(32 - threat.position.y) - 1)
            slot = hand.index("Skeletons")
            return encode_action(slot, tx, cy)

        # Ice Spirit to freeze
        if elixir >= 1 and "IceSpirit" in hand:
            if self.player_id == 0:
                cy = max(0, min(14, int(threat.position.y) - 2))
            else:
                cy = min(14, int(32 - threat.position.y) - 2)
            slot = hand.index("IceSpirit")
            return encode_action(slot, tx, cy)

        return None

    def _attack_hog(self, battle, hand, elixir) -> Optional[int]:
        """Deploy Hog Rider at the bridge."""
        slot = hand.index("HogRider")

        # Pick which bridge — prefer the one with a weaker or dead tower
        opp = battle.players[self.opp_id]
        if opp.left_tower_hp <= 0:
            # Left tower dead, attack left side (path to king)
            bx = 3
        elif opp.right_tower_hp <= 0:
            bx = 14
        elif opp.left_tower_hp < opp.right_tower_hp:
            bx = 3
        else:
            bx = 14

        # Deploy at y=14 (right at the bridge) for player 0
        by = 14 if self.player_id == 0 else 0
        return encode_action(slot, bx, by)

    def _ice_golem_hog(self, battle, hand) -> Optional[int]:
        """Deploy Ice Golem in front of Hog as a tank."""
        slot = hand.index("IceGolem")
        # Same bridge logic as Hog
        opp = battle.players[self.opp_id]
        if opp.left_tower_hp <= opp.right_tower_hp or opp.right_tower_hp <= 0:
            bx = 3
        else:
            bx = 14
        by = 14 if self.player_id == 0 else 0
        return encode_action(slot, bx, by)

    def _cycle_cheapest(self, battle, hand, elixir) -> Optional[int]:
        """Cycle the cheapest card to avoid elixir leak."""
        affordable = [(c, ELIXIR_COST.get(c, 10)) for c in hand if ELIXIR_COST.get(c, 10) <= elixir]
        if not affordable:
            return None

        affordable.sort(key=lambda x: x[1])
        card_name, cost = affordable[0]
        slot = hand.index(card_name)

        # Play in the back (safe position)
        if self.player_id == 0:
            return encode_action(slot, 9, 3)
        else:
            return encode_action(slot, 9, 11)

        return None


def generate_demonstrations(
    n_games: int = 500,
    verbose: bool = True,
) -> Tuple[list, list]:
    """Generate expert demonstration data for behavioral cloning.

    Returns:
        observations: list of np.ndarray (OBS_SIZE,)
        actions: list of int
    """
    import numpy as np
    from collections import deque
    from .env import ClashRoyaleEnv

    env = ClashRoyaleEnv(opponent="rule_bot")
    expert = ExpertPolicy(player_id=0)

    observations = []
    actions = []
    wins = 0

    for game in range(n_games):
        obs, info = env.reset()
        done = False

        while not done:
            action = expert.act(env.battle)

            # Validate action against mask
            mask = env.action_masks()
            if not mask[action]:
                action = 0  # fall back to WAIT if expert chose invalid action

            observations.append(obs.copy())
            actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if info.get("winner") == 0:
            wins += 1

        if verbose and (game + 1) % 50 == 0:
            wr = wins / (game + 1) * 100
            print(f"[Demo {game+1}/{n_games}] Expert WR={wr:.1f}% ({wins}W)")

    if verbose:
        wr = wins / n_games * 100
        print(f"Generated {len(observations)} frames from {n_games} games. Expert WR={wr:.1f}%")

    return observations, actions
