"""Smart 2.6 Hog Cycle bot for behavioral cloning demonstrations.

Plays with human-like decision priorities:
1. Defend threats first (reactive)
2. Attack when safe (proactive)
3. Manage elixir (never leak, never overcommit)

This bot plays for either player (player_id parameter).
"""

from __future__ import annotations

import random as _random
from typing import List, Optional

from .arena import Position
from .battle import BattleState
from .entities import Building, Troop

ELIXIR_COST = {
    "HogRider": 4, "Musketeer": 4, "IceGolem": 2, "IceSpirit": 1,
    "Cannon": 3, "Fireball": 4, "Skeletons": 1, "TheLog": 2,
}


class SmartBot:
    """Competent 2.6 Hog Cycle player."""

    def __init__(self, player_id: int = 1):
        self.pid = player_id
        self.opp_id = 1 - player_id
        # Player 0 (bottom): towers at y=6.5, king at y=2.5, deploy y=0-14
        # Player 1 (top): towers at y=25.5, king at y=29.5, deploy y=17-31
        self.is_top = player_id == 1

    def act(self, battle: BattleState) -> None:
        player = battle.players[self.pid]
        hand = player.hand
        elixir = player.elixir

        # Never play below 2 elixir
        if elixir < 2:
            return

        # ── 1. DEFEND threats ─────────────────────────────────────────
        threats = self._get_threats(battle)
        if threats:
            action = self._defend(battle, threats, hand, elixir)
            if action:
                return

        # ── 2. ATTACK when safe ───────────────────────────────────────
        if elixir >= 7 and not threats:
            action = self._attack(battle, hand, elixir)
            if action:
                return

        # ── 3. CYCLE at high elixir ───────────────────────────────────
        if elixir >= 9.0:
            self._cycle(battle, player, hand)

    def _get_threats(self, battle: BattleState) -> List[Troop]:
        """Find enemy troops on our side of the map."""
        threats = []
        for e in battle.entities.values():
            if not isinstance(e, Troop) or not e.is_alive:
                continue
            if e.player_id != self.opp_id:
                continue
            # Check if on our side
            if self.is_top and e.position.y > 16:
                threats.append(e)
            elif not self.is_top and e.position.y < 16:
                threats.append(e)

        # Sort by distance to our towers (closest = most urgent)
        if self.is_top:
            threats.sort(key=lambda t: -t.position.y)  # highest y = closest to top towers
        else:
            threats.sort(key=lambda t: t.position.y)  # lowest y = closest to bottom towers
        return threats

    def _defend(self, battle, threats, hand, elixir) -> bool:
        """React to enemy threats."""
        threat = threats[0]
        tx = threat.position.x
        is_left = tx < 9

        targets_buildings = getattr(threat.card_stats, 'targets_only_buildings', False)

        # Cannon for building-targeting troops (Hog, Ice Golem, Giant)
        if targets_buildings and "Cannon" in hand and elixir >= 3:
            # 4-3 pull position: center, between our towers
            cy = 22.0 if self.is_top else 10.0
            battle.deploy_card(self.pid, "Cannon", Position(9.0, cy))
            return True

        # Musketeer for ranged defense (behind tower, same side as threat)
        if "Musketeer" in hand and elixir >= 4:
            mx = 5.0 if is_left else 13.0
            my = 27.0 if self.is_top else 5.0
            battle.deploy_card(self.pid, "Musketeer", Position(mx, my))
            return True

        # Ice Spirit to freeze approaching troops
        if "IceSpirit" in hand and elixir >= 1:
            # Place between threat and our tower
            if self.is_top:
                sy = min(28.0, threat.position.y + 2)
            else:
                sy = max(4.0, threat.position.y - 2)
            battle.deploy_card(self.pid, "IceSpirit", Position(tx, sy))
            return True

        # Skeletons to distract
        if "Skeletons" in hand and elixir >= 1:
            if self.is_top:
                sy = min(28.0, threat.position.y + 1)
            else:
                sy = max(4.0, threat.position.y - 1)
            battle.deploy_card(self.pid, "Skeletons", Position(tx, sy))
            return True

        # Log on ground swarms
        if "TheLog" in hand and elixir >= 2 and not targets_buildings:
            ly = threat.position.y
            battle.deploy_card(self.pid, "TheLog", Position(tx, ly))
            return True

        # Fireball on clustered enemies (2+ nearby)
        if "Fireball" in hand and elixir >= 4:
            cluster = self._find_cluster(battle)
            if cluster:
                battle.deploy_card(self.pid, "Fireball", cluster)
                return True

        # Ice Golem as emergency tank
        if "IceGolem" in hand and elixir >= 2:
            if self.is_top:
                gy = min(28.0, threat.position.y + 1)
            else:
                gy = max(4.0, threat.position.y - 1)
            battle.deploy_card(self.pid, "IceGolem", Position(tx, gy))
            return True

        return False

    def _attack(self, battle, hand, elixir) -> bool:
        """Push when we have elixir advantage."""
        opp = battle.players[self.opp_id]
        bridge_y = 18.0 if self.is_top else 14.0

        # Pick attack lane: prefer weaker tower
        if opp.left_tower_hp <= 0:
            bx = 3.5  # left dead, go for king
        elif opp.right_tower_hp <= 0:
            bx = 14.5  # right dead, go for king
        elif opp.left_tower_hp < opp.right_tower_hp:
            bx = 3.5
        elif opp.right_tower_hp < opp.left_tower_hp:
            bx = 14.5
        else:
            bx = _random.choice([3.5, 14.5])

        # Ice Golem + Hog combo (9+ elixir)
        if "IceGolem" in hand and "HogRider" in hand and elixir >= 9:
            # Ice Golem first as tank
            battle.deploy_card(self.pid, "IceGolem", Position(bx, bridge_y))
            return True

        # Solo Hog (7+ elixir, leaving 3 for Cannon defense)
        if "HogRider" in hand and elixir >= 7:
            battle.deploy_card(self.pid, "HogRider", Position(bx, bridge_y))
            return True

        # If Hog not in hand, push with Ice Golem to cycle
        if "IceGolem" in hand and elixir >= 8:
            battle.deploy_card(self.pid, "IceGolem", Position(bx, bridge_y))
            return True

        return False

    def _cycle(self, battle, player, hand):
        """Play cheapest card behind king to avoid elixir leak."""
        affordable = [(c, ELIXIR_COST.get(c, 10)) for c in hand
                      if ELIXIR_COST.get(c, 10) <= player.elixir]
        if not affordable:
            return
        affordable.sort(key=lambda x: x[1])
        card = affordable[0][0]

        # Don't cycle Hog or Musketeer (too expensive, save for purpose)
        if card in ("HogRider", "Musketeer", "Fireball"):
            if len(affordable) > 1:
                card = affordable[1][0]
            else:
                return

        # Behind king tower
        ky = 28.0 if self.is_top else 4.0
        battle.deploy_card(self.pid, card, Position(9.0, ky))

    def _find_cluster(self, battle) -> Optional[Position]:
        """Find 2+ enemy troops within 3 tiles of each other."""
        enemies = [e for e in battle.entities.values()
                   if isinstance(e, Troop) and e.is_alive and e.player_id == self.opp_id]
        for t in enemies:
            nearby = sum(1 for o in enemies if t.position.distance_to(o.position) < 3.0)
            if nearby >= 2:
                return Position(t.position.x, t.position.y)
        return None


def make_smart_bot_policy(player_id: int = 1):
    """Create a smart bot policy function."""
    bot = SmartBot(player_id=player_id)
    def policy(battle: BattleState):
        bot.act(battle)
    return policy
