"""Improved training bot for warmup phase.

Plays 2.6 Hog Cycle with basic strategy:
- Hog pushes at bridge (sometimes with Ice Golem tank)
- Defensive Cannon pulls for building-targeting troops
- Musketeer for ranged defense
- Skeletons/Ice Spirit for cheap distraction
- Fireball on clustered enemies
- Log on small ground troops
- Cycles cheap cards behind king tower to avoid elixir leak
- Never plays below 2 elixir (saves for defense)

This bot is player 1 (top side, y=17-31 deploy zone).
"""

from __future__ import annotations

import random as _random
from typing import List

from .arena import Position
from .battle import BattleState
from .entities import Building, Troop

ELIXIR_COST = {
    "HogRider": 4, "Musketeer": 4, "IceGolem": 2, "IceSpirit": 1,
    "Cannon": 3, "Fireball": 4, "Skeletons": 1, "TheLog": 2,
}


def training_bot_policy(battle: BattleState) -> None:
    """Competent 2.6 Hog Cycle bot for player 1."""
    player = battle.players[1]
    hand = player.hand
    elixir = player.elixir

    # Don't play below 2 elixir (save for defense)
    if elixir < 2:
        return

    # ── DEFENSE FIRST ─────────────────────────────────────────────────
    threats = _get_threats(battle)

    if threats:
        threat = threats[0]  # closest to our towers

        # Cannon pull for building-targeting troops (Hog, Ice Golem)
        if (elixir >= 3 and "Cannon" in hand
                and _is_building_targeting(threat)):
            # 4-3 pull position: center of arena, between towers
            battle.deploy_card(1, "Cannon", Position(9.0, 22.0))
            return

        # Musketeer for ranged defense (behind tower)
        if elixir >= 4 and "Musketeer" in hand:
            # Place behind the threatened tower
            if threat.position.x < 9:
                battle.deploy_card(1, "Musketeer", Position(5.0, 27.0))
            else:
                battle.deploy_card(1, "Musketeer", Position(13.0, 27.0))
            return

        # Skeletons for cheap distraction
        if elixir >= 1 and "Skeletons" in hand:
            battle.deploy_card(1, "Skeletons",
                               Position(threat.position.x, threat.position.y + 1))
            return

        # Ice Spirit to freeze
        if elixir >= 1 and "IceSpirit" in hand:
            battle.deploy_card(1, "IceSpirit",
                               Position(threat.position.x, threat.position.y + 2))
            return

        # Fireball on clustered troops (2+ enemies close together)
        if elixir >= 4 and "Fireball" in hand:
            cluster = _find_cluster(battle, player_id=0, min_count=2)
            if cluster:
                battle.deploy_card(1, "Fireball", cluster)
                return

        # Log on small ground troops
        if elixir >= 2 and "TheLog" in hand:
            if not _is_building_targeting(threat):
                battle.deploy_card(1, "TheLog",
                                   Position(threat.position.x, 18.0))
                return

    # ── OFFENSE ───────────────────────────────────────────────────────

    # Ice Golem + Hog combo at bridge
    if (elixir >= 6 and "IceGolem" in hand and "HogRider" in hand
            and _random.random() < 0.3):
        bridge_x = _pick_attack_lane(battle)
        battle.deploy_card(1, "IceGolem", Position(bridge_x, 18.0))
        # Hog follows behind (will be played next call)
        return

    # Solo Hog push
    if elixir >= 4 and "HogRider" in hand and _random.random() < 0.4:
        bridge_x = _pick_attack_lane(battle)
        battle.deploy_card(1, "HogRider", Position(bridge_x, 18.0))
        return

    # ── CYCLE / ELIXIR MANAGEMENT ────────────────────────────────────

    # Don't leak elixir — cycle cheap cards behind king tower
    if elixir >= 9.0:
        _cycle_cheapest(battle, player, hand)
    elif elixir >= 7.0 and _random.random() < 0.2:
        _cycle_cheapest(battle, player, hand)


def _get_threats(battle: BattleState) -> List[Troop]:
    """Find enemy troops on player 1's side (y > 16), sorted by proximity to towers."""
    threats = []
    for e in battle.entities.values():
        if isinstance(e, Troop) and e.is_alive and e.player_id == 0:
            if e.position.y > 16:
                threats.append(e)
    # Sort by y descending (closest to P1's towers = highest y)
    threats.sort(key=lambda t: -t.position.y)
    return threats


def _is_building_targeting(troop: Troop) -> bool:
    """Check if a troop targets buildings only (Hog, Ice Golem)."""
    return getattr(troop.card_stats, 'targets_only_buildings', False)


def _find_cluster(battle: BattleState, player_id: int, min_count: int = 2) -> Position:
    """Find a position where multiple enemy troops are clustered."""
    troops = [e for e in battle.entities.values()
              if isinstance(e, Troop) and e.is_alive and e.player_id == player_id]

    for t in troops:
        nearby = sum(1 for other in troops
                     if t.position.distance_to(other.position) < 3.0)
        if nearby >= min_count:
            return Position(t.position.x, t.position.y)
    return None


def _pick_attack_lane(battle: BattleState) -> float:
    """Pick which bridge to attack: prefer the side with lower tower HP."""
    opp = battle.players[0]
    if opp.left_tower_hp <= 0:
        return 3.5  # left tower dead, push left to king
    if opp.right_tower_hp <= 0:
        return 14.5  # right tower dead, push right to king
    if opp.left_tower_hp < opp.right_tower_hp:
        return 3.5
    elif opp.right_tower_hp < opp.left_tower_hp:
        return 14.5
    return _random.choice([3.5, 14.5])


def _cycle_cheapest(battle: BattleState, player, hand):
    """Play cheapest card behind king tower."""
    affordable = [(c, ELIXIR_COST.get(c, 10)) for c in hand
                  if ELIXIR_COST.get(c, 10) <= player.elixir]
    if not affordable:
        return
    affordable.sort(key=lambda x: x[1])
    card_name = affordable[0][0]
    # Behind king tower (P1 king is at y=29.5)
    battle.deploy_card(1, card_name, Position(9.0, 28.0))
