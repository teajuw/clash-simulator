"""Tournament Standard (Level 11) stat overrides for the 2.6 Hog Cycle deck.

These values are sourced from official Clash Royale stat sheets (DeckShop.pro,
RoyaleAPI) and override the simulator's scaling formula to match real friendly
battle values exactly.

Usage:
    from clasher.tournament_standard import apply_tournament_overrides
    engine = BattleEngine()
    battle = engine.create_battle()
    apply_tournament_overrides(battle)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .battle import BattleState

# ── Troop / Building overrides ────────────────────────────────────────────────
# Keys match internal gamedata names (post-alias resolution).
# Values are tournament standard (Level 11) stats from real Clash Royale.
TROOP_OVERRIDES: dict[str, dict] = {
    "HogRider": {
        "hitpoints": 1697,
        "damage": 317,
        # hit_speed: 1600ms, range: 0.8 tiles, speed: 120 — unchanged
    },
    "Musketeer": {
        "hitpoints": 720,
        "damage": 218,
        # hit_speed: 1000ms, range: 6.0 tiles, speed: 60 — unchanged
    },
    "IceGolemite": {  # internal name for Ice Golem
        "hitpoints": 1197,
        "damage": 84,
    },
    "IceSpirits": {  # internal name for Ice Spirit
        "hitpoints": 230,
        "damage": 109,
    },
    "Cannon": {
        "hitpoints": 846,
        "damage": 264,
        "hit_speed": 800,  # ms — gamedata has 1000, real CR is 800
    },
    "Skeleton": {
        "hitpoints": 81,
        "damage": 81,
    },
    "Skeletons": {
        "hitpoints": 81,
        "damage": 81,
    },
}

# ── Spell overrides ──────────────────────────────────────────────────────────
# Spells bypass the card-loader scaling entirely, so these are critical.
SPELL_OVERRIDES: dict[str, dict] = {
    "Fireball": {
        "damage": 689,
        "crown_tower_damage_multiplier": 0.35,
    },
    "Log": {
        "damage": 290,
    },
}

# ── Tower overrides ──────────────────────────────────────────────────────────
TOWER_OVERRIDES: dict[str, dict] = {
    "princess_tower": {
        "hitpoints": 3052,
        "damage": 126,
    },
    "king_tower": {
        "hitpoints": 4824,
        "damage": 109,
    },
}


def apply_tournament_overrides(battle: "BattleState") -> None:
    """Patch a BattleState so all entities and spells use tournament standard stats.

    Call this AFTER ``create_battle()`` but BEFORE deploying any cards.
    """
    from .entities import Troop, Building
    from .spells import SPELL_REGISTRY

    # ── Patch card loader stats so future deploys use correct values ──────
    # Set level=1 so that scaled_hitpoints = hitpoints * 1.1^0 = hitpoints.
    # Then set hitpoints/damage to tournament standard values directly.
    # Patch both internal name AND alias-resolved name (they may be different objects).
    from .card_aliases import CARD_NAME_ALIASES
    card_defs = battle.card_loader.load_card_definitions()

    for internal_name, overrides in TROOP_OVERRIDES.items():
        # Collect all names that resolve to this card
        names_to_patch = {internal_name}
        for alias, target in CARD_NAME_ALIASES.items():
            if target == internal_name:
                names_to_patch.add(alias)
        # Also check direct card def keys
        for key in card_defs:
            cs = battle.card_loader.get_card(key)
            if cs and cs.name == internal_name:
                names_to_patch.add(key)

        for name in names_to_patch:
            card_stats = battle.card_loader.get_card(name)
            if card_stats is None:
                continue
            card_stats.level = 1  # disable scaling (1.1^0 = 1.0)
            for attr, value in overrides.items():
                if hasattr(card_stats, attr):
                    setattr(card_stats, attr, value)

    # ── Patch already-spawned towers ─────────────────────────────────────
    for entity in battle.entities.values():
        if isinstance(entity, Building):
            # Princess towers
            if entity.position.y in (6.5, 25.5):
                ov = TOWER_OVERRIDES["princess_tower"]
                entity.hitpoints = ov["hitpoints"]
                entity.max_hitpoints = ov["hitpoints"]
                entity.damage = ov["damage"]
                # Update player state
                player = battle.players[entity.player_id]
                if entity.position.x < 9:
                    player.left_tower_hp = ov["hitpoints"]
                else:
                    player.right_tower_hp = ov["hitpoints"]
            # King towers
            elif entity.position.y in (2.5, 29.5):
                ov = TOWER_OVERRIDES["king_tower"]
                entity.hitpoints = ov["hitpoints"]
                entity.max_hitpoints = ov["hitpoints"]
                entity.damage = ov["damage"]
                player = battle.players[entity.player_id]
                player.king_tower_hp = ov["hitpoints"]

    # ── Patch spell registry ─────────────────────────────────────────────
    for spell_name, overrides in SPELL_OVERRIDES.items():
        spell = SPELL_REGISTRY.get(spell_name)
        if spell is None:
            continue
        for attr, value in overrides.items():
            if hasattr(spell, attr):
                setattr(spell, attr, value)
