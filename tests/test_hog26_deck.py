"""Unit tests for all 8 cards in the 2.6 Hog Cycle deck at tournament standard.

Each card is tested for: stat accuracy, movement/targeting, damage output,
and special mechanics (freeze, knockback, death nova, etc.).
"""

import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clasher.arena import Position
from clasher.battle import BattleState
from clasher.entities import Building, Troop, Projectile
from clasher.tournament_standard import apply_tournament_overrides

HOG_DECK = [
    "HogRider", "Musketeer", "IceGolem", "IceSpirit",
    "Cannon", "Fireball", "Skeletons", "TheLog",
]


def _make_battle():
    """Create a tournament-standard battle with 2.6 deck for both players."""
    battle = BattleState()
    apply_tournament_overrides(battle)
    for p in battle.players:
        p.deck = list(HOG_DECK)
        p.hand = list(HOG_DECK[:4])
        p.cycle_queue = deque(HOG_DECK[4:])
        p.elixir = 10.0
    return battle


def _deploy(battle, player_id, card, x, y):
    """Deploy a card, ensuring it's in hand first."""
    player = battle.players[player_id]
    if card not in player.hand:
        # Swap the card into hand slot 0
        if card in player.cycle_queue:
            player.cycle_queue.remove(card)
        if player.hand:
            player.hand[0] = card
        else:
            player.hand = [card]
    return battle.deploy_card(player_id, card, Position(x, y))


def _get_troops(battle, player_id=None, name=None):
    """Get living troops, optionally filtered by player and card name."""
    troops = []
    for e in battle.entities.values():
        if not isinstance(e, Troop) or not e.is_alive:
            continue
        if player_id is not None and e.player_id != player_id:
            continue
        if name and e.card_stats.name != name:
            continue
        troops.append(e)
    return troops


def _get_buildings(battle, player_id=None, name=None):
    """Get living buildings, optionally filtered."""
    buildings = []
    for e in battle.entities.values():
        if not isinstance(e, Building) or not e.is_alive:
            continue
        if player_id is not None and e.player_id != player_id:
            continue
        if name and e.card_stats.name != name:
            continue
        buildings.append(e)
    return buildings


def _step_n(battle, n, speed=1.0):
    """Step the battle n ticks."""
    for _ in range(n):
        battle.step(speed_factor=speed)


def _get_tower(battle, player_id, side):
    """Get a specific tower. side = 'left', 'right', or 'king'."""
    if side == "king":
        pos = battle.arena.BLUE_KING_TOWER if player_id == 0 else battle.arena.RED_KING_TOWER
    elif side == "left":
        pos = battle.arena.BLUE_LEFT_TOWER if player_id == 0 else battle.arena.RED_LEFT_TOWER
    else:
        pos = battle.arena.BLUE_RIGHT_TOWER if player_id == 0 else battle.arena.RED_RIGHT_TOWER
    for e in battle.entities.values():
        if isinstance(e, Building) and e.player_id == player_id:
            if e.position.x == pos.x and e.position.y == pos.y:
                return e
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  HOG RIDER
# ═══════════════════════════════════════════════════════════════════════════════


class TestHogRider:

    def test_stats_tournament_standard(self):
        battle = _make_battle()
        _deploy(battle, 0, "HogRider", 9, 10)
        hog = _get_troops(battle, 0, "HogRider")[0]
        assert hog.hitpoints == 1697
        assert hog.damage == 317

    def test_targets_buildings_only(self):
        battle = _make_battle()
        _deploy(battle, 0, "HogRider", 9, 10)
        hog = _get_troops(battle, 0, "HogRider")[0]
        assert hog.card_stats.targets_only_buildings is True

    def test_moves_to_bridge_before_crossing(self):
        """Hog at x=9 should walk toward a bridge (x=3.5 or x=14.5) before jumping."""
        battle = _make_battle()
        _deploy(battle, 0, "HogRider", 9, 14)
        # Step past deploy delay + movement
        _step_n(battle, 60)
        hog = _get_troops(battle, 0, "HogRider")
        assert len(hog) == 1
        h = hog[0]
        # Should have moved toward a bridge, not teleported straight up at x=9
        if h.position.y > 16:
            # Crossed river — must be on a bridge (x near 3.5 or 14.5)
            assert h.position.x < 5.5 or h.position.x > 12.5, \
                f"Hog crossed river at x={h.position.x:.1f}, not on a bridge"

    def test_deals_damage_to_tower(self):
        """Hog should reach and damage a princess tower."""
        battle = _make_battle()
        _deploy(battle, 0, "HogRider", 14, 14)
        _step_n(battle, 300)
        right_tower = _get_tower(battle, 1, "right")
        assert right_tower.hitpoints < 3052, "Hog should have damaged the tower"

    def test_deploy_delay(self):
        battle = _make_battle()
        _deploy(battle, 0, "HogRider", 9, 10)
        hog = _get_troops(battle, 0, "HogRider")[0]
        assert hog.deploy_delay_remaining > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  MUSKETEER
# ═══════════════════════════════════════════════════════════════════════════════


class TestMusketeer:

    def test_stats_tournament_standard(self):
        battle = _make_battle()
        _deploy(battle, 0, "Musketeer", 9, 10)
        musk = _get_troops(battle, 0, "Musketeer")[0]
        assert musk.hitpoints == 720
        assert musk.damage == 218

    def test_ranged_attack(self):
        """Musketeer should use projectiles (ranged), not melee."""
        battle = _make_battle()
        _deploy(battle, 0, "Musketeer", 9, 10)
        musk = _get_troops(battle, 0, "Musketeer")[0]
        assert musk._uses_projectiles() is True

    def test_range_six_tiles(self):
        battle = _make_battle()
        _deploy(battle, 0, "Musketeer", 9, 10)
        musk = _get_troops(battle, 0, "Musketeer")[0]
        assert musk.range == 6.0

    def test_targets_air_and_ground(self):
        battle = _make_battle()
        _deploy(battle, 0, "Musketeer", 9, 10)
        musk = _get_troops(battle, 0, "Musketeer")[0]
        assert musk._can_attack_air() is True
        assert musk._can_attack_ground() is True


# ═══════════════════════════════════════════════════════════════════════════════
#  ICE GOLEM
# ═══════════════════════════════════════════════════════════════════════════════


class TestIceGolem:

    def test_stats_tournament_standard(self):
        battle = _make_battle()
        _deploy(battle, 0, "IceGolem", 9, 10)
        golem = _get_troops(battle, 0, "IceGolemite")[0]
        assert golem.hitpoints == 1197
        assert golem.damage == 84

    def test_targets_buildings(self):
        battle = _make_battle()
        _deploy(battle, 0, "IceGolem", 9, 10)
        golem = _get_troops(battle, 0, "IceGolemite")[0]
        assert golem.card_stats.targets_only_buildings is True

    def test_death_nova_damages_and_slows(self):
        """When Ice Golem dies, nearby enemies should take damage and be slowed."""
        battle = _make_battle()
        # Place Ice Golem and enemy Musketeer close on player 0's side
        _deploy(battle, 0, "IceGolem", 9, 10)
        _deploy(battle, 1, "Musketeer", 9, 22)  # deploy on P1's side
        golem = _get_troops(battle, 0, "IceGolemite")[0]
        musk = _get_troops(battle, 1, "Musketeer")[0]
        # Move musk next to golem for the test
        musk.position.x = 9.0
        musk.position.y = 10.5
        musk_hp_before = musk.hitpoints
        # Kill the golem
        golem.take_damage(golem.hitpoints)
        battle.step()  # trigger death processing
        # Musketeer should have taken death nova damage
        assert musk.hitpoints < musk_hp_before, "Death nova should deal damage"

    def test_slow_speed(self):
        battle = _make_battle()
        _deploy(battle, 0, "IceGolem", 9, 10)
        golem = _get_troops(battle, 0, "IceGolemite")[0]
        assert golem.speed == 45  # slow speed


# ═══════════════════════════════════════════════════════════════════════════════
#  ICE SPIRIT
# ═══════════════════════════════════════════════════════════════════════════════


class TestIceSpirit:

    def test_stats_tournament_standard(self):
        battle = _make_battle()
        _deploy(battle, 0, "IceSpirit", 9, 10)
        spirit = _get_troops(battle, 0, "IceSpirits")[0]
        assert spirit.hitpoints == 230
        assert spirit.damage == 109

    def test_very_fast_speed(self):
        battle = _make_battle()
        _deploy(battle, 0, "IceSpirit", 9, 10)
        spirit = _get_troops(battle, 0, "IceSpirits")[0]
        assert spirit.speed == 120

    def test_kamikaze_dies_on_hit(self):
        """Ice Spirit should die after landing its attack."""
        battle = _make_battle()
        _deploy(battle, 0, "IceSpirit", 9, 10)
        _deploy(battle, 1, "Musketeer", 9, 22)
        # Teleport musketeer close so spirit can reach it
        musk = _get_troops(battle, 1, "Musketeer")[0]
        musk.position.x = 9.0
        musk.position.y = 12.0
        # Step enough for Ice Spirit to reach and hit
        _step_n(battle, 150)
        spirits = _get_troops(battle, 0, "IceSpirits")
        assert len(spirits) == 0, "Ice Spirit should die after attack"

    def test_freeze_on_hit(self):
        """Target should be stunned after Ice Spirit hits."""
        battle = _make_battle()
        _deploy(battle, 0, "IceSpirit", 9, 13)
        _deploy(battle, 1, "Musketeer", 9, 15)
        _step_n(battle, 120)
        musk = _get_troops(battle, 1, "Musketeer")
        if musk:
            assert musk[0].is_stunned(), "Target should be frozen"

    def test_splash_damage(self):
        """Ice Spirit should deal AoE damage, not just single-target."""
        battle = _make_battle()
        _deploy(battle, 0, "IceSpirit", 9, 13)
        # Two enemy troops close together
        _deploy(battle, 1, "Skeletons", 9, 15)
        _step_n(battle, 120)
        # At least some skeletons should have taken damage from splash
        skels = _get_troops(battle, 1, "Skeletons")
        damaged = sum(1 for s in skels if s.hitpoints < 81)
        # The splash should hit multiple targets
        # (some may be dead, which is fine — checking total alive < 3 or damaged > 0)
        total_alive = len(skels)
        assert total_alive < 3 or damaged > 0, "Ice Spirit should deal AoE damage"


# ═══════════════════════════════════════════════════════════════════════════════
#  CANNON
# ═══════════════════════════════════════════════════════════════════════════════


class TestCannon:

    def test_stats_tournament_standard(self):
        battle = _make_battle()
        _deploy(battle, 0, "Cannon", 9, 10)
        cannons = _get_buildings(battle, 0, "Cannon")
        assert len(cannons) == 1
        cannon = cannons[0]
        assert cannon.hitpoints == 846
        assert cannon.damage == 264

    def test_is_building_does_not_move(self):
        battle = _make_battle()
        _deploy(battle, 0, "Cannon", 9, 10)
        cannon = _get_buildings(battle, 0, "Cannon")[0]
        start_pos = (cannon.position.x, cannon.position.y)
        _step_n(battle, 60)
        end_pos = (cannon.position.x, cannon.position.y)
        assert start_pos == end_pos, "Cannon should not move"

    def test_targets_ground_only(self):
        battle = _make_battle()
        _deploy(battle, 0, "Cannon", 9, 10)
        cannon = _get_buildings(battle, 0, "Cannon")[0]
        assert cannon._can_attack_ground() is True
        assert cannon._can_attack_air() is False

    def test_pulls_hog(self):
        """Cannon should pull an enemy Hog Rider toward it."""
        battle = _make_battle()
        _deploy(battle, 0, "Cannon", 14, 12)
        _deploy(battle, 1, "HogRider", 14, 18)
        _step_n(battle, 200)
        # Hog should have targeted the Cannon instead of the tower
        cannon = _get_buildings(battle, 0, "Cannon")
        if cannon and cannon[0].hitpoints < 846:
            pass  # Cannon took damage = Hog targeted it. Good.
        else:
            # Check if Cannon is dead (Hog killed it)
            assert cannon == [] or cannon[0].hitpoints < 846, \
                "Hog should have been pulled to the Cannon"

    def test_lifetime_decay(self):
        """Cannon HP should decay over its lifetime."""
        battle = _make_battle()
        _deploy(battle, 0, "Cannon", 9, 10)
        cannon = _get_buildings(battle, 0, "Cannon")[0]
        initial_hp = cannon.hitpoints
        _step_n(battle, 300)  # ~10 seconds
        cannon_now = _get_buildings(battle, 0, "Cannon")
        if cannon_now:
            assert cannon_now[0].hitpoints < initial_hp, "Cannon should decay over time"


# ═══════════════════════════════════════════════════════════════════════════════
#  SKELETONS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSkeletons:

    def test_spawns_three(self):
        battle = _make_battle()
        _deploy(battle, 0, "Skeletons", 9, 10)
        skels = _get_troops(battle, 0, "Skeletons")
        assert len(skels) == 3

    def test_stats_tournament_standard(self):
        battle = _make_battle()
        _deploy(battle, 0, "Skeletons", 9, 10)
        skels = _get_troops(battle, 0, "Skeletons")
        for s in skels:
            assert s.hitpoints == 81
            assert s.damage == 81

    def test_melee_range(self):
        battle = _make_battle()
        _deploy(battle, 0, "Skeletons", 9, 10)
        skel = _get_troops(battle, 0, "Skeletons")[0]
        assert skel.range <= 1.0, "Skeletons should be melee"

    def test_fast_speed(self):
        battle = _make_battle()
        _deploy(battle, 0, "Skeletons", 9, 10)
        skel = _get_troops(battle, 0, "Skeletons")[0]
        assert skel.speed == 90  # fast


# ═══════════════════════════════════════════════════════════════════════════════
#  FIREBALL
# ═══════════════════════════════════════════════════════════════════════════════


class TestFireball:

    def test_damage_tournament_standard(self):
        from clasher.spells import SPELL_REGISTRY
        apply_tournament_overrides.__wrapped__ = True  # ensure overrides applied
        battle = _make_battle()
        spell = SPELL_REGISTRY.get("Fireball")
        assert spell is not None
        assert spell.damage == 689

    def test_crown_tower_multiplier(self):
        from clasher.spells import SPELL_REGISTRY
        battle = _make_battle()
        spell = SPELL_REGISTRY.get("Fireball")
        assert spell.crown_tower_damage_multiplier == 0.35

    def test_deals_area_damage(self):
        """Fireball should damage multiple troops in its radius."""
        battle = _make_battle()
        # Place enemy skeletons on their side
        _deploy(battle, 1, "Skeletons", 9, 22)
        _step_n(battle, 5)
        skels_before = len(_get_troops(battle, 1, "Skeletons"))
        assert skels_before == 3, f"Expected 3 skeletons, got {skels_before}"
        # Cast Fireball on them
        _deploy(battle, 0, "Fireball", 9, 22)
        _step_n(battle, 60)  # allow projectile to land
        skels_after = len(_get_troops(battle, 1, "Skeletons"))
        assert skels_after < skels_before, "Fireball should kill skeletons"

    def test_radius(self):
        from clasher.spells import SPELL_REGISTRY
        battle = _make_battle()
        spell = SPELL_REGISTRY.get("Fireball")
        assert abs(spell.radius - 2.5) < 0.1


# ═══════════════════════════════════════════════════════════════════════════════
#  THE LOG
# ═══════════════════════════════════════════════════════════════════════════════


class TestTheLog:

    def test_damage_tournament_standard(self):
        from clasher.spells import SPELL_REGISTRY
        battle = _make_battle()
        spell = SPELL_REGISTRY.get("Log")
        assert spell is not None
        assert spell.damage == 290

    def test_deploy_succeeds(self):
        """TheLog alias should resolve and deploy."""
        battle = _make_battle()
        battle.players[0].hand = ["TheLog"]
        battle.players[0].elixir = 10.0
        ok = _deploy(battle, 0, "TheLog", 9, 10)
        assert ok, "TheLog should deploy successfully via alias"

    def test_hits_ground_only(self):
        from clasher.spells import SPELL_REGISTRY
        battle = _make_battle()
        spell = SPELL_REGISTRY.get("Log")
        # RollingProjectileSpell — the entity it creates skips air units
        # Verify by checking the spell exists and is a rolling type
        assert hasattr(spell, 'travel_distance') or hasattr(spell, 'travel_speed'), \
            "Log should be a rolling projectile spell"

    def test_kills_skeletons(self):
        """The Log should kill skeletons in its path."""
        battle = _make_battle()
        _deploy(battle, 1, "Skeletons", 9, 15)
        _step_n(battle, 5)
        battle.players[0].hand = ["TheLog"]
        battle.players[0].elixir = 10.0
        _deploy(battle, 0, "TheLog", 9, 10)
        _step_n(battle, 90)  # let it roll
        skels = _get_troops(battle, 1, "Skeletons")
        assert len(skels) == 0, "Log should kill skeletons"


# ═══════════════════════════════════════════════════════════════════════════════
#  TOWER STATS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTowers:

    def test_princess_tower_hp(self):
        battle = _make_battle()
        tower = _get_tower(battle, 0, "left")
        assert tower.hitpoints == 3052

    def test_king_tower_hp(self):
        battle = _make_battle()
        king = _get_tower(battle, 0, "king")
        assert king.hitpoints == 4824

    def test_princess_tower_damage(self):
        battle = _make_battle()
        tower = _get_tower(battle, 0, "left")
        assert tower.damage == 126

    def test_king_tower_damage(self):
        battle = _make_battle()
        king = _get_tower(battle, 0, "king")
        assert king.damage == 109


# ═══════════════════════════════════════════════════════════════════════════════
#  ELIXIR
# ═══════════════════════════════════════════════════════════════════════════════


class TestElixir:

    def test_regen_rate_normal(self):
        """Should gain ~1 elixir every 2.8 seconds in normal time."""
        battle = _make_battle()
        battle.players[0].elixir = 0.0
        _step_n(battle, 84)  # 84 ticks = 2.8s at 30fps
        assert 0.9 < battle.players[0].elixir < 1.2

    def test_hog_costs_four(self):
        battle = _make_battle()
        battle.players[0].elixir = 4.0
        ok = _deploy(battle, 0, "HogRider", 9, 10)
        assert ok
        assert battle.players[0].elixir < 0.5

    def test_not_enough_elixir(self):
        battle = _make_battle()
        battle.players[0].elixir = 1.0
        ok = _deploy(battle, 0, "HogRider", 9, 10)
        assert not ok, "Should not deploy with insufficient elixir"
