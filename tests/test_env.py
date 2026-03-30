"""Tests for the two-step Gymnasium environment."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from clasher.env import (
    ClashRoyaleEnv, OBS_SIZE, N_CARD_CHOICES, N_REGIONS, DECK,
    region_to_position, position_to_region,
)


class TestEnvBasics:

    def test_reset_returns_correct_shapes(self):
        env = ClashRoyaleEnv(opponent="rule_bot")
        obs, info = env.reset(seed=42)
        assert obs.shape == (OBS_SIZE,)
        assert obs.dtype == np.float32
        assert 0.0 <= obs.min()
        assert obs.max() <= 1.0

    def test_action_space_is_multi_discrete(self):
        env = ClashRoyaleEnv()
        assert env.action_space.shape == (2,)
        assert env.action_space.nvec[0] == N_CARD_CHOICES  # 5
        assert env.action_space.nvec[1] == N_REGIONS  # 30

    def test_wait_action(self):
        env = ClashRoyaleEnv()
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(np.array([0, 0]))
        assert not terminated

    def test_step_returns_correct_types(self):
        env = ClashRoyaleEnv()
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(np.array([0, 0]))
        assert obs.shape == (OBS_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert "battle_time" in info

    def test_full_game_terminates(self):
        env = ClashRoyaleEnv()
        env.reset(seed=42)
        steps = 0
        while steps < 500:
            obs, reward, terminated, truncated, info = env.step(np.array([0, 0]))
            steps += 1
            if terminated:
                break
        assert terminated, "Game should end within 500 steps"

    def test_starting_elixir(self):
        env = ClashRoyaleEnv()
        env.reset(seed=1)
        assert env.battle.players[0].elixir == 5.0

    def test_hand_has_four_cards(self):
        env = ClashRoyaleEnv()
        env.reset(seed=1)
        assert len(env.battle.players[0].hand) == 4
        for card in env.battle.players[0].hand:
            assert card in DECK

    def test_deploy_card_at_region(self):
        """Deploying a card at a valid region should work."""
        env = ClashRoyaleEnv(opponent="none")
        env.reset(seed=42)
        env.battle.players[0].elixir = 10.0
        # Play card slot 0 at region 25 (left bridge approach)
        obs, reward, terminated, truncated, info = env.step(np.array([1, 25]))
        # Should not crash
        assert not terminated


class TestRegions:

    def test_30_regions(self):
        assert N_REGIONS == 30

    def test_region_centers_valid(self):
        for r in range(N_REGIONS):
            pos = region_to_position(r)
            assert 0 <= pos.x <= 18
            assert 0 <= pos.y <= 15

    def test_position_to_region_roundtrip(self):
        for r in range(N_REGIONS):
            pos = region_to_position(r)
            r2 = position_to_region(pos.x, pos.y)
            assert r2 == r, f"Region {r} -> pos ({pos.x}, {pos.y}) -> region {r2}"

    def test_bridge_regions(self):
        """Regions 25 and 28 should be near the bridges."""
        left_bridge = region_to_position(25)  # x=3-5, y=12-14
        right_bridge = region_to_position(28)  # x=12-14, y=12-14
        assert 3 <= left_bridge.x <= 6
        assert 12 <= left_bridge.y <= 15
        assert 12 <= right_bridge.x <= 15
        assert 12 <= right_bridge.y <= 15


class TestActionMasks:

    def test_masks_return_two_arrays(self):
        env = ClashRoyaleEnv(backend="python")
        env.reset(seed=1)
        card_mask, region_mask = env.action_masks()
        assert card_mask.shape == (N_CARD_CHOICES,)
        assert region_mask.shape == (N_REGIONS,)
        assert card_mask[0] is np.True_  # WAIT always valid

    def test_low_elixir_masks_expensive_cards(self):
        env = ClashRoyaleEnv(backend="python")
        env.reset(seed=1)
        env.battle.players[0].elixir = 0.5
        card_mask, _ = env.action_masks()
        assert card_mask[0] is np.True_  # WAIT
        # All cards cost >= 1, so none should be affordable at 0.5
        assert not card_mask[1:].any()


class TestReward:

    def test_reward_on_tower_damage(self):
        """Deploying Hog at bridge should eventually produce positive reward."""
        env = ClashRoyaleEnv(opponent="none")
        env.reset(seed=42)
        env.battle.players[0].elixir = 10.0

        # Find Hog in hand
        hog_slot = None
        for i, card in enumerate(env.battle.players[0].hand):
            if card == "HogRider":
                hog_slot = i + 1  # card_choice is 1-indexed
                break

        if hog_slot is not None:
            # Play Hog at region 28 (right bridge approach)
            env.step(np.array([hog_slot, 28]))
            total_reward = 0
            for _ in range(200):
                obs, reward, terminated, truncated, info = env.step(np.array([0, 0]))
                total_reward += reward
                if terminated:
                    break
            assert total_reward > 0, f"Expected positive reward from Hog, got {total_reward}"
