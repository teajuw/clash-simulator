"""Tests for the two-step Gymnasium environment."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from clasher.env import (
    ClashRoyaleEnv, OBS_SIZE, N_CARD_CHOICES, GRID_X, GRID_Y, DECK,
    tile_to_position,
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
        assert env.action_space.shape == (3,)
        assert env.action_space.nvec[0] == N_CARD_CHOICES  # 5
        assert env.action_space.nvec[1] == GRID_X  # 18
        assert env.action_space.nvec[2] == GRID_Y  # 15

    def test_wait_action(self):
        env = ClashRoyaleEnv()
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0]))
        assert not terminated

    def test_step_returns_correct_types(self):
        env = ClashRoyaleEnv()
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0]))
        assert obs.shape == (OBS_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert "battle_time" in info

    def test_full_game_terminates(self):
        env = ClashRoyaleEnv()
        env.reset(seed=42)
        steps = 0
        while steps < 500:
            obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0]))
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

    def test_deploy_card_at_tile(self):
        """Deploying a card at a valid tile should work."""
        env = ClashRoyaleEnv(opponent="none")
        env.reset(seed=42)
        env.battle.players[0].elixir = 10.0
        # Play card slot 0 at tile (14, 14) — right bridge approach
        obs, reward, terminated, truncated, info = env.step(np.array([1, 14, 14]))
        # Should not crash
        assert not terminated


class TestTileGrid:

    def test_grid_dimensions(self):
        assert GRID_X == 18
        assert GRID_Y == 15

    def test_tile_to_position(self):
        pos = tile_to_position(0, 0)
        assert pos.x == 0.5 and pos.y == 0.5
        pos = tile_to_position(17, 14)
        assert pos.x == 17.5 and pos.y == 14.5

    def test_bridge_tiles(self):
        """Tiles (3, 14) and (14, 14) should be at bridge approach."""
        left = tile_to_position(3, 14)
        right = tile_to_position(14, 14)
        assert 3 <= left.x <= 4
        assert 14 <= left.y <= 15
        assert 14 <= right.x <= 15
        assert 14 <= right.y <= 15


class TestActionMasks:

    def test_masks_return_three_arrays(self):
        env = ClashRoyaleEnv(backend="python")
        env.reset(seed=1)
        card_mask, x_mask, y_mask = env.action_masks()
        assert card_mask.shape == (N_CARD_CHOICES,)
        assert x_mask.shape == (GRID_X,)
        assert y_mask.shape == (GRID_Y,)
        assert card_mask[0] is np.True_  # WAIT always valid

    def test_low_elixir_masks_expensive_cards(self):
        env = ClashRoyaleEnv(backend="python")
        env.reset(seed=1)
        env.battle.players[0].elixir = 0.5
        card_mask, _, _ = env.action_masks()
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
            # Play Hog at tile (14, 14) — right bridge approach
            env.step(np.array([hog_slot, 14, 14]))
            total_reward = 0
            for _ in range(200):
                obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0]))
                total_reward += reward
                if terminated:
                    break
            assert total_reward > 0, f"Expected positive reward from Hog, got {total_reward}"
