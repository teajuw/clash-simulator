"""Tests for the Gymnasium environment."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from clasher.env import (
    ClashRoyaleEnv, OBS_SIZE, N_ACTIONS, DECK,
    decode_action, encode_action,
)


class TestEnvBasics:

    def test_reset_returns_correct_shapes(self):
        env = ClashRoyaleEnv(opponent="rule_bot")
        obs, info = env.reset(seed=42)
        assert obs.shape == (OBS_SIZE,)
        assert obs.dtype == np.float32
        assert 0.0 <= obs.min()
        assert obs.max() <= 1.0

    def test_action_space_size(self):
        env = ClashRoyaleEnv()
        assert env.action_space.n == N_ACTIONS  # 1081

    def test_wait_action_always_valid(self):
        env = ClashRoyaleEnv()
        env.reset(seed=1)
        mask = env.valid_action_mask()
        assert mask[0] is np.True_

    def test_step_returns_correct_types(self):
        env = ClashRoyaleEnv()
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (OBS_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "battle_time" in info

    def test_mask_prevents_overspend(self):
        """With low elixir, expensive cards should be masked out."""
        env = ClashRoyaleEnv()
        env.reset(seed=1)
        env.battle.players[0].elixir = 0.5
        mask = env.valid_action_mask()
        # Only WAIT should be valid with 0.5 elixir (cheapest card is 1 elixir)
        assert mask[0] is np.True_
        # Most deploy actions should be masked
        assert mask[1:].sum() < 300  # very few valid at low elixir

    def test_full_game_terminates(self):
        env = ClashRoyaleEnv()
        obs, _ = env.reset(seed=42)
        steps = 0
        while steps < 500:  # safety limit
            obs, reward, terminated, truncated, info = env.step(0)  # always WAIT
            steps += 1
            if terminated:
                break
        assert terminated, "Game should end within 500 steps (~500 seconds)"
        assert info["winner"] in (0, 1, None)

    def test_starting_elixir(self):
        env = ClashRoyaleEnv()
        env.reset(seed=1)
        assert env.battle.players[0].elixir == 5.0
        assert env.battle.players[1].elixir == 5.0

    def test_hand_has_four_cards(self):
        env = ClashRoyaleEnv()
        env.reset(seed=1)
        assert len(env.battle.players[0].hand) == 4
        for card in env.battle.players[0].hand:
            assert card in DECK


class TestActionEncoding:

    def test_wait_is_zero(self):
        """Action 0 = WAIT."""
        env = ClashRoyaleEnv()
        env.reset(seed=1)
        # Step with WAIT should not crash
        obs, reward, terminated, truncated, info = env.step(0)
        assert not terminated

    def test_encode_decode_roundtrip(self):
        for slot in range(4):
            for tx in range(18):
                for ty in range(15):
                    action = encode_action(slot, tx, ty)
                    s, x, y = decode_action(action)
                    assert s == slot and x == tx and y == ty

    def test_action_range(self):
        assert encode_action(0, 0, 0) == 1
        assert encode_action(3, 17, 14) == N_ACTIONS - 1


class TestReward:

    def test_reward_on_tower_damage(self):
        """Deploying a Hog should eventually produce positive reward."""
        env = ClashRoyaleEnv(opponent="none")
        env.reset(seed=42)
        # Deploy Hog Rider
        mask = env.valid_action_mask()
        # Find a Hog action
        hog_slot = None
        for i, card in enumerate(env.battle.players[0].hand):
            if card == "HogRider":
                hog_slot = i
                break
        if hog_slot is not None:
            action = encode_action(hog_slot, 14, 14)
            if mask[action]:
                total_reward = 0
                env.step(action)
                for _ in range(200):
                    obs, reward, terminated, truncated, info = env.step(0)
                    total_reward += reward
                    if terminated:
                        break
                # Hog should have done some tower damage → positive reward
                assert total_reward > 0, f"Expected positive reward from Hog, got {total_reward}"
