"""Expert kickstarting wrapper — injects smart bot actions into PPO training.

Wraps the env so that a percentage of actions come from the expert bot
instead of the agent. PPO learns from the rewards of expert actions
as if it chose them itself. The expert ratio decays over episodes.

Usage:
    env = KickstartWrapper(env, expert_episodes=500, decay_episodes=500)
    # Episodes 0-500: 100% expert actions
    # Episodes 500-1000: linear decay from 100% to 0%
    # Episodes 1000+: 0% expert (pure agent)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Optional

from .smart_bot import SmartBot
from .env_v3 import CARD_TO_IDX, GRID_X, GRID_Y


class KickstartWrapper(gym.Wrapper):
    """Injects expert actions with decaying probability."""

    def __init__(self, env, expert_episodes: int = 500, decay_episodes: int = 500):
        super().__init__(env)
        self.expert = SmartBot(player_id=0)
        self.expert_episodes = expert_episodes
        self.decay_episodes = decay_episodes
        self.total_episodes = 0
        self._last_expert_action = None

    def reset(self, **kwargs):
        self.total_episodes += 1
        self._last_expert_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        # Decide: expert or agent action?
        ratio = self._expert_ratio()

        if ratio > 0 and np.random.random() < ratio:
            # Use expert action instead
            expert_action = self._get_expert_action()
            if expert_action is not None:
                action = expert_action

        return self.env.step(action)

    def _expert_ratio(self) -> float:
        """Current expert action probability (1.0 → 0.0 over time)."""
        if self.total_episodes <= self.expert_episodes:
            return 1.0
        elif self.total_episodes <= self.expert_episodes + self.decay_episodes:
            progress = (self.total_episodes - self.expert_episodes) / self.decay_episodes
            return 1.0 - progress
        else:
            return 0.0

    def _get_expert_action(self) -> Optional[np.ndarray]:
        """Get the expert's action without executing it."""
        battle = self.env.battle
        if battle is None:
            return None

        player = battle.players[0]
        hand_before = list(player.hand)
        elixir_before = player.elixir
        entities_before = set(battle.entities.keys())

        # Let expert act
        self.expert.act(battle)

        # Detect what happened
        hand_after = list(player.hand)
        entities_after = set(battle.entities.keys())
        new_entities = entities_after - entities_before

        if hand_before == hand_after or not new_entities:
            # Expert chose WAIT (or deploy failed)
            return np.array([0, 0, 0], dtype=np.int64)

        # Find what was played and where
        new_id = min(new_entities)
        new_entity = battle.entities[new_id]
        deploy_x = int(new_entity.position.x)
        deploy_y = int(new_entity.position.y)

        # Find hand slot
        card_slot = 0
        for i in range(min(len(hand_before), len(hand_after))):
            if hand_before[i] != hand_after[i]:
                card_slot = i
                break

        return np.array([
            min(4, card_slot + 1),
            max(0, min(GRID_X - 1, deploy_x)),
            max(0, min(GRID_Y - 1, deploy_y)),
        ], dtype=np.int64)
