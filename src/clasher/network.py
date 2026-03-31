"""Custom CNN + Scalar feature extractor for the V2 environment.

Processes the Dict observation space:
  "spatial": (3, 32, 18) — CNN extracts spatial features
  "scalars": (15,) — FC layer processes scalar features
  Both concatenated → shared representation for actor + critic

Compatible with SB3's MultiInputPolicy.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CRFeatureExtractor(BaseFeaturesExtractor):
    """CNN for spatial grid + FC for scalars → concatenated features."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        spatial_shape = observation_space["spatial"].shape  # (3, 32, 18)
        scalar_shape = observation_space["scalars"].shape

        # Lighter CNN: 16→32→32 filters (was 32→64→64)
        # Input: (batch, 3, 32, 18)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (16, 16, 9)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 8, 5)
            nn.ReLU(),
            nn.Flatten(),  # 32 * 8 * 5 = 1280
        )

        # Compute CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, *spatial_shape)
            cnn_out_size = self.cnn(sample).shape[1]

        # FC for scalar features
        self.scalar_fc = nn.Sequential(
            nn.Linear(scalar_shape[0], 32),
            nn.ReLU(),
        )

        # Combined projection
        self.combined_fc = nn.Sequential(
            nn.Linear(cnn_out_size + 32, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        spatial = observations["spatial"]
        scalars = observations["scalars"]

        spatial_features = self.cnn(spatial)
        scalar_features = self.scalar_fc(scalars)

        combined = torch.cat([spatial_features, scalar_features], dim=1)
        return self.combined_fc(combined)
