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

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Must call super with the total features_dim
        super().__init__(observation_space, features_dim)

        spatial_shape = observation_space["spatial"].shape  # (3, 32, 18)
        scalar_shape = observation_space["scalars"].shape  # (15,)

        # CNN for spatial features
        # Input: (batch, 3, 32, 18)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # (32, 32, 18)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 16, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # (64, 8, 5)
            nn.ReLU(),
            nn.Flatten(),  # 64 * 8 * 5 = 2560
        )

        # Compute CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, *spatial_shape)
            cnn_out_size = self.cnn(sample).shape[1]

        # FC for scalar features
        self.scalar_fc = nn.Sequential(
            nn.Linear(scalar_shape[0], 64),
            nn.ReLU(),
        )

        # Combined projection
        self.combined_fc = nn.Sequential(
            nn.Linear(cnn_out_size + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        spatial = observations["spatial"]
        scalars = observations["scalars"]

        spatial_features = self.cnn(spatial)
        scalar_features = self.scalar_fc(scalars)

        combined = torch.cat([spatial_features, scalar_features], dim=1)
        return self.combined_fc(combined)
