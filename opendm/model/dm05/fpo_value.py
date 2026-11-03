"""State-value head for DM05 Flow Policy Optimization."""

from collections.abc import Sequence

import torch
from torch import nn


class DM05FPOValueHead(nn.Module):
    """Predict V(s) from frozen DM05 prefix features and robot state."""

    def __init__(
        self,
        prefix_dim: int,
        state_dim: int = 8,
        hidden_dims: Sequence[int] = (512, 256),
    ) -> None:
        super().__init__()
        if prefix_dim <= 0 or state_dim < 0:
            raise ValueError("prefix_dim must be positive and state_dim non-negative.")

        dimensions = [prefix_dim + state_dim, *hidden_dims, 1]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend(
                [
                    nn.Linear(input_dim, output_dim),
                    nn.ReLU(),
                ]
            )
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.mlp = nn.Sequential(*layers)
        self.state_dim = state_dim

    @staticmethod
    def pool_prefix(
        prefix_features: torch.Tensor,
        prefix_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prefix_features.ndim != 3:
            raise ValueError("prefix_features must have shape [B, L, D].")
        if prefix_valid_mask.shape != prefix_features.shape[:2]:
            raise ValueError("prefix_valid_mask must have shape [B, L].")
        mask = prefix_valid_mask.to(
            device=prefix_features.device,
            dtype=prefix_features.dtype,
        ).unsqueeze(-1)
        return (prefix_features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def forward(
        self,
        prefix_features: torch.Tensor,
        prefix_valid_mask: torch.Tensor,
        states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled_prefix = self.pool_prefix(prefix_features, prefix_valid_mask)
        if self.state_dim:
            if states is None:
                raise ValueError("states are required when state_dim is non-zero.")
            if states.shape != (pooled_prefix.shape[0], self.state_dim):
                raise ValueError(
                    "states must have shape "
                    f"{(pooled_prefix.shape[0], self.state_dim)}, got {states.shape}."
                )
            pooled_prefix = torch.cat(
                [
                    pooled_prefix,
                    states.to(
                        device=pooled_prefix.device,
                        dtype=pooled_prefix.dtype,
                    ),
                ],
                dim=-1,
            )
        return self.mlp(pooled_prefix).squeeze(-1).float()
