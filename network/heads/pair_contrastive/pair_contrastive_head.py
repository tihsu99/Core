"""Trainable projection head for pair contrastive learning."""

import torch.nn.functional as F
from torch import Tensor, nn


class PairContrastiveHead(nn.Module):
    """Project selected pair states into normalized contrastive embeddings."""

    def __init__(
        self,
        pair_dim: int,
        projection_dim: int,
        symmetrize_pair: bool = True,
    ) -> None:
        super().__init__()
        if pair_dim <= 0 or projection_dim <= 0:
            raise ValueError("Pair and contrastive dimensions must be positive")
        self.symmetrize_pair = symmetrize_pair
        self.projector = nn.Sequential(
            nn.Linear(pair_dim, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, projection_dim),
        )

    def forward(
        self,
        pair_state: Tensor,
        batch: Tensor,
        i: Tensor,
        j: Tensor,
    ) -> Tensor:
        pair_values = pair_state[batch, i, j]
        if self.symmetrize_pair and batch.numel():
            pair_values = 0.5 * (pair_values + pair_state[batch, j, i])
        return F.normalize(self.projector(pair_values), dim=-1, eps=1.0e-8)
