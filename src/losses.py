"""
Loss functions for pattern-family proxy metric learning.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TRAIN_CONFIG


class TripletLoss(nn.Module):
    """Batch-hard triplet loss on L2-normalized embeddings (pattern-family labels)."""

    def __init__(
        self,
        margin: float | None = None,
        p: int | None = None,
    ) -> None:
        super().__init__()
        self.margin = margin if margin is not None else TRAIN_CONFIG.triplet_margin
        self.p = p if p is not None else TRAIN_CONFIG.triplet_p
        self.triplet = nn.TripletMarginLoss(margin=self.margin, p=self.p)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return batch_hard_triplet_loss(embeddings, labels, self.margin, self.p)


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    p: int = 2,
) -> torch.Tensor:
    """
    For each anchor: hardest positive (same pattern family), hardest negative (different family).
    Safely skips anchors that lack a second positive or any negative in the batch.
    """
    if embeddings.size(0) < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    dist = torch.cdist(embeddings, embeddings, p=p)
    losses: list[torch.Tensor] = []
    n = labels.size(0)

    for i in range(n):
        same_mask = labels == labels[i]
        diff_mask = labels != labels[i]
        same_mask[i] = False
        if not same_mask.any() or not diff_mask.any():
            continue
        pos_dist = dist[i][same_mask].max()
        neg_dist = dist[i][diff_mask].min()
        losses.append(F.relu(pos_dist - neg_dist + margin))

    if not losses:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    return torch.stack(losses).mean()