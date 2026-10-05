"""
EfficientNet-B0 models for pattern-family classification and metric learning.

Proxy supervision only — pattern families are NOT fine-grained design IDs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def _build_backbone(pretrained: bool = True) -> tuple[nn.Module, int]:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = efficientnet_b0(weights=weights)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    return backbone, in_features


class SareeClassifier(nn.Module):
    """Experiment A: EfficientNet-B0 + pattern-family classification head."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone, in_features = _build_backbone(pretrained)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class SareeEmbeddingModel(nn.Module):
    """
    Experiment B: EfficientNet-B0 + 256-D L2-normalized embedding + optional logits.

    Embedding and classification heads are separate modules.
    """

    def __init__(
        self,
        num_classes: int = 4,
        embedding_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone, in_features = _build_backbone(pretrained)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim

    def forward(
        self, x: torch.Tensor, return_logits: bool = True
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        out: dict[str, torch.Tensor] = {"embedding": embedding}
        if return_logits:
            out["logits"] = self.classifier(embedding)
        return out