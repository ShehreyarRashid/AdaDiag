"""Methodology step 4 — the two diagnosis heads.

RCL (root cause localization) scores every instance from its concatenated node embeddings
and is trained with a cross-entropy over the instances of each graph.
FTI (failure type identification) classifies the concatenated graph embeddings.

Both are plain MLPs, matching TVDiag's `Voter` and `Classifier`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class Locator(nn.Module):
    """One logit per instance -> ranked list."""

    def __init__(self, in_dim: int, hidden: list[int]):
        super().__init__()
        self.net = _mlp(in_dim, hidden, 1)

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        return self.net(node_emb).squeeze(-1)


class TypeClassifier(nn.Module):
    """One logit per failure type."""

    def __init__(self, in_dim: int, hidden: list[int], n_types: int):
        super().__init__()
        self.net = _mlp(in_dim, hidden, n_types)

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        return self.net(graph_emb)
