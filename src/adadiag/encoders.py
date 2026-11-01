"""Methodology step 3 — per-modality graph encoders.

One 2-layer GraphSAGE (mean aggregator, no bias) per data source, then a max-pool readout,
exactly as TVDiag configures `dgl.nn.SAGEConv`. DGL is deliberately NOT a dependency: these
graphs have 10-21 nodes, so a dense batched adjacency matmul is faster, has no version
pinning problems, and is far easier to inspect.

dgl's SAGEConv(aggregator_type='mean', bias=False) computes
    h' = W_self . h + W_neigh . mean_{u in N(v)} h_u
which is what `SAGEEncoder` implements below.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGEEncoder(nn.Module):
    def __init__(self, dims: list[int], feat_drop: float = 0.0):
        super().__init__()
        self.self_fc = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims, dims[1:]))
        self.neigh_fc = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims, dims[1:]))
        self.drop = nn.Dropout(feat_drop)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # clamp: an instance with no in-edges would otherwise give a 0/0 neighbour mean
        deg = A.sum(-1, keepdim=True).clamp(min=1.0)
        last = len(self.self_fc) - 1
        for i, (sf, nf) in enumerate(zip(self.self_fc, self.neigh_fc)):
            h = self.drop(h)
            h = sf(h) + nf(torch.bmm(A, h) / deg)
            if i < last:
                h = F.relu(h)
        return h


def max_pool(node_emb: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Graph-level readout. Dropped nodes are excluded, not pooled as zeros."""
    masked = node_emb.masked_fill(~keep[..., None], float("-inf"))
    return torch.nan_to_num(masked.max(1).values, neginf=0.0)
