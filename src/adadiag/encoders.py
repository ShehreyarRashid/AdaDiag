"""Methodology step 3 — per-modality graph encoders.

One 2-layer GraphSAGE (mean aggregator, no bias) per data source, then a max-pool readout,
exactly as TVDiag configures `dgl.nn.SAGEConv`. DGL is deliberately NOT a dependency: these
graphs have 10-21 nodes, so a dense batched adjacency matmul is faster, has no version
pinning problems, and is far easier to inspect.

dgl's SAGEConv(aggregator_type='mean', bias=False) computes
    h' = W_self . h + W_neigh . mean_{u in N(v)} h_u
which is what `SAGEEncoder` implements below.

`aggregator="lstm"` (README section 6.18) is the aggregator the TVDiag PAPER describes; its
released code uses "mean". It reproduces dgl's SAGEConv(aggregator_type='lstm'): each node's
in-neighbours, in index order, are fed through an LSTM whose hidden size equals its input
size, from a zero initial state, and the final hidden state replaces the mean:
    h' = W_self . h + W_neigh . LSTM(h_u1, ..., h_uk)[-1]
A node with no in-neighbour gets a zero message, as in dgl. The default stays "mean", so
every committed result is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def lstm_neighbours(lstm: nn.LSTM, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """dgl's LSTM reducer, dense and batched. h: (B, N, D); A[b, i, j] > 0: j -> i.

    Returns (B, N, D): for each node, the LSTM's final hidden state over its in-neighbours in
    ascending index order; zero for a node with none. Nodes are grouped by in-degree and each
    group runs as one dense LSTM call, as dgl's degree bucketing does (faster on CPU than
    packed sequences, and identical to a node-by-node loop: tests/test_lstm_aggregator.py).
    """
    B, N, D = h.shape
    mask = A > 0
    deg = mask.sum(-1).reshape(-1)                               # (B*N,)
    out = h.new_zeros(B * N, D)
    k = int(deg.max()) if deg.numel() else 0
    if k == 0:
        return out.view(B, N, D)
    # neighbour indices in ascending order, padded: a stable sort puts the True entries first
    order = torch.argsort((~mask).to(torch.int8), dim=-1, stable=True)[..., :k]   # (B, N, k)
    nb = torch.gather(h.unsqueeze(1).expand(B, N, N, D), 2,
                      order.unsqueeze(-1).expand(B, N, k, D)).reshape(B * N, k, D)
    for d in torch.unique(deg).tolist():
        if d == 0:
            continue
        sel = (deg == d).nonzero().squeeze(1)
        _, (hn, _) = lstm(nb[sel, :d])
        out = out.index_copy(0, sel, hn[-1])
    return out.view(B, N, D)


class SAGEEncoder(nn.Module):
    """`alg1=True` (README section 6.19) is Algorithm 1 of the TVDiag PAPER, lines 6-7:
    sigma after EVERY layer, then each node embedding L2-normalised, at every layer. The
    released code (and the default here) applies ReLU between layers only and never
    normalises. The weights and their initialisation are the same either way."""

    def __init__(self, dims: list[int], feat_drop: float = 0.0, aggregator: str = "mean",
                 alg1: bool = False):
        super().__init__()
        if aggregator not in ("mean", "lstm"):
            raise ValueError(f"unknown aggregator {aggregator!r}; expected 'mean' or 'lstm'")
        self.aggregator = aggregator
        self.alg1 = bool(alg1)
        self.self_fc = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims, dims[1:]))
        self.neigh_fc = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims, dims[1:]))
        if aggregator == "lstm":     # built after the linear layers, so they initialise as before
            self.lstm = nn.ModuleList(nn.LSTM(a, a, batch_first=True) for a in dims[:-1])
        self.drop = nn.Dropout(feat_drop)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # clamp: an instance with no in-edges would otherwise give a 0/0 neighbour mean
        deg = A.sum(-1, keepdim=True).clamp(min=1.0)
        last = len(self.self_fc) - 1
        for i, (sf, nf) in enumerate(zip(self.self_fc, self.neigh_fc)):
            h = self.drop(h)
            if self.aggregator == "lstm":
                h = sf(h) + nf(lstm_neighbours(self.lstm[i], h, A))
            else:
                h = sf(h) + nf(torch.bmm(A, h) / deg)
            if self.alg1:
                h = F.normalize(F.relu(h), dim=-1)    # Algorithm 1, lines 6-7
            elif i < last:
                h = F.relu(h)
        return h


def max_pool(node_emb: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Graph-level readout. Dropped nodes are excluded, not pooled as zeros."""
    masked = node_emb.masked_fill(~keep[..., None], float("-inf"))
    return torch.nan_to_num(masked.max(1).values, neginf=0.0)
