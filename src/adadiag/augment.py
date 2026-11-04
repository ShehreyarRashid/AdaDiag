"""Methodology step 2 — graph augmentation (TVDiag's AUG).

Drops a fraction of the NON-root instances from each incident graph, keeping the culprit.
TVDiag mixes these augmented copies into every batch alongside the raw graphs; their own
ablation shows AUG is the component that carries most of the gain, so it is reproduced here
rather than simplified away.

Reference: core/aug.py in WHU-AISE/TVDiag (`aug_drop_node`).
"""
from __future__ import annotations

import random

import torch


def drop_non_root(X: torch.Tensor, A: torch.Tensor, root: torch.Tensor,
                  pct: float, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (X_aug, A_aug, keep_mask).

    Dropped nodes are zeroed in the features and disconnected in the adjacency; `keep` is
    returned so the readout and the RCL softmax can mask them out. Leaving them unmasked
    would let a dropped instance compete as a candidate root cause.
    """
    B, _, N, _ = X.shape
    k = int(N * pct)
    keep = torch.ones(B, N, dtype=torch.bool)
    for b in range(B):
        candidates = [j for j in range(N) if j != int(root[b])]
        for j in rng.sample(candidates, min(k, len(candidates))):
            keep[b, j] = False
    m = keep.float()
    return X * m[:, None, :, None], A * m[:, None, :] * m[:, :, None], keep


def drop_modality(X: torch.Tensor, start: int, pct: float, rng: random.Random) -> torch.Tensor:
    """README section 6.21: TVDiag's unused `aug_loss_modality` (core/aug.py), applied to the
    augmented rows X[start:]. In each, int(N * pct) instances, any instance including the
    root, each lose one modality chosen at random: its features are zeroed. The graph and
    the instance stay, so the model must diagnose from the other two sources."""
    X = X.clone()
    _, M, N, _ = X.shape
    k = int(N * pct)
    for b in range(start, X.shape[0]):
        for j in rng.sample(range(N), k):
            X[b, rng.randint(0, M - 1), j] = 0.0
    return X
