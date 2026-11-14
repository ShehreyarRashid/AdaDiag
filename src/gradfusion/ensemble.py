"""v1.22 README section 6.25: GradFusion -- several GradFusion+ models that differ only in their seed.

Each member is trained independently (`train(arm, seed, ...)`). At diagnosis time every member
scores the incident and the softmax outputs are averaged: over instances for root cause
localization, over failure types for failure type identification (deep ensembles,
Lakshminarayanan et al., NeurIPS 2017). Nothing is trained jointly and no member is changed,
so cost is linear in the number of members, for training and for diagnosis.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def predict(models, ds, idx) -> tuple[torch.Tensor, torch.Tensor]:
    """(instance scores (len(idx), N), failure-type scores (len(idx), T)), member-averaged."""
    keep = torch.ones(len(idx), ds.n_nodes, dtype=torch.bool)
    node, typ = 0.0, 0.0
    for m in models:
        m.eval()
        _, _, nl, tl = m(ds.X[idx], ds.A[idx], keep, ds.cnt[idx])
        node = node + torch.softmax(nl, dim=1)
        typ = typ + torch.softmax(tl, dim=1)
    return node / len(models), typ / len(models)
