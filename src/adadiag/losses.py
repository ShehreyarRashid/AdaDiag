"""Methodology step 5 — losses.

Three pieces, all reproduced from TVDiag so that only the weighting differs between arms:

* `supcon_per_sample` — supervised contrastive loss, refactored to return the PER-SAMPLE
  term. TVDiag's `SupConLoss` returns one scalar for the whole batch, which makes a
  per-incident weight impossible. Summing what this returns and dividing by `nz`
  reproduces their scalar exactly; `tests/test_supcon_equivalence.py` asserts it.
* `uspcon` — the unsupervised cross-modal (CM) term, unchanged.
* `AutomaticWeightedLoss` — the outer Eq. 13 weighting, reproduced *including the fact that
  it does not learn* in the released implementation (see the class docstring).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def supcon_per_sample(emb: torch.Tensor, labels: torch.Tensor,
                      temperature: float) -> tuple[torch.Tensor, int]:
    """TVDiag's supervised contrastive loss, per sample.

    Returns:
        per_sample: (n,) loss contribution of each incident in the batch.
        nz: the non-zero count TVDiag divides by.

    `per_sample.sum() / nz` equals TVDiag's `SupConLoss(emb, labels)` to machine precision.
    """
    n = labels.shape[0]
    z = F.normalize(emb, dim=1)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    mask = (labels.view(-1, 1) == labels.view(1, -1)).float()
    mask_no_sim = 1.0 - mask
    eye = torch.eye(n, device=emb.device)
    sim = torch.exp(sim / temperature) * (1.0 - eye)
    pos = mask * sim
    neg = sim - pos
    denom = pos + neg.sum(1).repeat(n, 1).T
    partial = -torch.log(mask_no_sim + pos / denom.clamp(min=1e-12) + eye)
    nz = max(int((partial != 0).sum().item()), 1)
    return partial.sum(1), nz


def uspcon(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    """Cross-modal association (CM): InfoNCE between two modality views of the same incidents."""
    za, zb = F.normalize(a, dim=1), F.normalize(b, dim=1)
    logits = za @ zb.T / temperature
    return F.cross_entropy(logits, torch.arange(za.shape[0], device=a.device))


class AutomaticWeightedLoss(nn.Module):
    """Liebel & Koerner uncertainty weighting over the four loss terms (TVDiag's Eq. 13).

    FIDELITY NOTE, and this matters. In `core/TVDiag.py` the optimizer is built over
    `model.parameters()` on line 50 and this module is constructed on line 52, so its
    parameters are never added to any param group and are never stepped. They stay at their
    initial 1.0, and the module reduces to

        0.5 * sum(losses) + 4 * log(2)

    i.e. plain equal weighting plus a constant. Eq. 13's "dynamic" weighting therefore does
    nothing in the released implementation. We reproduce that behaviour exactly -- adding
    these parameters to the optimizer would mean no longer reproducing TVDiag -- and keep it
    identical across every arm so the comparison stays controlled.
    """

    def __init__(self, n: int = 4, learnable: bool = False):
        super().__init__()
        p = torch.ones(n)
        self.p = nn.Parameter(p, requires_grad=learnable)
        self.learnable = learnable

    def forward(self, *losses: torch.Tensor) -> torch.Tensor:
        return sum(0.5 / self.p[i] ** 2 * x + torch.log(1 + self.p[i] ** 2)
                   for i, x in enumerate(losses))
