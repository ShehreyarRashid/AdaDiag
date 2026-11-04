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


def supcon_weighted(emb: torch.Tensor, labels: torch.Tensor, temperature: float,
                    pos_beta: float = 0.0, neg_beta: float = 0.0,
                    keep_frac: float = 1.0) -> tuple[torch.Tensor, int]:
    """TVDiag's SupCon with pair- and anchor-level weights (README section 6.13).

    TVDiag's term for anchor i is a sum over its positives p of

        l_ip = -log( e_ip / (e_ip + sum_n e_in) ),     e_ij = exp(cos(z_i, z_j) / tau)

    Three weightings of it, each with weights of MEAN ONE so the term's scale is unchanged,
    each computed from detached similarities (they choose emphasis, they are not optimised):

    * `pos_beta` (type 1, weight positive pairs): l_ip -> u_ip * l_ip with
      u_ip proportional to exp(-pos_beta * cos_ip) over i's positives -- the positives the
      encoder still holds far away count more.
    * `neg_beta` (type 2, weight negative samples): sum_n e_in -> sum_n v_in * e_in with
      v_in proportional to exp(neg_beta * cos_in) over i's negatives -- the negatives that
      already sit close count more.
    * `keep_frac` < 1 (type 5, curriculum): only the `keep_frac` easiest anchors (lowest mean
      l_ip) contribute, rescaled so the term stays a mean over the included positive pairs;
      the trainer raises it to 1.

    All three at their defaults returns `supcon_per_sample` itself, bit for bit.
    """
    if pos_beta == 0.0 and neg_beta == 0.0 and keep_frac >= 1.0:
        return supcon_per_sample(emb, labels, temperature)
    n = labels.shape[0]
    z = F.normalize(emb, dim=1)
    cos = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    eye = torch.eye(n, device=emb.device)
    same = (labels.view(-1, 1) == labels.view(1, -1)).float()
    posm, negm = same * (1.0 - eye), 1.0 - same
    e = torch.exp(cos / temperature) * (1.0 - eye)
    c = cos.detach()
    if neg_beta:
        v = torch.exp(neg_beta * c) * negm
        v = v * (negm.sum(1, keepdim=True) / v.sum(1, keepdim=True).clamp_min(1e-12))
    else:
        v = negm
    negsum = (v * e).sum(1, keepdim=True)
    # -log(e_ip / (e_ip + negsum)) = log(e_ip + negsum) - cos_ip / tau, on positive pairs only
    lp = (torch.log((e + negsum).clamp_min(1e-30)) - cos / temperature) * posm
    nz = max(int((lp.detach() != 0).sum().item()), 1)
    if pos_beta:
        u = torch.exp(-pos_beta * c) * posm
        u = u * (posm.sum(1, keepdim=True) / u.sum(1, keepdim=True).clamp_min(1e-12))
        lp = lp * u
    per = lp.sum(1)
    if keep_frac < 1.0:
        npos = posm.sum(1)
        valid = npos > 0
        n_valid = int(valid.sum())
        if n_valid > 1:
            lbar = per.detach() / npos.clamp_min(1.0)
            k = max(1, int(round(keep_frac * n_valid)))
            thr = lbar[valid].sort().values[k - 1]
            inc = valid & (lbar <= thr)
            # rescale so the term stays a mean over the positive pairs that are included
            per = per * inc.to(per.dtype) * (npos[valid].sum() / npos[inc].sum())
    return per, nz


def supcon_anchor_mean(emb: torch.Tensor, labels: torch.Tensor,
                       temperature: float) -> tuple[torch.Tensor, int]:
    """Eq. 3 as the TVDiag PAPER writes it (README section 6.19, `to_norm="paper"`).

    The paper's term for anchor i is -1/|P_i| * sum_{j in P_i} log(...): a mean over i's OWN
    positives. The released code divides the sum over all positive pairs by their count
    instead, so an anchor with many positives counts for more. Same numerator and
    denominator; only the normalisation differs.

    Returns per_sample (n,) = each anchor's mean over its positives (0 for an anchor with
    none), and nz = the number of anchors that have a positive, so per_sample.sum() / nz is
    the mean over anchors. (Eq. 3 literally SUMS over anchors, i.e. n times this; with
    TVDiag's omega = 0.1 that would scale the term by the batch size, 256-1024x, which the
    1/2n of Eq. 5 suggests is not intended. A constant factor, recorded here.)
    """
    per, _ = supcon_per_sample(emb, labels, temperature)
    npos = (labels.view(-1, 1) == labels.view(1, -1)).sum(1) - 1
    valid = npos > 0
    per = torch.where(valid, per / npos.clamp_min(1).to(per.dtype), torch.zeros_like(per))
    return per, max(int(valid.sum()), 1)


def uspcon(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    """Cross-modal association (CM): InfoNCE between two modality views of the same incidents."""
    za, zb = F.normalize(a, dim=1), F.normalize(b, dim=1)
    logits = za @ zb.T / temperature
    return F.cross_entropy(logits, torch.arange(za.shape[0], device=a.device))


def cm_paper(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    """Eqs. 5-6 of the TVDiag PAPER (README section 6.19, `cm_mode="paper"`).

        L^{D1~D2} = 1/(2n) sum_i [ l(F_i^D1, F_i^D2) + l(F_i^D2, F_i^D1) ]
        l(F_i^D1, F_i^D2) = -log  phi(i1, i2) /
                                  ( sum_{a != i} phi(i1, a1) + sum_b phi(i1, b2) )

    with phi = exp(cos / tau). Two things the released `uspcon` leaves out: the second
    direction, and the intra-modal negatives sum_{a != i} phi(i1, a1).
    """
    za, zb = F.normalize(a, dim=1), F.normalize(b, dim=1)
    n = za.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=a.device)
    target = torch.arange(n, device=a.device) + n

    def one(x, y):
        intra = (x @ x.T / temperature).masked_fill(eye, float("-inf"))
        return F.cross_entropy(torch.cat([intra, x @ y.T / temperature], 1), target)

    return 0.5 * (one(za, zb) + one(zb, za))


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

    `learnable=True` (README section 6.19, `awl_learnable`) is Eq. 13 as the PAPER states
    it: `train` then adds these parameters to the optimizer.
    """

    def __init__(self, n: int = 4, learnable: bool = False):
        super().__init__()
        p = torch.ones(n)
        self.p = nn.Parameter(p, requires_grad=learnable)
        self.learnable = learnable

    def forward(self, *losses: torch.Tensor) -> torch.Tensor:
        return sum(0.5 / self.p[i] ** 2 * x + torch.log(1 + self.p[i] ** 2)
                   for i, x in enumerate(losses))
