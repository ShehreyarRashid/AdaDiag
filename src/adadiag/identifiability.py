"""Methodology step 10 — why the learned arms cannot work as specified.

This module exists because of a review finding, and it is the most important thing in the
repository. It turns the null result from "we tried adaptive weighting and the metrics did
not move" into a structural statement that can be checked in a few seconds.

The claim, in three parts
-------------------------

**(1) The weighting module receives no gradient from either diagnosis loss.**

``w`` is used in exactly one place: the task-oriented contrastive term ``l_to``. It is not
on the forward path that produces node logits or type logits (that is the whole reason
AdaDiag adds no inference cost). So for ``v2_global`` and ``v3_incident``::

    d l_rcl / d theta_w  ==  0        exactly
    d l_fti / d theta_w  ==  0        exactly

``measure_gradient_paths`` verifies this numerically. The module's only learning signal is
``sigma * l_to``, and sigma = 0.1.

**(2) l_to is LINEAR in w, so on a simplex its minimiser is a vertex.**

    l_to = sum_m sum_t  w[m, t] * L_{m,t}

with each task column constrained to the simplex. A linear function on a simplex attains
its minimum at a vertex, so the only minimisers are one-hot columns -- "put everything on
one modality per task". The smooth interior allocation the proposal is built around is not
a solution of the objective as written; it is only ever a point the optimiser passes
through. ``measure_collapse`` reports how close the learned columns get to a vertex.

**(3) Therefore the simplex fixes the wrong degree of freedom.**

The simplex was introduced to make ``w`` identifiable against the outer scale sigma, and it
does that. But identifiability is not alignment: nothing ties the chosen vertex to
diagnosis accuracy, because by (1) no diagnosis gradient reaches the module. The vertex is
selected by the auxiliary loss landscape. Per-incident conditioning therefore does not buy
a graded "who to listen to" -- it buys a per-incident choice of one modality per task,
which is the same KIND of rule as TVDiag's hard mapping, chosen by auxiliary-loss
convenience instead of by domain knowledge.

What this does and does not prove
---------------------------------
Verified: (1) exactly, and (2) empirically -- every GAIA seed collapses to >= 0.979 on a
single modality for both tasks, as does Hotel. NOT claimed: that the selected vertex is
always ``argmin L_{m,t}``. Measured after training it matches on RCL but not on FTI, where
the three FTI terms sit within 3% of each other and the encoders have co-adapted to
whatever ``w`` became. The collapse is the robust finding; which vertex is picked is not.

What would be needed instead
----------------------------
Any fix has to get task gradient into the module without putting ``w`` on the prediction
path (``v4_fusion`` does the latter and is worse -- README section "Why V4 exists"):

* a bi-level / hypergradient step, updating ``w`` on held-out *task* loss rather than on
  ``l_to``;
* an entropy or temperature penalty on the columns, which would prevent vertex collapse but
  only by overriding the objective rather than fixing it;
* raising sigma so that ``l_to`` has enough influence for its allocation to matter at all --
  the gradient-share diagnostic (``diagnostics.py``) puts it at 10.4% on GAIA and 1.7% on
  Hotel, so there is very little to reallocate in the first place.

All three change TVDiag's objective, which is out of scope for a controlled comparison
against it. They are the honest next step, not a result claimed here.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from .losses import supcon_per_sample
from .model import AdaDiag
from .train import train
from .weighting import MODALITIES, TASKS


def measure_gradient_paths(ds, cfg: dict, arm: str = "v3_incident", seed: int = 0) -> dict:
    """Claim (1): total |grad| reaching the weighting parameters from each loss term.

    Returns a dict of term -> total absolute gradient at the weighting module's parameters.
    ``l_rcl`` and ``l_fti`` should be exactly 0.0.
    """
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = AdaDiag(arm, len(ds.types), cfg)
    if not model.weighting.is_learned:
        raise ValueError(f"arm {arm!r} has no learned weighting parameters")

    b = ds.train_idx[: cfg["batch_size"]]
    keep = torch.ones(len(b), ds.n_nodes, dtype=torch.bool)
    fs, f, node_logits, type_logits = model(ds.X[b], ds.A[b], keep, ds.cnt[b])
    w = model.weights(f, ds.cnt[b])
    labels = {"rcl": ds.root[b], "fti": ds.ytype[b]}

    l_to = torch.zeros((), dtype=torch.float32)
    for mi, m in enumerate(MODALITIES):
        for ti, t in enumerate(TASKS):
            per_sample, nz = supcon_per_sample(fs[m], labels[t], cfg["temperature"])
            l_to = l_to + (w[:, mi, ti] * per_sample).sum() / nz

    terms = {
        "l_rcl": F.cross_entropy(node_logits, labels["rcl"]),
        "l_fti": F.cross_entropy(type_logits, labels["fti"]),
        "sigma*l_to": cfg["contrastive_loss_scale"] * l_to,
    }
    params = list(model.weighting.parameters())
    out = {}
    for name, term in terms.items():
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        out[name] = sum(0.0 if g is None else float(g.abs().sum()) for g in grads)
    return {"dataset": ds.name, "arm": arm, "grad_to_weighting": out}


def measure_collapse(ds, cfg: dict, arm: str = "v3_incident", seed: int = 0) -> dict:
    """Claim (2): train the arm, then report how close each task column is to a vertex.

    ``max_weight`` near 1.0 means the column collapsed onto a single modality. Also reports
    the per-(modality, task) SupCon values at convergence and whether the selected vertex
    coincides with their argmin -- which is informative but, as the module docstring says,
    not a claim that survives on both tasks.
    """
    res, model = train(arm, seed, ds, cfg)
    b = ds.train_idx
    keep = torch.ones(len(b), ds.n_nodes, dtype=torch.bool)
    with torch.no_grad():
        fs, f, _, _ = model(ds.X[b], ds.A[b], keep, ds.cnt[b])
        w = model.weights(f, ds.cnt[b]).mean(0)
    labels = {"rcl": ds.root[b], "fti": ds.ytype[b]}

    cols = {}
    for ti, t in enumerate(TASKS):
        sup = []
        for m in MODALITIES:
            per_sample, nz = supcon_per_sample(fs[m], labels[t], cfg["temperature"])
            sup.append(float(per_sample.sum() / nz))
        col = [float(w[mi, ti]) for mi in range(len(MODALITIES))]
        cols[t] = {
            "weights": dict(zip(MODALITIES, [round(c, 4) for c in col])),
            "supcon": dict(zip(MODALITIES, [round(s, 4) for s in sup])),
            "max_weight": round(max(col), 4),
            "argmax_w": MODALITIES[col.index(max(col))],
            "argmin_supcon": MODALITIES[sup.index(min(sup))],
            "vertex_is_argmin": MODALITIES[col.index(max(col))] == MODALITIES[sup.index(min(sup))],
        }
    return {"dataset": ds.name, "arm": arm, "seed": seed,
            "HR@1": res["HR@1"], "f1": res["f1"], "columns": cols}
