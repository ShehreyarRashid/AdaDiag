"""Methodology step 9 — the gradient-share diagnostic.

A null result is only interpretable if you know how much influence the thing you changed
actually has. AdaDiag reweights the task-oriented contrastive term; this measures what
fraction of the encoder gradient that term supplies.

If `sigma * l_to` contributes a small share, then `w` can only ever reallocate within that
share and NO weighting scheme -- fixed or learned -- can move the end metrics. That is a
property of TVDiag's objective, not of the weighting module, and it is the difference
between "AdaDiag does not work" and "the task-oriented loss is too weakly coupled to the
heads for any reweighting of it to matter".

CAVEAT on "share". Gradient norms do not add linearly -- two terms can reinforce or cancel
at the parameter level -- so `norm_i / sum_j norm_j` is a proxy for influence, not an exact
decomposition of the applied update. It is reported because the ordering it gives is large
and stable (on Hotel, 1.7% against 45-58%), not because the percentages are exact.

See also `identifiability.py`, which shows the complementary and sharper problem: the
weighting module receives no gradient from either diagnosis loss at all.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from .augment import drop_non_root
from .losses import AutomaticWeightedLoss, uspcon
from .model import AdaDiag
from .train import task_oriented_loss


def gradient_shares(arm: str, ds, cfg: dict, seed: int = 0, epochs: int = 120) -> dict:
    """Train briefly and report the gradient norm each loss term induces at the encoders."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    rng = random.Random(seed)
    model = AdaDiag(arm, len(ds.types), cfg)
    awl = AutomaticWeightedLoss(4, learnable=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    enc_params = list(model.encoders.parameters())
    track = []

    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(ds.train_idx)
        for s in range(0, len(perm), cfg["batch_size"]):
            b = perm[s:s + cfg["batch_size"]]
            X, A = ds.X[b], ds.A[b]
            root, yt, cnt = ds.root[b], ds.ytype[b], ds.cnt[b]
            keep = torch.ones(len(b), ds.n_nodes, dtype=torch.bool)
            Xa, Aa, ka = drop_non_root(X, A, root, cfg["aug_percent"], rng)
            X, A = torch.cat([X, Xa]), torch.cat([A, Aa])
            keep = torch.cat([keep, ka])
            root, yt, cnt = torch.cat([root] * 2), torch.cat([yt] * 2), torch.cat([cnt] * 2)

            fs, f, node_logits, type_logits = model(X, A, keep, cnt)
            w = model.weights(f, cnt)
            l_to = task_oriented_loss(model, fs, {"rcl": root, "fti": yt}, w, cfg["temperature"])
            l_cm = sum(uspcon(fs["metric"], fs[m], cfg["temperature"]) for m in ("trace", "log"))
            l_rcl = F.cross_entropy(node_logits.masked_fill(~keep, float("-inf")), root)
            l_fti = F.cross_entropy(type_logits, yt)
            sigma = cfg["contrastive_loss_scale"]

            def gnorm(term):
                gs = torch.autograd.grad(term, enc_params, retain_graph=True, allow_unused=True)
                return float(torch.sqrt(sum((g ** 2).sum() for g in gs if g is not None)))

            track.append([gnorm(l_rcl), gnorm(l_fti), gnorm(sigma * l_to), gnorm(sigma * l_cm)])

            loss = awl(l_rcl, l_fti, sigma * l_to, sigma * l_cm)
            opt.zero_grad(); loss.backward(); opt.step()

    T = np.asarray(track)[len(track) // 2:]        # second half of training
    g = T.mean(0)
    names = ["l_rcl", "l_fti", "sigma*l_to", "sigma*l_cm"]
    return {"dataset": ds.name, "arm": arm,
            "grad_norm": dict(zip(names, g.round(4).tolist())),
            "share": dict(zip(names, (100 * g / g.sum()).round(1).tolist()))}
