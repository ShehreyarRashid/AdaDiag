"""Methodology step 8 — evaluation, matching TVDiag's own metric definitions.

RCL: HR@1..HR@5 and MRR@3 (reciprocal rank, zero beyond rank 3), as in helper/eval.py.
FTI: weighted precision and recall, and -- important -- TVDiag reports F1 as the HARMONIC
     MEAN of those two weighted figures, not sklearn's weighted F1. The published numbers
     are that quantity, so `f1` below is computed their way and `f1_sklearn_weighted` is
     reported alongside because the two differ. `macro_f1` is added because on GAIA two
     classes are 94% of the data, so weighted F1 hides exactly the per-class effects this
     work is about.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


@torch.no_grad()
def evaluate(model, ds, idx: np.ndarray, batch_size: int = 512) -> dict:
    model.eval()
    hits = {k: 0 for k in range(1, 6)}
    rr, n = 0.0, 0
    y_pred, y_true = [], []
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        keep = torch.ones(len(b), ds.n_nodes, dtype=torch.bool)
        _, _, node_logits, type_logits = model(ds.X[b], ds.A[b], keep, ds.cnt[b])
        order = node_logits.argsort(dim=1, descending=True)
        root = ds.root[b]
        rank = (order == root[:, None]).float().argmax(1) + 1
        for k in range(1, 6):
            hits[k] += int((rank <= k).sum())
        rr += float(torch.where(rank <= 3, 1.0 / rank, torch.zeros_like(rank, dtype=torch.float)).sum())
        y_pred.append(type_logits.argmax(1))
        y_true.append(ds.ytype[b])
        n += len(b)

    yp = torch.cat(y_pred).numpy()
    yt = torch.cat(y_true).numpy()
    p = precision_score(yt, yp, average="weighted", zero_division=0)
    r = recall_score(yt, yp, average="weighted", zero_division=0)
    out = {f"HR@{k}": hits[k] / n for k in range(1, 6)}
    out["MRR@3"] = rr / n
    out["precision"] = p
    out["recall"] = r
    out["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0      # TVDiag's definition
    out["f1_sklearn_weighted"] = f1_score(yt, yp, average="weighted", zero_division=0)
    out["macro_f1"] = f1_score(yt, yp, average="macro", zero_division=0)
    return out


@torch.no_grad()
def per_class(model, ds) -> tuple[dict, dict]:
    """Per-failure-class HR@1/HR@3/FTI-recall, and the mean learned weight matrix per class.

    The premise of this work is that the useful source varies BY FAILURE CLASS, so an
    aggregate table can neither confirm nor refute it. This is the table that can.
    """
    model.eval()
    te = ds.test_idx
    metrics, weights = {}, {}
    for ci, cname in enumerate(ds.types):
        sel = te[ds.ytype[te].numpy() == ci]
        if len(sel) == 0:
            continue
        keep = torch.ones(len(sel), ds.n_nodes, dtype=torch.bool)
        _, f, node_logits, type_logits = model(ds.X[sel], ds.A[sel], keep, ds.cnt[sel])
        order = node_logits.argsort(dim=1, descending=True)
        rank = (order == ds.root[sel][:, None]).float().argmax(1) + 1
        metrics[cname] = dict(
            n=len(sel),
            hr1=float((rank <= 1).float().mean()),
            hr3=float((rank <= 3).float().mean()),
            fti_recall=float((type_logits.argmax(1) == ci).float().mean()),
        )
        if model.weighting.is_learned:
            weights[cname] = model.weights(f, ds.cnt[sel]).mean(0).tolist()
    return metrics, weights
