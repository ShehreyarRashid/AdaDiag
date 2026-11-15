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


def _ranks(node_logits: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
    """1-based rank of the true root cause among each incident's instances."""
    order = node_logits.argsort(dim=1, descending=True)
    return (order == root[:, None]).float().argmax(1) + 1


def fti_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """FTI metrics, TVDiag's way: F1 is the harmonic mean of weighted precision and recall."""
    p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    return {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if (p + r) > 0 else 0.0,
        "f1_sklearn_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def metrics_from_scores(node_p: np.ndarray, type_p: np.ndarray, root: np.ndarray,
                        y: np.ndarray) -> dict:
    """`evaluate`'s metrics from score arrays instead of a model (used for ensembles).

    node_p (n, N) instance scores, type_p (n, T) failure-type scores; ties in node_p are
    broken by instance index (stable sort).
    """
    if len(root) == 0:
        raise ValueError("no incidents to score")
    order = np.argsort(-node_p, axis=1, kind="stable")
    rank = (order == root[:, None]).argmax(1) + 1
    out = {f"HR@{k}": float((rank <= k).mean()) for k in range(1, 6)}
    out["MRR@3"] = float(np.where(rank <= 3, 1.0 / rank, 0.0).mean())
    out.update(fti_metrics(y, type_p.argmax(1)))
    return out


@torch.no_grad()
def evaluate(model, ds, idx: np.ndarray, batch_size: int = 512) -> dict:
    if len(idx) == 0:
        raise ValueError("no incidents to evaluate")
    model.eval()
    hits = {k: 0 for k in range(1, 6)}
    rr, n = 0.0, 0
    y_pred, y_true = [], []
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        keep = torch.ones(len(b), ds.n_nodes, dtype=torch.bool)
        _, _, node_logits, type_logits = model(ds.X[b], ds.A[b], keep, ds.cnt[b])
        rank = _ranks(node_logits, ds.root[b])
        for k in range(1, 6):
            hits[k] += int((rank <= k).sum())
        # float64, like metrics_from_scores (float32 left MRR@3 off by ~1e-8)
        rr += float(torch.where(rank <= 3, 1.0 / rank.double(), 0.0).sum())
        y_pred.append(type_logits.argmax(1))
        y_true.append(ds.ytype[b])
        n += len(b)

    out = {f"HR@{k}": hits[k] / n for k in range(1, 6)}
    out["MRR@3"] = rr / n
    out.update(fti_metrics(torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()))
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
        rank = _ranks(node_logits, ds.root[sel])
        metrics[cname] = dict(
            n=len(sel),
            hr1=float((rank <= 1).float().mean()),
            hr3=float((rank <= 3).float().mean()),
            fti_recall=float((type_logits.argmax(1) == ci).float().mean()),
        )
        if model.weighting.is_learned:
            weights[cname] = model.weights(f, ds.cnt[sel]).mean(0).tolist()
    return metrics, weights
