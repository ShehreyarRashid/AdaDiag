"""Measurements behind the comparison with the TVDiag paper (results/paper_comparison/).

TVDiag's evaluation (ACM TOSEM 35(2), Art. 40, Sec. 5) reports aggregate tables and a set of
figures. This module measures, on a trained model and the TEST split, the quantities those
figures plot, so each can be redrawn from this re-implementation and set beside the paper:

    per_class_k     HR@1/3/5 and FTI precision/recall per failure type      (Fig. 14)
    modality_shapley  exact Shapley value of each modality, per incident     (Fig. 18)
    to_agreement    same-label vs different-label cosine similarity          (Fig. 11)
    example_outputs FTI and RCL probability vectors for a few incidents      (Fig. 13)
    weight_matrix   the 3x2 modality x task matrix the arm used              (Fig. 11)

`VARIANTS` names the configuration changes behind Table 6 (w/o DW) and Fig. 16 (the four
hyper-parameter sweeps). Everything else in the comparison reuses runs already in results/.
"""
from __future__ import annotations

import itertools
import math

import torch

from .weighting import MODALITIES

# --------------------------------------------------------------------------- variants
#: name -> config overrides. "default" is TVDiag's own setting (config.py).
VARIANTS: dict[str, dict] = {
    "default": {},
    # Table 6, on TVDiag as shipped (v0_raw)
    "no_aug": {"aug_times": 0},
    "no_to": {"use_to": False},
    "no_cm": {"use_cm": False},
    "no_dw": {"dynamic_weights": False},
}
ABLATION_VARIANTS = ("no_aug", "no_to", "no_cm", "no_dw")
#: Six of the paper's nine points per sweep -- dense around the paper's stated optima
#: (p = 0.2, omega = 0.1, tau = 0.2), sparser above -- to keep Fig. 16 to ~2 CPU-hours.
_GRID = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
for _v in _GRID:
    VARIANTS[f"p={_v}"] = {"aug_percent": _v}                    # Fig. 16(a)
for _v in [0.0] + _GRID:
    VARIANTS[f"omega={_v}"] = {"contrastive_loss_scale": _v}     # Fig. 16(b)
for _v in _GRID:
    VARIANTS[f"tau={_v}"] = {"temperature": _v}                  # Fig. 16(c)
for _v in (1, 2, 3, 4):
    VARIANTS[f"l={_v}"] = {"graph_layers": _v}                   # Fig. 16(d)

#: Fig. 16 sweep name -> (config key, paper symbol).
SWEEPS = {"p": ("aug_percent", "p"), "omega": ("contrastive_loss_scale", "ω"),
          "tau": ("temperature", "τ"), "l": ("graph_layers", "l")}


def is_default(name: str, cfg: dict) -> bool:
    """True when a sweep point coincides with the dataset's default (run once, as 'default')."""
    over = VARIANTS[name]
    return (bool(over) and name not in ABLATION_VARIANTS
            and all(cfg.get(k) == v for k, v in over.items()))


# --------------------------------------------------------------------------- forward helpers
@torch.no_grad()
def _embed(model, ds, idx):
    """Per-modality node embeddings (n, N, d) and graph embeddings (n, d), as in forward()."""
    if model.arm == "v4_fusion":
        raise ValueError("paper_compare measures the published arms; v4 gates the heads")
    model.eval()
    X, A = ds.X[idx], ds.A[idx]
    keep = torch.ones(len(idx), ds.n_nodes, dtype=torch.bool)
    fs, es = {}, {}
    for j, m in enumerate(MODALITIES):
        if model.active[j]:
            es[m] = model.encoders[m](X[:, j], A)
        else:
            es[m] = X.new_zeros(X.shape[0], X.shape[2], model.d_out)
        fs[m] = es[m].max(1).values                      # keep is all-true at test time
    return es, fs


def _hr(rank, k):
    return float((rank <= k).float().mean())


# --------------------------------------------------------------------------- Fig. 14
@torch.no_grad()
def per_class_k(model, ds) -> dict:
    """HR@1/3/5, Avg@3 and FTI precision/recall for each failure type (test split)."""
    te = ds.test_idx
    es, fs = _embed(model, ds, te)
    node_logits = model.locator(torch.cat([es[m] for m in MODALITIES], -1))
    type_logits = model.classifier(torch.cat([fs[m] for m in MODALITIES], -1))
    order = node_logits.argsort(1, descending=True)
    rank = (order == ds.root[te][:, None]).float().argmax(1) + 1
    pred, yt = type_logits.argmax(1), ds.ytype[te]
    out = {}
    for ci, c in enumerate(ds.types):
        sel = yt == ci
        if int(sel.sum()) == 0:
            continue
        r = rank[sel]
        n_pred = int((pred == ci).sum())
        out[c] = dict(n=int(sel.sum()), hr1=_hr(r, 1), hr3=_hr(r, 3), hr5=_hr(r, 5),
                      avg3=(_hr(r, 1) + _hr(r, 2) + _hr(r, 3)) / 3,
                      fti_recall=float((pred[sel] == ci).float().mean()),
                      fti_precision=(int(((pred == ci) & sel).sum()) / n_pred) if n_pred else 0.0)
    return out


# --------------------------------------------------------------------------- Fig. 18
def _shapley3(v: dict) -> torch.Tensor:
    """Exact Shapley values for 3 players from a coalition -> value(n,) table."""
    n, phi = 3, []
    for m in range(n):
        others = [k for k in range(n) if k != m]
        tot = 0.0
        for r in range(n):
            for S in itertools.combinations(others, r):
                wgt = math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
                tot = tot + wgt * (v[frozenset(S) | {m}] - v[frozenset(S)])
        phi.append(tot)
    return torch.stack(phi, 1)


@torch.no_grad()
def modality_shapley(model, ds) -> dict:
    """Each modality's share of the prediction, per test incident, for FTI and for RCL.

    TVDiag (Sec. 5.5) computes SHAP on the 96-d fused feature and averages each modality's 32
    entries. Here the three modalities are the players and the Shapley value is EXACT (all
    2^3 coalitions): a modality left out is replaced by its mean embedding on the training
    split. The value is the probability of the model's own prediction (the predicted type
    for FTI; the top-ranked instance for RCL). Reported as |phi_m| / sum_k |phi_k|, so the
    three shares are in [0, 1] and sum to 1 -- the scale of the paper's Fig. 18 x-axis.
    """
    tr, te = ds.train_idx, ds.test_idx
    es_tr, fs_tr = _embed(model, ds, tr)
    es, fs = _embed(model, ds, te)
    base_f = {m: fs_tr[m].mean(0) for m in MODALITIES}
    base_e = {m: es_tr[m].mean((0, 1)) for m in MODALITIES}

    full_t = model.classifier(torch.cat([fs[m] for m in MODALITIES], -1))
    full_n = model.locator(torch.cat([es[m] for m in MODALITIES], -1))
    t_hat, n_hat = full_t.argmax(1), full_n.argmax(1)
    ar = torch.arange(len(te))

    v_fti, v_rcl = {}, {}
    for r in range(4):
        for S in itertools.combinations(range(3), r):
            S = frozenset(S)
            f = torch.cat([fs[m] if j in S else base_f[m].expand_as(fs[m])
                           for j, m in enumerate(MODALITIES)], -1)
            e = torch.cat([es[m] if j in S else base_e[m].expand_as(es[m])
                           for j, m in enumerate(MODALITIES)], -1)
            v_fti[S] = model.classifier(f).softmax(1)[ar, t_hat]
            v_rcl[S] = model.locator(e).softmax(1)[ar, n_hat]

    out = {}
    for task, v in (("fti", v_fti), ("rcl", v_rcl)):
        phi = _shapley3(v)
        share = phi.abs() / phi.abs().sum(1, keepdim=True).clamp_min(1e-12)
        out[task] = dict(share=[[round(float(x), 4) for x in row] for row in share],
                         mean_phi=[float(x) for x in phi.mean(0)],
                         mean_share=[float(x) for x in share.mean(0)])
    return out


# --------------------------------------------------------------------------- Fig. 11
@torch.no_grad()
def to_agreement(model, ds) -> dict:
    """Mean cosine similarity of same-label vs different-label pairs, per modality and label.

    Fig. 11 illustrates TO as "maximise agreement" between same-label samples of a modality.
    This measures whether that happened on the test split: for each modality's graph
    embedding and each label (root cause for RCL, failure type for FTI), the mean pairwise
    cosine of same-label pairs (pos), different-label pairs (neg) and the gap pos - neg.
    """
    te = ds.test_idx
    _, fs = _embed(model, ds, te)
    labels = {"rcl": ds.root[te], "fti": ds.ytype[te]}
    off = ~torch.eye(len(te), dtype=torch.bool)
    out = {}
    for m in MODALITIES:
        z = torch.nn.functional.normalize(fs[m], dim=1)
        S = z @ z.T
        out[m] = {}
        for t, y in labels.items():
            same = (y[:, None] == y[None, :]) & off
            diff = y[:, None] != y[None, :]
            pos, neg = float(S[same].mean()), float(S[diff].mean())
            out[m][t] = dict(pos=pos, neg=neg, gap=pos - neg)
    return out


# --------------------------------------------------------------------------- Fig. 13
@torch.no_grad()
def example_outputs(model, ds, k: int = 3) -> list:
    """FTI and RCL probability vectors for the first k test incidents (no selection)."""
    te = ds.test_idx[:k]
    es, fs = _embed(model, ds, te)
    pt = model.classifier(torch.cat([fs[m] for m in MODALITIES], -1)).softmax(1)
    pn = model.locator(torch.cat([es[m] for m in MODALITIES], -1)).softmax(1)
    return [dict(position=int(i), root=ds.instances[int(ds.root[i])],
                 type=ds.types[int(ds.ytype[i])],
                 fti=[round(float(x), 4) for x in pt[j]],
                 rcl=[round(float(x), 4) for x in pn[j]])
            for j, i in enumerate(te)]


@torch.no_grad()
def weight_matrix(model, ds) -> list:
    """The 3x2 matrix the arm applied, averaged over the test split."""
    te = ds.test_idx
    _, fs = _embed(model, ds, te)
    f = torch.cat([fs[m] for m in MODALITIES], -1)
    return model.weights(f, ds.cnt[te]).mean(0).tolist()
