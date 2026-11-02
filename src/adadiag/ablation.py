"""Methodology step 11 — ablation studies.

Each group tests a specific claim the README makes, rather than ablating for its own sake:

  A  TVDiag's components (AUG, TO, CM).
     Claim tested (section 6.4): the task-oriented term AdaDiag reweights carries little of
     the learning signal. If removing TO entirely barely moves the metrics, then NO way of
     reweighting it can matter either -- an independent check on the gradient-share result.

  B  Modality removal (drop one, keep one).
     Claim tested (the premise): which source matters for which task, and for which failure
     class. This is the empirical "ground truth" of who to listen to, measured by taking each
     source away. Per-class results come with every run at no extra training cost.

  C  AdaDiag's own design choices, on the full method.
     Claims tested (sections 1 and 6.6): that the simplex is needed for identifiability
     (C_softplus), that the descriptor and detach matter (C_desc_*, C_no_detach), and that
     the vertex collapse is what the objective produces (C_entropy_* stop it -- does
     accuracy change when it stops?).

  D  Scale of the task-oriented term, sigma_to, for the prior and for AdaDiag.
     Claim tested (section 8, the named follow-up): do the arms separate once l_to carries
     real weight? Only l_to is rescaled; the CM term keeps TVDiag's 0.1.

Every configuration runs under TVDiag's hyperparameters with the same seeds as its
reference, so differences are paired. Early stopping is TVDiag's rule (on training loss),
which means removing a loss term also shifts where training stops -- that is how TVDiag's
own ablation behaves too, and it is part of what "removing the term" means here.
"""
from __future__ import annotations

import random

import torch

from .evaluate import per_class
from .train import compute_terms, make_batch, train

#: name -> (group, arm, cfg overrides, short description)
ABLATIONS: dict[str, tuple[str, str, dict, str]] = {
    # references -----------------------------------------------------------------------
    "ref_prior":     ("ref", "v0_simplex",  {}, "TVDiag prior (V0-simplex), full model"),
    "ref_adadiag":   ("ref", "v3_incident", {}, "AdaDiag (V3-incident), full model"),
    # A: TVDiag components, on the prior ------------------------------------------------
    "A_no_aug":      ("A", "v0_simplex", {"aug_times": 0}, "w/o AUG"),
    "A_no_to":       ("A", "v0_simplex", {"use_to": False}, "w/o TO (the term AdaDiag reweights)"),
    "A_no_cm":       ("A", "v0_simplex", {"use_cm": False}, "w/o CM"),
    "A_no_to_cm":    ("A", "v0_simplex", {"use_to": False, "use_cm": False},
                      "w/o TO and CM (diagnosis losses + AUG only)"),
    # B: modality removal, on the prior -------------------------------------------------
    "B_no_metric":   ("B", "v0_simplex", {"drop_modalities": ("metric",)}, "w/o metric"),
    "B_no_trace":    ("B", "v0_simplex", {"drop_modalities": ("trace",)}, "w/o trace"),
    "B_no_log":      ("B", "v0_simplex", {"drop_modalities": ("log",)}, "w/o log"),
    "B_only_metric": ("B", "v0_simplex", {"drop_modalities": ("trace", "log")}, "metric only"),
    "B_only_trace":  ("B", "v0_simplex", {"drop_modalities": ("metric", "log")}, "trace only"),
    "B_only_log":    ("B", "v0_simplex", {"drop_modalities": ("metric", "trace")}, "log only"),
    # C: AdaDiag design, on the full method --------------------------------------------
    "C_softplus":    ("C", "v3_incident", {"w_constraint": "softplus"},
                      "no simplex (unconstrained positive weights)"),
    "C_no_detach":   ("C", "v3_incident", {"detach": False}, "descriptor not detached"),
    "C_desc_emb":    ("C", "v3_incident", {"descriptor": "emb"}, "descriptor: embeddings only"),
    "C_desc_counts": ("C", "v3_incident", {"descriptor": "counts"}, "descriptor: alert counts only"),
    "C_entropy_01":  ("C", "v3_incident", {"entropy_reg": 0.01}, "+ entropy penalty, lambda=0.01"),
    "C_entropy_1":   ("C", "v3_incident", {"entropy_reg": 0.1}, "+ entropy penalty, lambda=0.1"),
    # D: scale of l_to ------------------------------------------------------------------
    "D_prior_s0.5":  ("D", "v0_simplex",  {"to_scale": 0.5}, "prior, sigma_to=0.5"),
    "D_prior_s1":    ("D", "v0_simplex",  {"to_scale": 1.0}, "prior, sigma_to=1"),
    "D_prior_s2":    ("D", "v0_simplex",  {"to_scale": 2.0}, "prior, sigma_to=2"),
    "D_ada_s0.5":    ("D", "v3_incident", {"to_scale": 0.5}, "AdaDiag, sigma_to=0.5"),
    "D_ada_s1":      ("D", "v3_incident", {"to_scale": 1.0}, "AdaDiag, sigma_to=1"),
    "D_ada_s2":      ("D", "v3_incident", {"to_scale": 2.0}, "AdaDiag, sigma_to=2"),
}

GROUPS = ("A", "B", "C", "D")


def reference_for(name: str) -> str | None:
    """The configuration a given ablation is compared against (same seeds)."""
    group, arm, _, _ = ABLATIONS[name]
    if group == "ref":
        return None
    return "ref_adadiag" if arm == "v3_incident" else "ref_prior"


def select(groups=None, only=None) -> list[str]:
    """Names to run: the requested groups plus the references they need."""
    if only:
        names = list(only)
    else:
        groups = set(groups or GROUPS)
        names = [n for n, (g, *_) in ABLATIONS.items() if g in groups]
    refs = {reference_for(n) for n in names} - {None}
    out = [n for n in ABLATIONS if n in refs or n in names]
    unknown = set(names) - set(ABLATIONS)
    if unknown:
        raise KeyError(f"unknown ablation(s): {sorted(unknown)}")
    return out


@torch.no_grad()
def _weight_summary(model, ds) -> dict | None:
    """Mean weight matrix on the training set, its largest entry per column and the
    column sums (which only move away from 1 for the unconstrained C_softplus)."""
    if not model.weighting.is_learned:
        return None
    model.eval()
    b = ds.train_idx
    keep = torch.ones(len(b), ds.n_nodes, dtype=torch.bool)
    _, f, _, _ = model(ds.X[b], ds.A[b], keep, ds.cnt[b])
    w = model.weights(f, ds.cnt[b]).mean(0)
    return {"w": w.tolist(),
            "col_max": w.max(0).values.tolist(),
            "col_sum": w.sum(0).tolist()}


def gradient_snapshot(model, arm, ds, cfg, seed: int = 0) -> dict:
    """Share of the encoder gradient each loss term supplies, on the trained model.

    One full training batch (with AUG copies), eval mode. Same proxy as diagnostics.py --
    norm_i / sum norm_j -- measured at convergence instead of averaged over training, so
    it costs one backward pass per term rather than a second training run.
    """
    model.eval()
    rng = random.Random(seed)
    batch = make_batch(ds, ds.train_idx, cfg, rng)
    terms, _ = compute_terms(model, arm, batch, cfg)
    sigma = cfg["contrastive_loss_scale"]
    sigma_to = cfg.get("to_scale") or sigma
    scaled = {"l_rcl": terms["l_rcl"], "l_fti": terms["l_fti"],
              "sigma*l_to": sigma_to * terms["l_to"], "sigma*l_cm": sigma * terms["l_cm"]}
    params = [p for p in model.encoders.parameters() if p.requires_grad]
    norms = {}
    for k, t in scaled.items():
        if not t.requires_grad:
            norms[k] = 0.0
            continue
        g = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
        norms[k] = float(torch.sqrt(sum((x ** 2).sum() for x in g if x is not None)))
    tot = sum(norms.values()) or 1.0
    return {k: round(100 * v / tot, 2) for k, v in norms.items()}


def run_one(name: str, seed: int, ds, base_cfg: dict) -> dict:
    """Train one ablation configuration with one seed; return everything worth keeping."""
    group, arm, overrides, desc = ABLATIONS[name]
    cfg = {**base_cfg, **overrides}
    res, model = train(arm, seed, ds, cfg)
    pc, _ = per_class(model, ds)
    res.update(name=name, group=group, description=desc,
               overrides={k: list(v) if isinstance(v, tuple) else v for k, v in overrides.items()},
               per_class=pc,
               grad_share=gradient_snapshot(model, arm, ds, cfg, seed),
               weights=_weight_summary(model, ds))
    res.pop("w_final", None)        # superseded by `weights`, measured on the train set
    return res
