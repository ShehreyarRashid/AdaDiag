"""Methodology step 7 — the training loop, and the block AdaDiag replaces.

Everything here is TVDiag's except `task_oriented_loss`, which is the one function that
differs between arms. Augmentation, the cross-modal term, the two diagnosis losses, the
optimiser and the early-stopping rule are all held fixed so that any difference between
arms is attributable to the weighting and to nothing else.
"""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .augment import drop_non_root
from .losses import AutomaticWeightedLoss, supcon_per_sample, uspcon
from .model import AdaDiag
from .evaluate import evaluate
from .weighting import MODALITIES, PRIOR_SIMPLEX, TASKS


def task_oriented_loss(model, fs, labels, w, temperature, active=None) -> torch.Tensor:
    """Six weighted terms in place of TVDiag's three unit-weight ones.

    TVDiag::

        l_to  = SupCon(f_metric, instance_labels)     # metric -> RCL
              + SupCon(f_log,    type_labels)         # log    -> FTI
              + SupCon(f_trace,  instance_labels)     # trace  -> RCL

    AdaDiag::

        l_to  = sum_m sum_t  w[:, m, t] . SupCon_per_sample(f_m, y_t) / nz

    A zero weight short-circuits the term, so the fixed arms cost no more than TVDiag does.
    `active` (ablation) skips removed modalities outright.
    """
    total = torch.zeros((), dtype=torch.float32, device=w.device)
    for mi, m in enumerate(MODALITIES):
        if active is not None and not active[mi]:
            continue
        for ti, t in enumerate(TASKS):
            # Fixed arms only: their w is constant across the batch, so row 0 decides for all.
            # .detach() because w carries grad for the learned arms (harmless, but float() on a
            # grad-tracking tensor warns).
            if not model.weighting.is_learned and float(w[0, mi, ti].detach()) == 0.0:
                continue
            per_sample, nz = supcon_per_sample(fs[m], labels[t], temperature)
            total = total + (w[:, mi, ti] * per_sample).sum() / nz
    return total


def make_batch(ds, b, cfg, rng):
    """One training batch, with TVDiag's AUG copies mixed in when aug_times > 0."""
    X, A = ds.X[b], ds.A[b]
    root, yt, cnt = ds.root[b], ds.ytype[b], ds.cnt[b]
    keep = torch.ones(len(b), ds.n_nodes, dtype=torch.bool)
    if cfg["aug_times"] > 0:                       # TVDiag mixes AUG copies into the batch
        Xa, Aa, ka = drop_non_root(X, A, root, cfg["aug_percent"], rng)
        X, A = torch.cat([X, Xa]), torch.cat([A, Aa])
        keep = torch.cat([keep, ka])
        root, yt, cnt = torch.cat([root] * 2), torch.cat([yt] * 2), torch.cat([cnt] * 2)
    return X, A, keep, root, yt, cnt


def compute_terms(model, arm, batch, cfg) -> tuple[dict, torch.Tensor]:
    """Every loss term for one batch, plus the weight matrix.

    Shared by the training loop and the ablation gradient snapshot so the two can never
    disagree about what the objective is. Ablation switches (all default to the method):

        use_to=False    drop the task-oriented contrastive term (TVDiag "w/o TO")
        use_cm=False    drop the cross-modal term (TVDiag "w/o CM")
        drop_modalities handled in the model; the CM anchor moves to the first remaining one
        entropy_reg     reported as `negent`; the caller adds lambda * negent
    """
    X, A, keep, root, yt, cnt = batch
    fs, f, node_logits, type_logits = model(X, A, keep, cnt)
    labels = {"rcl": root, "fti": yt}
    zero = torch.zeros((), dtype=torch.float32)

    # --- the replaced block -------------------------------------------------
    w = model.weights(f, cnt)
    if arm == "v4_fusion":
        # v4 moves the weights to the fusion path, so the LOSS keeps TVDiag's prior
        w_loss = torch.tensor(PRIOR_SIMPLEX, device=f.device).expand_as(w)
    else:
        w_loss = w
    active = None if all(model.active) else model.active
    l_to = (task_oriented_loss(model, fs, labels, w_loss, cfg["temperature"], active)
            if cfg.get("use_to", True) else zero)
    # ------------------------------------------------------------------------

    present = [m for m, a in zip(MODALITIES, model.active) if a]
    if cfg.get("use_cm", True) and len(present) > 1:
        anchor = present[0]                         # "metric" unless metric was removed
        l_cm = sum(uspcon(fs[anchor], fs[m], cfg["temperature"]) for m in present[1:])
    else:
        l_cm = zero
    l_rcl = F.cross_entropy(node_logits.masked_fill(~keep, float("-inf")), root)
    l_fti = F.cross_entropy(type_logits, yt)

    terms = {"l_rcl": l_rcl, "l_fti": l_fti, "l_to": l_to, "l_cm": l_cm}
    if cfg.get("entropy_reg", 0.0) and model.weighting.is_learned:
        # negative entropy of each task column, mean over the batch, summed over tasks
        terms["negent"] = (w * torch.log(w.clamp_min(1e-12))).sum(1).mean(0).sum()
    return terms, w


def total_loss(awl, terms, cfg) -> torch.Tensor:
    sigma = cfg["contrastive_loss_scale"]
    sigma_to = cfg.get("to_scale") or sigma          # ablation: scale l_to on its own
    parts = (terms["l_rcl"], terms["l_fti"], sigma_to * terms["l_to"], sigma * terms["l_cm"])
    if cfg.get("dynamic_weights", True):
        loss = awl(*parts)
    else:
        # TVDiag's "w/o DW" (Table 6): Eq. 12 with static unit weights instead of Eq. 13.
        loss = parts[0] + parts[1] + parts[2] + parts[3]
    if "negent" in terms:
        loss = loss + cfg["entropy_reg"] * terms["negent"]
    return loss


def train(arm: str, seed: int, ds, cfg: dict,
          verbose: bool = False) -> tuple[dict, AdaDiag]:
    """Train one arm with one seed.

    Returns:
        (metrics, model) -- the test metrics dict, and the trained model so callers can
        inspect the learned weights (see `evaluate.per_class`, `identifiability`).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = random.Random(seed)

    model = AdaDiag(arm, len(ds.types), cfg)
    # Eq. 13's outer weighting. Reproduced INERT, exactly as TVDiag ships it -- see
    # losses.AutomaticWeightedLoss. Its parameters are deliberately not optimised.
    awl = AutomaticWeightedLoss(4, learnable=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    tr = ds.train_idx
    best, bad, state = None, 0, copy.deepcopy(model.state_dict())
    w_trace, t0, epoch = [], time.time(), 0

    for epoch in range(cfg["epochs"]):
        model.train()
        perm = np.random.permutation(tr)
        running, nb = 0.0, 0
        for s in range(0, len(perm), cfg["batch_size"]):
            b = perm[s:s + cfg["batch_size"]]
            batch = make_batch(ds, b, cfg, rng)
            terms, w = compute_terms(model, arm, batch, cfg)
            loss = total_loss(awl, terms, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.detach())
            nb += 1
            if model.weighting.is_learned:
                w_trace.append(w.mean(0).detach().tolist())

        mean_loss = running / nb
        if best is None or mean_loss < best - 1e-4:
            best, bad, state = mean_loss, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg["patience"]:
                if verbose:
                    print(f"    early stop @ epoch {epoch}")
                break

    model.load_state_dict(state)
    res = evaluate(model, ds, ds.test_idx)
    res.update(arm=arm, seed=seed, dataset=ds.name,
               epochs=epoch + 1, train_s=round(time.time() - t0, 1),
               weighting_params=model.weighting.n_parameters(),
               env={"torch": torch.__version__, "threads": torch.get_num_threads()})
    if w_trace:
        res["w_final"] = w_trace[-1]
    return res, model
