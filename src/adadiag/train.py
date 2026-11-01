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
from .weighting import MODALITIES, TASKS


def task_oriented_loss(model, fs, labels, w, temperature) -> torch.Tensor:
    """Six weighted terms in place of TVDiag's three unit-weight ones.

    TVDiag::

        l_to  = SupCon(f_metric, instance_labels)     # metric -> RCL
              + SupCon(f_log,    type_labels)         # log    -> FTI
              + SupCon(f_trace,  instance_labels)     # trace  -> RCL

    AdaDiag::

        l_to  = sum_m sum_t  w[:, m, t] . SupCon_per_sample(f_m, y_t) / nz

    A zero weight short-circuits the term, so the fixed arms cost no more than TVDiag does.
    """
    total = torch.zeros((), dtype=torch.float32, device=w.device)
    for mi, m in enumerate(MODALITIES):
        for ti, t in enumerate(TASKS):
            if float(w[0, mi, ti]) == 0.0 and not model.weighting.is_learned:
                continue
            per_sample, nz = supcon_per_sample(fs[m], labels[t], temperature)
            total = total + (w[:, mi, ti] * per_sample).sum() / nz
    return total


def train(arm: str, seed: int, ds, cfg: dict, verbose: bool = False) -> dict:
    """Train one arm with one seed and return the test metrics."""
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
            X, A = ds.X[b], ds.A[b]
            root, yt, cnt = ds.root[b], ds.ytype[b], ds.cnt[b]
            keep = torch.ones(len(b), ds.n_nodes, dtype=torch.bool)

            if cfg["aug_times"] > 0:                       # TVDiag mixes AUG copies into the batch
                Xa, Aa, ka = drop_non_root(X, A, root, cfg["aug_percent"], rng)
                X, A = torch.cat([X, Xa]), torch.cat([A, Aa])
                keep = torch.cat([keep, ka])
                root, yt, cnt = torch.cat([root] * 2), torch.cat([yt] * 2), torch.cat([cnt] * 2)

            fs, f, node_logits, type_logits = model(X, A, keep, cnt)
            labels = {"rcl": root, "fti": yt}

            # --- the replaced block -------------------------------------------------
            w = model.weights(f, cnt)
            if arm == "v4_fusion":
                # v4 moves the weights to the fusion path, so the LOSS keeps TVDiag's prior
                from .weighting import PRIOR_SIMPLEX
                w_loss = torch.tensor(PRIOR_SIMPLEX, device=f.device).expand_as(w)
            else:
                w_loss = w
            l_to = task_oriented_loss(model, fs, labels, w_loss, cfg["temperature"])
            # ------------------------------------------------------------------------

            l_cm = sum(uspcon(fs["metric"], fs[m], cfg["temperature"]) for m in ("trace", "log"))
            l_rcl = F.cross_entropy(node_logits.masked_fill(~keep, float("-inf")), root)
            l_fti = F.cross_entropy(type_logits, yt)

            sigma = cfg["contrastive_loss_scale"]
            loss = awl(l_rcl, l_fti, sigma * l_to, sigma * l_cm)
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
               weighting_params=model.weighting.n_parameters())
    if w_trace:
        res["w_final"] = w_trace[-1]
    return res, model
