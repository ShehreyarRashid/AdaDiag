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
from .losses import AutomaticWeightedLoss, supcon_per_sample, supcon_weighted, uspcon
from .model import AdaDiag
from .evaluate import evaluate
from .weighting import MODALITIES, PRIOR_SIMPLEX, TASKS


def task_oriented_loss(model, fs, labels, w, temperature, active=None,
                       pos_beta: float = 0.0, neg_beta: float = 0.0,
                       keep_frac: float = 1.0) -> torch.Tensor:
    """Six weighted terms in place of TVDiag's three unit-weight ones.

    TVDiag::

        l_to  = SupCon(f_metric, instance_labels)     # metric -> RCL
              + SupCon(f_log,    type_labels)         # log    -> FTI
              + SupCon(f_trace,  instance_labels)     # trace  -> RCL

    AdaDiag::

        l_to  = sum_m sum_t  w[:, m, t] . SupCon_per_sample(f_m, y_t) / nz

    A zero weight short-circuits the term, so the fixed arms cost no more than TVDiag does.
    `active` (ablation) skips removed modalities outright. `pos_beta`, `neg_beta` and
    `keep_frac` are the pair- and anchor-level weightings of README section 6.13
    (`losses.supcon_weighted`); at their defaults the term is TVDiag's, bit for bit.
    """
    total = torch.zeros((), dtype=torch.float32, device=w.device)
    for mi, m in enumerate(MODALITIES):
        if active is not None and not active[mi]:
            continue
        for ti, t in enumerate(TASKS):
            # Fixed arms only: their w is constant across the batch, so row 0 decides for all.
            # .detach() because w carries grad for the learned arms (harmless, but float() on a
            # grad-tracking tensor warns).
            # (all rows, not row 0: v0_oracle's per-class matrix varies across the batch)
            if not model.weighting.is_learned and float(w[:, mi, ti].detach().abs().sum()) == 0.0:
                continue
            per_sample, nz = supcon_weighted(fs[m], labels[t], temperature,
                                             pos_beta, neg_beta, keep_frac)
            total = total + (w[:, mi, ti] * per_sample).sum() / nz
    return total


def _flat(grads, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(grads, params)])


@torch.no_grad()
def _set_align(model, cos, cfg):
    wm = model.weighting
    rho = float(cfg.get("align_ema", 0.9))
    wm.align_ema.mul_(rho).add_((1.0 - rho) * cos)
    a = wm.active.to(cos.dtype)[:, None]
    w = wm.align_ema.clamp_min(0.0) * a
    col = w.sum(0, keepdim=True)
    w = torch.where(col > 0, w / col.clamp_min(1e-12), (a / a.sum()).expand_as(w))
    wm.align_w.copy_(w)


def align_weights(model, fs, labels, diag, cfg) -> torch.Tensor:
    """v5_align (README section 6.13): set the 3x2 matrix from the DIAGNOSIS losses.

    A step of size eta on term (m, t) changes task t's diagnosis loss, to first order, by

        -eta * < grad_theta_m L_t , grad_theta_m L_supcon(f_m, y_t) >

    where theta_m is modality m's encoder (the only parameters the term touches). So the
    alignment of the two gradients says whether the term helps the task it is assigned to.
    The cosine is used (scale-free across terms), smoothed by a running mean
    (`align_ema`, default 0.9), clipped at zero and normalised down each task column:
    terms that hurt their task get weight 0; a column with no helpful term is uniform.
    Returns the (3, 2) cosines of this batch; the module's matrix is updated in place.
    """
    cos = torch.zeros(len(MODALITIES), len(TASKS))
    groups = {m: [p for p in model.encoders[m].parameters() if p.requires_grad]
              for j, m in enumerate(MODALITIES) if model.active[j]}
    allp = [p for ps in groups.values() for p in ps]
    for ti, t in enumerate(TASKS):
        gd = torch.autograd.grad(diag[t], allp, retain_graph=True, allow_unused=True)
        k = 0
        for mi, m in enumerate(MODALITIES):
            if m not in groups:
                continue
            ps = groups[m]
            g_task = _flat(gd[k:k + len(ps)], ps)
            k += len(ps)
            per, nz = supcon_per_sample(fs[m], labels[t], cfg["temperature"])
            g_term = _flat(torch.autograd.grad(per.sum() / nz, ps, retain_graph=True,
                                               allow_unused=True), ps)
            if g_task.norm() > 0 and g_term.norm() > 0:
                cos[mi, ti] = F.cosine_similarity(g_task, g_term, dim=0).detach()
    _set_align(model, cos, cfg)
    return cos


def build_aug_pool(ds, cfg, rng) -> list:
    """TVDiag's AUG pool: `aug_times` node-dropped copies of every training incident, made
    once before training (EventProcess.build_dataset). Stored as (incident, keep mask)."""
    pool = []
    for _ in range(cfg["aug_times"]):
        for i in ds.train_idx:
            i = int(i)
            _, _, keep = drop_non_root(ds.X[i:i + 1], ds.A[i:i + 1], ds.root[i:i + 1],
                                       cfg["aug_percent"], rng)
            pool.append((i, keep[0]))
    return pool


def make_batch(ds, b, cfg, rng, pool=None):
    """One training batch, with TVDiag's AUG copies mixed in when aug_times > 0.

    aug_mode "paired" (default, the committed results): the augmented half is a fresh
    node-drop of the SAME incidents, so every incident's own copy is in the batch.
    aug_mode "pool" (TVDiag's code, README section 6.11): the augmented half is
    len(b) copies drawn at random from the pre-built pool, i.e. mostly OTHER incidents.
    """
    X, A = ds.X[b], ds.A[b]
    root, yt, cnt = ds.root[b], ds.ytype[b], ds.cnt[b]
    keep = torch.ones(len(b), ds.n_nodes, dtype=torch.bool)
    if cfg["aug_times"] > 0 and cfg.get("aug_mode", "paired") == "pool":
        picks = rng.sample(pool, len(b))
        idx = torch.tensor([i for i, _ in picks])
        ka = torch.stack([k for _, k in picks])
        m = ka.float()
        Xa = ds.X[idx] * m[:, None, :, None]
        Aa = ds.A[idx] * m[:, None, :] * m[:, :, None]
        X, A = torch.cat([X, Xa]), torch.cat([A, Aa])
        keep = torch.cat([keep, ka])
        root = torch.cat([root, ds.root[idx]])
        yt = torch.cat([yt, ds.ytype[idx]])
        cnt = torch.cat([cnt, ds.cnt[idx]])
    elif cfg["aug_times"] > 0:                     # TVDiag mixes AUG copies into the batch
        Xa, Aa, ka = drop_non_root(X, A, root, cfg["aug_percent"], rng)
        X, A = torch.cat([X, Xa]), torch.cat([A, Aa])
        keep = torch.cat([keep, ka])
        root, yt, cnt = torch.cat([root] * 2), torch.cat([yt] * 2), torch.cat([cnt] * 2)
    return X, A, keep, root, yt, cnt


def compute_terms(model, arm, batch, cfg, capture: dict | None = None,
                  warmup: bool = False, epoch: int | None = None) -> tuple[dict, torch.Tensor]:
    """Every loss term for one batch, plus the weight matrix.

    Shared by the training loop and the ablation gradient snapshot so the two can never
    disagree about what the objective is. Ablation switches (all default to the method):

        use_to=False    drop the task-oriented contrastive term (TVDiag "w/o TO")
        use_cm=False    drop the cross-modal term (TVDiag "w/o CM")
        drop_modalities handled in the model; the CM anchor moves to the first remaining one
        entropy_reg     reported as `negent`; the caller adds lambda * negent

    `capture` (diagnostics only): if a dict is passed, the detached per-modality graph
    embeddings, node embeddings and labels are stored in it. Nothing else changes.

    `warmup` (the `weight_warmup` option, README section 6.10): while True, a learned arm's
    l_to uses uniform weights over the present modalities instead of its own, so every
    encoder is trained on both labels before the module starts choosing between them, and
    the module itself receives no gradient. Fixed arms ignore it.

    README section 6.13, the five weightings of the contrastive term (all off by default):
    `cl_pos_beta` (1, positive pairs), `cl_neg_beta` (2, negative samples), `cl_importance`
    (3, anchors weighted by their detached diagnosis loss), arm `v5_align` (4, the matrix
    set from the diagnosis losses), `cl_curriculum` (5, easiest anchors first; needs
    `epoch`).
    """
    X, A, keep, root, yt, cnt = batch
    fs, f, node_logits, type_logits = model(X, A, keep, cnt)
    labels = {"rcl": root, "fti": yt}
    if capture is not None:
        capture.update(fs={m: v.detach() for m, v in fs.items()}, labels=labels, keep=keep)
    zero = torch.zeros((), dtype=torch.float32)

    # --- the replaced block -------------------------------------------------
    l_rcl = l_fti = None
    if arm == "v5_align" and model.training and torch.is_grad_enabled():
        l_rcl = F.cross_entropy(node_logits.masked_fill(~keep, float("-inf")), root)
        l_fti = F.cross_entropy(type_logits, yt)
        align_weights(model, fs, labels, {"rcl": l_rcl, "fti": l_fti}, cfg)
    w = model.weights(f, cnt)
    if arm == "v4_fusion":
        # v4 moves the weights to the fusion path, so the LOSS keeps TVDiag's prior
        w_loss = torch.tensor(PRIOR_SIMPLEX, device=f.device).expand_as(w)
    elif arm == "v0_oracle" and cfg.get("oracle_class_w") is not None:
        # README section 6.12: one matrix per failure type, applied by the training label
        W = torch.tensor(cfg["oracle_class_w"], dtype=w.dtype, device=w.device)   # (T, 3, 2)
        w_loss = W[yt]
    elif warmup and model.weighting.is_learned:
        a = model.weighting.active.to(w.dtype)
        w_loss = (a / a.sum())[None, :, None].expand_as(w)
    else:
        w_loss = w
    gamma = float(cfg.get("cl_importance", 0.0) or 0.0)
    if gamma:
        # type 3: each anchor weighted by how badly the heads currently diagnose it
        with torch.no_grad():
            nl = node_logits.detach().masked_fill(~keep, float("-inf"))
            ce = torch.stack([F.cross_entropy(nl, root, reduction="none"),
                              F.cross_entropy(type_logits.detach(), yt, reduction="none")], -1)
        a = ce.clamp_min(1e-12) ** gamma
        w_loss = w_loss * (a / a.mean(0, keepdim=True))[:, None, :]
    K = int(cfg.get("cl_curriculum", 0) or 0)
    keep_frac = 1.0
    if K and epoch is not None:
        q0 = float(cfg.get("cl_curriculum_start", 0.5))
        keep_frac = q0 + (1.0 - q0) * min(1.0, epoch / K)
    active = None if all(model.active) else model.active
    l_to = (task_oriented_loss(model, fs, labels, w_loss, cfg["temperature"], active,
                               float(cfg.get("cl_pos_beta", 0.0) or 0.0),
                               float(cfg.get("cl_neg_beta", 0.0) or 0.0), keep_frac)
            if cfg.get("use_to", True) else zero)
    # ------------------------------------------------------------------------

    present = [m for m, a in zip(MODALITIES, model.active) if a]
    if cfg.get("use_cm", True) and len(present) > 1:
        anchor = present[0]                         # "metric" unless metric was removed
        l_cm = sum(uspcon(fs[anchor], fs[m], cfg["temperature"]) for m in present[1:])
    else:
        l_cm = zero
    if l_rcl is None:          # created here, after l_to and l_cm, exactly as before
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
          verbose: bool = False, monitor=None) -> tuple[dict, AdaDiag]:
    """Train one arm with one seed.

    `monitor` (diagnostics only): called after every optimiser step as
    monitor(epoch, capture, w) with the batch's detached embeddings and weights. It reads,
    never writes, so a monitored run is bit-identical to an unmonitored one.

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
    warm_n = int(cfg.get("weight_warmup", 0) or 0)     # 0 = the method as published here
    # early stopping is paused while the objective is still changing (warm-up, curriculum)
    pause_n = max(warm_n, int(cfg.get("cl_curriculum", 0) or 0))
    pool = (build_aug_pool(ds, cfg, rng)
            if cfg["aug_times"] > 0 and cfg.get("aug_mode", "paired") == "pool" else None)

    for epoch in range(cfg["epochs"]):
        model.train()
        in_warmup = epoch < warm_n
        in_pause = epoch < pause_n
        perm = np.random.permutation(tr)
        running, nb = 0.0, 0
        for s in range(0, len(perm), cfg["batch_size"]):
            b = perm[s:s + cfg["batch_size"]]
            batch = make_batch(ds, b, cfg, rng, pool)
            capture = {} if monitor is not None else None
            terms, w = compute_terms(model, arm, batch, cfg, capture, warmup=in_warmup,
                                     epoch=epoch)
            loss = total_loss(awl, terms, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if monitor is not None:
                monitor(epoch, capture, w.detach())

            running += float(loss.detach())
            nb += 1
            if model.weighting.is_learned or arm == "v5_align":
                w_trace.append(w.mean(0).detach().tolist())

        mean_loss = running / nb
        if pause_n and epoch == pause_n:
            best, bad = None, 0          # the objective changes here: restart the clock
        if best is None or mean_loss < best - 1e-4:
            best, bad, state = mean_loss, 0, copy.deepcopy(model.state_dict())
        elif not in_pause:               # no early stop while the objective is changing
            bad += 1
            if bad >= cfg["patience"]:
                if verbose:
                    print(f"    early stop @ epoch {epoch}")
                break

    if cfg.get("restore_best", True):   # TVDiag itself evaluates the LAST epoch's model
        model.load_state_dict(state)
    res = evaluate(model, ds, ds.test_idx)
    res.update(arm=arm, seed=seed, dataset=ds.name,
               epochs=epoch + 1, train_s=round(time.time() - t0, 1),
               weighting_params=model.weighting.n_parameters(),
               env={"torch": torch.__version__, "threads": torch.get_num_threads()})
    if w_trace:
        res["w_final"] = w_trace[-1]
    return res, model
