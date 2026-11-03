"""Methodology step 6 — THE CONTRIBUTION: the modality x task weighting module.

What TVDiag does (core/TVDiag.py, inside `if self.config.TO:`)::

    if 'metric' in modalities: l_to += supConLoss(fs['metric'], instance_labels)   # -> RCL
    if 'log'    in modalities: l_to += supConLoss(fs['log'],    type_labels)       # -> FTI
    if 'trace'  in modalities: l_to += supConLoss(fs['trace'],  instance_labels)   # -> RCL

Three terms. The modality -> task mapping is an ``if``/``else`` on the data type and the
losses are summed with equal weight -- the same rule for every system, failure class and
incident. (The authors' own ``metric -> FTI`` variant sits commented out directly above.)

AdaDiag replaces that with six terms, one per (modality, task), each scaled by an entry of a
3x2 matrix whose columns are simplex-constrained::

    l_to = sum_m sum_t  w[m, t] * L_supcon(f_m, y_t)

Why the simplex. An unconstrained positive weight is unidentifiable: the outer scale
(``contrastive_loss_scale`` and the Eq. 13 term) absorbs any uniform rescaling of ``w``, so
the optimiser can lower the loss by shrinking every weight instead of reallocating between
them. Constraining each task column to sum to one separates the two degrees of freedom --
sigma decides how loud the loss is, ``w`` decides who speaks.

Arms
----
``v0_raw``       the three unit-weight terms exactly as TVDiag ships them.
``v0_soft``      TVDiag's prior softened (README section 6.11): 0.9 where it has 1 and 0.1
                 where it has 0, unnormalised like v0_raw -- RCL = [.9, .9, .1],
                 FTI = [.1, .1, .9]. A fixed matrix, proposed for testing in review.
``v0_simplex``   the same prior written on the simplex: RCL = [.5, .5, 0], FTI = [0, 0, 1].
``v0_oracle``    a matrix read off the source-removal ablations (README section 6.12), given
                 in the config: one per dataset (`oracle_w`) or one per failure type
                 (`oracle_class_w`, applied by the incident's training label). An upper-bound
                 probe built from test-split evidence, not a method.
``v1_uniform``   every column 1/3.
``v2_global``    one learned 3x2 matrix, shared by all incidents.
``v3_incident``  full AdaDiag: w = softmax_col(MLP(e)), conditioned on the incident.
``v4_fusion``    exploratory: the same per-incident matrix applied to the FUSION path
                 instead of the contrastive loss (see README, "Why V4 exists").
``v5_align``     README section 6.13: one matrix set from the DIAGNOSIS losses. Each step,
                 w[m, t] is proportional to the positive part of a running mean of
                 cos(grad L_t, grad L_supcon(f_m, y_t)) over modality m's encoder, i.e. how
                 much a step on that term also lowers task t's diagnosis loss (to first
                 order). The training loop computes it (`train.align_weights`); the module
                 only holds it. No parameters, so `is_learned` is False.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MODALITIES = ("metric", "trace", "log")
TASKS = ("rcl", "fti")

ARMS = ("v0_raw", "v0_soft", "v0_simplex", "v0_oracle", "v1_uniform", "v2_global",
        "v3_incident", "v4_fusion", "v5_align")
LEARNED_ARMS = ("v2_global", "v3_incident", "v4_fusion")

#: TVDiag's hand-written mapping, as unit weights (what the code literally does).
PRIOR_RAW = [[1.0, 0.0],   # metric -> RCL
             [1.0, 0.0],   # trace  -> RCL
             [0.0, 1.0]]   # log    -> FTI
#: The prior softened: every modality contributes a little to every task (section 6.11).
PRIOR_SOFT = [[0.9, 0.1],   # metric
              [0.9, 0.1],   # trace
              [0.1, 0.9]]   # log
#: The same prior renormalised onto the simplex, for an equal-loss-mass comparison.
PRIOR_SIMPLEX = [[0.5, 0.0],
                 [0.5, 0.0],
                 [0.0, 1.0]]


class WeightingModule(nn.Module):
    """Produces w of shape (B, 3, 2) with each TASK COLUMN on the simplex.

    For the fixed arms this is a constant. For ``v2_global`` it is one learned matrix. For
    ``v3_incident`` and ``v4_fusion`` it is produced from the incident descriptor

        e = F^metric (+) F^trace (+) F^log (+) c     in R^(3*d_out + 3)

    where ``F^m`` are the graph-level embeddings the encoders already produce and ``c`` is
    the standardised log1p alert count per source. With d_out = 32 that is R^99, and the
    MLP has roughly 1e4 parameters. It is used during TRAINING ONLY: the contrastive terms
    are not on the forward path that produces predictions, so AdaDiag adds no inference cost.
    """

    def __init__(self, arm: str, fuse_dim: int, hidden: int = 64,
                 constraint: str = "simplex", active: tuple[bool, ...] | None = None,
                 fixed: list | None = None, fixed_trace: list | None = None):
        """
        Args:
            constraint: "simplex" (the method) or "softplus" -- ABLATION ONLY: unconstrained
                positive weights, initialised so each column sums to ~1 like the simplex.
                Tests the identifiability argument of README section 1.
            active: which modalities are present; ABLATION ONLY (modality removal). A removed
                modality gets weight exactly 0 in every arm.
            fixed_trace: README section 6.14, learned arms only. [t_rcl, t_fti]: trace's
                weight in each task column is held at t, never learned, and the module
                learns how to split the remaining 1 - t between metric and log.
        """
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        if constraint not in ("simplex", "softplus"):
            raise ValueError(f"unknown constraint {constraint!r}")
        self.arm, self.constraint = arm, constraint
        self.fixed = fixed            # v0_oracle's global matrix (3 x 2), if given
        self.register_buffer("fixed_trace", None, persistent=False)
        if fixed_trace is not None:
            if arm not in ("v2_global", "v3_incident"):
                raise ValueError("fixed_trace applies to the learned arms v2_global / v3_incident")
            if active is not None and not all(active):
                raise ValueError("fixed_trace is not combined with modality removal")
            t = torch.tensor([float(x) for x in fixed_trace])
            if t.shape != (len(TASKS),) or (t < 0).any() or (t > 1).any():
                raise ValueError(f"fixed_trace must be {len(TASKS)} values in [0, 1], got {fixed_trace}")
            self.register_buffer("fixed_trace", t, persistent=False)
        self.n_m, self.n_t = len(MODALITIES), len(TASKS)
        act = tuple(active) if active is not None else (True,) * self.n_m
        if len(act) != self.n_m or not any(act):
            raise ValueError(f"active must name at least one of {self.n_m} modalities, got {act}")
        self.register_buffer("active", torch.tensor(act, dtype=torch.bool), persistent=False)
        self.all_active = all(act)
        if arm == "v5_align":
            # the current matrix and the running mean of the alignments it is built from
            self.register_buffer("align_w", torch.full((self.n_m, self.n_t), 1.0 / self.n_m))
            self.register_buffer("align_ema", torch.zeros(self.n_m, self.n_t))
        if arm == "v2_global":
            self.logits = nn.Parameter(torch.zeros(self.n_m, self.n_t))
        elif arm in ("v3_incident", "v4_fusion"):
            self.net = nn.Sequential(
                nn.Linear(fuse_dim + len(MODALITIES), hidden), nn.ReLU(),
                nn.Linear(hidden, self.n_m * self.n_t),
            )

    # -- helpers for the ablation switches; no-ops at the defaults ----------------------
    def _mask_fixed(self, w: torch.Tensor) -> torch.Tensor:
        """Zero removed modalities in a fixed prior and renormalise each column over the
        remaining ones; a column left empty (its only source was removed) becomes uniform
        over what remains. v0_raw and v0_soft keep their raw weights rather than
        renormalising."""
        if self.all_active:
            return w
        a = self.active.to(w.dtype)[:, None]
        w = w * a
        empty = w.sum(0, keepdim=True) == 0
        raw = self.arm in ("v0_raw", "v0_soft")
        fill = a.expand_as(w) if raw else a.expand_as(w) / a.sum()
        w = torch.where(empty, fill, w)
        if not raw:
            w = w / w.sum(0, keepdim=True)
        return w

    def _with_fixed_trace(self, raw: torch.Tensor, dim: int) -> torch.Tensor:
        """Softmax over metric and log only, scaled into the 1 - t that trace leaves."""
        ti = MODALITIES.index("trace")
        shape = [1] * raw.dim(); shape[dim] = self.n_m
        is_trace = torch.zeros(self.n_m, dtype=torch.bool, device=raw.device)
        is_trace[ti] = True
        ml = raw.masked_fill(is_trace.view(shape), float("-inf")).softmax(dim)
        t = self.fixed_trace.to(raw.dtype)                   # (n_t,), the last axis
        return ml * (1.0 - t) + is_trace.view(shape).to(raw.dtype) * t

    def _normalise(self, raw: torch.Tensor, dim: int) -> torch.Tensor:
        """raw logits -> weights, along the modality axis `dim`."""
        if self.fixed_trace is not None:
            return self._with_fixed_trace(raw, dim)
        if self.constraint == "softplus":
            n_act = int(self.active.sum())
            w = nn.functional.softplus(raw) / (n_act * 0.6931471805599453)   # ~1/n at init
            if not self.all_active:
                shape = [1] * raw.dim(); shape[dim] = self.n_m
                w = w * self.active.view(shape).to(w.dtype)
            return w
        if not self.all_active:
            shape = [1] * raw.dim(); shape[dim] = self.n_m
            raw = raw.masked_fill(~self.active.view(shape), float("-inf"))
        return raw.softmax(dim)

    def forward(self, graph_emb: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """graph_emb: (B, 3*d_out) -- pass it DETACHED for the conditioned arms.

        Detaching matters: the module should choose an allocation given the incident, not
        reshape the encoders so that its own job becomes easier.
        """
        B, dev = graph_emb.shape[0], graph_emb.device
        if self.arm == "v0_raw":
            w = self._mask_fixed(torch.tensor(PRIOR_RAW, device=dev))
            return w.expand(B, self.n_m, self.n_t)
        if self.arm == "v0_soft":
            w = self._mask_fixed(torch.tensor(PRIOR_SOFT, device=dev))
            return w.expand(B, self.n_m, self.n_t)
        if self.arm == "v0_simplex":
            w = self._mask_fixed(torch.tensor(PRIOR_SIMPLEX, device=dev))
            return w.expand(B, self.n_m, self.n_t)
        if self.arm == "v0_oracle":
            base = self.fixed if self.fixed is not None else [[1 / self.n_m] * self.n_t] * self.n_m
            w = self._mask_fixed(torch.tensor(base, dtype=torch.float32, device=dev))
            return w.expand(B, self.n_m, self.n_t)
        if self.arm == "v5_align":
            return self._mask_fixed(self.align_w).expand(B, self.n_m, self.n_t)
        if self.arm == "v1_uniform":
            if self.all_active:
                return torch.full((B, self.n_m, self.n_t), 1.0 / self.n_m, device=dev)
            w = self._mask_fixed(torch.full((self.n_m, self.n_t), 1.0, device=dev))
            return w.expand(B, self.n_m, self.n_t)
        if self.arm == "v2_global":
            return self._normalise(self.logits, 0).expand(B, self.n_m, self.n_t)
        e = torch.cat([graph_emb, counts], dim=-1)
        raw = self.net(e).view(B, self.n_m, self.n_t)
        return self._normalise(raw, 1)                                # DOWN each column

    @property
    def is_learned(self) -> bool:
        return self.arm in LEARNED_ARMS

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
