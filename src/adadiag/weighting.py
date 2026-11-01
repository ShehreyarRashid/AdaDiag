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
``v0_simplex``   the same prior written on the simplex: RCL = [.5, .5, 0], FTI = [0, 0, 1].
``v1_uniform``   every column 1/3.
``v2_global``    one learned 3x2 matrix, shared by all incidents.
``v3_incident``  full AdaDiag: w = softmax_col(MLP(e)), conditioned on the incident.
``v4_fusion``    exploratory: the same per-incident matrix applied to the FUSION path
                 instead of the contrastive loss (see README, "Why V4 exists").
"""
from __future__ import annotations

import torch
import torch.nn as nn

MODALITIES = ("metric", "trace", "log")
TASKS = ("rcl", "fti")

ARMS = ("v0_raw", "v0_simplex", "v1_uniform", "v2_global", "v3_incident", "v4_fusion")
LEARNED_ARMS = ("v2_global", "v3_incident", "v4_fusion")

#: TVDiag's hand-written mapping, as unit weights (what the code literally does).
PRIOR_RAW = [[1.0, 0.0],   # metric -> RCL
             [1.0, 0.0],   # trace  -> RCL
             [0.0, 1.0]]   # log    -> FTI
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

    def __init__(self, arm: str, fuse_dim: int, hidden: int = 64):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        self.n_m, self.n_t = len(MODALITIES), len(TASKS)
        if arm == "v2_global":
            self.logits = nn.Parameter(torch.zeros(self.n_m, self.n_t))
        elif arm in ("v3_incident", "v4_fusion"):
            self.net = nn.Sequential(
                nn.Linear(fuse_dim + len(MODALITIES), hidden), nn.ReLU(),
                nn.Linear(hidden, self.n_m * self.n_t),
            )

    def forward(self, graph_emb: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """graph_emb: (B, 3*d_out) -- pass it DETACHED for the conditioned arms.

        Detaching matters: the module should choose an allocation given the incident, not
        reshape the encoders so that its own job becomes easier.
        """
        B, dev = graph_emb.shape[0], graph_emb.device
        if self.arm == "v0_raw":
            return torch.tensor(PRIOR_RAW, device=dev).expand(B, self.n_m, self.n_t)
        if self.arm == "v0_simplex":
            return torch.tensor(PRIOR_SIMPLEX, device=dev).expand(B, self.n_m, self.n_t)
        if self.arm == "v1_uniform":
            return torch.full((B, self.n_m, self.n_t), 1.0 / self.n_m, device=dev)
        if self.arm == "v2_global":
            return self.logits.softmax(0).expand(B, self.n_m, self.n_t)
        e = torch.cat([graph_emb, counts], dim=-1)
        return self.net(e).view(B, self.n_m, self.n_t).softmax(1)   # softmax DOWN each column

    @property
    def is_learned(self) -> bool:
        return self.arm in LEARNED_ARMS

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
