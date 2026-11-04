"""Assembles steps 3, 4 and 6 into one network.

The forward pass is TVDiag's: three graph encoders -> max-pool readout -> concatenate ->
a node-level head (RCL) and a graph-level head (FTI). The only structural addition is the
`WeightingModule`, which does not touch the forward path for the published arms -- it acts
inside the training objective (see `train.py`). The exploratory `v4_fusion` arm is the one
exception and gates the fused representation; it is flagged wherever it appears.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import SAGEEncoder, max_pool
from .heads import Locator, TypeClassifier
from .weighting import MODALITIES, WeightingModule


class AdaDiag(nn.Module):
    def __init__(self, arm: str, n_types: int, cfg: dict):
        super().__init__()
        self.arm = arm
        d_in, d_h, d_o = cfg["alert_embedding_dim"], cfg["graph_hidden_dim"], cfg["graph_out"]
        self.encoders = nn.ModuleDict({
            m: SAGEEncoder([d_in] + [d_h] * (cfg["graph_layers"] - 1) + [d_o], cfg["feat_drop"],
                           aggregator=cfg.get("aggregator", "mean"),
                           alg1=bool(cfg.get("sage_alg1", False)))
            for m in MODALITIES
        })
        fuse = len(MODALITIES) * d_o
        self.locator = Locator(fuse, cfg["linear_hidden"])
        self.classifier = TypeClassifier(fuse, cfg["linear_hidden"], n_types)

        # Ablation switches (see ablation.py). Every default reproduces the method exactly.
        dropped = set(cfg.get("drop_modalities", ()) or ())
        unknown = dropped - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown modalities to drop: {sorted(unknown)}")
        self.active = tuple(m not in dropped for m in MODALITIES)
        self.detach = cfg.get("detach", True)
        # Fidelity switch (README section 6.11): TVDiag gives every node with no in-edge a
        # self-loop, in the raw graphs and again after AUG drops nodes.
        self.self_loops = bool(cfg.get("self_loops", False))
        # Paper switch (README section 6.19): "the edges in E are bidirectional" (TVDiag
        # paper, Sec. 4.2). The released graphs are used as given; GAIA's are half one-way.
        self.bidirectional = bool(cfg.get("bidirectional", False))
        self.descriptor = cfg.get("descriptor", "full")
        if self.descriptor not in ("full", "emb", "counts"):
            raise ValueError(f"unknown descriptor {self.descriptor!r}")

        # README section 6.15. Building a learned module draws from torch's random stream
        # (its MLP initialisation), which shifts every later draw -- the encoders' dropout
        # masks -- relative to an arm without one. `isolate_weighting_rng` rewinds the stream
        # afterwards, so every arm trains with the same dropout masks for the same seed. Off
        # by default: the committed results were produced without it.
        rng_state = torch.get_rng_state() if cfg.get("isolate_weighting_rng") else None
        self.weighting = WeightingModule(arm, fuse, constraint=cfg.get("w_constraint", "simplex"),
                                         active=self.active, fixed=cfg.get("oracle_w"),
                                         fixed_trace=cfg.get("fixed_trace"))
        if rng_state is not None:
            torch.set_rng_state(rng_state)
        self.d_out = d_o

    def forward(self, X, A, keep, counts):
        """Returns (per-modality graph embeddings, fused embedding, node logits, type logits).

        A removed modality (ablation) contributes an all-zero embedding: its encoder is not
        run, so it receives no gradient, and the heads see nothing from it.
        """
        if self.bidirectional:
            A = torch.maximum(A, A.transpose(1, 2))
        if self.self_loops:
            A = A + torch.diag_embed((A.sum(-1) == 0).to(A.dtype))
        fs, es = {}, {}
        for j, m in enumerate(MODALITIES):
            if self.active[j]:
                node_emb = self.encoders[m](X[:, j], A)
            else:
                node_emb = X.new_zeros(X.shape[0], X.shape[2], self.d_out)
            es[m] = node_emb
            fs[m] = max_pool(node_emb, keep)
        f = torch.cat([fs[m] for m in MODALITIES], dim=-1)          # (B, 3*d_out)
        e = torch.cat([es[m] for m in MODALITIES], dim=-1)          # (B, N, 3*d_out)

        f_fti = f
        if self.arm == "v4_fusion":
            # Exploratory arm: move the weights onto the path that carries most of the
            # gradient. Scaled by n_modalities so a uniform w is the identity.
            w = self.weighting(f.detach(), counts)
            rep = self.d_out
            e = e * (w[:, :, 0].repeat_interleave(rep, 1) * len(MODALITIES))[:, None, :]
            f_fti = f * (w[:, :, 1].repeat_interleave(rep, 1) * len(MODALITIES))

        return fs, f, self.locator(e), self.classifier(f_fti)

    def weights(self, f: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """The 3x2 weight matrix for this batch. Detached input for the conditioned arms.

        `descriptor` (ablation): "full" = embeddings + counts (the method); "emb" zeroes the
        counts; "counts" zeroes the embeddings. Zeroing keeps the MLP's input size fixed, so
        the parameter count is identical across the three.
        """
        conditioned = self.weighting.arm in ("v3_incident", "v4_fusion")
        cond = f.detach() if (conditioned and self.detach) else f
        if self.descriptor == "emb":
            counts = torch.zeros_like(counts)
        elif self.descriptor == "counts":
            cond = torch.zeros_like(cond)
        return self.weighting(cond, counts)
