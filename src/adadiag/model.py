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
            m: SAGEEncoder([d_in] + [d_h] * (cfg["graph_layers"] - 1) + [d_o], cfg["feat_drop"])
            for m in MODALITIES
        })
        fuse = len(MODALITIES) * d_o
        self.locator = Locator(fuse, cfg["linear_hidden"])
        self.classifier = TypeClassifier(fuse, cfg["linear_hidden"], n_types)
        self.weighting = WeightingModule(arm, fuse)
        self.d_out = d_o

    def forward(self, X, A, keep, counts):
        """Returns (per-modality graph embeddings, fused embedding, node logits, type logits)."""
        fs, es = {}, {}
        for j, m in enumerate(MODALITIES):
            node_emb = self.encoders[m](X[:, j], A)
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
        """The 3x2 weight matrix for this batch. Detached input for the conditioned arms."""
        cond = f.detach() if self.weighting.arm in ("v3_incident", "v4_fusion") else f
        return self.weighting(cond, counts)
