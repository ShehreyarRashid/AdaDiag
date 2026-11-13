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
        # README section 6.21 (both off by default):
        #   rcl_context  "global": each node is also scored with the incident's fused graph
        #                feature; "neighbour": with its own features minus its neighbours'
        #                mean (a root differs from the services its fault spreads to);
        #                "both".
        #   fti_readout  "root_attn": FTI also reads the node embeddings pooled by the RCL
        #                softmax, i.e. the failure type is read off the likely culprit.
        self.rcl_context = cfg.get("rcl_context", "none") or "none"
        if self.rcl_context not in ("none", "global", "neighbour", "both"):
            raise ValueError(f"unknown rcl_context {self.rcl_context!r}")
        self.fti_readout = cfg.get("fti_readout", "max") or "max"
        if self.fti_readout not in ("max", "root_attn"):
            raise ValueError(f"unknown fti_readout {self.fti_readout!r}")
        # README section 6.24: with "root_attn", `root_detach` stops the FTI loss from reaching the
        # RCL scores through the readout's softmax, so FTI reads the culprit without reshaping
        # the ranking
        self.root_detach = bool(cfg.get("root_detach", False))
        k_rcl = 1 + (self.rcl_context in ("global", "both")) + (self.rcl_context in ("neighbour", "both"))
        k_fti = 2 if self.fti_readout == "root_attn" else 1
        # README section 6.25 (both off by default; both need alert_bag, section 6.24):
        #   bag_drop  dropout on the alert-type count columns of X during training
        #   bag_head  the heads also read the raw counts: the locator each instance's own
        #             counts, the classifier their max and mean over instances (a wide path
        #             beside the graph encoders)
        self.n_emb = int(cfg.get("embedding_cols", 128))
        self.bag_drop = float(cfg.get("bag_drop", 0.0) or 0.0)
        self.bag_head = bool(cfg.get("bag_head", False))
        n_bag = len(MODALITIES) * (d_in - self.n_emb)
        if (self.bag_drop or self.bag_head) and n_bag <= 0:
            raise ValueError("bag_drop / bag_head need alert_bag=True (section 6.24)")
        x_rcl = n_bag if self.bag_head else 0
        x_fti = 2 * n_bag if self.bag_head else 0
        self.locator = Locator(fuse * k_rcl + x_rcl, cfg["linear_hidden"])
        self.classifier = TypeClassifier(fuse * k_fti + x_fti, cfg["linear_hidden"], n_types)

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
        # Encoder analysis (README section 6.23): drop every edge, so each instance is encoded
        # from its own alerts only -- how much does the encoder use the service graph?
        self.no_edges = bool(cfg.get("no_edges", False))
        # README section 6.24: a linear path from each instance's own features straight to
        # its embedding (a skip around the graph layers), so a local fault is not averaged
        # away by its neighbours. Built after everything else, so nothing else's
        # initialisation moves.
        self.self_res = bool(cfg.get("self_res", False))
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
        if self.self_res:
            self.res = nn.ModuleDict({m: nn.Linear(d_in, d_o, bias=False) for m in MODALITIES})

    def forward(self, X, A, keep, counts):
        """Returns (per-modality graph embeddings, fused embedding, node logits, type logits).

        A removed modality (ablation) contributes an all-zero embedding: its encoder is not
        run, so it receives no gradient, and the heads see nothing from it.
        """
        if self.no_edges:
            A = torch.zeros_like(A)
        if self.bidirectional:
            A = torch.maximum(A, A.transpose(1, 2))
        if self.self_loops:
            A = A + torch.diag_embed((A.sum(-1) == 0).to(A.dtype))
        if self.bag_drop and self.training:
            X = torch.cat([X[..., :self.n_emb],
                           nn.functional.dropout(X[..., self.n_emb:], self.bag_drop)], dim=-1)
        fs, es = {}, {}
        for j, m in enumerate(MODALITIES):
            if self.active[j]:
                node_emb = self.encoders[m](X[:, j], A)
                if self.self_res:
                    node_emb = node_emb + self.res[m](X[:, j])
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

        e_in = e
        if self.rcl_context in ("global", "both"):
            e_in = torch.cat([e_in, f[:, None, :].expand(-1, e.shape[1], -1)], dim=-1)
        if self.rcl_context in ("neighbour", "both"):
            deg = A.sum(-1, keepdim=True).clamp(min=1.0)
            e_in = torch.cat([e_in, e - torch.bmm(A, e) / deg], dim=-1)
        if self.bag_head:
            raw = X[..., self.n_emb:].permute(0, 2, 1, 3).flatten(2)        # (B, N, 3V)
            e_in = torch.cat([e_in, raw], dim=-1)
        node_logits = self.locator(e_in)
        if self.fti_readout == "root_attn":
            p = torch.softmax(node_logits.masked_fill(~keep, float("-inf")), dim=1)
            if self.root_detach:
                p = p.detach()
            f_fti = torch.cat([f_fti, (p[..., None] * e).sum(1)], dim=-1)
        if self.bag_head:
            f_fti = torch.cat([f_fti, raw.max(1).values, raw.mean(1)], dim=-1)
        return fs, f, node_logits, self.classifier(f_fti)

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
