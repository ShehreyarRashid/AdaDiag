#!/usr/bin/env python3
"""README section 6.13: how much does each weighting change the contrastive term's gradient?

Trains TVDiag default for 80 epochs (seed 0), then, on one training batch, compares the
encoder gradient of each of TVDiag's three task-oriented terms with and without each
pair- or anchor-level weighting (types 1, 2, 3, 5). A cosine near 1 means the weighting
barely changes what the term asks of the encoder. Also reports the spread (coefficient of
variation) of the pair weights at beta = 1. Writes results/cl_variants/gradient_change.json.

    python scripts/cl_gradient_change.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adadiag import config, data                                     # noqa: E402
from adadiag.losses import supcon_weighted                           # noqa: E402
from adadiag.train import make_batch, train                          # noqa: E402

TERMS = (("metric", "rcl"), ("trace", "rcl"), ("log", "fti"))         # TVDiag's three
SETTINGS = {"1 · pos, β=1": dict(pos_beta=1.0), "1 · pos, β=1/τ": dict(pos_beta=1 / 0.3),
            "2 · neg, β=1": dict(neg_beta=1.0), "2 · neg, β=1/τ": dict(neg_beta=1 / 0.3),
            "3 · importance": "importance", "5 · curriculum at 50 %": dict(keep_frac=0.5)}


def main() -> None:
    torch.set_num_threads(1)
    out = {}
    for name in ("gaia", "hotel"):
        ds = data.load(name)
        cfg = config.get(name, epochs=80)
        _, model = train("v0_raw", 0, ds, cfg)
        model.eval()
        b = ds.train_idx[:cfg["batch_size"]]
        X, A, keep, root, yt, cnt = make_batch(ds, b, cfg, random.Random(0))
        labels = {"rcl": root, "fti": yt}
        with torch.no_grad():
            _, _, nl, tl = model(X, A, keep, cnt)
            ce = {"rcl": F.cross_entropy(nl.masked_fill(~keep, float("-inf")), root, reduction="none"),
                  "fti": F.cross_entropy(tl, yt, reduction="none")}
        res = {}
        for m, t in TERMS:
            ps = list(model.encoders[m].parameters())

            def grad(kw):
                fsm = model(X, A, keep, cnt)[0][m]
                if kw == "importance":
                    per, nz = supcon_weighted(fsm, labels[t], cfg["temperature"])
                    per = per * (ce[t] / ce[t].mean())
                else:
                    per, nz = supcon_weighted(fsm, labels[t], cfg["temperature"], **kw)
                g = torch.autograd.grad(per.sum() / nz, ps, allow_unused=True)
                return torch.cat([x.reshape(-1) for x in g if x is not None])

            g0 = grad({})
            row = {k: float(F.cosine_similarity(g0, grad(kw), dim=0)) for k, kw in SETTINGS.items()}
            with torch.no_grad():
                z = F.normalize(model(X, A, keep, cnt)[0][m], dim=1)
                c = z @ z.T
                lab = labels[t]
                same = lab[:, None] == lab[None, :]
                eye = torch.eye(len(lab), dtype=torch.bool)

                def cv(mask, sign):
                    v = [torch.exp(sign * c[i][mask[i]]) for i in range(len(lab)) if mask[i].sum() > 1]
                    return float(np.mean([(x.std() / x.mean()).item() for x in v]))
                row["weight_cv_pos_beta1"] = cv(same & ~eye, -1.0)
                row["weight_cv_neg_beta1"] = cv(~same, 1.0)
            res[f"{m}->{t}"] = row
            print(name, m, t, {k: round(v, 3) for k, v in row.items()}, flush=True)
        out[name] = res
    path = ROOT / "results" / "cl_variants" / "gradient_change.json"
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
