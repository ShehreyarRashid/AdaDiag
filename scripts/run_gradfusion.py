#!/usr/bin/env python3
"""Train and evaluate GradFusion on one dataset.

GradFusion = three GradFusion+ models that differ only in their seed (s, s+100, s+200), with
their softmax outputs averaged. GradFusion+ = TVDiag + adaptive modality weights (GradFusion-DL)
+ alert-type counts + Eq. 13's learnable loss weights.

    python scripts/run_gradfusion.py --dataset hotel --seed 5              # 5-8 min on one core
    python scripts/run_gradfusion.py --dataset gaia --seed 5 --members 1   # one GradFusion+ model
    python scripts/run_gradfusion.py --dataset sockshop --fast             # faster, not bit-exact

--fast computes the contrastive similarity matrix with one matrix product (config option
`fast_supcon`). Same mathematics, several times faster, but the floating-point rounding
differs, so the numbers are close to, not identical with, the committed results.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:                                              # installed with `pip install -e .`
    import gradfusion                             # noqa: F401
except ImportError:                               # or run from a plain clone
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from gradfusion import config, data
from gradfusion.alerts import with_alert_bag
from gradfusion.ensemble import predict
from gradfusion.evaluate import metrics_from_scores
from gradfusion.train import train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["gaia", "sockshop", "hotel"])
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--members", type=int, default=len(config.MEMBER_SEED_OFFSETS),
                    choices=range(1, len(config.MEMBER_SEED_OFFSETS) + 1))
    ap.add_argument("--fast", action="store_true", help="matrix-product SupCon (not bit-exact)")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch CPU threads (the committed results used 1)")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    ds = with_alert_bag(data.load(a.dataset))
    cfg = config.get(a.dataset, **config.GRADFUSION, fast_supcon=a.fast or None)
    models = []
    for offset in config.MEMBER_SEED_OFFSETS[:a.members]:
        seed = a.seed + offset
        res, model = train(config.GRADFUSION_ARM, seed, ds, cfg)
        models.append(model)
        print(f"member seed {seed:3d}: HR@1 {res['HR@1']:.3f}  F1 {res['f1']:.3f}  "
              f"({res['epochs']} epochs, {res['train_s']:.0f}s)", flush=True)

    te = ds.test_idx
    node, typ = predict(models, ds, te)
    m = metrics_from_scores(node.numpy(), typ.numpy(), ds.root[te].numpy(), ds.ytype[te].numpy())
    print(f"\nGradFusion ({a.members} member(s)) on {a.dataset}, seed {a.seed}:")
    print("  RCL  " + "  ".join(f"{k} {m[k]:.3f}" for k in ("HR@1", "HR@2", "HR@3", "HR@5", "MRR@3")))
    print(f"  FTI  precision {m['precision']:.3f}  recall {m['recall']:.3f}  F1 {m['f1']:.3f}")


if __name__ == "__main__":
    main()
