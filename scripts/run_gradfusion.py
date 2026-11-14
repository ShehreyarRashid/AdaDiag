#!/usr/bin/env python3
"""Train and evaluate GradFusion on one dataset.

GradFusion = three GradFusion+ models that differ only in their seed (s, s+100, s+200), with
their softmax outputs averaged. GradFusion+ = TVDiag + adaptive modality weights (GradFusion-DL)
+ alert-type counts + Eq. 13's learnable loss weights.

    python scripts/run_gradfusion.py --dataset hotel --seed 5          # 5-8 min on one core
    python scripts/run_gradfusion.py --dataset gaia --seed 5 --members 1   # one GradFusion+ model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch                                                         # noqa: E402
from gradfusion import config, data                                     # noqa: E402
from gradfusion.alerts import with_alert_bag                            # noqa: E402
from gradfusion.ensemble import predict                                 # noqa: E402
from gradfusion.train import train                                      # noqa: E402
from plus2 import MEMBERS, PLUS, metrics                             # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["gaia", "sockshop", "hotel"])
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--members", type=int, default=3, choices=[1, 2, 3])
    a = ap.parse_args()
    torch.set_num_threads(1)
    ds = with_alert_bag(data.load(a.dataset))
    cfg = config.get(a.dataset, isolate_weighting_rng=True, **PLUS)
    models = []
    for off in MEMBERS[:a.members]:
        res, model = train("v5_align", a.seed + off, ds, cfg)
        models.append(model)
        print(f"member seed {a.seed + off:3d}: HR@1 {res['HR@1']:.3f}  F1 {res['f1']:.3f}  "
              f"({res['epochs']} epochs, {res['train_s']:.0f}s)", flush=True)
    te = ds.test_idx
    node, typ = predict(models, ds, te)
    m = metrics(node.numpy(), typ.numpy(), ds.root[te].numpy(), ds.ytype[te].numpy())
    print(f"\nGradFusion ({a.members} member(s)) on {a.dataset}, seed {a.seed}:")
    print("  RCL  " + "  ".join(f"{k} {m[k]:.3f}" for k in ("HR@1", "HR@2", "HR@3", "HR@5", "MRR@3")))
    print(f"  FTI  precision {m['precision']:.3f}  recall {m['recall']:.3f}  F1 {m['f1']:.3f}")


if __name__ == "__main__":
    main()
