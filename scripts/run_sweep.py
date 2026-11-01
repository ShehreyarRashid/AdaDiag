#!/usr/bin/env python3
"""Train every arm across several seeds on one dataset and write results to JSON.

    python scripts/run_sweep.py --dataset hotel --seeds 5 --repo /path/to/TVDiag
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adadiag import config, data                          # noqa: E402
from adadiag.train import train                           # noqa: E402
from adadiag.weighting import ARMS                        # noqa: E402

DEFAULT_ARMS = ["v0_raw", "v0_simplex", "v1_uniform", "v2_global", "v3_incident"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="gaia", choices=["gaia", "hotel", "sockshop"])
    ap.add_argument("--repo", required=True, help="path to a WHU-AISE/TVDiag checkout")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help=f"comma-separated; available: {','.join(ARMS)}")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = config.get(a.dataset, epochs=a.epochs)
    ds = data.load(a.dataset, a.repo)
    print(ds)
    out_path = Path(a.out or f"results/{a.dataset}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, list] = {}
    for arm in a.arms.split(","):
        results[arm] = []
        for seed in range(a.seeds):
            res, _ = train(arm, seed, ds, cfg, verbose=True)
            results[arm].append(res)
            print(f"  {arm:12s} seed {seed}  HR@1 {res['HR@1']:.3f}  MRR@3 {res['MRR@3']:.3f}  "
                  f"F1 {res['f1']:.3f}  macroF1 {res['macro_f1']:.3f}  "
                  f"({res['epochs']} ep, {res['train_s']}s)")
            json.dump(results, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
