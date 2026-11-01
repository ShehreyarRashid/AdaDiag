#!/usr/bin/env python3
"""Per-failure-class breakdown and the learned weight matrix per class.

The premise is that the useful source varies BY FAILURE CLASS, so this is the table that
can actually test it -- the aggregate sweep cannot.

    python scripts/run_perclass.py --dataset hotel --repo /path/to/TVDiag
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adadiag import config, data                          # noqa: E402
from adadiag.evaluate import per_class                    # noqa: E402
from adadiag.train import train                           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="gaia", choices=["gaia", "hotel", "sockshop"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--arms", default="v0_simplex,v2_global,v3_incident")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = config.get(a.dataset, epochs=a.epochs)
    ds = data.load(a.dataset, a.repo)
    out_path = Path(a.out or f"results/{a.dataset}_perclass.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = {}
    for arm in a.arms.split(","):
        _, model = train(arm, a.seed, ds, cfg)
        metrics, weights = per_class(model, ds)
        out[arm] = {"per_class": metrics, "weights_per_class": weights}
        print(f"\n=== {arm} ===")
        print(f"  {'failure class':38s}{'n':>5s}{'HR@1':>8s}{'HR@3':>8s}{'FTI rec':>9s}")
        for c, v in metrics.items():
            print(f"  {c[:38]:38s}{v['n']:>5d}{v['hr1']:>8.3f}{v['hr3']:>8.3f}{v['fti_recall']:>9.3f}")
        for c, w in weights.items():
            flat = "  ".join(f"{w[i][j]:.2f}" for i in range(3) for j in range(2))
            print(f"    w[{c[:30]:30s}] {flat}")
        json.dump(out, open(out_path, "w"), indent=1)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
