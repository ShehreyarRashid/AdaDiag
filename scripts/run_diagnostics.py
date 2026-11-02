#!/usr/bin/env python3
"""Measure what share of the encoder gradient each loss term supplies.

Run this before concluding anything from a null result -- see README, "Why the result is a
null, and how we know".

    python scripts/run_diagnostics.py --dataset hotel
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
ROOT = Path(__file__).resolve().parents[1]
import torch                                                      # noqa: E402
from adadiag import config, data                          # noqa: E402
from adadiag.diagnostics import gradient_shares           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="gaia", choices=["gaia", "hotel", "sockshop"])
    ap.add_argument("--repo", default=str(ROOT), help="folder that contains data/ (default: this repository, which ships it; or a TVDiag checkout)")
    ap.add_argument("--arms", default="v0_simplex,v3_incident")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads; results/ used 2 (see results/ENVIRONMENT.md)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    cfg = config.get(a.dataset)
    ds = data.load(a.dataset, a.repo)
    out_path = Path(a.out or f"results/{a.dataset}_gradshare.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = []
    for arm in a.arms.split(","):
        r = gradient_shares(arm, ds, cfg, epochs=a.epochs)
        out.append(r)
        print(f"\n=== {a.dataset} / {arm} ===")
        print(f"  {'term':14s}{'grad norm':>12s}{'share':>9s}")
        for k in r["grad_norm"]:
            print(f"  {k:14s}{r['grad_norm'][k]:>12.4f}{r['share'][k]:>8.1f}%")
        print(f"  -> the task-oriented term supplies {r['share']['sigma*l_to']:.1f}% of the "
              f"encoder gradient; w only reallocates within that share.")
        json.dump(out, open(out_path, "w"), indent=1)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
