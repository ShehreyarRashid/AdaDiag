#!/usr/bin/env python3
"""Check why the learned arms cannot work as specified -- see README section 6.6.

Two measurements:
  1. the gradient reaching the weighting module from each loss term (expect 0.0 from both
     diagnosis losses), and
  2. how far each learned task column collapses onto a single modality.

    python scripts/run_identifiability.py --dataset hotel --repo /path/to/TVDiag
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adadiag import config, data, identifiability                 # noqa: E402

MODALITIES = ("metric", "trace", "log")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["gaia", "hotel", "sockshop"])
    ap.add_argument("--repo", required=True, help="path to a TVDiag checkout")
    ap.add_argument("--arm", default="v3_incident", choices=["v2_global", "v3_incident"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ds = data.load(a.dataset, a.repo)
    cfg = config.get(a.dataset, epochs=a.epochs)
    print(ds, "\n")

    print("(1) gradient reaching the weighting module, by loss term")
    paths = identifiability.measure_gradient_paths(ds, cfg, a.arm)
    for term, g in paths["grad_to_weighting"].items():
        verdict = "  <-- no learning signal" if g == 0.0 else ""
        print(f"    {term:12s} total |grad| = {g:.6e}{verdict}")
    print("\n    l_to is linear in w and each column is on the simplex, so its minimiser")
    print("    is a VERTEX: one modality per task, chosen without any task gradient.\n")

    print(f"(2) collapse of the learned columns ({a.seeds} seeds)")
    runs = []
    for seed in range(a.seeds):
        r = identifiability.measure_collapse(ds, cfg, a.arm, seed)
        runs.append(r)
        print(f"    seed {seed}  HR@1={r['HR@1']:.3f}  f1={r['f1']:.3f}")
        for task, c in r["columns"].items():
            w = "  ".join(f"{m}={c['weights'][m]:.3f}" for m in MODALITIES)
            print(f"      {task.upper():4s} w: {w}   max={c['max_weight']:.3f}"
                  f"   argmin SupCon={c['argmin_supcon']}"
                  f"{'  (= argmax w)' if c['vertex_is_argmin'] else ''}")

    mx = [c["max_weight"] for r in runs for c in r["columns"].values()]
    print(f"\n    max column weight across all seeds/tasks: "
          f"min={min(mx):.3f} mean={sum(mx)/len(mx):.3f} max={max(mx):.3f}")
    print("    (1.0 = fully collapsed onto one modality; 0.333 = the uniform allocation)")

    payload = {"gradient_paths": paths, "collapse": runs}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
