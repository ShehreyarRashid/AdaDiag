#!/usr/bin/env python3
"""Re-run committed experiments and compare against results/*.json.

    python scripts/verify_results.py                       # quick: Hotel, every arm, seed 0 (~4 min)
    python scripts/verify_results.py --dataset gaia --arms v0_simplex v3_incident
    python scripts/verify_results.py --dataset hotel --seeds 0 1 2 3 4   # the full Hotel sweep

Each selected (arm, seed) is retrained from scratch with the shipped code and the bundled
data/, and its metrics are printed next to the committed ones.

What to expect. On the machine and software that produced results/ (2 CPU threads,
PyTorch 2.15, see results/ENVIRONMENT.md) every number matches EXACTLY. On other hardware,
PyTorch versions or thread counts, floating-point reductions happen in a different order,
training trajectories drift, and single-seed numbers can differ by a few points -- that is
why every claim in the README rests on five-seed means and paired tests, not on single
runs. Pass --threads 2 to remove the thread-count difference.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
ROOT = Path(__file__).resolve().parents[1]
import torch                                                      # noqa: E402
from adadiag import config, data                                  # noqa: E402
from adadiag.train import train                                   # noqa: E402

KEYS = ["HR@1", "HR@3", "MRR@3", "f1", "macro_f1", "epochs"]


def committed(dataset: str) -> dict:
    """{arm: {seed: record}} from results/<dataset>.json and results/<dataset>_v4_fusion.json."""
    out = {}
    for f in (ROOT / "results" / f"{dataset}.json", ROOT / "results" / f"{dataset}_v4_fusion.json"):
        if f.exists():
            for arm, runs in json.loads(f.read_text()).items():
                out.setdefault(arm, {}).update({r["seed"]: r for r in runs})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hotel", choices=["gaia", "hotel"])
    ap.add_argument("--arms", nargs="+", default=None, help="default: every committed arm")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--repo", default=str(ROOT), help="folder that contains data/")
    ap.add_argument("--threads", type=int, default=None, help="torch threads (results/ used 2)")
    ap.add_argument("--strict", action="store_true", help="exit 1 unless every number matches")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    ref = committed(a.dataset)
    arms = a.arms or list(ref)
    ds = data.load(a.dataset, a.repo)
    cfg = config.get(a.dataset)
    print(ds, f"| torch {torch.__version__}, {torch.get_num_threads()} threads\n")
    print(f"{'arm':12s} {'seed':>4s}  " + "  ".join(f"{k:>16s}" for k in KEYS))

    exact, worst = True, 0.0
    for arm in arms:
        for seed in a.seeds:
            if seed not in ref.get(arm, {}):
                print(f"{arm:12s} {seed:4d}  (no committed run)")
                continue
            got, _ = train(arm, seed, ds, cfg)
            want = ref[arm][seed]
            cells = []
            for k in KEYS:
                g, w = got[k], want[k]
                same = g == w
                exact &= same
                if k != "epochs":
                    worst = max(worst, abs(g - w))
                cells.append(f"{g:.4f}={w:.4f}" if same and k != "epochs" else
                             f"{g}={w}" if same else
                             f"{g:.4f}≠{w:.4f}" if k != "epochs" else f"{g}≠{w}")
            print(f"{arm:12s} {seed:4d}  " + "  ".join(f"{c:>16s}" for c in cells), flush=True)

    print("\nRESULT:", "EXACT MATCH — every number reproduced bit-for-bit" if exact else
          f"differs (largest metric gap {worst:.4f}); see the docstring for why this happens "
          f"across machines")
    if a.strict and not exact:
        sys.exit(1)


if __name__ == "__main__":
    main()
