#!/usr/bin/env python3
"""Runs behind the comparison with the TVDiag paper (results/paper_comparison/).

Each job trains one (variant, arm, seed) and stores the test metrics plus the per-figure
measurements in `adadiag.paper_compare`. Results are appended after every job, so an
interrupted run resumes where it stopped.

    # Tables 4-5, Figs 11/13/14/18: the three datasets, default settings, 5 seeds
    python scripts/run_paper_comparison.py --dataset sockshop --threads 2

    # Fig. 16: hyper-parameter sweeps on GAIA, 3 seeds, in two halves
    python scripts/run_paper_comparison.py --dataset gaia --variants sweeps --seeds 3 \\
        --arms v0_raw,v3_incident --threads 1 --shard 0/2 --out results/paper_comparison/sweep_gaia_s0.json

    # Fig. 15: alert-field ablation on re-embedded alerts (scripts/reembed_alerts.py)
    python scripts/run_paper_comparison.py --dataset gaia --data-root <dir> --tag no_op ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch                                                     # noqa: E402
from adadiag import config, data, paper_compare as pc           # noqa: E402
from adadiag.train import train                                  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["gaia", "hotel", "sockshop"])
    ap.add_argument("--data-root", default=str(ROOT),
                    help="folder containing data/<dataset>/ (default: this repository)")
    ap.add_argument("--arms", default="v0_raw,v0_simplex,v3_incident")
    ap.add_argument("--variants", default="default",
                    help="comma list of names in paper_compare.VARIANTS, or 'sweeps'")
    ap.add_argument("--drop", default="", help="comma list of modalities to remove")
    ap.add_argument("--tag", default="", help="label stored with each record (e.g. embedding)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400, help="same cap as results/ (run_sweep)")
    ap.add_argument("--light", action="store_true",
                    help="metrics and per-class only (skip Shapley/agreement/examples)")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads; results/ used 2 (see results/ENVIRONMENT.md)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    ds = data.load(a.dataset, a.data_root)
    base = config.get(a.dataset, epochs=a.epochs)
    if a.drop:
        base["drop_modalities"] = tuple(a.drop.split(","))
    if a.variants == "sweeps":
        variants = [v for v in pc.VARIANTS if v not in pc.ABLATION_VARIANTS]
    elif a.variants == "ablations":
        variants = ["default", *pc.ABLATION_VARIANTS]
    else:
        variants = a.variants.split(",")
    unknown = set(variants) - set(pc.VARIANTS)
    if unknown:
        sys.exit(f"unknown variants: {sorted(unknown)}")
    variants = [v for v in variants if not pc.is_default(v, base)]   # run each point once

    jobs = [(v, arm, s) for v in variants for arm in a.arms.split(",") for s in range(a.seeds)]
    i, n = map(int, a.shard.split("/"))
    jobs = jobs[i::n]

    out = Path(a.out or ROOT / "results" / "paper_comparison" / f"runs_{a.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    recs = json.load(open(out)) if out.exists() else []
    done = {(r["variant"], r["arm"], r["seed"], r.get("tag", "")) for r in recs}
    print(f"{ds!r} | {len(jobs)} jobs in shard {a.shard}, {len(done)} already in {out.name}")

    for v, arm, seed in jobs:
        if (v, arm, seed, a.tag) in done:
            continue
        cfg = {**base, **pc.VARIANTS[v]}
        res, model = train(arm, seed, ds, cfg)
        res.pop("w_final", None)
        res.update(variant=v, tag=a.tag, overrides=pc.VARIANTS[v],
                   drop=list(base.get("drop_modalities", ())),
                   per_class=pc.per_class_k(model, ds),
                   weights=pc.weight_matrix(model, ds))
        if not a.light:
            sh = pc.modality_shapley(model, ds)
            if seed != 0:                       # per-incident shares: seed 0 only (size)
                for t in sh:
                    sh[t].pop("share")
            res.update(shapley=sh, agreement=pc.to_agreement(model, ds))
            if seed == 0:
                res["examples"] = pc.example_outputs(model, ds)
        recs.append(res)
        json.dump(recs, open(out, "w"))
        print(f"  {v:10s} {arm:12s} s{seed}  HR@1 {res['HR@1']:.3f}  F1 {res['f1']:.3f}"
              f"  ({res['epochs']} ep, {res['train_s']}s)", flush=True)
    print("done", out)


if __name__ == "__main__":
    main()
