#!/usr/bin/env python3
"""README section 6.13: five ways to weight TVDiag's task-oriented contrastive term.

Each variant changes one thing on top of TVDiag's default (v0_raw), with one setting fixed
in advance (none is tuned on the test split):

    pos         1  weight positive pairs     cl_pos_beta = 1   (harder positives count more)
    neg         2  weight negative samples   cl_neg_beta = 1   (harder negatives count more)
    importance  3  weight anchors            cl_importance = 1 (by their diagnosis loss)
    align       4  weights from the diagnosis losses  arm v5_align (gradient alignment)
    curriculum  5  easiest anchors first     cl_curriculum = 50 (50 % -> 100 % over 50 epochs)
    pos_strong, neg_strong   types 1 and 2 at beta = 1/tau (sensitivity check)
    --capped-from importance TVDiag stopped at type 3's epochs (training-length control)
    trace_third, trace_tvdiag  README section 6.14: AdaDiag (v3_incident) with trace's weight
                held at 1/3 of both columns, or at TVDiag's role (RCL 0.5, FTI 0), and only
                metric and log learned; --check-adadiag checks AdaDiag itself first

Same settings as the references it is paired with (README sections 6.8, 6.12): 1 thread,
the 500-epoch cap, TVDiag's early stopping, seeds 0-2 (GAIA) / 0-4 (Hotel).

    python scripts/run_cl_variants.py --dataset hotel --seeds 5 --check
    python scripts/run_cl_variants.py --dataset gaia  --seeds 3 --sigma-to 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch                                                         # noqa: E402
from adadiag import config, data                                     # noqa: E402
from adadiag.evaluate import per_class                               # noqa: E402
from adadiag.train import train                                      # noqa: E402

VARIANTS = {
    "pos":        ("v0_raw", dict(cl_pos_beta=1.0)),
    "neg":        ("v0_raw", dict(cl_neg_beta=1.0)),
    "importance": ("v0_raw", dict(cl_importance=1.0)),
    "align":      ("v5_align", dict()),
    "curriculum": ("v0_raw", dict(cl_curriculum=50, cl_curriculum_start=0.5)),
    # TVDiag's own rule at the same sigma_to (the reference for sigma_to != 0.1; at 0.1 it
    # is TVDiag default itself)
    "none":       ("v0_raw", dict()),
    # sensitivity, not tuning: types 1 and 2 at beta = 1/tau, the scale of the logits
    # themselves (weight proportional to e_ij); both settings are reported
    "pos_strong": ("v0_raw", dict(cl_pos_beta=1 / 0.3)),
    "neg_strong": ("v0_raw", dict(cl_neg_beta=1 / 0.3)),
    # README section 6.14: AdaDiag with trace's weight held fixed, metric and log learned
    "trace_third":  ("v3_incident", dict(fixed_trace=[1 / 3, 1 / 3])),
    "trace_tvdiag": ("v3_incident", dict(fixed_trace=[0.5, 0.0])),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["gaia", "hotel"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sigma-to", type=float, default=0.1)
    ap.add_argument("--variants", default="pos,neg,importance,align,curriculum")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--check", action="store_true",
                    help="first re-train TVDiag default (seed 0) and check it reproduces the "
                         "committed reference bit-for-bit")
    ap.add_argument("--check-adadiag", action="store_true",
                    help="first re-train AdaDiag (v3_incident, seed 0) and check it reproduces "
                         "the section 6.8 ref_adadiag record bit-for-bit")
    ap.add_argument("--capped-from", default=None, metavar="VARIANT",
                    help="control: train TVDiag default with each seed's epoch cap set to the "
                         "epoch count that VARIANT stopped at on that seed (records variant "
                         "'capped:<VARIANT>'), to separate the weighting from training length")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ds = data.load(a.dataset)

    if a.check:
        res, _ = train("v0_raw", 0, ds, config.get(a.dataset))
        f = ("sweep_gaia_tv0.json" if a.dataset == "gaia" else "ablate_hotel.json")
        ref = [r for r in json.load(open(ROOT / "results" / "paper_comparison" / f))
               if r["arm"] == "v0_raw" and r.get("variant") == "default" and r["seed"] == 0
               and not r.get("tag")][0]
        keys = ("HR@1", "HR@3", "f1", "epochs")
        same = all(res[k] == ref[k] for k in keys)
        print("TVDiag-default check:", {k: (res[k], ref[k]) for k in keys},
              "IDENTICAL" if same else "DIFFERENT", flush=True)

    if a.check_adadiag:
        res, _ = train("v3_incident", 0, ds, config.get(a.dataset))
        ref = []
        for f in sorted((ROOT / "results").glob(f"ablation_{a.dataset}_s*.json")):
            ref += [r for r in json.load(open(f)) if r["name"] == "ref_adadiag" and r["seed"] == 0]
        keys = ("HR@1", "HR@3", "f1", "epochs")
        same = all(res[k] == ref[0][k] for k in keys)
        print("AdaDiag check:", {k: (res[k], ref[0][k]) for k in keys},
              "IDENTICAL" if same else "DIFFERENT", flush=True)

    out = Path(a.out or ROOT / "results" / "cl_variants" / f"runs_{a.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    recs = json.load(open(out)) if out.exists() else []
    done = {(r["variant"], r["sigma_to"], r["seed"]) for r in recs}
    if a.capped_from:
        src = []
        for f in sorted(out.parent.glob(f"runs_{a.dataset}*.json")):
            src += [r for r in json.load(open(f)) if r["variant"] == a.capped_from
                    and r["sigma_to"] == a.sigma_to]
        name = f"capped:{a.capped_from}"
        for r0 in sorted(src, key=lambda r: r["seed"]):
            if (name, a.sigma_to, r0["seed"]) in done:
                continue
            cfg = config.get(a.dataset, to_scale=a.sigma_to, epochs=r0["epochs"])
            res, model = train("v0_raw", r0["seed"], ds, cfg)
            res.update(variant=name, sigma_to=a.sigma_to, overrides={"epochs": r0["epochs"]},
                       per_class=per_class(model, ds)[0])
            recs = json.load(open(out)) if out.exists() else []
            recs.append(res)
            json.dump(recs, open(out, "w"))
            print(f"  {name} σ_to={a.sigma_to:<4} s{r0['seed']}  HR@1 {res['HR@1']:.3f}  "
                  f"F1 {res['f1']:.3f}  ({res['epochs']} ep; {a.capped_from}: HR@1 "
                  f"{r0['HR@1']:.3f}, F1 {r0['f1']:.3f})", flush=True)
        print("done", out)
        return
    for v in a.variants.split(","):
        arm, over = VARIANTS[v]
        for seed in range(a.seeds):
            if (v, a.sigma_to, seed) in done:
                continue
            cfg = config.get(a.dataset, to_scale=a.sigma_to, **over)
            res, model = train(arm, seed, ds, cfg)
            res.update(variant=v, sigma_to=a.sigma_to, overrides=over,
                       per_class=per_class(model, ds)[0])
            recs = json.load(open(out)) if out.exists() else []    # another lane may write
            recs.append(res)
            json.dump(recs, open(out, "w"))
            w = res.get("w_final")
            extra = (" w RCL " + "/".join(f"{x[0]:.2f}" for x in w) + " FTI "
                     + "/".join(f"{x[1]:.2f}" for x in w)) if w else ""
            print(f"  {v:10s} σ_to={a.sigma_to:<4} s{seed}  HR@1 {res['HR@1']:.3f}  "
                  f"F1 {res['f1']:.3f}  ({res['epochs']} ep, {res['train_s']:.0f}s){extra}",
                  flush=True)
    print("done", out)


if __name__ == "__main__":
    main()
