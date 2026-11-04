#!/usr/bin/env python3
"""README section 6.17: which AdaDiag is best? Screening, then confirmation on fresh seeds.

Every candidate is compared with TVDiag trained under IDENTICAL settings (same loop, thread
count, epoch cap, seeds), and every run uses `isolate_weighting_rng`, so dropout masks match
across arms (section 6.15). Runs are written to results/selection/runs_<ds>.json, keyed by
(variant, seed); the script resumes.

Stages:
  screen   seeds 0-4 (Hotel) / 0-2 (GAIA): the candidates whose committed runs were not
           yet stream-aligned or not 1-thread (fidelity loop, fixed trace, warm-up)
  confirm  fresh seeds 5-9, never used to rank anything, both datasets: the finalists and
           their matched TVDiag

The selection rule is fixed before the confirmation runs (see SELECTION_RULE below and
README section 6.17).

    python scripts/select_adadiag.py --stage screen  --dataset hotel
    python scripts/select_adadiag.py --stage confirm --dataset gaia --variants tvdiag,adadiag
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

FID = dict(config.FIDELITY)
#: name -> (arm, cfg overrides, matched TVDiag reference)
VARIANTS = {
    "tvdiag":       ("v0_raw", {}, None),
    "tvdiag_fid":   ("v0_raw", FID, None),
    "adadiag":      ("v3_incident", {}, "tvdiag"),
    "adadiag_fid":  ("v3_incident", FID, "tvdiag_fid"),
    "v2_global":    ("v2_global", {}, "tvdiag"),
    "warmup":       ("v3_incident", dict(weight_warmup=100), "tvdiag"),
    "trace_third":  ("v3_incident", dict(fixed_trace=[1 / 3, 1 / 3]), "tvdiag"),
    "trace_tvdiag": ("v3_incident", dict(fixed_trace=[0.5, 0.0]), "tvdiag"),
    "align":        ("v5_align", {}, "tvdiag"),
    # README section 6.18: the LSTM aggregator the TVDiag paper describes, for both arms
    "tvdiag_lstm":  ("v0_raw", dict(aggregator="lstm"), None),
    "adadiag_lstm": ("v3_incident", dict(aggregator="lstm"), "tvdiag_lstm"),
    "align_lstm":   ("v5_align", dict(aggregator="lstm"), "tvdiag_lstm"),
    # README section 6.19: what the TVDiag paper specifies and its code does not do
    # (config.PAPER), all together for both arms, then one at a time for TVDiag
    "tvdiag_paper": ("v0_raw", dict(config.PAPER), None),
    "align_paper":  ("v5_align", dict(config.PAPER), "tvdiag_paper"),
    **{f"tvdiag+{k}": ("v0_raw", {k: v}, None) for k, v in config.PAPER.items()},
    # control: with theta learnable the loss never stops falling, so early stopping never
    # fires and every paper run trains to the 500-epoch cap; this is TVDiag (code) trained
    # to the same cap, to separate "theta learned" from "trained longer"
    "tvdiag_noes":  ("v0_raw", dict(patience=10 ** 6), None),
    # README section 6.19: AdaDiag-DL with only Eq. 13's learnable theta, the one paper
    # feature that did not hurt TVDiag; matched against TVDiag with the same switch
    "align+awl_learnable": ("v5_align", dict(awl_learnable=True), "tvdiag+awl_learnable"),
}
SELECTION_RULE = (
    "Score = mean of the four paired differences from the matched TVDiag (GAIA HR@1, GAIA F1, "
    "Hotel HR@1, Hotel F1) on the fresh seeds 5-9. The best AdaDiag is the highest score, "
    "provided no cell is significantly worse than TVDiag (paired t, p < 0.05). If no "
    "candidate's score is significantly above zero (one-sample t over the 10 per-seed scores, "
    "Holm over the finalists), the selection is reported as a tie on performance and decided "
    "by mechanism (learns from the diagnosis losses, graded weights), then by training cost.")
SCREEN = {"hotel": (range(5), ["adadiag_fid", "trace_third", "trace_tvdiag", "warmup"]),
          "gaia": (range(3), ["warmup"])}


def run(ds, variants, seeds, out, threads):
    torch.set_num_threads(threads)
    recs = json.load(open(out)) if out.exists() else []
    done = {(r["variant"], r["seed"]) for r in recs}
    for v in variants:
        arm, over, _ = VARIANTS[v]
        for seed in seeds:
            if (v, seed) in done:
                continue
            cfg = config.get(ds.name, isolate_weighting_rng=True, **over)
            res, model = train(arm, seed, ds, cfg)
            res.update(variant=v, overrides={k: (list(x) if isinstance(x, tuple) else x)
                                             for k, x in over.items()},
                       per_class=per_class(model, ds)[0])
            recs = json.load(open(out)) if out.exists() else []
            recs.append(res)
            json.dump(recs, open(out, "w"))
            w = res.get("w_final")
            wtxt = (" w RCL " + "/".join(f"{x[0]:.2f}" for x in w) + " FTI "
                    + "/".join(f"{x[1]:.2f}" for x in w)) if w else ""
            print(f"  {v:13s} s{seed}  HR@1 {res['HR@1']:.3f}  F1 {res['f1']:.3f}  "
                  f"({res['epochs']} ep, {res['train_s']:.0f}s){wtxt}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["screen", "confirm", "length"])
    ap.add_argument("--dataset", required=True, choices=["gaia", "hotel", "sockshop"])
    ap.add_argument("--variants", default=None, help="confirm: comma-separated finalists")
    ap.add_argument("--seeds", default="5-9", help="confirm: seed range, e.g. 5-9")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="", help="suffix for the output file, so two lanes never "
                                               "write the same file")
    a = ap.parse_args()
    ds = data.load(a.dataset)
    OUT = ROOT / "results" / "selection"
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "length":
        # control: TVDiag stopped, seed by seed, at the epoch each finalist stopped at on the
        # fresh seeds (README section 6.13: on GAIA the stopping point alone moves HR@1)
        torch.set_num_threads(a.threads)
        conf = []
        for f in sorted(OUT.glob(f"confirm_{a.dataset}*.json")):
            conf += json.load(open(f))
        out = OUT / f"length_{a.dataset}.json"
        recs = json.load(open(out)) if out.exists() else []
        done = {(r["variant"], r["seed"]) for r in recs}
        for v in a.variants.split(","):
            for r0 in sorted([r for r in conf if r["variant"] == v], key=lambda r: r["seed"]):
                if (f"capped:{v}", r0["seed"]) in done:
                    continue
                # the finalist's matched TVDiag, with its settings (e.g. the LSTM aggregator)
                ref = VARIANTS[v][2] or "tvdiag"
                arm_r, over_r, _ = VARIANTS[ref]
                cfg = config.get(ds.name, isolate_weighting_rng=True, **over_r)
                cfg["epochs"] = r0["epochs"]
                res, _ = train(arm_r, r0["seed"], ds, cfg)
                res.update(variant=f"capped:{v}", cap=r0["epochs"])
                recs = json.load(open(out)) if out.exists() else []
                recs.append(res)
                json.dump(recs, open(out, "w"))
                print(f"  TVDiag stopped at {v}'s epoch, s{r0['seed']} ({r0['epochs']} ep): HR@1 "
                      f"{res['HR@1']:.3f} F1 {res['f1']:.3f} | {v} HR@1 {r0['HR@1']:.3f} F1 "
                      f"{r0['f1']:.3f}", flush=True)
        print("done", out)
        return
    if a.stage == "screen":
        seeds, variants = SCREEN[a.dataset]
        out = OUT / f"screen_{a.dataset}.json"
    else:
        lo, hi = (int(x) for x in a.seeds.split("-"))
        seeds, variants = range(lo, hi + 1), a.variants.split(",")
        out = OUT / f"confirm_{a.dataset}{a.tag}.json"
    run(ds, variants, list(seeds), out, a.threads)
    print("done", out)


if __name__ == "__main__":
    main()
