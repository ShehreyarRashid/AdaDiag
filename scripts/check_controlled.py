#!/usr/bin/env python3
"""README section 6.15: is everything but the task-oriented (TO) block the same for TVDiag
and AdaDiag?

Three checks, each written to results/controlled/<part>_<dataset>.json:

  null   With the TO term switched off (use_to=False), TVDiag (v0_raw) and AdaDiag
         (v3_incident) should train identically, bit for bit. Run with and without
         `isolate_weighting_rng`; compares metrics, epoch count and every encoder/head
         parameter.
  rng    AdaDiag re-trained with `isolate_weighting_rng` (same dropout masks as TVDiag),
         every seed, to measure what the random-stream difference did to the committed
         comparison.
  length TVDiag stopped, seed by seed, at the epoch AdaDiag stopped at, to separate the TO
         change from the training length it causes (TVDiag's stopping rule watches the
         total training loss, which includes the TO term).

    python scripts/check_controlled.py --part null   --dataset hotel
    python scripts/check_controlled.py --part rng    --dataset hotel --seeds 5
    python scripts/check_controlled.py --part length --dataset gaia  --seeds 3
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

KEYS = ("HR@1", "HR@3", "MRR@3", "f1", "macro_f1", "epochs")
OUT = ROOT / "results" / "controlled"


def ablation_refs(ds_name, name):
    recs = []
    for f in sorted((ROOT / "results").glob(f"ablation_{ds_name}_s*.json")):
        recs += [r for r in json.load(open(f)) if r["name"] == name]
    return {r["seed"]: r for r in recs}


def shared_params(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith("weighting")}


def part_null(ds, a):
    out = []
    for iso in (False, True):
        cfg = config.get(ds.name, use_to=False, isolate_weighting_rng=iso)
        r0, m0 = train("v0_raw", 0, ds, cfg)
        r1, m1 = train("v3_incident", 0, ds, cfg)
        p0, p1 = shared_params(m0), shared_params(m1)
        same_params = all(torch.equal(p0[k], p1[k]) for k in p0)
        same = all(r0[k] == r1[k] for k in KEYS) and same_params
        rec = dict(dataset=ds.name, isolate_weighting_rng=iso, identical=same,
                   identical_params=same_params,
                   tvdiag={k: r0[k] for k in KEYS}, adadiag={k: r1[k] for k in KEYS},
                   max_param_diff=max(float((p0[k] - p1[k]).abs().max()) for k in p0))
        out.append(rec)
        print(f"  TO off, isolate_weighting_rng={iso}: "
              f"{'IDENTICAL' if same else 'DIFFERENT'}  TVDiag HR@1 {r0['HR@1']:.3f} F1 "
              f"{r0['f1']:.3f} ({r0['epochs']} ep) | AdaDiag HR@1 {r1['HR@1']:.3f} F1 "
              f"{r1['f1']:.3f} ({r1['epochs']} ep) | max |Δparam| {rec['max_param_diff']:.2e}",
              flush=True)
    return out


def part_rng(ds, a):
    ref = ablation_refs(ds.name, "ref_adadiag")
    out = []
    for seed in range(a.seeds):
        cfg = config.get(ds.name, isolate_weighting_rng=True)
        res, model = train("v3_incident", seed, ds, cfg)
        w = res.pop("w_final", None)
        res.update(per_class=per_class(model, ds)[0], w_final=w,
                   committed={k: ref[seed][k] for k in KEYS} if seed in ref else None)
        out.append(res)
        c = res["committed"]
        print(f"  AdaDiag, isolated RNG, s{seed}: HR@1 {res['HR@1']:.3f} F1 {res['f1']:.3f} "
              f"({res['epochs']} ep) | committed HR@1 {c['HR@1']:.3f} F1 {c['f1']:.3f} "
              f"({c['epochs']} ep){'  IDENTICAL' if all(res[k] == c[k] for k in KEYS) else ''}",
              flush=True)
    return out


def part_length(ds, a):
    if a.epochs_from == "rng":
        # the AdaDiag runs with the stream aligned (part rng), the controlled comparison
        ref = {r["seed"]: r for r in json.load(open(OUT / f"rng_{ds.name}.json"))}
    else:
        ref = ablation_refs(ds.name, "ref_adadiag")
    out = []
    for seed in range(a.seeds):
        ep = ref[seed]["epochs"]
        res, model = train("v0_raw", seed, ds, config.get(ds.name, epochs=ep))
        res.update(adadiag={k: ref[seed][k] for k in KEYS}, cap=ep, epochs_from=a.epochs_from)
        out.append(res)
        print(f"  TVDiag stopped at AdaDiag's epoch, s{seed} ({ep} ep): HR@1 {res['HR@1']:.3f} "
              f"F1 {res['f1']:.3f} | AdaDiag HR@1 {ref[seed]['HR@1']:.3f} F1 "
              f"{ref[seed]['f1']:.3f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=["null", "rng", "length"])
    ap.add_argument("--dataset", required=True, choices=["gaia", "hotel"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--epochs-from", choices=["committed", "rng"], default="committed",
                    help="length: take AdaDiag's epochs from the committed runs or from the "
                         "aligned-stream re-run (part rng); the latter writes length_rng_<ds>.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ds = data.load(a.dataset)
    out = {"null": part_null, "rng": part_rng, "length": part_length}[a.part](ds, a)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_rng" if a.part == "length" and a.epochs_from == "rng" else ""
    path = OUT / f"{a.part}{tag}_{a.dataset}.json"
    json.dump(out, open(path, "w"), indent=1)
    print("done", path)


if __name__ == "__main__":
    main()
