#!/usr/bin/env python3
"""Why does AdaDiag give the trace modality zero weight? (README section 6.10)

For each dataset: trains AdaDiag (v3_incident) and TVDiag as shipped (v0_raw) while recording
the weight matrix and the six task-oriented contrastive losses every epoch; repeats AdaDiag
with the trace and metric data swapped between slots (does the zero follow the data or the
slot?); records the losses at initialisation for five seeds; and tests the obvious fix,
a warm-up in which every encoder trains on both labels before the module may choose.

    python scripts/run_trace_audit.py --threads 2        # ~50 min on 2 cores

With 2 threads the audited runs are the committed ones (checked, and stored as `identical`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch                                                          # noqa: E402
from adadiag import config, data                                     # noqa: E402
from adadiag import identifiability as ident                         # noqa: E402

KEYS = ("HR@1", "HR@3", "MRR@3", "f1", "macro_f1", "epochs")


def committed(ds_name: str, arm: str, seed: int):
    for f in (ROOT / "results" / f"{ds_name}.json",
              ROOT / "results" / "paper_comparison" / f"runs_{ds_name}.json"):
        if not f.exists():
            continue
        d = json.load(open(f))
        recs = d.get(arm, []) if isinstance(d, dict) else [r for r in d if r["arm"] == arm
                                                           and r.get("variant") == "default"]
        for r in recs:
            if r["seed"] == seed:
                return r
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="hotel,gaia,sockshop")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=100,
                    help="epochs of uniform weights before AdaDiag's module may choose")
    ap.add_argument("--warmup-seeds", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "results" / "trace_audit.json"))
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    out_path = Path(a.out)
    out = json.load(open(out_path)) if out_path.exists() else {}

    for name in a.datasets.split(","):
        ds = data.load(name)
        cfg = config.get(name, epochs=400)
        rec = out.setdefault(name, {})
        if "initial" not in rec:
            rec["initial"] = {s: ident.initial_losses(ds, cfg, "v3_incident", s)
                              for s in range(5)}
        for arm in ("v3_incident", "v0_raw"):
            if arm in rec:
                continue
            r = ident.trace_audit(ds, cfg, arm, 0)
            ref = committed(name, arm, 0)
            r["identical"] = None if ref is None else all(r["metrics"][k] == ref[k] for k in KEYS)
            rec[arm] = r
            json.dump(out, open(out_path, "w"), indent=1)
            print(f"{name} {arm}: settled {r['settled']} identical={r['identical']}", flush=True)
        if "swapped" not in rec:
            order = ("trace", "metric", "log")
            r = ident.trace_audit(ident.permute_modalities(ds, order), cfg, "v3_incident", 0)
            r["slot_holds"] = list(order)
            rec["swapped"] = r
            json.dump(out, open(out_path, "w"), indent=1)
            print(f"{name} swapped: final w {r['w'][-1]}", flush=True)
        # the obvious fix to test: let every encoder train on both labels before choosing
        wkey = f"warmup{a.warmup}"
        rec.setdefault(wkey, {})
        for seed in range(a.warmup_seeds):
            if str(seed) in rec[wkey]:
                continue
            r = ident.trace_audit(ds, {**cfg, "weight_warmup": a.warmup}, "v3_incident", seed)
            rec[wkey][str(seed)] = r
            json.dump(out, open(out_path, "w"), indent=1)
            print(f"{name} warm-up {a.warmup} seed {seed}: final w {r['w'][-1]} "
                  f"HR@1 {r['metrics']['HR@1']:.3f} F1 {r['metrics']['f1']:.3f}", flush=True)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
