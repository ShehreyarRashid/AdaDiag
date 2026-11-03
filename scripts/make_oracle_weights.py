#!/usr/bin/env python3
"""Oracle weight matrices for the headroom check (README section 6.12).

The question: if the task-oriented term were weighted the way the evidence says it should
be, would TVDiag do better? The evidence is the source-removal ablation (README section 6.8,
group B, results/ablation_<ds>_*.json): how much each task loses when one modality is
removed. Each oracle column is that loss, clipped at zero and normalised onto the simplex:

    w[m, t] = max(0, cost(m, t)) / sum_k max(0, cost(k, t))      (uniform if all <= 0)

    cost(m, RCL) = HR@1(full) - HR@1(without m)
    cost(m, FTI) = F1(full)   - F1(without m)

* `global` -- one matrix per dataset, from the aggregate metrics.
* `class`  -- one matrix per FAILURE TYPE, from the per-class metrics (HR@1, and FTI recall
  for that class). Applied during training by the incident's true type, which is known at
  training time; the contrastive term is not used at inference, so nothing leaks into it.

The costs are measured on the TEST split, so these are ORACLES -- upper bounds on what a
weighting could achieve, not methods. Writes results/oracle/oracle_weights.json.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adadiag.weighting import MODALITIES                               # noqa: E402

DROP = {"metric": "B_no_metric", "trace": "B_no_trace", "log": "B_no_log"}


def column(costs):
    c = np.clip(np.asarray(costs, float), 0, None)
    return (c / c.sum()).tolist() if c.sum() > 0 else [1 / len(c)] * len(c)


def by_seed(recs, name):
    return {r["seed"]: r for r in recs if r["name"] == name}


def main() -> None:
    out = {}
    for ds in ("gaia", "hotel"):
        recs = []
        for f in glob.glob(str(ROOT / "results" / f"ablation_{ds}_s*.json")):
            recs += json.load(open(f))
        ref = by_seed(recs, "ref_prior")
        drops = {m: by_seed(recs, n) for m, n in DROP.items()}
        seeds = sorted(set(ref).intersection(*[set(d) for d in drops.values()]))

        def cost(m, get):
            return float(np.mean([get(ref[s]) - get(drops[m][s]) for s in seeds]))

        g_rcl = [cost(m, lambda r: r["HR@1"]) for m in MODALITIES]
        g_fti = [cost(m, lambda r: r["f1"]) for m in MODALITIES]
        glob_w = np.array([column(g_rcl), column(g_fti)]).T.tolist()       # (3, 2)

        classes = list(ref[seeds[0]]["per_class"])
        cls_w, cls_cost = {}, {}
        for c in classes:
            rc = [cost(m, lambda r: r["per_class"][c]["hr1"]) for m in MODALITIES]
            fc = [cost(m, lambda r: r["per_class"][c]["fti_recall"]) for m in MODALITIES]
            cls_w[c] = np.array([column(rc), column(fc)]).T.tolist()
            cls_cost[c] = {"rcl": rc, "fti": fc, "n": ref[seeds[0]]["per_class"][c]["n"]}
        out[ds] = {"seeds": seeds, "global": glob_w,
                   "global_cost": {"rcl": g_rcl, "fti": g_fti},
                   "class": cls_w, "class_cost": cls_cost}
        print(f"== {ds} (seeds {seeds})")
        print("  global RCL", np.round(np.array(glob_w)[:, 0], 2), " FTI", np.round(np.array(glob_w)[:, 1], 2))
        for c in classes:
            w = np.array(cls_w[c])
            print(f"  {c[:34]:34s} RCL {np.round(w[:, 0], 2)}  FTI {np.round(w[:, 1], 2)}")
    path = ROOT / "results" / "oracle" / "oracle_weights.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
