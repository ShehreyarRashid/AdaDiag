#!/usr/bin/env python3
"""Turn results/*.json into the comparison tables (Markdown or LaTeX).

No number in the README is typed by hand; everything comes from here.

    python scripts/make_report.py results/gaia.json --dataset gaia
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
from scipy import stats as st

# TVDiag, ACM TOSEM 35(2) Art. 40, 2026 -- the published figures, per dataset.
PAPER = {"gaia": {"HR@1": 0.759, "f1": 0.936}, "aiops22": {"HR@1": 0.894, "f1": 0.876},
         "sockshop": {"HR@1": 0.863, "f1": 0.636}, "hotel": {"HR@1": 0.821, "f1": 0.822}}
ORDER = ["v0_raw", "v0_simplex", "v1_uniform", "v2_global", "v3_incident", "v4_fusion"]
NAME = {"v0_raw": "V0-raw (TVDiag as shipped)", "v0_simplex": "V0-simplex (prior)",
        "v1_uniform": "V1-uniform (1/3 each)", "v2_global": "V2-global (learned matrix)",
        "v3_incident": "V3-incident (AdaDiag)", "v4_fusion": "V4-fusion (exploratory)"}
METRICS = ["HR@1", "HR@3", "MRR@3", "f1", "macro_f1"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--baseline", default="v0_simplex")
    a = ap.parse_args()

    R: dict[str, list] = {}
    for p in a.results:
        R.update(json.load(open(p)))
    arms = [x for x in ORDER if x in R and R[x]]
    g = lambda arm, m: np.array([r[m] for r in R[arm]])

    print(f"\n### {a.dataset} — {len(R[arms[0]])} seeds, mean ± sd\n")
    print("| arm | " + " | ".join(METRICS) + " |")
    print("|" + "---|" * (len(METRICS) + 1))
    for arm in arms:
        cells = " | ".join(f"{g(arm,m).mean():.3f} ± {g(arm,m).std(ddof=1):.3f}" for m in METRICS)
        print(f"| {NAME[arm]} | {cells} |")

    if a.baseline in arms:
        print(f"\n### Paired t-test vs {NAME[a.baseline]} (same seeds)\n")
        print("| arm | Δ HR@1 | Δ MRR@3 | Δ macro-F1 |")
        print("|---|---|---|---|")
        for arm in arms:
            if arm == a.baseline:
                continue
            cells = []
            for m in ["HR@1", "MRR@3", "macro_f1"]:
                x, y = g(arm, m), g(a.baseline, m)
                d = (x - y).mean()
                if np.allclose(x - y, 0):
                    cells.append("0.000 (identical)")
                else:
                    _, p = st.ttest_rel(x, y)
                    cells.append(f"{d:+.3f} (p={p:.2f}{'*' if p < 0.05 else ''})")
            print(f"| {NAME[arm]} | " + " | ".join(cells) + " |")

    if "v0_raw" in arms and a.dataset in PAPER:
        print(f"\n### Fidelity: V0-raw vs TVDiag's published {a.dataset} numbers\n")
        print("| metric | paper | this re-implementation | gap |")
        print("|---|---|---|---|")
        for m, pv in PAPER[a.dataset].items():
            v = g("v0_raw", m)
            print(f"| {m} | {pv:.3f} | {v.mean():.3f} ± {v.std(ddof=1):.3f} | {v.mean()-pv:+.3f} |")

    for arm in ("v2_global", "v3_incident"):
        if arm in arms and "w_final" in R[arm][0]:
            W = np.array([r["w_final"] for r in R[arm] if "w_final" in r])
            print(f"\n### Learned weights — {NAME[arm]} (mean of {len(W)} seeds)\n")
            print("| source | RCL | FTI | TVDiag's prior |")
            print("|---|---|---|---|")
            prior = [("metric", "0.5", "0"), ("trace", "0.5", "0"), ("log", "0", "1")]
            for i, (m, pr, pf) in enumerate(prior):
                print(f"| {m} | {W[:,i,0].mean():.3f} | {W[:,i,1].mean():.3f} | RCL={pr}, FTI={pf} |")


if __name__ == "__main__":
    main()
