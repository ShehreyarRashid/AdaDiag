#!/usr/bin/env python3
"""README section 6.11: the softened prior and the TVDiag-fidelity switches.

Reads results/fidelity/*.json (scripts/fidelity_runs.sh) and the committed runs, writes
results/FIDELITY.md and results/fidelity_table.md (the compact table the README quotes).

    python scripts/make_fidelity_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adadiag import paper_numbers as P                                 # noqa: E402

OUT = ROOT / "results"
DS = ("gaia", "sockshop", "hotel")
NAME = {"gaia": "A · GAIA", "sockshop": "C · Sock Shop", "hotel": "D · Hotel"}
LETTER = {"gaia": "A", "sockshop": "C", "hotel": "D"}
ARMS = ("v0_raw", "v0_soft", "v0_simplex", "v3_incident")
LABEL = {"v0_raw": "TVDiag as shipped (1 / 1 / 0, 0 / 0 / 1)",
         "v0_soft": "**softened prior (.9 / .9 / .1, .1 / .1 / .9)**",
         "v0_simplex": "TVDiag rule on the simplex (.5 / .5 / 0, 0 / 0 / 1)",
         "v3_incident": "AdaDiag (learned)"}
METRICS = (("HR@1", "HR@1"), ("HR@3", "HR@3"), ("MRR@3", "MRR@3"), ("f1", "F1"),
           ("macro_f1", "macro-F1"))


def committed(ds: str) -> dict:
    out = {}
    if ds in ("gaia", "hotel"):
        d = json.load(open(OUT / f"{ds}.json"))
        for arm in ("v0_raw", "v0_simplex", "v3_incident"):
            out[arm] = d[arm]
    else:
        recs = json.load(open(OUT / "paper_comparison" / "runs_sockshop.json"))
        for arm in ("v0_raw", "v0_simplex", "v3_incident"):
            out[arm] = [r for r in recs if r["arm"] == arm and r["variant"] == "default"]
    f = OUT / "fidelity" / f"soft_{ds}.json"
    if f.exists():
        out["v0_soft"] = json.load(open(f))
    return out


def fidelity(ds: str) -> dict:
    recs = []
    for f in sorted((OUT / "fidelity").glob(f"fidelity_{ds}_*.json")):
        recs += json.load(open(f))
    return {arm: sorted([r for r in recs if r["arm"] == arm], key=lambda r: r["seed"])
            for arm in ARMS}


def ms(rs, k):
    v = np.array([r[k] for r in rs])
    if len(v) == 0:
        return "—"
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def paired(a, b, k):
    sa = {r["seed"]: r[k] for r in a}
    sb = {r["seed"]: r[k] for r in b}
    s = sorted(set(sa) & set(sb))
    if len(s) < 2:
        return float("nan"), float("nan")
    x, y = np.array([sa[i] for i in s]), np.array([sb[i] for i in s])
    d = y - x
    if np.allclose(d, d[0]):
        return float(d.mean()), 1.0 if d[0] == 0 else 0.0
    return float(d.mean()), float(stats.ttest_rel(y, x).pvalue)


def holm(ps):
    ps = np.asarray(ps, float)
    out = np.full_like(ps, np.nan)
    ok = ~np.isnan(ps)
    order = np.argsort(np.where(ok, ps, np.inf))
    run = 0.0
    for r, i in enumerate(order[: ok.sum()]):
        run = max(run, min(1.0, (ok.sum() - r) * ps[i]))
        out[i] = run
    return out


def arm_table(get, title):
    """One block per dataset: the four arms, then softened prior vs each reference."""
    L = [f"### {title}\n"]
    comps, keys = [], []
    for ds in DS:
        runs = get(ds)
        for ref in ("v0_raw", "v0_simplex", "v3_incident"):
            for k, _ in METRICS:
                if runs.get("v0_soft") and runs.get(ref):
                    comps.append(paired(runs[ref], runs["v0_soft"], k))
                    keys.append((ds, ref, k))
    adj = dict(zip(keys, holm([p for _, p in comps])))
    raw = dict(zip(keys, comps))
    facts = {}
    for ds in DS:
        runs = get(ds)
        if not any(runs.values()):
            continue
        L.append(f"**{NAME[ds]}**\n")
        L.append("| arm | seeds | " + " | ".join(lab for _, lab in METRICS) + " |")
        L.append("|---|---|" + "---|" * len(METRICS))
        for arm in ARMS:
            rs = runs.get(arm, [])
            if rs:
                L.append(f"| {LABEL[arm]} | {len(rs)} | " + " | ".join(ms(rs, k) for k, _ in METRICS) + " |")
        for ref in ("v0_raw", "v0_simplex", "v3_incident"):
            if (ds, ref, "HR@1") not in raw:
                continue
            cells = []
            for k, _ in METRICS:
                d, p = raw[(ds, ref, k)]
                a = adj[(ds, ref, k)]
                star = "**" if a < 0.05 else ""
                cells.append(f"{star}{d:+.3f}{star} (Holm {a:.2f})")
            L.append(f"| *softened − {ref}* | | " + " | ".join(cells) + " |")
        L.append("")
        facts[ds] = {arm: {k: float(np.mean([r[k] for r in runs[arm]])) for k, _ in METRICS}
                     for arm in ARMS if runs.get(arm)}
        facts[ds]["_diff"] = {(ref, k): (raw[(ds, ref, k)][0], adj[(ds, ref, k)])
                              for ref in ("v0_raw", "v0_simplex", "v3_incident")
                              for k, _ in METRICS if (ds, ref, k) in raw}
    L.append("Bold: significant after Holm correction over every comparison in this table "
             "(3 datasets × 3 references × 5 metrics). Paired t-test over seeds.\n")
    return "\n".join(L), facts


def weights_table():
    rows = []
    for ds in DS:
        f = fidelity(ds).get("v3_incident", [])
        if not f:
            continue
        W = np.mean([np.array(r["weights"]) for r in f], 0)
        # committed AdaDiag, test-set weights (the re-trained, identical runs of §6.9)
        c = [r for r in json.load(open(OUT / "paper_comparison" / f"runs_{ds}.json"))
             if r["arm"] == "v3_incident"]
        Wc = np.mean([np.array(r["weights"]) for r in c], 0)
        rows.append(f"| {NAME[ds]} | " + " / ".join(f"{x:.2f}" for x in Wc[:, 0]) + " | "
                    + " / ".join(f"{x:.2f}" for x in Wc[:, 1]) + " | "
                    + " / ".join(f"{x:.2f}" for x in W[:, 0]) + " | "
                    + " / ".join(f"{x:.2f}" for x in W[:, 1]) + " |")
    if not rows:
        return ""
    return ("| dataset | committed: RCL m / t / l | committed: FTI m / t / l | fidelity: RCL m / t / l "
            "| fidelity: FTI m / t / l |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n")


def paper_gap():
    rows = []
    for ds in DS:
        L = LETTER[ds]
        p4 = dict(zip(P.TABLE4_COLS, P.TABLE4[L][-1][3:]))
        p5 = P.TABLE5[L][-1][5]
        c, f = committed(ds)["v0_raw"], fidelity(ds)["v0_raw"]
        if not f:
            continue
        m = lambda rs, k: float(np.mean([r[k] for r in rs]))
        rows.append(f"| {NAME[ds]} | {p4['HR@1']:.3f} / {p5:.3f} | "
                    f"{m(c, 'HR@1'):.3f} / {m(c, 'f1'):.3f} ({m(c, 'HR@1') - p4['HR@1']:+.3f} / "
                    f"{m(c, 'f1') - p5:+.3f}) | {m(f, 'HR@1'):.3f} / {m(f, 'f1'):.3f} "
                    f"({m(f, 'HR@1') - p4['HR@1']:+.3f} / {m(f, 'f1') - p5:+.3f}) |")
    return ("| dataset | paper HR@1 / F1 | committed re-run (gap) | with the fixes (gap) |\n"
            "|---|---|---|---|\n" + "\n".join(rows) + "\n")


def adadiag_table():
    """AdaDiag against the fixed rules once the loop matches TVDiag's (Holm over the table)."""
    keys, comps = [], []
    for ds in DS:
        runs = fidelity(ds)
        for ref in ("v0_raw", "v0_simplex"):
            for k, _ in METRICS:
                if runs.get(ref) and runs.get("v3_incident"):
                    comps.append(paired(runs[ref], runs["v3_incident"], k)); keys.append((ds, ref, k))
    if not keys:
        return ""
    adj = dict(zip(keys, holm([p for _, p in comps])))
    raw = dict(zip(keys, comps))
    rows = []
    for ds in DS:
        for ref in ("v0_raw", "v0_simplex"):
            if (ds, ref, "HR@1") in raw:
                rows.append(f"| {NAME[ds]} | AdaDiag − {ref} | " + " | ".join(
                    f"{raw[(ds, ref, k)][0]:+.3f} (p {raw[(ds, ref, k)][1]:.3f}, Holm "
                    f"{adj[(ds, ref, k)]:.2f})" for k, _ in METRICS) + " |")
    return ("| dataset | comparison | " + " | ".join(l for _, l in METRICS) + " |\n|---|---|"
            + "---|" * len(METRICS) + "\n" + "\n".join(rows) + "\n")


def main() -> None:
    t1, f1 = arm_table(committed, "1. The committed settings (2 threads; the other three "
                                  "arms are the committed runs)")
    t2, f2 = arm_table(fidelity, "2. With the three fidelity fixes (1 thread, every arm "
                                 "re-run)")
    L = ["# The softened prior, and three fixes to the training loop — README §6.11\n",
         "Produced by `scripts/fidelity_runs.sh` and `scripts/make_fidelity_report.py`.\n",
         "*Softened prior* = TVDiag's rule with 0.9 where it has 1 and 0.1 where it has 0 "
         "(RCL column metric / trace / log = .9 / .9 / .1; FTI = .1 / .1 / .9), as proposed "
         "in review. The *fidelity fixes* (`config.FIDELITY`) make the training loop match "
         "TVDiag's released code: self-loops on nodes with no in-edge, the augmented half of "
         "each batch drawn from a pre-built pool, the last epoch's model evaluated, a 500-"
         "epoch cap.\n", t1, t2,
         "### 2b. AdaDiag against the fixed rules, with the fixes\n", adadiag_table(),
         "### 3. Does the fixed loop move the re-run closer to the paper?\n", paper_gap(),
         "### 4. AdaDiag's learned matrix, before and after the fixes (5-seed means)\n",
         weights_table()]
    (OUT / "FIDELITY.md").write_text("\n".join(L))
    json.dump({"committed": {d: {a: v for a, v in x.items() if a != "_diff"} for d, x in f1.items()},
               "fidelity": {d: {a: v for a, v in x.items() if a != "_diff"} for d, x in f2.items()}},
              open(OUT / "fidelity" / "summary.json", "w"), indent=1)
    print("wrote results/FIDELITY.md")


if __name__ == "__main__":
    main()
