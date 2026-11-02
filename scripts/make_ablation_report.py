#!/usr/bin/env python3
"""Render the ablation results (README section 6.8) as Markdown tables.

    python scripts/make_ablation_report.py results/ablation_hotel_s0.json results/ablation_hotel_s1.json

Accepts any number of JSON files written by run_ablation.py (e.g. one per shard) and
merges them. Every comparison is paired by seed against the configuration's reference.
"""
from __future__ import annotations

import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adadiag.ablation import ABLATIONS, reference_for              # noqa: E402

MODS = ("metric", "trace", "log")


def load(paths):
    runs = {}
    for p in paths:
        for r in json.loads(Path(p).read_text()):
            runs[(r["name"], r["seed"])] = r
    by = defaultdict(dict)
    for (n, s), r in runs.items():
        by[n][s] = r
    return by


def paired(by, a, b, key):
    """mean(a - b) and paired-t p over the seeds both have."""
    seeds = sorted(set(by[a]) & set(by[b]))
    x = np.array([by[a][s][key] for s in seeds])
    y = np.array([by[b][s][key] for s in seeds])
    d = x - y
    if len(seeds) < 2 or np.allclose(d, d[0]):
        p = 1.0 if np.allclose(d, 0) else float("nan")
    else:
        p = float(stats.ttest_rel(x, y).pvalue)
    return float(d.mean()), p, len(seeds)


def ms(by, n, key):
    v = np.array([r[key] for r in by[n].values()])
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def holm(pvals: dict) -> dict:
    """Holm–Bonferroni adjusted p-values for a family {key: p}; NaNs are left out."""
    items = sorted(((p, k) for k, p in pvals.items() if not math.isnan(p)))
    m, out, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return {k: out.get(k, float("nan")) for k in pvals}


def fmt_d(d, p, p_adj=None):
    """Bold only when significant AFTER Holm correction (or raw p if no family given)."""
    test = p if p_adj is None else p_adj
    star = "**" if (not math.isnan(test) and test < 0.05) else ""
    ps = "—" if math.isnan(p) else f"{p:.2f}"
    adj = "" if p_adj is None or math.isnan(p_adj) else f", Holm {p_adj:.2f}"
    return f"{star}{d:+.3f}{star} (p={ps}{adj})"


KEYS = ("HR@1", "f1", "macro_f1")


def family_holm(by) -> dict:
    """Holm over EVERY ablation comparison in this file, per metric: (name, key) -> p_adj."""
    out = {}
    names = [n for n, (g, *_) in ABLATIONS.items() if g != "ref" and n in by
             and reference_for(n) in by]
    for key in KEYS:
        raw = {n: paired(by, n, reference_for(n), key)[1] for n in names}
        out.update({(n, key): p for n, p in holm(raw).items()})
    return out


def share(by, n):
    return np.mean([r["grad_share"]["sigma*l_to"] for r in by[n].values()])


def group_table(by, group, title, adj):
    names = [n for n, (g, *_) in ABLATIONS.items() if g == group and n in by]
    if not names:
        return
    refs = list(dict.fromkeys(reference_for(n) for n in names))
    refs = [r for r in refs if r in by]
    k = min(len(by[r]) for r in refs)
    print(f"\n#### {title}\n")
    print("Reference: " + " / ".join(f"**{ABLATIONS[r][3]}**" for r in refs)
          + f" · {k} seeds · Δ = configuration − its own reference (same arm), paired by seed. "
          "σ·l_to share is measured on the trained model (one full batch), so it is a "
          "snapshot at convergence, not the training average of section 6.4. "
          "**Bold** = significant after Holm–Bonferroni correction over every ablation "
          "comparison of that metric in this file.\n")
    print("| configuration | HR@1 | MRR@3 | F1 | macro-F1 | Δ HR@1 | Δ F1 | Δ macro-F1 | σ·l_to share |")
    print("|---|---|---|---|---|---|---|---|---|")
    m = lambda n, key: "{:.3f} ± {:.3f}".format(*ms(by, n, key))
    for ref in refs:
        print(f"| *{ABLATIONS[ref][3]}* | {m(ref,'HR@1')} | {m(ref,'MRR@3')} | {m(ref,'f1')} | "
              f"{m(ref,'macro_f1')} | — | — | — | {share(by, ref):.1f} % |")
    for n in names:
        ref = reference_for(n)
        cells = [fmt_d(*paired(by, n, ref, key)[:2], adj.get((n, key))) for key in KEYS]
        print(f"| {ABLATIONS[n][3]} | {m(n,'HR@1')} | {m(n,'MRR@3')} | {m(n,'f1')} | "
              f"{m(n,'macro_f1')} | " + " | ".join(cells) + f" | {share(by, n):.1f} % |")


def per_class_modality(by):
    """Group B, per failure class: what removing each source costs."""
    drops = [f"B_no_{m}" for m in MODS if f"B_no_{m}" in by]
    if "ref_prior" not in by or not drops:
        return
    classes = list(next(iter(by["ref_prior"].values()))["per_class"])
    print("\n#### B, per failure class — what removing each source costs\n")
    print("Mean over seeds of (without source − full model). The most damaging removal "
          "per class is in **bold**: that is the source this class actually depends on.\n")
    for metric, label in (("hr1", "RCL HR@1"), ("fti_recall", "FTI recall")):
        print(f"\n{label}:\n")
        print("| failure class | n | full model | " + " | ".join(f"w/o {d[5:]}" for d in drops) + " |")
        print("|---|---|---|" + "---|" * len(drops))
        for c in classes:
            seeds = sorted(set(by["ref_prior"]).intersection(*(by[d] for d in drops)))
            base = np.mean([by["ref_prior"][s]["per_class"][c][metric] for s in seeds])
            deltas = [np.mean([by[d][s]["per_class"][c][metric] for s in seeds]) - base for d in drops]
            worst = int(np.argmin(deltas))
            cells = [(f"**{x:+.3f}**" if i == worst and x < -0.01 else f"{x:+.3f}")
                     for i, x in enumerate(deltas)]
            n = next(iter(by["ref_prior"].values()))["per_class"][c]["n"]
            print(f"| {c} | {n} | {base:.3f} | " + " | ".join(cells) + " |")


def weight_table(by):
    names = ["ref_adadiag"] + [n for n, (g, *_) in ABLATIONS.items() if g == "C" and n in by]
    names = [n for n in names if n in by and any(r.get("weights") for r in by[n].values())]
    if not names:
        return
    print("\n#### C — what the weighting module actually learned\n")
    print("Largest weight in each task column (1.0 = collapsed onto one source, 0.333 = "
          "uniform) and the column sums (1.0 by construction on the simplex).\n")
    print("| configuration | RCL max weight | FTI max weight | RCL column sum | FTI column sum | RCL argmax | FTI argmax |")
    print("|---|---|---|---|---|---|---|")
    for n in names:
        W = np.array([r["weights"]["w"] for r in by[n].values()])          # (seeds, 3, 2)
        cm = np.array([r["weights"]["col_max"] for r in by[n].values()])
        cs = np.array([r["weights"]["col_sum"] for r in by[n].values()])
        am = [MODS[int(np.argmax(W.mean(0)[:, t]))] for t in range(2)]
        print(f"| {ABLATIONS[n][3]} | {cm[:,0].mean():.3f} | {cm[:,1].mean():.3f} | "
              f"{cs[:,0].mean():.3f} | {cs[:,1].mean():.3f} | {am[0]} | {am[1]} |")


def sigma_table(by):
    rows = [(0.1, "ref_prior", "ref_adadiag")] + [
        (s, f"D_prior_s{t}", f"D_ada_s{t}") for s, t in ((0.5, "0.5"), (1.0, "1"), (2.0, "2"))]
    rows = [r for r in rows if r[1] in by and r[2] in by]
    if len(rows) < 2:
        return
    print("\n#### D — does AdaDiag separate from the prior once l_to carries real weight?\n")
    print("Only l_to is rescaled (σ_to); CM keeps TVDiag's 0.1. Δ = AdaDiag − prior at the "
          "same σ_to, paired by seed.\n")
    print("| σ_to | σ·l_to share (prior) | prior HR@1 | AdaDiag HR@1 | Δ HR@1 | prior F1 | AdaDiag F1 | Δ F1 |")
    print("|---|---|---|---|---|---|---|---|")
    ph_all = holm({s: paired(by, a, p, "HR@1")[1] for s, p, a in rows})
    pf_all = holm({s: paired(by, a, p, "f1")[1] for s, p, a in rows})
    for s, p, a in rows:
        dh, ph, _ = paired(by, a, p, "HR@1")
        df, pf, _ = paired(by, a, p, "f1")
        print(f"| {s} | {share(by, p):.1f} % | {ms(by,p,'HR@1')[0]:.3f} | {ms(by,a,'HR@1')[0]:.3f} | "
              f"{fmt_d(dh, ph, ph_all[s])} | {ms(by,p,'f1')[0]:.3f} | {ms(by,a,'f1')[0]:.3f} | "
              f"{fmt_d(df, pf, pf_all[s])} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    a = ap.parse_args()
    by = load(a.results)
    ds = next(iter(next(iter(by.values())).values()))["dataset"]
    adj = family_holm(by)
    print(f"\n### Ablations — {ds}")
    group_table(by, "A", "A — TVDiag's components (on the prior)", adj)
    group_table(by, "B", "B — modality removal (on the prior)", adj)
    per_class_modality(by)
    group_table(by, "C", "C — AdaDiag's design choices (on AdaDiag)", adj)
    weight_table(by)
    group_table(by, "D", "D — scale of the task-oriented term", adj)
    sigma_table(by)


if __name__ == "__main__":
    main()
