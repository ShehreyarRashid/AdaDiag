#!/usr/bin/env python3
"""README section 6.12: the oracle headroom check.

Reads results/oracle/runs_<ds>.json (scripts/run_oracle.py) and the paired references,
all trained with 1 thread and the 500-epoch cap on the same seeds:

* TVDiag default -- v0_raw, sigma 0.1: the `default` records of
  results/paper_comparison/sweep_gaia_tv*.json and ablate_hotel.json (README section 6.9)
* TVDiag's rule on the simplex, and AdaDiag, at the same sigma_to -- the section 6.8
  ablation records ref_prior / D_prior_s0.5 / D_prior_s1 and ref_adadiag / D_ada_s0.5 /
  D_ada_s1 in results/ablation_<ds>_s*.json

Writes results/oracle/ORACLE.md, results/oracle/readme_table.md and
results/oracle/oracle_headroom.png.

    python scripts/make_oracle_report.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adadiag.weighting import MODALITIES                               # noqa: E402

OUT = ROOT / "results" / "oracle"
DS = ("gaia", "hotel")
NAME = {"gaia": "A · GAIA", "hotel": "D · Hotel"}
SIGMAS = (0.1, 0.5, 1.0)
METRICS = (("HR@1", "HR@1"), ("HR@3", "HR@3"), ("MRR@3", "MRR@3"), ("f1", "F1"),
           ("macro_f1", "macro-F1"))
HEAD = ("HR@1", "f1")                         # the two the verdict is read from
PRIOR = {0.1: "ref_prior", 0.5: "D_prior_s0.5", 1.0: "D_prior_s1"}
ADA = {0.1: "ref_adadiag", 0.5: "D_ada_s0.5", 1.0: "D_ada_s1"}
ORACLES = (("global", "oracle, one matrix"), ("class", "oracle, one matrix per failure type"))


# --------------------------------------------------------------------------- loading
def by_seed(rs):
    return sorted(rs, key=lambda r: r["seed"])


def tvdiag_default(ds):
    if ds == "gaia":
        recs = []
        for f in sorted((ROOT / "results" / "paper_comparison").glob("sweep_gaia_tv*.json")):
            recs += json.load(open(f))
    else:
        recs = json.load(open(ROOT / "results" / "paper_comparison" / "ablate_hotel.json"))
    return by_seed([r for r in recs if r["arm"] == "v0_raw" and r.get("variant") == "default"
                    and not r.get("tag")])


def ablation(ds):
    recs = []
    for f in glob.glob(str(ROOT / "results" / f"ablation_{ds}_s*.json")):
        recs += json.load(open(f))
    return recs


def load(ds):
    """{config label: [records by seed]} for one dataset, in table order."""
    ab = ablation(ds)
    runs = json.load(open(OUT / f"runs_{ds}.json")) if (OUT / f"runs_{ds}.json").exists() else []
    out = {("tvdiag", 0.1): tvdiag_default(ds)}
    for s in SIGMAS:
        out[("prior", s)] = by_seed([r for r in ab if r["name"] == PRIOR[s]])
        out[("adadiag", s)] = by_seed([r for r in ab if r["name"] == ADA[s]])
        for o, _ in ORACLES:
            out[(o, s)] = by_seed([r for r in runs if r["oracle"] == o and r["sigma_to"] == s])
    return out


# --------------------------------------------------------------------------- statistics
def mean(rs, k):
    return float(np.mean([r[k] for r in rs])) if rs else float("nan")


def ms(rs, k):
    v = np.array([r[k] for r in rs])
    if len(v) == 0:
        return "—"
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def paired(a, b, k):
    """Mean of b - a over shared seeds, and the paired t-test p-value."""
    sa = {r["seed"]: r[k] for r in a}
    sb = {r["seed"]: r[k] for r in b}
    s = sorted(set(sa) & set(sb))
    if len(s) < 2:
        return float("nan"), float("nan"), len(s)
    x, y = np.array([sa[i] for i in s]), np.array([sb[i] for i in s])
    d = y - x
    if np.allclose(d, d[0]):
        return float(d.mean()), 1.0 if d[0] == 0 else 0.0, len(s)
    return float(d.mean()), float(stats.ttest_rel(y, x).pvalue), len(s)


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


def comparisons(data):
    """Every oracle configuration against TVDiag default and against the prior at the same
    sigma_to, on every metric; Holm over the whole family (both datasets)."""
    keys, raw = [], []
    for ds in DS:
        for o, _ in ORACLES:
            for s in SIGMAS:
                for ref in ("tvdiag", "prior"):
                    a = data[ds][("tvdiag", 0.1)] if ref == "tvdiag" else data[ds][("prior", s)]
                    b = data[ds][(o, s)]
                    for k, _ in METRICS:
                        if a and b:
                            keys.append((ds, o, s, ref, k))
                            raw.append(paired(a, b, k))
    adj = holm([p for _, p, _ in raw])
    return {k: (d, p, n, h) for k, (d, p, n), h in zip(keys, raw, adj)}


# --------------------------------------------------------------------------- tables
def fmt_matrix(W):
    W = np.asarray(W)
    return (" / ".join(f"{x:.2f}" for x in W[:, 0]), " / ".join(f"{x:.2f}" for x in W[:, 1]))


def matrices_md(ow):
    L = ["| dataset | matrix | RCL: metric / trace / log | FTI: metric / trace / log |",
         "|---|---|---|---|",
         "| (both) | TVDiag's rule, on the simplex | 0.50 / 0.50 / 0.00 | 0.00 / 0.00 / 1.00 |"]
    for ds in DS:
        r, f = fmt_matrix(ow[ds]["global"])
        L.append(f"| {NAME[ds]} | **oracle, one matrix** | {r} | {f} |")
        for c, W in ow[ds]["class"].items():
            r, f = fmt_matrix(W)
            L.append(f"| | {c} (n = {ow[ds]['class_cost'][c]['n']}) | {r} | {f} |")
    return "\n".join(L) + "\n"


def costs_md(ow):
    L = ["| dataset | task | cost of removing metric | trace | log |", "|---|---|---|---|---|"]
    for ds in DS:
        g = ow[ds]["global_cost"]
        L.append(f"| {NAME[ds]} | RCL (HR@1 lost) | " + " | ".join(f"{x:+.3f}" for x in g["rcl"]) + " |")
        L.append(f"| | FTI (F1 lost) | " + " | ".join(f"{x:+.3f}" for x in g["fti"]) + " |")
    return "\n".join(L) + "\n"


def label(cfg):
    kind, s = cfg
    return {"tvdiag": "TVDiag default (1 / 1 / 0, 0 / 0 / 1)", "prior": "TVDiag's rule on the simplex",
            "adadiag": "AdaDiag (learned)", "global": "**oracle, one matrix**",
            "class": "**oracle, per failure type**"}[kind], s


def results_md(data, C):
    L = []
    for ds in DS:
        L.append(f"**{NAME[ds]}**\n")
        L.append("| weighting | σ_to | seeds | " + " | ".join(l for _, l in METRICS) + " |")
        L.append("|---|---|---|" + "---|" * len(METRICS))
        order = [("tvdiag", 0.1)] + [(k, s) for s in SIGMAS
                                     for k in ("prior", "adadiag", "global", "class")]
        for cfg in order:
            rs = data[ds][cfg]
            if not rs:
                continue
            name, s = label(cfg)
            L.append(f"| {name} | {s:g} | {len(rs)} | " + " | ".join(ms(rs, k) for k, _ in METRICS) + " |")
        L.append("")
        L.append("| oracle − reference | σ_to | " + " | ".join(l for _, l in METRICS) + " |")
        L.append("|---|---|" + "---|" * len(METRICS))
        for o, _ in ORACLES:
            for s in SIGMAS:
                for ref in ("tvdiag", "prior"):
                    if (ds, o, s, ref, "HR@1") not in C:
                        continue
                    cells = []
                    for k, _ in METRICS:
                        d, p, n, h = C[(ds, o, s, ref, k)]
                        b = "**" if h < 0.05 else ""
                        cells.append(f"{b}{d:+.3f}{b} (p {p:.2f}, Holm {h:.2f})")
                    rname = "TVDiag default" if ref == "tvdiag" else "prior, same σ_to"
                    L.append(f"| {o} − {rname} | {s:g} | " + " | ".join(cells) + " |")
        L.append("")
    n = sum(1 for _ in C)
    L.append(f"Paired t-test over seeds; Holm correction over all {n} comparisons in this table "
             "(both datasets, both oracles, three σ_to, two references, five metrics). "
             "Bold: significant after correction.\n")
    return "\n".join(L)


def headroom(data, C):
    """For each dataset and headline metric: the best oracle configuration (chosen AFTER
    seeing the test results, so optimistic) and its gain over TVDiag default."""
    out = {}
    for ds in DS:
        out[ds] = {}
        for k in HEAD:
            cands = [((o, s), C[(ds, o, s, "tvdiag", k)]) for o, _ in ORACLES for s in SIGMAS
                     if (ds, o, s, "tvdiag", k) in C]
            if not cands:
                continue
            (o, s), (d, p, n, h) = max(cands, key=lambda c: c[1][0])
            # 95% CI of the paired difference, for the best configuration
            a = {r["seed"]: r[k] for r in data[ds][("tvdiag", 0.1)]}
            b = {r["seed"]: r[k] for r in data[ds][(o, s)]}
            dd = np.array([b[i] - a[i] for i in sorted(set(a) & set(b))])
            half = (stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
                    if len(dd) > 1 and dd.std() > 0 else 0.0)
            out[ds][k] = dict(oracle=o, sigma=s, diff=d, p=p, holm=h, n=n,
                              ci=(d - half, d + half),
                              tvdiag=mean(data[ds][("tvdiag", 0.1)], k),
                              best=mean(data[ds][(o, s)], k),
                              any_sig=any(C[(ds, oo, ss, "tvdiag", k)][3] < 0.05
                                          for oo, _ in ORACLES for ss in SIGMAS
                                          if (ds, oo, ss, "tvdiag", k) in C))
    return out


def headroom_md(H):
    L = ["| dataset | metric | TVDiag default | best oracle (chosen on test) | gain | 95% CI | p | Holm |",
         "|---|---|---|---|---|---|---|---|"]
    for ds in DS:
        for k in HEAD:
            if k not in H[ds]:
                continue
            h = H[ds][k]
            L.append(f"| {NAME[ds]} | {dict(METRICS)[k]} | {h['tvdiag']:.3f} | {h['oracle']}, σ_to "
                     f"{h['sigma']:g}: {h['best']:.3f} | {h['diff']:+.3f} | "
                     f"[{h['ci'][0]:+.3f}, {h['ci'][1]:+.3f}] | {h['p']:.2f} | {h['holm']:.2f} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- figure
def figure(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
    SERIES = (("prior", "TVDiag's rule on the simplex", "#2a78d6", "o"),
              ("adadiag", "AdaDiag (learned)", "#eb6834", "s"),
              ("global", "oracle, one matrix", "#1baf7a", "D"),
              ("class", "oracle, per failure type", "#eda100", "^"))
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), facecolor=SURF)
    xs = np.arange(len(SIGMAS))
    for i, ds in enumerate(DS):
        for j, (k, lab) in enumerate((("HR@1", "HR@1 (root-cause localisation)"),
                                      ("f1", "F1 (failure-type identification)"))):
            ax = axes[i, j]
            ax.set_facecolor(SURF)
            base = data[ds][("tvdiag", 0.1)]
            if base:
                ax.axhline(mean(base, k), color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
                ax.text(xs[-1] + 0.32, mean(base, k), "TVDiag\ndefault", color=INK2,
                        fontsize=7.5, va="center", ha="left")
            for n, (kind, name, col, mk) in enumerate(SERIES):
                off = (n - 1.5) * 0.09
                m = [mean(data[ds][(kind, s)], k) for s in SIGMAS]
                ax.plot(xs + off, m, color=col, lw=2, marker=mk, ms=7, mec=SURF, mew=1.5,
                        label=name, zorder=3)
                for x, s in zip(xs + off, SIGMAS):
                    v = [r[k] for r in data[ds][(kind, s)]]
                    ax.scatter([x] * len(v), v, s=9, color=col, alpha=0.35, lw=0, zorder=2)
            ax.set_xticks(xs, [f"{s:g}" for s in SIGMAS])
            ax.set_xlim(-0.4, xs[-1] + 0.75)
            ax.set_title(f"{NAME[ds]} — {lab}", fontsize=9.5, color=INK, loc="left")
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.tick_params(colors=INK2, labelsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(MUTED)
            if i == 1:
                ax.set_xlabel("σ_to — weight of the task-oriented contrastive term", fontsize=8.5,
                              color=INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "Lines: means over seeds (GAIA 3, Hotel 5; AdaDiag at σ_to = 1 on GAIA: 1 seed). "
             "Faint dots: single seeds. Oracle matrices are fitted on the test split — upper "
             "bounds, not methods.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=SURF)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main() -> None:
    ow = json.load(open(OUT / "oracle_weights.json"))
    data = {ds: load(ds) for ds in DS}
    C = comparisons(data)
    H = headroom(data, C)
    figure(data, OUT / "oracle_headroom.png")

    n_sig = sum(1 for v in C.values() if v[3] < 0.05)
    n_sig_pos = sum(1 for v in C.values() if v[3] < 0.05 and v[0] > 0)
    md = [
        "# Oracle headroom check (README §6.12)\n",
        "Question: if the task-oriented contrastive term were weighted the way the evidence says "
        "it should be, would TVDiag do better? If even a matrix fitted on the test split cannot "
        "beat TVDiag's default, no weighting learned from training data can, and the weighting "
        "is not where the headroom is.\n",
        "## 1. The oracle matrices\n",
        "Each column is the cost of removing one modality (README §6.8, group B, measured on "
        "the test split against `ref_prior`), clipped at zero and normalised onto the simplex. "
        "`scripts/make_oracle_weights.py` → `oracle_weights.json`.\n",
        costs_md(ow), "", matrices_md(ow), "",
        "The per-failure-type matrices are applied during training by each incident's true type "
        "(known at training time). The contrastive term is not used at inference, so no label "
        "reaches the test predictions through it.\n",
        "## 2. Results\n",
        "Every run: 1 thread, 500-epoch cap, TVDiag's early stopping, seeds paired with the "
        "references (GAIA 0–2, Hotel 0–4). σ_to scales the task-oriented term only; the "
        "cross-modal term stays at σ = 0.1.\n",
        results_md(data, C),
        f"Significant after correction: {n_sig} of {len(C)} comparisons, {n_sig_pos} of them "
        "in the oracle's favour.\n",
        "## 3. Headroom — the best oracle against TVDiag default\n",
        "The best of the six oracle configurations per dataset and metric, chosen after seeing "
        "the test results. That choice is itself optimistic, so this gain is an upper bound "
        "on what any re-weighting of the task-oriented term can add.\n",
        headroom_md(H),
        "![oracle headroom](oracle_headroom.png)\n",
    ]
    (OUT / "ORACLE.md").write_text("\n".join(md))
    (OUT / "readme_table.md").write_text(headroom_md(H))
    json.dump({"headroom": {ds: {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                                     for kk, vv in v.items()} for k, v in H[ds].items()}
                            for ds in DS},
               "n_comparisons": len(C), "n_significant": n_sig,
               "n_significant_oracle_better": n_sig_pos},
              open(OUT / "summary.json", "w"), indent=1)
    print((OUT / "ORACLE.md").read_text())


if __name__ == "__main__":
    main()
