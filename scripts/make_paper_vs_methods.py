#!/usr/bin/env python3
"""README section 6.16: every method of sections 6.11-6.15 against the TVDiag paper only.

The reference is the TVDiag row of the paper's Table 4 (RCL: HR@1, HR@3, Avg@3, MRR@3) and
Table 5 (FTI: precision, recall, F1), transcribed in `adadiag.paper_numbers`. No re-run of
TVDiag is used as a reference. Nothing is trained: every number comes from committed runs.

Writes, in results/paper_vs_methods/:
  PAPER_VS_METHODS.md     the tables, one per dataset, and the notes
  table_<A|C|D>.tex       the same tables in LaTeX (booktabs)
  all_methods.csv         every method x dataset x metric: mean, sd, n, paper, gap, p, Holm p
  fig_p1_values.png       HR@1 and F1 of every method, seeds and means, paper line
  fig_p2_gap.png          gap from the paper on all seven metrics, every method and dataset

    python scripts/make_paper_vs_methods.py
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adadiag import paper_numbers as P                                 # noqa: E402

RES = ROOT / "results"
OUT = RES / "paper_vs_methods"
DS = ("gaia", "sockshop", "hotel")
LETTER = {"gaia": "A", "sockshop": "C", "hotel": "D"}
NAME = {"gaia": "A · GAIA", "sockshop": "C · Sock Shop", "hotel": "D · Hotel"}
METRICS = (("HR@1", lambda r: r["HR@1"]), ("HR@3", lambda r: r["HR@3"]),
           ("Avg@3", lambda r: (r["HR@1"] + r["HR@2"] + r["HR@3"]) / 3),
           ("MRR@3", lambda r: r["MRR@3"]), ("P", lambda r: r["precision"]),
           ("R", lambda r: r["recall"]), ("F1", lambda r: r["f1"]))


def paper(ds):
    t4 = dict(zip(P.TABLE4_COLS, P.TABLE4[LETTER[ds]][-1][3:]))
    p, r, f = P.TABLE5[LETTER[ds]][-1][3:]
    assert P.TABLE4[LETTER[ds]][-1][1] == "TVDiag" and P.TABLE5[LETTER[ds]][-1][1] == "TVDiag"
    return {**t4, "P": p, "R": r, "F1": f}


# --------------------------------------------------------------------------- sources
def _load(pattern):
    recs = []
    for f in sorted(glob.glob(str(RES / pattern))):
        d = json.load(open(f))
        recs += d if isinstance(d, list) else []
    return recs


def _seeds(rs):
    out = {}
    for r in rs:
        out.setdefault(r["seed"], r)          # one record per seed
    return [out[s] for s in sorted(out)]


def ablation(name):
    return lambda ds: _seeds([r for r in _load(f"ablation_{ds}_s*.json") if r["name"] == name])


def where(pattern, **flt):
    return lambda ds: _seeds([r for r in _load(pattern.format(ds=ds))
                              if all(r.get(k) == v for k, v in flt.items())])


def first(*srcs):
    def f(ds):
        for s in srcs:
            rs = s(ds)
            if rs:
                return rs
        return []
    return f


def only(dsets, src):
    return lambda ds: src(ds) if ds in dsets else []


GH = ("gaia", "hotel")
SOCK_ADA = where("paper_comparison/runs_{ds}.json", arm="v3_incident", variant="default")
#: (key, label, group, source). Groups: colour in the figures, a block in the tables.
METHODS = [
    ("ada", "AdaDiag", "AdaDiag",
     first(only(GH, ablation("ref_adadiag")), only(("sockshop",), SOCK_ADA))),
    ("ada_aligned", "AdaDiag, random stream aligned (§6.15)", "AdaDiag",
     first(only(GH, where("controlled/rng_{ds}.json")), only(("sockshop",), SOCK_ADA))),
    ("ada_fid", "AdaDiag, TVDiag-fidelity loop (§6.11)", "AdaDiag",
     where("fidelity/fidelity_{ds}_*.json", arm="v3_incident")),
    ("trace_third", "AdaDiag, trace fixed at 1/3 (§6.14)", "AdaDiag",
     only(GH, where("trace_fixed/runs_{ds}*.json", variant="trace_third", sigma_to=0.1))),
    ("trace_tvdiag", "AdaDiag, trace fixed at TVDiag's role (§6.14)", "AdaDiag",
     only(GH, where("trace_fixed/runs_{ds}*.json", variant="trace_tvdiag", sigma_to=0.1))),
    ("soft", "Softened prior .9/.9/.1, .1/.1/.9 (§6.11)", "Fixed weights",
     where("fidelity/soft_{ds}.json")),
    ("pos", "1 · weight positive pairs, β = 1 (§6.13)", "Contrastive weighting",
     only(GH, where("cl_variants/runs_{ds}*.json", variant="pos", sigma_to=0.1))),
    ("pos_strong", "1 · weight positive pairs, β = 1/τ", "Contrastive weighting",
     only(GH, where("cl_variants/runs_{ds}*.json", variant="pos_strong", sigma_to=0.1))),
    ("neg", "2 · weight negative samples, β = 1", "Contrastive weighting",
     only(GH, where("cl_variants/runs_{ds}*.json", variant="neg", sigma_to=0.1))),
    ("neg_strong", "2 · weight negative samples, β = 1/τ", "Contrastive weighting",
     only(GH, where("cl_variants/runs_{ds}*.json", variant="neg_strong", sigma_to=0.1))),
    ("importance", "3 · weight by sample importance", "Contrastive weighting",
     only(GH, where("cl_variants/runs_{ds}*.json", variant="importance", sigma_to=0.1))),
    ("align", "4 · weights from the diagnosis losses", "Contrastive weighting",
     only(GH, where("cl_variants/runs_{ds}*.json", variant="align", sigma_to=0.1))),
    ("curriculum", "5 · curriculum (easy → hard)", "Contrastive weighting",
     only(GH, where("cl_variants/runs_{ds}*.json", variant="curriculum", sigma_to=0.1))),
    ("oracle", "Oracle, one matrix (§6.12, fitted on test)", "Upper bound",
     only(GH, where("oracle/runs_{ds}.json", oracle="global", sigma_to=0.1))),
    ("oracle_cls", "Oracle, per failure type (fitted on test)", "Upper bound",
     only(GH, where("oracle/runs_{ds}.json", oracle="class", sigma_to=0.1))),
]
GROUP_COLOR = {"AdaDiag": "#2a78d6", "Fixed weights": "#eb6834",
               "Contrastive weighting": "#1baf7a", "Upper bound": "#8a8984"}


# --------------------------------------------------------------------------- statistics
def holm(ps):
    ps = np.asarray(ps, float)
    out = np.full_like(ps, np.nan)
    ok = ~np.isnan(ps)
    run = 0.0
    for r, i in enumerate(np.argsort(np.where(ok, ps, np.inf))[: ok.sum()]):
        run = max(run, min(1.0, (ok.sum() - r) * ps[i]))
        out[i] = run
    return out


def collect():
    rows = []
    for key, label, group, src in METHODS:
        for ds in DS:
            rs = src(ds)
            if not rs:
                continue
            pp = paper(ds)
            for m, get in METRICS:
                v = np.array([get(r) for r in rs], float)
                sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
                if len(v) > 1 and sd > 0:
                    p = float(stats.ttest_1samp(v, pp[m]).pvalue)
                else:
                    p = float("nan")
                rows.append(dict(key=key, method=label, group=group, dataset=ds, metric=m,
                                 n=len(v), mean=float(v.mean()), sd=sd, paper=pp[m],
                                 gap=float(v.mean() - pp[m]), p=p, values=v.tolist(),
                                 threads=sorted({(r.get("env") or {}).get("threads", 2)
                                                 for r in rs}),
                                 epochs_cap=None))
    adj = holm([r["p"] for r in rows])
    for r, h in zip(rows, adj):
        r["holm"] = float(h)
    return rows


# --------------------------------------------------------------------------- tables
def mark(r):
    """▲ / ▼: different from the paper after Holm correction; †: p < 0.05 before it."""
    if not np.isnan(r["holm"]) and r["holm"] < 0.05:
        return " ▲" if r["gap"] > 0 else " ▼"
    if not np.isnan(r["p"]) and r["p"] < 0.05:
        return " †"
    return ""


def table_md(rows, ds):
    pp = paper(ds)
    cols = [m for m, _ in METRICS]
    L = [f"**{NAME[ds]}**\n",
         "| method | seeds | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols),
         "| **TVDiag (paper)** | — | " + " | ".join(f"**{pp[m]:.3f}**" for m in cols) + " |"]
    group = None
    for key, label, g, _ in METHODS:
        cell = {r["metric"]: r for r in rows if r["key"] == key and r["dataset"] == ds}
        if not cell:
            continue
        if g != group:
            L.append(f"| *{g}* | | " + " | ".join("" for _ in cols) + " |")
            group = g
        n = cell["HR@1"]["n"]
        L.append(f"| {label} | {n} | " + " | ".join(
            f"{cell[m]['mean']:.3f} ({cell[m]['gap']:+.3f}){mark(cell[m])}" for m in cols) + " |")
    return "\n".join(L) + "\n"


def table_tex(rows, ds):
    pp = paper(ds)
    cols = [m for m, _ in METRICS]
    esc = lambda s: (s.replace("§", r"\S").replace("β", r"$\beta$").replace("τ", r"$\tau$")
                     .replace("→", r"$\rightarrow$").replace(" · ", ": ").replace("&", r"\&"))
    L = [r"\begin{table}[t]", r"\centering\small",
         rf"\caption{{Dataset {LETTER[ds]} ({NAME[ds].split(' · ')[1]}): every method against "
         r"TVDiag's published results (Tables 4--5). Mean over seeds, gap to the paper in "
         r"parentheses; $\blacktriangle$/$\blacktriangledown$: different from the paper after "
         r"Holm correction (one-sample $t$-test over seeds); $^{\dagger}$: $p<0.05$ before "
         r"correction.}",
         rf"\label{{tab:paper-vs-methods-{LETTER[ds]}}}",
         r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{l c " + " ".join("c" for _ in cols) + "}", r"\toprule",
         r"Method & $n$ & " + " & ".join(c.replace("@", "@") for c in cols) + r" \\",
         r"\midrule",
         r"\textbf{TVDiag (paper)} & -- & " + " & ".join(rf"\textbf{{{pp[m]:.3f}}}" for m in cols) + r" \\",
         r"\midrule"]
    group = None
    for key, label, g, _ in METHODS:
        cell = {r["metric"]: r for r in rows if r["key"] == key and r["dataset"] == ds}
        if not cell:
            continue
        if g != group:
            if group is not None:
                L.append(r"\addlinespace")
            L.append(rf"\multicolumn{{{len(cols) + 2}}}{{l}}{{\emph{{{esc(g)}}}}} \\")
            group = g
        def c(m):
            r = cell[m]
            g = f"{r['gap']:+.3f}".replace("-", "$-$").replace("+", "$+$")
            s = rf"{r['mean']:.3f} ({g})"
            mk = mark(r)
            if "▲" in mk:
                s += r"$^{\blacktriangle}$"
            elif "▼" in mk:
                s += r"$^{\blacktriangledown}$"
            elif "†" in mk:
                s += r"$^{\dagger}$"
            return s
        L.append(f"{esc(label)} & {cell['HR@1']['n']} & " + " & ".join(c(m) for m in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L) + "\n"


def summary_md(rows):
    """HR@1 and F1 gaps only, all datasets side by side: the compact table the README quotes."""
    L = ["| method | A · GAIA HR@1 / F1 | C · Sock Shop HR@1 / F1 | D · Hotel HR@1 / F1 |",
         "|---|---|---|---|",
         "| **TVDiag (paper)** | " + " | ".join(
             f"**{paper(ds)['HR@1']:.3f} / {paper(ds)['F1']:.3f}**" for ds in DS) + " |"]
    for key, label, g, _ in METHODS:
        cells = []
        for ds in DS:
            c = {r["metric"]: r for r in rows if r["key"] == key and r["dataset"] == ds}
            cells.append("—" if not c else
                         f"{c['HR@1']['gap']:+.3f}{mark(c['HR@1'])} / {c['F1']['gap']:+.3f}{mark(c['F1'])}")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def counts(rows):
    """For each method: of the paper numbers it was compared on, how many it reaches."""
    out = []
    for key, label, g, _ in METHODS:
        rs = [r for r in rows if r["key"] == key]
        if not rs:
            continue
        out.append(dict(method=label, n=len(rs), reach=sum(r["gap"] >= 0 for r in rs),
                        above=sum(mark(r) == " ▲" for r in rs),
                        below=sum(mark(r) == " ▼" for r in rs),
                        raw_above=sum(r["p"] < 0.05 and r["gap"] > 0 for r in rs),
                        raw_below=sum(r["p"] < 0.05 and r["gap"] < 0 for r in rs)))
    return out


# --------------------------------------------------------------------------- figures
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"


def fig_values(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    labels = [lab for _, lab, _, _ in METHODS]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.2), facecolor=SURF, sharey=True)
    for i, m in enumerate(("HR@1", "F1")):
        for j, ds in enumerate(DS):
            ax = axes[i, j]
            ax.set_facecolor(SURF)
            pv = paper(ds)[m]
            ax.axvline(pv, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=1)
            ax.text(pv, len(METHODS) - 0.35, " TVDiag (paper)", color=INK, fontsize=7.5,
                    va="bottom", ha="left")
            for k, (key, lab, g, _) in enumerate(METHODS):
                y = len(METHODS) - 1 - k
                c = [r for r in rows if r["key"] == key and r["dataset"] == ds and r["metric"] == m]
                if not c:
                    ax.text(pv, y, "not run", color=MUTED, fontsize=6.5, va="center", ha="center",
                            alpha=0.8)
                    continue
                r = c[0]
                col = GROUP_COLOR[g]
                ax.scatter(r["values"], [y] * len(r["values"]), s=10, color=col, alpha=0.35,
                           lw=0, zorder=2)
                ax.scatter([r["mean"]], [y], s=48, color=col, edgecolor=SURF, lw=1.5, zorder=3)
            ax.set_title(f"{NAME[ds]} — {m}", fontsize=10, color=INK, loc="left")
            ax.grid(axis="x", color=GRID, lw=0.8)
            ax.tick_params(colors=INK2, labelsize=8)
            ax.tick_params(axis="y", length=0)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(MUTED)
            ax.set_ylim(-0.7, len(METHODS) + 0.2)
    axes[0, 0].set_yticks(range(len(METHODS)), labels[::-1], fontsize=8)
    axes[1, 0].set_yticks(range(len(METHODS)), labels[::-1], fontsize=8)
    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=7, label=g)
               for g, c in GROUP_COLOR.items()]
    handles.append(Line2D([], [], color=INK, ls=(0, (4, 3)), label="TVDiag, published"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.004, "Large dot: mean over seeds. Faint dots: single seeds. Dashed line: TVDiag's "
             "published number (paper Tables 4–5). 'not run': the method was not run on that dataset.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(path, dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_gap(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    # diverging blue (above the paper) <-> red (below), neutral grey midpoint
    cmap = LinearSegmentedColormap.from_list(
        "gap", ["#9b2423", "#e34948", "#f6c4c3", "#f0efec", "#b7d3f6", "#2a78d6", "#0d366b"])
    cols = [(ds, m) for ds in DS for m, _ in METRICS]
    M = np.full((len(METHODS), len(cols)), np.nan)
    S = [["" for _ in cols] for _ in METHODS]
    for i, (key, *_r) in enumerate(METHODS):
        for j, (ds, m) in enumerate(cols):
            c = [r for r in rows if r["key"] == key and r["dataset"] == ds and r["metric"] == m]
            if c:
                M[i, j] = c[0]["gap"]
                S[i][j] = mark(c[0]).strip()
    lim = float(np.nanmax(np.abs(M)))
    fig, ax = plt.subplots(figsize=(15, 7.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "·", ha="center", va="center", color=MUTED, fontsize=9)
                continue
            dark = abs(M[i, j]) > 0.55 * lim
            ax.text(j, i, f"{M[i, j]:+.3f}{S[i][j]}", ha="center", va="center", fontsize=6.6,
                    color="#ffffff" if dark else INK)
    ax.set_xticks(range(len(cols)), [m for _, m in cols], fontsize=8, color=INK2)
    ax.set_yticks(range(len(METHODS)), [lab for _, lab, _, _ in METHODS], fontsize=8, color=INK2)
    for k, ds in enumerate(DS):
        x0 = k * len(METRICS)
        ax.text(x0 + (len(METRICS) - 1) / 2, -1.15, NAME[ds], ha="center", fontsize=10, color=INK)
        if k:
            ax.axvline(x0 - 0.5, color=SURF, lw=4)
        ax.axvline(x0 + 3.5, color=SURF, lw=1.5)          # RCL | FTI within a dataset
    for i in range(1, len(METHODS)):
        if METHODS[i][2] != METHODS[i - 1][2]:
            ax.axhline(i - 0.5, color=SURF, lw=4)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(METHODS) - 0.5, -1.6)
    fig.text(0.5, 0.01, "Each cell: method mean − TVDiag's published value (paper Tables 4–5). "
             "Blue: above the paper, red: below, grey: equal. †: p < 0.05 before correction "
             "(none survives Holm). ·: not run. Left four columns of each dataset: RCL; right three: FTI.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURF)
    plt.close(fig)


def rerun_gap() -> str:
    """This repository's TVDiag (v0_raw) re-run against the paper, HR@1 / F1 -- for the note."""
    src = {"gaia": where("paper_comparison/sweep_gaia_tv*.json", arm="v0_raw", variant="default"),
           "hotel": where("paper_comparison/ablate_hotel.json", arm="v0_raw", variant="default"),
           "sockshop": where("paper_comparison/runs_{ds}.json", arm="v0_raw", variant="default")}
    parts = []
    for ds in DS:
        rs = [r for r in src[ds](ds) if not r.get("tag")]
        h = np.mean([r["HR@1"] for r in rs]) - paper(ds)["HR@1"]
        f = np.mean([r["f1"] for r in rs]) - paper(ds)["F1"]
        parts.append(f"{h:+.3f} HR@1 / {f:+.3f} F1 on {NAME[ds].split(' · ')[1]} ({len(rs)} seeds)")
    return ", ".join(parts)


# --------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect()
    fig_values(rows, OUT / "fig_p1_values.png")
    fig_gap(rows, OUT / "fig_p2_gap.png")
    with open(OUT / "all_methods.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "group", "dataset", "metric", "n", "mean", "sd",
                                          "paper", "gap", "p", "holm"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 6) if isinstance(r[k], float) else r[k]) for k in w.fieldnames})
    for ds in DS:
        (OUT / f"table_{LETTER[ds]}.tex").write_text(table_tex(rows, ds))
    cnt = counts(rows)
    n_above = sum(1 for r in rows if mark(r) == " ▲")
    n_below = sum(1 for r in rows if mark(r) == " ▼")
    raw = [r for r in rows if not np.isnan(r["p"]) and r["p"] < 0.05]
    n_raw_up = sum(r["gap"] > 0 for r in raw)
    n_raw_dn = len(raw) - n_raw_up
    md = ["# Every method against the TVDiag paper (README §6.16)\n",
          "Reference: the **TVDiag row of the paper's Table 4** (RCL: HR@1, HR@3, Avg@3, MRR@3) "
          "and **Table 5** (FTI: precision, recall, F1, the paper's definition) only. No re-run of "
          "TVDiag is used. Each cell: mean over seeds (gap to the paper). ▲ / ▼: different from "
          "the paper after Holm correction over all "
          f"{len(rows)} cells (one-sample t-test over seeds against the published value); "
          "†: p < 0.05 before correction.\n",
          "## Summary: HR@1 and F1 gap to the paper\n", summary_md(rows),
          "## Full tables (Tables 4 and 5 together, per dataset)\n",
          *[table_md(rows, ds) for ds in DS],
          "## How many of the paper's numbers each method reaches\n",
          "| method | paper numbers compared | reached (gap ≥ 0) | above the paper: p < 0.05 before (after) Holm | below the paper: p < 0.05 before (after) Holm |",
          "|---|---|---|---|---|",
          *[f"| {c['method']} | {c['n']} | {c['reach']} | {c['raw_above']} ({c['above']}) | "
            f"{c['raw_below']} ({c['below']}) |" for c in cnt],
          "",
          f"Across all {len(rows)} cells: {n_above} significantly above the paper and {n_below} "
          f"significantly below after Holm correction; before it, {n_raw_up} above and "
          f"{n_raw_dn} below (p < 0.05).\n",
          "## Notes\n",
          "- **Settings.** GAIA and Hotel rows come from the 1-thread, 500-epoch runs of "
          "§6.12–6.15 (GAIA 3 seeds, Hotel 5), except the softened prior (committed settings, "
          "2 threads, 5 seeds) and the fidelity-loop AdaDiag (§6.11, 1 thread, 5 seeds). "
          "Sock Shop rows come from §6.9 / §6.11 (5 seeds). Only four methods were run on Sock Shop.",
          "- **The paper's spread is not published**, so the test asks whether a method's seed "
          "mean differs from one fixed number. That is generous to both sides: with 3–5 seeds, "
          "most gaps under ~0.02 cannot be told from zero.",
          "- **The re-implementation's own gap.** A method's gap to the paper mixes its effect "
          "with how closely this repository reproduces TVDiag (§6.1, §6.9): this repository's "
          "TVDiag re-run sits at " + rerun_gap() + " against the paper (GAIA and Hotel: the "
          "1-thread runs; Sock Shop: §6.9). It is not a row of the tables, which compare with "
          "the paper only. The method-against-method comparisons with the same seeds are in "
          "§6.12–6.15.",
          "- **The oracle rows are upper bounds**, fitted on the test split (§6.12), not methods.\n",
          "![values](fig_p1_values.png)\n", "![gap](fig_p2_gap.png)\n",
          "LaTeX: `table_A.tex`, `table_C.tex`, `table_D.tex` (need `booktabs`, `amssymb`, "
          "`graphicx`). Every cell with its p-values: `all_methods.csv`.\n"]
    (OUT / "PAPER_VS_METHODS.md").write_text("\n".join(md))
    (OUT / "readme_table.md").write_text(summary_md(rows))
    json.dump({"n_cells": len(rows), "above": n_above, "below": n_below,
               "raw_above": int(n_raw_up), "raw_below": int(n_raw_dn), "counts": cnt},
              open(OUT / "summary.json", "w"), indent=1)
    print((OUT / "PAPER_VS_METHODS.md").read_text())


if __name__ == "__main__":
    main()
