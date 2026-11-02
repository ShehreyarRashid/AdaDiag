#!/usr/bin/env python3
"""Tables and figures that set this re-implementation beside the TVDiag paper.

Reads the published numbers (adadiag/paper_numbers.py), the committed main runs
(results/gaia.json, results/hotel.json) and the comparison runs in results/paper_comparison/
(scripts/run_paper_comparison.py), and writes, into results/paper_comparison/:

    PAPER_COMPARISON.md       every table, the figures, and what each comparison shows
    fig11_task_routing.png    fig13_worked_example.png   fig14_per_failure_type.png
    fig15_alert_fields.png    fig16_hyperparameters.png  fig17_modalities.png
    fig18_modality_shapley.png
    readme_table.md           the compact table quoted in README §6.9

    python scripts/make_paper_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.patches import Patch                                # noqa: E402
from adadiag import paper_numbers as P                              # noqa: E402
from adadiag.weighting import PRIOR_RAW                             # noqa: E402

OUT = ROOT / "results" / "paper_comparison"
RUN_FILES = ("runs_*.json", "sweep_*.json", "ablate_*.json", "fig15_*.json")

# ---- palette: validated categorical slots 1-3 (dataviz reference palette), recessive ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
MOD_COL = {"metric": BLUE, "trace": ORANGE, "log": AQUA}
PAPER_HATCH = "///"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "legend.fontsize": 8.5,
})

LETTER = {"gaia": "A", "sockshop": "C", "hotel": "D"}
DSNAME = {"gaia": "GAIA", "sockshop": "Sock Shop", "hotel": "Hotel Reservation"}
ARMNAME = {"v0_raw": "TVDiag (re-run)", "v0_simplex": "TVDiag prior on the simplex (V0-simplex)",
           "v3_incident": "AdaDiag (V3-incident)"}
SHORT = {"v0_raw": "TVDiag re-run", "v0_simplex": "V0-simplex", "v3_incident": "AdaDiag"}


def pretty(t, n=22):
    """Failure-type label for a figure: GAIA's are bracketed and underscored."""
    t = t.strip("[]").replace("_", " ")
    return t if len(t) <= n else t[:n - 1] + "…"


# =========================================================================== data
def avg3(r):
    return (r["HR@1"] + r["HR@2"] + r["HR@3"]) / 3


GET = {"HR@1": lambda r: r["HR@1"], "HR@3": lambda r: r["HR@3"], "HR@5": lambda r: r["HR@5"],
       "Avg@3": avg3, "MRR@3": lambda r: r["MRR@3"], "Precision": lambda r: r["precision"],
       "Recall": lambda r: r["recall"], "F1": lambda r: r["f1"],
       "macro-F1": lambda r: r["macro_f1"], "F1 (sklearn)": lambda r: r["f1_sklearn_weighted"]}


def load_runs() -> list[dict]:
    recs = []
    for pat in RUN_FILES:
        for f in sorted(OUT.glob(pat)):
            recs += json.load(open(f))
    return recs


def committed(ds: str) -> dict:
    f = ROOT / "results" / f"{ds}.json"
    return json.load(open(f)) if f.exists() else {}


def sel(recs, ds, arm, variant="default", tag="", threads=None) -> list[dict]:
    """One record per seed, sorted by seed."""
    by = {}
    for r in recs:
        if (r["dataset"] == ds and r["arm"] == arm and r.get("variant") == variant
                and r.get("tag", "") == tag
                and (threads is None or r.get("env", {}).get("threads") == threads)):
            by.setdefault(r["seed"], r)
    return [by[s] for s in sorted(by)]


def vals(rs, m):
    return np.array([GET[m](r) for r in rs], dtype=float)


def ms(rs, m, d=3):
    v = vals(rs, m)
    if len(v) == 0:
        return "—"
    if len(v) == 1:
        return f"{v[0]:.{d}f}"
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"


def mean(rs, m):
    v = vals(rs, m)
    return float(v.mean()) if len(v) else float("nan")


def paired(a, b, m):
    """Paired t-test over common seeds: returns (mean diff b - a, p)."""
    sa = {r["seed"]: GET[m](r) for r in a}
    sb = {r["seed"]: GET[m](r) for r in b}
    common = sorted(set(sa) & set(sb))
    if len(common) < 2:                       # too few pairs to test: difference of means
        if not sa or not sb:
            return float("nan"), float("nan")
        return float(np.mean(list(sb.values())) - np.mean(list(sa.values()))), float("nan")
    x = np.array([sa[s] for s in common]); y = np.array([sb[s] for s in common])
    d = y - x
    if np.allclose(d, d[0]):
        return float(d.mean()), (1.0 if d[0] == 0 else 0.0)
    return float(d.mean()), float(stats.ttest_rel(y, x).pvalue)


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    out = np.full_like(ps, np.nan)
    ok = ~np.isnan(ps)
    idx = np.argsort(np.where(ok, ps, np.inf))
    m, running = ok.sum(), 0.0
    for rank, i in enumerate(idx[:m]):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        out[i] = running
    return out


def fp(p):
    if p != p:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def sgn(x, d=3):
    return "—" if x != x else f"{x:+.{d}f}"


def table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    sep = ["---:" if a == "r" else (":---:" if a == "c" else "---") for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return name


# =========================================================================== Table 3
def dataset_stats(ds: str) -> dict:
    import csv
    d = ROOT / "data" / ds
    rows = list(csv.DictReader(open(d / "label.csv")))
    nodes = json.load(open(d / "raw" / "nodes.json"))
    inst = list(next(iter(nodes.values())))

    def service(n):
        if ds == "gaia":
            return n.rstrip("0123456789")
        return n.rsplit("-", 1)[0]

    used = {str(r["index"]) for r in rows}
    al = {m: json.load(open(d / "raw" / f"{m}s.json")) for m in ("metric", "trace", "log")}
    n_alerts = {m: sum(len(v) for k, v in al[m].items() if k in used) for m in al}
    mt = {(e[0], e[-2]) for k, v in al["metric"].items() if k in used for e in v}
    tt = {tuple(e[:3]) for k, v in al["trace"].items() if k in used for e in v}
    lk = {e[1] for k, v in al["log"].items() if k in used for e in v}
    from adadiag import data
    dd = data.load(ds)
    return dict(microservices=len({service(n) for n in inst}), instances=len(inst),
                failure_types=len(dd.types), incidents=len(dd.split),
                train=len(dd.train_idx), test=len(dd.test_idx),
                alerts=n_alerts, types=dict(metric=len(mt), trace=len(tt), log=len(lk)))


def table3() -> str:
    rows = []
    for L in "ABCD":
        p = P.TABLE3[L]
        rows.append([f"{L} · {P.DATASET_NAME[L]}", "paper", p["microservices"], p["instances"],
                     p["failure_types"], "—",
                     " / ".join(f"{p['records'][m]:,}" for m in ("metric", "trace", "log")),
                     " / ".join(f"{p['types'][m]:,}" for m in ("metric", "trace", "log"))])
        ds = P.DATASET_KEY[L]
        if ds == "aiops22":
            rows.append(["", "this repository", "not distributed", "", "", "", "", ""])
            continue
        s = dataset_stats(ds)
        rows.append(["", "this repository (`data/`)", s["microservices"], s["instances"],
                     s["failure_types"], f"{s['incidents']} ({s['train']}/{s['test']})",
                     " / ".join(f"{s['alerts'][m]:,}" for m in ("metric", "trace", "log")),
                     " / ".join(f"{s['types'][m]:,}" for m in ("metric", "trace", "log"))])
    return table(["Dataset", "Source", "Micro-services", "Instances", "Failure types",
                  "Incidents (train/test)", "Records: metric / trace / log",
                  "Types: metric / trace / LogKeys"], rows,
                 ["l", "l", "r", "r", "r", "r", "r", "r"])


# =========================================================================== Tables 4-5
def main_rows(recs, ds):
    """{arm: records} for the default setting, 2 threads (= results/<ds>.json)."""
    out = {}
    com = committed(ds)
    for arm in ("v0_raw", "v0_simplex", "v3_incident"):
        rs = sel(recs, ds, arm, threads=2)
        if not rs and arm in com:
            rs = [dict(r, variant="default") for r in com[arm]]
        out[arm] = rs
    return out


def repro_check(recs) -> list[str]:
    """The comparison runs re-train the committed arms: say whether they match exactly."""
    lines = []
    for ds in ("gaia", "hotel"):
        com = committed(ds)
        for arm in ("v0_raw", "v3_incident"):
            new = sel(recs, ds, arm, threads=2)
            old = {r["seed"]: r for r in com.get(arm, [])}
            if not new:
                continue
            same = all(all(r[k] == old[r["seed"]][k] for k in ("HR@1", "HR@3", "MRR@3", "f1",
                                                              "macro_f1", "epochs"))
                       for r in new if r["seed"] in old)
            lines.append(f"{DSNAME[ds]} {ARMNAME[arm]}: {len(new)} seeds, "
                         f"{'identical to' if same else 'DIFFERENT from'} results/{ds}.json")
    return lines


def table45(recs, which: str) -> tuple[str, dict]:
    if which == "rcl":
        cols, paper = ["HR@1", "HR@3", "Avg@3", "MRR@3"], P.TABLE4
    else:
        cols, paper = ["Precision", "Recall", "F1"], P.TABLE5
    blocks, facts = [], {}
    for L in "ACD":
        ds = P.DATASET_KEY[L]
        arms = main_rows(recs, ds)
        rows = []
        for entry in paper[L]:
            mod, meth, pv, *v = entry
            name = f"**{meth} (paper)**" if meth == "TVDiag" else meth
            rows.append([name, mod, pv] + [f"{x:.3f}" for x in v])
        tv_paper = dict(zip(cols, paper[L][-1][3:]))
        best_base = {c: max(e[3 + i] for e in paper[L][:-1]) for i, c in enumerate(cols)}
        ps = []
        for arm in ("v0_raw", "v0_simplex", "v3_incident"):
            rs = arms[arm]
            if not rs:
                continue
            rows.append([f"{ARMNAME[arm]}, {len(rs)} seeds", "Multimodal", "—"]
                        + [ms(rs, c) for c in cols])
        tv, ada = arms["v0_raw"], arms["v3_incident"]
        if tv and which == "fti":
            rows.append(["*TVDiag (re-run), F1 the baselines' way ²*", "", "", "", "",
                         ms(tv, "F1 (sklearn)")])
        if tv:
            rows.append(["*re-run − paper*", "", ""]
                        + [sgn(mean(tv, c) - tv_paper[c]) for c in cols])
        if tv and ada:
            diffs = [paired(tv, ada, c) for c in cols]
            ps = [p for _, p in diffs]
            rows.append(["*AdaDiag − re-run (paired p)*", "", ""]
                        + [f"{sgn(d)} ({fp(p)})" for d, p in diffs])
        blocks.append(f"**{L} · {DSNAME[ds]}**\n\n" + table(
            ["Method", "Modality", "p-value (paper)"] + cols, rows,
            ["l", "l", "r"] + ["r"] * len(cols)))
        facts[L] = dict(paper=tv_paper, best_base=best_base,
                        rerun={c: mean(tv, c) for c in cols} if tv else {},
                        ada={c: mean(ada, c) for c in cols} if ada else {},
                        simplex={c: mean(arms["v0_simplex"], c) for c in cols}
                        if arms["v0_simplex"] else {},
                        rerun_min={c: float(vals(tv, c).min()) for c in cols} if tv else {},
                        rerun_max={c: float(vals(tv, c).max()) for c in cols} if tv else {},
                        p=dict(zip(cols, ps)),
                        f1_sklearn=mean(tv, "F1 (sklearn)") if tv and which == "fti" else None)
    if which == "fti":
        below = [(L, e[1], e[5] - 2 * e[3] * e[4] / (e[3] + e[4])) for L in "ABCD"
                 for e in paper[L]]
        facts["f1_def"] = dict(
            tvdiag_max_gap=max(abs(g) for L, m, g in below if m == "TVDiag"),
            baselines_below=[(L, m, g) for L, m, g in below if m != "TVDiag" and g < -0.005],
            n_baselines=sum(1 for L, m, g in below if m != "TVDiag"))
    return "\n\n".join(blocks), facts


# =========================================================================== Table 6
ABL = [("no_aug", "w/o AUG"), ("no_to", "w/o TO"), ("no_cm", "w/o CM"), ("no_dw", "w/o DW"),
       ("default", "TVDiag")]
M4 = ["HR@1", "HR@3", "Avg@3", "F1"]


#: dataset -> True once Table 6 has every variant and its reference at full seed count
complete: dict = {}


def table6(recs) -> tuple[str, dict]:
    blocks, facts = [], {}
    # A: paper vs re-run
    full = {ds: sel(recs, ds, "v0_raw", "default", threads=1) for ds in ("gaia", "hotel")}
    for ds in ("gaia", "hotel"):
        if not full[ds]:
            continue
        comps = [(v, sel(recs, ds, "v0_raw", v, threads=1)) for v, _ in ABL[:-1]]
        comps = [(v, rs) for v, rs in comps if rs]
        raw = {(v, m): paired(full[ds], rs, m) for v, rs in comps for m in M4}
        keys = list(raw)
        adj = dict(zip(keys, holm([raw[k][1] for k in keys])))
        rows = []
        for v, lab in ABL:
            rs = full[ds] if v == "default" else sel(recs, ds, "v0_raw", v, threads=1)
            if not rs:
                continue
            row = [lab]
            if ds == "gaia":
                pv = P.TABLE6[lab]["A"]
                pref = P.TABLE6["TVDiag"]["A"]
                row += [f"{x:.3f}" if v == "default" else f"{x:.3f} ({x - y:+.3f})"
                        for x, y in zip(pv, pref)]
            for m in M4:
                if v == "default":
                    row.append(ms(rs, m))
                else:
                    d, _ = raw[(v, m)]
                    star = "**" if adj[(v, m)] < 0.05 else ""
                    row.append(f"{mean(rs, m):.3f} ({star}{d:+.3f}{star})")
            rows.append(row)
        n = len(full[ds])
        if ds == "gaia":
            head = ["Variant"] + [f"paper A · {m}" for m in M4] + [f"re-run A · {m}" for m in M4]
        else:
            head = ["Variant"] + [f"re-run D · {m}" for m in M4]
        blocks.append(f"**{LETTER[ds]} · {DSNAME[ds]}** — re-run: TVDiag as shipped "
                      f"(V0-raw), {n} seed{'s' if n != 1 else ''}\n\n" + table(head, rows))
        facts[ds] = {v: {m: (raw[(v, m)][0], adj[(v, m)], raw[(v, m)][1]) for m in M4}
                     for v, _ in comps}
        want = {"gaia": 3, "hotel": 5}[ds]
        complete[ds] = len(full[ds]) >= want and len(comps) == 4 and all(
            len(rs) >= want for _, rs in comps)
    return "\n\n".join(blocks), facts


# =========================================================================== Table 7
def table7(recs) -> tuple[str, dict]:
    rows, facts = [], {}
    run = {("gaia", "Trace"): "drop:metric,log", ("gaia", "Log"): "drop:metric,trace",
           ("hotel", "Trace"): "drop:metric,log", ("hotel", "Log"): "drop:metric,trace"}
    for L, ds in (("A", "gaia"), ("D", "hotel")):
        paper = {(e[0], e[1]): e[2:] for e in P.TABLE7.get(L, [])}
        for ext, mod in (("IForest", "Trace"), ("3-sigma", "Trace"), ("Rule-based", "Log"),
                         ("Sampling", "Log")):
            pv = paper.get((ext, mod))
            own = ext in ("IForest", "Rule-based")
            rs = sel(recs, ds, "v0_raw", "default", tag=run[(ds, mod)], threads=1) if own else []
            row = [f"{L} · {DSNAME[ds]}", ext, mod]
            row += [f"{x:.3f}" for x in pv] if pv else ["—"] * 4
            if own and rs:
                row += [ms(rs, m) for m in M4]
                facts[(L, ext)] = {m: mean(rs, m) for m in M4}
            elif own:
                row += ["pending"] * 4
            else:
                row += ["not reproducible ¹"] + [""] * 3
            rows.append(row)
    head = (["Dataset", "Extractor", "Modality"] + [f"paper · {m}" for m in M4]
            + [f"re-run · {m}" for m in M4])
    return table(head, rows, ["l", "l", "l"] + ["r"] * 8), facts


# =========================================================================== Fig. 11
def fig11(recs) -> tuple[str, dict]:
    dss = [d for d in ("gaia", "sockshop", "hotel") if sel(recs, d, "v3_incident", threads=2)]
    if not dss:
        return "", {}
    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 1 + len(dss), height_ratios=[1, 1.15], hspace=0.55, wspace=0.35)
    mats = [("TVDiag: the fixed rule", np.array(PRIOR_RAW))]
    facts = {}
    for ds in dss:
        W = np.mean([np.array(r["weights"]) for r in sel(recs, ds, "v3_incident", threads=2)], 0)
        mats.append((f"AdaDiag, learned · {LETTER[ds]} {DSNAME[ds]}", W))
        facts[ds] = W
    for k, (title, W) in enumerate(mats):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(W, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(3):
            for j in range(2):
                ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if W[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["RCL", "FTI"]); ax.set_yticks([0, 1, 2], ["metric", "trace", "log"])
        ax.set_title(title, fontsize=8.5)
    pairs = [(m, t) for m in ("metric", "trace", "log") for t in ("rcl", "fti")]
    targeted = {("metric", "rcl"), ("trace", "rcl"), ("log", "fti")}
    for k, ds in enumerate(dss):
        ax = fig.add_subplot(gs[1, k if len(dss) < 3 else k] if False else gs[1, k])
        x = np.arange(len(pairs))
        for off, arm, col in ((-0.2, "v0_raw", BLUE), (0.2, "v3_incident", ORANGE)):
            rs = sel(recs, ds, arm, threads=2)
            g = np.array([[r["agreement"][m][t]["gap"] for m, t in pairs] for r in rs])
            ax.bar(x + off, g.mean(0), 0.38, yerr=g.std(0, ddof=1) if len(g) > 1 else None,
                   color=col, error_kw=dict(ecolor=INK2, lw=0.8, capsize=0),
                   label=SHORT[arm])
            facts.setdefault(("gap", ds, arm), dict(zip(pairs, g.mean(0))))
        ax.set_xticks(x, [f"{m}·{t.upper()}" + (" ★" if (m, t) in targeted else "")
                          for m, t in pairs], fontsize=7.5, rotation=40, ha="right")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{LETTER[ds]} · {DSNAME[ds]}")
        if k == 0:
            ax.set_ylabel("same-label − different-label\ncosine similarity (test)")
            ax.legend(loc="upper left")
    ax = fig.add_subplot(gs[1, len(dss)]); ax.axis("off")
    ax.text(0, 0.95, "★ = a pair TVDiag's TO term\npulls together\n(metric→RCL, trace→RCL,\n"
            "log→FTI).\n\nBars: how much closer\nsame-label incidents sit\nthan different-label ones\n"
            "in that modality's\nembedding. Mean ± sd\nover 5 seeds.", va="top", fontsize=8,
            color=INK2)
    fig.suptitle("Fig. 11 counterpart — which modality is pulled towards which task label, "
                 "and whether it happened", fontsize=11, fontweight="bold", color=INK, y=0.99)
    return save(fig, "fig11_task_routing.png"), facts


# =========================================================================== Fig. 13
def fig13(recs) -> str:
    dss = [d for d in ("gaia", "sockshop", "hotel")
           if sel(recs, d, "v0_raw", threads=2) and sel(recs, d, "v3_incident", threads=2)]
    if not dss:
        return ""
    from adadiag import data
    fig, axes = plt.subplots(len(dss), 2, figsize=(11, 2.9 * len(dss)),
                             gridspec_kw=dict(width_ratios=[1, 1.25], hspace=0.75, wspace=0.25))
    axes = np.atleast_2d(axes)
    for k, ds in enumerate(dss):
        d = data.load(ds)
        ex = {arm: sel(recs, ds, arm, threads=2)[0]["examples"][0]
              for arm in ("v0_raw", "v3_incident")}
        e0 = ex["v0_raw"]
        ax = axes[k, 0]
        x = np.arange(len(d.types))
        for off, arm, col in ((-0.2, "v0_raw", BLUE), (0.2, "v3_incident", ORANGE)):
            ax.bar(x + off, ex[arm]["fti"], 0.38, color=col, label=SHORT[arm])
        ax.set_xticks(x, [pretty(t, 18) for t in d.types], rotation=20, ha="right", fontsize=7.5)
        for t in ax.get_xticklabels():
            if t.get_text() == pretty(e0["type"], 18):
                t.set_fontweight("bold"); t.set_color(INK)
        ax.set_ylim(0, 1); ax.set_ylabel("probability")
        ax.set_title(f"{LETTER[ds]} · FTI — true type: {pretty(e0['type'], 30)}", fontsize=9)
        if k == 0:
            ax.legend(loc="upper left")
        ax = axes[k, 1]
        top = np.argsort(-np.maximum(ex["v0_raw"]["rcl"], ex["v3_incident"]["rcl"]))[:6]
        x = np.arange(len(top))
        for off, arm, col in ((-0.2, "v0_raw", BLUE), (0.2, "v3_incident", ORANGE)):
            ax.bar(x + off, np.array(ex[arm]["rcl"])[top], 0.38, color=col)
        ax.set_xticks(x, [d.instances[i] for i in top], rotation=20, ha="right", fontsize=7.5)
        for t in ax.get_xticklabels():
            if t.get_text() == e0["root"]:
                t.set_fontweight("bold"); t.set_color(INK)
        ax.set_ylim(0, 1)
        ax.set_title(f"{LETTER[ds]} · RCL — true root cause: {e0['root']} (six most likely)",
                     fontsize=9)
    fig.suptitle("Fig. 13 counterpart — the two heads' outputs on the first test incident of "
                 "each dataset (seed 0; true label in bold)", fontsize=11, fontweight="bold",
                 color=INK, y=0.95 if len(dss) > 2 else 1.02)
    return save(fig, "fig13_worked_example.png")


# =========================================================================== Fig. 14
def fig14(recs) -> tuple[str, dict]:
    dss = [d for d in ("gaia", "sockshop", "hotel")
           if sel(recs, d, "v0_raw", threads=2) and sel(recs, d, "v3_incident", threads=2)]
    if not dss:
        return "", {}
    fig, axes = plt.subplots(len(dss), 3, figsize=(10.5, 3.5 * len(dss)),
                             subplot_kw=dict(polar=True))
    axes = np.atleast_2d(axes)
    facts = {}
    for i, ds in enumerate(dss):
        rs = {arm: sel(recs, ds, arm, threads=2) for arm in ("v0_raw", "v3_incident")}
        types = list(rs["v0_raw"][0]["per_class"])
        ang = np.linspace(0, 2 * np.pi, len(types), endpoint=False)
        for j, k in enumerate(("hr1", "hr3", "hr5")):
            ax = axes[i, j]
            for arm, col in (("v0_raw", BLUE), ("v3_incident", ORANGE)):
                v = np.array([[r["per_class"][t][k] for t in types] for r in rs[arm]]).mean(0)
                facts[(ds, arm, k)] = dict(zip(types, v))
                ax.plot(np.r_[ang, ang[0]], np.r_[v, v[0]], color=col, lw=2,
                        label=SHORT[arm])
                ax.fill(np.r_[ang, ang[0]], np.r_[v, v[0]], color=col, alpha=0.08)
            ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["", "0.5", "", "1.0"],
                                             fontsize=7)
            ax.set_xticks(ang, [pretty(t, 16) for t in types], fontsize=7.2)
            ax.set_rlabel_position(100)
            ax.set_title(f"{LETTER[ds]} · {DSNAME[ds]} — HR@{k[2]}", pad=14, fontsize=9)
            ax.grid(color=GRID)
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(-0.35, 1.32))
    fig.suptitle("Fig. 14 counterpart — HR@k per failure type (mean of 5 seeds)",
                 fontsize=11, fontweight="bold", color=INK, y=1.01)
    fig.tight_layout()
    return save(fig, "fig14_per_failure_type.png"), facts


# =========================================================================== Fig. 15 / 17
def bars_vs_paper(ax, labels, paper, rerun, extra=None, ylim=(0, 1)):
    x = np.arange(len(labels))
    pv = [paper.get(l, np.nan) for l in labels]
    ax.bar(x - 0.2, pv, 0.38, color=AQUA, hatch=PAPER_HATCH, edgecolor=SURF, lw=0,
           label="paper (read off the figure)")
    mu = [np.mean(rerun[l]) if rerun.get(l) else np.nan for l in labels]
    sd = [np.std(rerun[l], ddof=1) if rerun.get(l) and len(rerun[l]) > 1 else 0 for l in labels]
    ax.bar(x + 0.2, mu, 0.38, yerr=sd, color=BLUE, error_kw=dict(ecolor=INK2, lw=0.8),
           label="re-run (mean ± sd)")
    if extra:
        for l, v in extra.items():
            if l in labels and v:
                ax.plot(labels.index(l) + 0.2, np.mean(v), "D", ms=6, color=INK,
                        markerfacecolor=SURF, label="re-run, TVDiag's shipped embeddings")
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(*ylim)


def fig17(recs) -> tuple[str, dict]:
    drops = {"w/ Metric": "drop:trace,log", "w/ Trace": "drop:metric,log",
             "w/ Log": "drop:metric,trace", "w/o Metric": "drop:metric",
             "w/o Trace": "drop:trace", "w/o Log": "drop:log", "TVDiag": ""}
    labels = list(drops)
    facts = {}
    rows = []
    for ds in ("gaia", "hotel"):
        rr = {l: sel(recs, ds, "v0_raw", "default", tag=t, threads=1) for l, t in drops.items()}
        if any(rr.values()):
            rows.append((ds, rr))
    if not rows:
        return "", {}
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.6 * len(rows)), squeeze=False)
    for i, (ds, rr) in enumerate(rows):
        paper = {l: v for l, v in P.FIG17_A.items()} if ds == "gaia" else {}
        for j, m in enumerate(M4):
            ax = axes[i, j]
            bars_vs_paper(ax, labels, {l: v[j] for l, v in paper.items()},
                          {l: list(vals(rs, m)) for l, rs in rr.items() if rs})
            ax.set_title(f"{LETTER[ds]} · {DSNAME[ds]} — {m}", fontsize=9)
            facts.update({(ds, l, m): mean(rs, m) for l, rs in rr.items() if rs})
        if i == 0:
            axes[i, 0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Fig. 17 counterpart — keeping (w/) or removing (w/o) one modality, TVDiag "
                 "as shipped", fontsize=11, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout()
    return save(fig, "fig17_modalities.png"), facts


def fig15(recs) -> tuple[str, dict]:
    tags = {"w/o opName": "emb:trace_no_op", "w/o abType": "emb:trace_no_abtype",
            "w/ Trace": "emb:trace_full", "w/o abDirection": "emb:metric_no_direction",
            "w/ Metric": "emb:metric_full"}
    rr = {l: sel(recs, "gaia", "v0_raw", "default", tag=t, threads=1) for l, t in tags.items()}
    if not any(rr.values()):
        return "", {}
    shipped = {"w/ Trace": sel(recs, "gaia", "v0_raw", "default", tag="drop:metric,log",
                               threads=1),
               "w/ Metric": sel(recs, "gaia", "v0_raw", "default", tag="drop:trace,log",
                                threads=1)}
    labels = list(tags)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    facts = {}
    for j, m in enumerate(M4):
        ax = axes[j]
        bars_vs_paper(ax, labels, {l: v[j] for l, v in P.FIG15_A.items()},
                      {l: list(vals(rs, m)) for l, rs in rr.items() if rs},
                      extra={l: list(vals(rs, m)) for l, rs in shipped.items() if rs})
        ax.axvline(2.5, color=MUTED, lw=0.8, ls=":")
        ax.set_title(f"A · GAIA — {m}", fontsize=9)
        facts.update({(l, m): mean(rs, m) for l, rs in rr.items() if rs})
        facts.update({("shipped " + l, m): mean(rs, m) for l, rs in shipped.items() if rs})
    h, l = axes[0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    axes[0].legend(uniq.values(), uniq.keys(), loc="upper left", fontsize=7.2)
    fig.suptitle("Fig. 15 counterpart — removing alert fields before embedding "
                 "(trace-only left of the dotted line, metric-only right)",
                 fontsize=11, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    return save(fig, "fig15_alert_fields.png"), facts


# =========================================================================== Fig. 16
SWEEP_KEY = {"p": "aug_percent", "omega": "contrastive_loss_scale", "tau": "temperature",
             "l": "graph_layers"}
SWEEP_LABEL = {"p": "inactivation probability p", "omega": "contrastive loss scale ω",
               "tau": "temperature τ", "l": "graph layers l"}


def sweep_curve(recs, ds, arm, name):
    from adadiag import config
    base = config.get(ds)
    pts = {}
    for r in recs:
        if (r["dataset"] != ds or r["arm"] != arm or r.get("tag", "")
                or r.get("env", {}).get("threads") != 1):
            continue
        v = r.get("variant", "")
        if v == "default":
            x = base[SWEEP_KEY[name]]
        elif v.startswith(name + "="):
            x = float(v.split("=")[1])
        else:
            continue
        pts.setdefault(x, {})[r["seed"]] = r
    return {x: [d[s] for s in sorted(d)] for x, d in sorted(pts.items())}


def fig16(recs) -> tuple[str, dict]:
    curves = {(arm, n): sweep_curve(recs, "gaia", arm, n)
              for arm in ("v0_raw", "v3_incident") for n in SWEEP_KEY}
    if not any(len(c) > 1 for c in curves.values()):
        return "", {}
    from adadiag import config
    fig, axes = plt.subplots(4, 4, figsize=(13, 10.5), sharey="row")
    facts = {}
    for j, n in enumerate(SWEEP_KEY):
        pv = P.FIG16_A[n]
        for i, m in enumerate(M4):
            ax = axes[i, j]
            ax.plot(pv["x"], pv[m], ls="--", marker="s", ms=4.5, color=AQUA, lw=1.6,
                    label="paper (read off the figure)")
            for arm, col, mk in (("v0_raw", BLUE, "o"), ("v3_incident", ORANGE, "^")):
                c = curves[(arm, n)]
                if len(c) < 2:
                    continue
                xs = np.array(list(c))
                mu = np.array([mean(c[x], m) for x in xs])
                sd = np.array([vals(c[x], m).std(ddof=1) if len(c[x]) > 1 else 0 for x in xs])
                ax.plot(xs, mu, marker=mk, ms=5, color=col, lw=2,
                        label=f"{SHORT[arm]} (mean ± sd)")
                ax.fill_between(xs, mu - sd, mu + sd, color=col, alpha=0.12, lw=0)
                facts[(arm, n, m)] = dict(zip(xs.tolist(), mu.tolist()))
                d0 = config.get("gaia")[SWEEP_KEY[n]]
                if d0 in c:
                    facts[("delta", arm, n, m)] = {x: paired(c[d0], c[x], m) for x in xs
                                                   if x != d0}
            from adadiag import config
            ax.axvline(config.get("gaia")[SWEEP_KEY[n]], color=MUTED, lw=0.9, ls=":")
            if n == "l":
                ax.set_xticks([1, 2, 3, 4])
            if i == 0:
                ax.set_title(SWEEP_LABEL[n])
            if j == 0:
                ax.set_ylabel(m)
            if i == 3:
                ax.set_xlabel(SWEEP_LABEL[n].split()[-1])
    for n in SWEEP_KEY:
        tv, ada = curves[("v0_raw", n)], curves[("v3_incident", n)]
        for m in M4:
            facts[("ada_vs_tv", n, m)] = {x: paired(tv[x], ada[x], m)
                                          for x in sorted(set(tv) & set(ada))}
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Fig. 16 counterpart — sensitivity to four hyper-parameters, A · GAIA "
                 "(dotted line: the default)", fontsize=11, fontweight="bold", color=INK,
                 y=1.03)
    fig.tight_layout()
    return save(fig, "fig16_hyperparameters.png"), facts


# =========================================================================== Fig. 18
def fig18(recs) -> tuple[str, dict]:
    dss = [d for d in ("gaia", "sockshop", "hotel")
           if sel(recs, d, "v0_raw", threads=2) and sel(recs, d, "v3_incident", threads=2)]
    if not dss:
        return "", {}
    fig, axes = plt.subplots(len(dss), 2, figsize=(9.5, 3.0 * len(dss)), squeeze=False)
    facts = {}
    mods = ("metric", "trace", "log")
    for i, ds in enumerate(dss):
        for j, task in enumerate(("fti", "rcl")):
            ax = axes[i, j]
            for arm, ls in (("v0_raw", "-"), ("v3_incident", "--")):
                rs = sel(recs, ds, arm, threads=2)
                if not rs or "share" not in rs[0]["shapley"][task]:
                    continue
                S = np.array(rs[0]["shapley"][task]["share"])
                for k, m in enumerate(mods):
                    v = np.sort(S[:, k])
                    ax.plot(v, np.arange(1, len(v) + 1) / len(v), ls=ls, color=MOD_COL[m],
                            lw=1.8, label=f"{m}" if arm == "v0_raw" else None)
                    facts[(ds, arm, task, m)] = (float(np.quantile(S[:, k], .25)),
                                                 float(np.median(S[:, k])),
                                                 float(np.quantile(S[:, k], .75)))
            if ds == "gaia":
                for m in mods:
                    q = P.FIG18_A[task][m]
                    ax.plot(q, (0.25, 0.5, 0.75), "o", ms=6, color=MOD_COL[m],
                            markeredgecolor=INK, markeredgewidth=0.8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            ax.set_title(f"{LETTER[ds]} · {DSNAME[ds]} — {task.upper()}", fontsize=9)
            ax.set_xlabel("modality's share of the prediction (|Shapley| / Σ)")
            if j == 0:
                ax.set_ylabel("CDF over test incidents")
    h, l = axes[0, 0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=INK2, ls="-"), plt.Line2D([], [], color=INK2, ls="--"),
          plt.Line2D([], [], color=SURF, marker="o", markerfacecolor=MUTED,
                     markeredgecolor=INK, ls="")]
    l += ["TVDiag re-run (solid)", "AdaDiag (dashed)", "paper's quartiles (A only)"]
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.01), fontsize=8)
    fig.suptitle("Fig. 18 counterpart — how much each modality contributes, per test incident "
                 "(seed 0)", fontsize=11, fontweight="bold", color=INK, y=1.05)
    fig.tight_layout()
    return save(fig, "fig18_modality_shapley.png"), facts


# =========================================================================== main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = load_runs()
    parts = {}
    parts["t3"] = table3()
    parts["t4"], f4 = table45(recs, "rcl")
    parts["t5"], f5 = table45(recs, "fti")
    parts["t6"], f6 = table6(recs)
    parts["t7"], f7 = table7(recs)
    parts["f11"], f11 = fig11(recs)
    parts["f13"] = fig13(recs)
    parts["f14"], f14 = fig14(recs)
    parts["f15"], f15 = fig15(recs)
    parts["f16"], f16 = fig16(recs)
    parts["f17"], f17 = fig17(recs)
    parts["f18"], f18 = fig18(recs)
    facts = dict(f4=f4, f5=f5, f6=f6, f7=f7, f11=f11, f14=f14, f15=f15, f16=f16, f17=f17,
                 f18=f18, repro=repro_check(recs), n_records=len(recs), f6_complete=complete)
    from report_paper_comparison import write_report      # prose lives beside this script
    write_report(OUT / "PAPER_COMPARISON.md", parts, facts)
    # the compact table README §6.9 quotes
    rows = []
    for L in "ACD":
        if not (f4.get(L, {}).get("rerun") and f4[L].get("ada")):
            continue
        for lab, src4, src5 in (("TVDiag, paper", f4[L]["paper"], f5[L]["paper"]),
                                ("TVDiag, re-run", f4[L]["rerun"], f5[L]["rerun"]),
                                ("AdaDiag", f4[L]["ada"], f5[L]["ada"])):
            rows.append([f"{L} · {DSNAME[P.DATASET_KEY[L]]}" if lab.endswith("paper") else "",
                         lab] + [f"{src4[m]:.3f}" for m in ("HR@1", "HR@3", "Avg@3", "MRR@3")]
                        + [f"{src5['F1']:.3f}"])
    (OUT / "readme_table.md").write_text(table(
        ["Dataset", "", "HR@1", "HR@3", "Avg@3", "MRR@3", "F1"], rows,
        ["l", "l", "r", "r", "r", "r", "r"]) + "\n")
    print("wrote", OUT / "PAPER_COMPARISON.md", "from", len(recs), "records")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
