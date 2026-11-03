#!/usr/bin/env python3
"""Render results/trace_audit.json (scripts/run_trace_audit.py) as a figure and a report.

    python scripts/make_trace_report.py      ->  results/trace_audit.png, results/TRACE_AUDIT.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from adadiag.weighting import MODALITIES, TASKS                      # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
COL = {"metric": BLUE, "trace": ORANGE, "log": AQUA}
NAME = {"gaia": "GAIA", "sockshop": "Sock Shop", "hotel": "Hotel Reservation"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
})


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def empty_trace_share(name):
    from adadiag import data
    ds = data.load(name)
    nz = ds.X.abs().sum(-1) > 0                                   # (M, 3, N)
    return {m: float((~nz[:, j].any(1)).float().mean()) for j, m in enumerate(MODALITIES)}


def figure(a, out):
    dss = [d for d in ("gaia", "sockshop", "hotel") if d in a and "v3_incident" in a[d]]
    fig, axes = plt.subplots(3, len(dss), figsize=(4.3 * len(dss), 8.6), squeeze=False)
    for j, ds in enumerate(dss):
        r3, r0 = a[ds]["v3_incident"], a[ds].get("v0_raw")
        ep = np.array(r3["epochs"]); W = np.array(r3["w"]); L = np.array(r3["L"])
        for ti, t in enumerate(TASKS):
            ax = axes[ti, j]
            for mi, m in enumerate(MODALITIES):
                ax.plot(ep, W[:, mi, ti], color=COL[m], lw=2.2 if m == "trace" else 1.4,
                        label=m)
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{NAME[ds]} — AdaDiag's weight, {t.upper()} column")
            if j == 0:
                ax.set_ylabel("weight")
        ax = axes[2, j]
        for mi, m in enumerate(MODALITIES):
            ax.plot(ep, L[:, mi, 0], color=COL[m], lw=1.4, label=f"{m}, AdaDiag")
        if r0:
            L0 = np.array(r0["L"])
            ax.plot(r0["epochs"], L0[:, 1, 0], color=COL["trace"], ls="--", lw=2,
                    label="trace, TVDiag (weight 1)")
        ax.axhline(r3["chance"], color=MUTED, ls=":", lw=1)
        ax.text(ep[-1], r3["chance"], " chance", va="bottom", ha="right", fontsize=7.5,
                color=INK2)
        ax.set_title(f"{NAME[ds]} — RCL contrastive loss L[m, RCL]")
        ax.set_xlabel("epoch")
        if j == 0:
            ax.set_ylabel("SupCon loss (lower = better clustered)")
    axes[0, 0].legend(loc="center right", fontsize=8)
    axes[2, 0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Why trace gets zero weight: the module follows the lowest contrastive loss, "
                 "and trace's starts highest", fontsize=11, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    a = json.load(open(ROOT / "results" / "trace_audit.json"))
    figure(a, ROOT / "results" / "trace_audit.png")
    L = ["# Why AdaDiag gives the trace modality zero weight — audit\n",
         "Produced by `scripts/run_trace_audit.py` and `scripts/make_trace_report.py`; "
         "discussed in README §6.10.\n", "![trace audit](trace_audit.png)\n"]

    rows = []
    for ds in ("gaia", "sockshop", "hotel"):
        if ds not in a:
            continue
        e = empty_trace_share(ds)
        r3 = a[ds].get("v3_incident", {})
        rows.append([NAME[ds], f"{e['trace']:.1%}",
                     "yes" if r3.get("identical") else ("—" if not r3 else "no"),
                     *(f"{a[ds]['swapped']['w'][-1][0][ti]:.3f}" if "swapped" in a[ds] else "—"
                       for ti in range(2))])
    L.append("## 1. It is not a wiring error\n")
    L.append(table(["dataset", "incidents with no trace alert", "audited run = committed run",
                    "weight on trace data moved to slot 0, RCL", "… FTI"], rows) + "\n")
    L.append("With the trace and metric data swapped between slots, the zero follows the "
             "trace **data** into the other slot. The trace plane holds `tmp/trace.pkl` "
             "(`tests/test_trace_weight.py`).\n")

    L.append("## 2. Trace's contrastive loss is the highest from the first step\n")
    rows = []
    for ds in ("gaia", "sockshop", "hotel"):
        if ds not in a or "initial" not in a[ds]:
            continue
        init = a[ds]["initial"]
        rank = lambda v, ti: int(np.argsort(-np.array(v["L"])[:, ti]).tolist().index(1)) + 1
        L0 = np.array(init["0"]["L"]) if "0" in init else np.array(next(iter(init.values()))["L"])
        r3 = a[ds].get("v3_incident", {})
        rows.append([NAME[ds], f"{init[next(iter(init))]['chance']:.2f}",
                     " / ".join(f"{x:.2f}" for x in L0[:, 0]),
                     " / ".join(f"{x:.2f}" for x in L0[:, 1]),
                     ", ".join(str(rank(v, 0)) for v in init.values()),
                     ", ".join(str(rank(v, 1)) for v in init.values()),
                     " / ".join(f"{r3['settled'][t]['epoch']}" for t in TASKS) if r3 else "—"])
    L.append(table(["dataset", "chance", "L RCL m / t / l (seed 0)", "L FTI m / t / l",
                    "trace's rank, RCL (5 seeds; 1 = highest)", "… FTI",
                    "epoch a column passes 0.9 (RCL / FTI)"], rows) + "\n")

    L.append("## 3. Under TVDiag, trace keeps training — and on GAIA ends up lowest\n")
    rows = []
    for ds in ("gaia", "sockshop", "hotel"):
        if ds not in a or "v0_raw" not in a[ds]:
            continue
        L0 = np.array(a[ds]["v0_raw"]["L"]); L3 = np.array(a[ds]["v3_incident"]["L"])
        first = [e for e, row in zip(a[ds]["v0_raw"]["epochs"], L0[:, :, 0])
                 if int(np.argmin(row)) == 1]
        rows.append([NAME[ds], f"{L0[0, 1, 0]:.2f} → {L0[-1, 1, 0]:.2f}",
                     f"{L3[0, 1, 0]:.2f} → {L3[-1, 1, 0]:.2f}",
                     " / ".join(f"{x:.2f}" for x in L0[-1, :, 0]),
                     first[0] if first else "never"])
    L.append(table(["dataset", "trace L[RCL], TVDiag (weight 1)", "… AdaDiag (weight 0)",
                    "TVDiag at the end, m / t / l", "first epoch trace is lowest (TVDiag)"],
                   rows) + "\n")

    for key in sorted({k for d in a.values() for k in d if k.startswith("warmup")}):
        L.append(f"## 4. The obvious fix: a {key[6:]}-epoch warm-up\n")
        rows = []
        for ds in ("gaia", "sockshop", "hotel"):
            runs = a.get(ds, {}).get(key, {})
            if not runs:
                continue
            Wf = np.array([r["w"][-1] for r in runs.values()])
            rows.append([NAME[ds], len(runs), f"{Wf[:, 1, 0].mean():.3f}", f"{Wf[:, 1, 1].mean():.3f}",
                         " / ".join(f"{x:.2f}" for x in Wf[:, :, 0].mean(0)),
                         " / ".join(f"{x:.2f}" for x in Wf[:, :, 1].mean(0)),
                         f"{np.mean([r['metrics']['HR@1'] for r in runs.values()]):.3f}",
                         f"{np.mean([r['metrics']['f1'] for r in runs.values()]):.3f}"])
        L.append(table(["dataset", "seeds", "trace weight, RCL", "trace weight, FTI",
                        "final RCL column m / t / l", "final FTI column m / t / l", "HR@1",
                        "F1"], rows) + "\n")
    (ROOT / "results" / "TRACE_AUDIT.md").write_text("\n".join(L))
    print("wrote results/TRACE_AUDIT.md and results/trace_audit.png")


if __name__ == "__main__":
    main()
