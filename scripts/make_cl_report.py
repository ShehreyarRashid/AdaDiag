#!/usr/bin/env python3
"""README section 6.13: five ways to weight the contrastive term, against TVDiag.

Reads results/cl_variants/runs_<ds>*.json (scripts/run_cl_variants.py),
results/cl_variants/gradient_change.json (scripts/cl_gradient_change.py) and the paired
references of section 6.12 (TVDiag default; AdaDiag and TVDiag's rule on the simplex from
the section 6.8 ablations; the oracle runs). Writes results/cl_variants/CL_VARIANTS.md,
readme_table.md, summary.json and cl_variants.png.

    python scripts/make_cl_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_oracle_report as MO                                        # noqa: E402

OUT = ROOT / "results" / "cl_variants"
DS = MO.DS
NAME = MO.NAME
METRICS = MO.METRICS
VARIANTS = (("pos", "1 · weight positive pairs (β = 1)"),
            ("pos_strong", "1 · weight positive pairs (β = 1/τ)"),
            ("neg", "2 · weight negative samples (β = 1)"),
            ("neg_strong", "2 · weight negative samples (β = 1/τ)"),
            ("importance", "3 · weight by sample importance"),
            ("align", "4 · weights from the diagnosis losses"),
            ("curriculum", "5 · curriculum (easy → hard)"))
#: training-length controls: TVDiag stopped, seed by seed, at the epoch the variant stopped at
CONTROLS = (("capped:importance", "importance", "control: TVDiag stopped at type 3's epochs"),
            ("capped:align", "align", "control: TVDiag stopped at type 4's epochs"))
#: references: TVDiag default (sigma 0.1); TVDiag's rule at the same sigma_to (only when it
#: differs from the default); AdaDiag as committed, at the same sigma_to
REFS = (("tvdiag", "TVDiag default"), ("none", "TVDiag, same σ_to"), ("adadiag", "AdaDiag"))


def runs(ds):
    recs = []
    for f in sorted(OUT.glob(f"runs_{ds}*.json")):
        recs += json.load(open(f))
    return recs


def load(ds):
    base = MO.load(ds)                     # tvdiag, prior, adadiag, oracle (global/class)
    rs = runs(ds)
    sig = sorted({r["sigma_to"] for r in rs if r["variant"] != "none"}) or [0.1]
    out = {("tvdiag", 0.1): base[("tvdiag", 0.1)], ("none", 0.1): base[("tvdiag", 0.1)]}
    for s in sig:
        for k in ("prior", "adadiag", "global"):
            out[(k, s)] = base.get((k, s), [])
        for v in ([v for v, _ in VARIANTS] + [c for c, _, _ in CONTROLS]
                  + (["none"] if s != 0.1 else [])):
            out[(v, s)] = MO.by_seed([r for r in rs if r["variant"] == v and r["sigma_to"] == s])
    return out, sig


def ref_runs(data, ds, ref, s):
    if ref == "tvdiag":
        return data[ds][("tvdiag", 0.1)]
    if ref == "none":
        return data[ds].get(("none", s), []) if s != 0.1 else []
    return data[ds].get((ref, s), [])


def comparisons(data, sigmas):
    """Every variant against each reference, every metric; Holm within each reference."""
    fam = {ref: ([], []) for ref, _ in REFS}
    for ds in DS:
        for s in sigmas[ds]:
            for v, _ in VARIANTS:
                b = data[ds].get((v, s), [])
                for ref, _ in REFS:
                    a = ref_runs(data, ds, ref, s)
                    for k, _ in METRICS:
                        if a and len(b) > 1:
                            fam[ref][0].append((ds, v, s, ref, k))
                            fam[ref][1].append(MO.paired(a, b, k))
    C = {}
    for ref, (keys, raw) in fam.items():
        adj = MO.holm([p for _, p, _ in raw])
        C.update({k: (d, p, n, h) for k, (d, p, n), h in zip(keys, raw, adj)})
    return C


def ci(a, b, k):
    sa = {r["seed"]: r[k] for r in a}
    sb = {r["seed"]: r[k] for r in b}
    d = np.array([sb[i] - sa[i] for i in sorted(set(sa) & set(sb))])
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


def summary_md(data, C, s=0.1):
    """The compact table the README quotes: HR@1 and F1 (difference from TVDiag default)."""
    head = "| weighting | GAIA HR@1 | GAIA F1 | Hotel HR@1 | Hotel F1 |"
    L = [head, "|---|---|---|---|---|"]
    base = {ds: data[ds][("tvdiag", 0.1)] for ds in DS}
    L.append("| TVDiag default | " + " | ".join(
        f"{MO.mean(base[ds], k):.3f}" for ds in DS for k in ("HR@1", "f1")) + " |")
    rows = [(("adadiag", s), "AdaDiag (current)")] + [((v, s), lab) for v, lab in VARIANTS]
    for key, lab in rows:
        cells = []
        for ds in DS:
            rs = data[ds].get(key, [])
            for k in ("HR@1", "f1"):
                if not rs:
                    cells.append("—")
                    continue
                d = MO.mean(rs, k) - MO.mean(base[ds], k)
                h = C.get((ds, key[0], s, "tvdiag", k), (0, 0, 0, 1.0))[3]
                b = "**" if h < 0.05 else ""
                cells.append(f"{MO.mean(rs, k):.3f} ({b}{d:+.3f}{b})")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def control_md(data):
    """GAIA: types 3 and 4 stop 2-5x earlier than TVDiag. Re-train TVDiag stopped at the same
    epoch, seed by seed, to separate the weighting from the training length."""
    L = []
    for ctl, v, lab in CONTROLS:
        var = {r["seed"]: r for r in data["gaia"].get((v, 0.1), [])}
        cap = {r["seed"]: r for r in data["gaia"].get((ctl, 0.1), [])}
        tv = {r["seed"]: r for r in data["gaia"][("tvdiag", 0.1)]}
        seeds = sorted(set(var) & set(cap) & set(tv))
        if not seeds:
            continue
        name = dict(VARIANTS)[v].split(" (")[0]
        L += [f"**{name}**\n",
              "| GAIA seed | epochs | variant: HR@1 / F1 | TVDiag, same epochs: HR@1 / F1 | "
              "TVDiag default (epochs): HR@1 / F1 |", "|---|---|---|---|---|"]
        for s in seeds:
            L.append(f"| {s} | {var[s]['epochs']} | {var[s]['HR@1']:.3f} / {var[s]['f1']:.3f} | "
                     f"{cap[s]['HR@1']:.3f} / {cap[s]['f1']:.3f} | {tv[s]['HR@1']:.3f} / "
                     f"{tv[s]['f1']:.3f} ({tv[s]['epochs']}) |")
        m = lambda d, k: np.mean([d[s][k] for s in seeds])
        L.append(f"| mean | | {m(var, 'HR@1'):.3f} / {m(var, 'f1'):.3f} | {m(cap, 'HR@1'):.3f} / "
                 f"{m(cap, 'f1'):.3f} | {m(tv, 'HR@1'):.3f} / {m(tv, 'f1'):.3f} |")
        L.append(f"| variant − same-epoch TVDiag | | {m(var, 'HR@1') - m(cap, 'HR@1'):+.3f} / "
                 f"{m(var, 'f1') - m(cap, 'f1'):+.3f} | | |\n")
    return "\n".join(L) + "\n" if L else ""


def gradient_md():
    f = OUT / "gradient_change.json"
    if not f.exists():
        return ""
    G = json.load(open(f))
    keys = [k for k in next(iter(next(iter(G.values())).values())) if not k.startswith("weight_cv")]
    L = ["| dataset | term | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for ds in DS:
        for term, row in G[ds].items():
            L.append(f"| {NAME[ds]} | {term} | " + " | ".join(f"{row[k]:.2f}" for k in keys) + " |")
    return "\n".join(L) + "\n"


def align_md(data, sigmas):
    L = ["| dataset | σ_to | matrix | RCL: metric / trace / log | FTI: metric / trace / log |",
         "|---|---|---|---|---|"]
    ow = json.load(open(ROOT / "results" / "oracle" / "oracle_weights.json"))
    for ds in DS:
        for s in sigmas[ds]:
            rs = [r for r in data[ds].get(("align", s), []) if r.get("w_final")]
            if not rs:
                continue
            W = np.mean([np.array(r["w_final"]) for r in rs], 0)
            r_, f_ = MO.fmt_matrix(W)
            L.append(f"| {NAME[ds]} | {s:g} | **4 · from the diagnosis losses** ({len(rs)} seeds) | {r_} | {f_} |")
            ada = [r for r in data[ds].get(("adadiag", s), []) if r.get("weights")]
            if ada:
                Wa = np.mean([np.array(r["weights"]["w"] if isinstance(r["weights"], dict)
                                       else r["weights"]) for r in ada], 0)
                r_, f_ = MO.fmt_matrix(Wa)
                L.append(f"| | | AdaDiag (current) | {r_} | {f_} |")
            r_, f_ = MO.fmt_matrix(ow[ds]["global"])
            L.append(f"| | | oracle (§6.12) | {r_} | {f_} |")
    return "\n".join(L) + "\n"


def table_md(data, sigmas, C):
    L = []
    for ds in DS:
        for s in sigmas[ds]:
            L.append(f"**{NAME[ds]}, σ_to = {s:g}**\n")
            L.append("| weighting | seeds | " + " | ".join(l for _, l in METRICS) + " |")
            L.append("|---|---|" + "---|" * len(METRICS))
            rows = [(("tvdiag", 0.1), "TVDiag default (σ = 0.1)")]
            if s != 0.1:
                rows.append((("none", s), f"TVDiag's rule at σ_to = {s:g}"))
            rows += [(("prior", s), "TVDiag's rule on the simplex"),
                     (("adadiag", s), "AdaDiag (current: learned from the contrastive term)"),
                     (("global", s), "*oracle, one matrix (§6.12, fitted on test)*")]
            rows += [((v, s), f"**{lab}**") for v, lab in VARIANTS]
            if s == 0.1:
                rows += [((c, s), lab) for c, _, lab in CONTROLS]
            for key, lab in rows:
                rs = data[ds].get(key, [])
                if rs:
                    L.append(f"| {lab} | {len(rs)} | " + " | ".join(MO.ms(rs, k) for k, _ in METRICS) + " |")
            L.append("")
            L.append("| difference | " + " | ".join(l for _, l in METRICS) + " |")
            L.append("|---|" + "---|" * len(METRICS))
            for v, lab in VARIANTS:
                for ref, rlab in REFS:
                    if (ds, v, s, ref, "HR@1") not in C:
                        continue
                    cells = []
                    for k, _ in METRICS:
                        d, p, n, h = C[(ds, v, s, ref, k)]
                        b = "**" if h < 0.05 else ""
                        cells.append(f"{b}{d:+.3f}{b} (p {p:.2f}, Holm {h:.2f})")
                    short = lab.split(" (")[0].split(" · ")[0] + ("s" if "1/τ" in lab else "")
                    L.append(f"| {short} − {rlab} | " + " | ".join(cells) + " |")
            L.append("")
    ns = {ref: sum(1 for k in C if k[3] == ref) for ref, _ in REFS}
    L.append("Paired t-test over seeds. Holm correction within each reference: "
             + ", ".join(f"{ns[r]} comparisons against {lab}" for r, lab in REFS if ns[r])
             + ". `1s`/`2s`: β = 1/τ. Bold: significant after correction.\n")
    return "\n".join(L)


def figure(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
    BLUE = "#2a78d6"
    rows = [("adadiag", "AdaDiag (current)", MUTED), ("global", "oracle (§6.12, fitted on test)", MUTED)]
    rows += [(v, lab, BLUE) for v, lab in VARIANTS]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 4.4), facecolor=SURF, sharey=True)
    for j, (ds, (k, lab)) in enumerate([(ds, m) for ds in DS for m in (("HR@1", "HR@1"), ("f1", "F1"))]):
        ax = axes[j]
        ax.set_facecolor(SURF)
        base = data[ds][("tvdiag", 0.1)]
        for i, (key, name, col) in enumerate(rows):
            rs = data[ds].get((key, 0.1), [])
            if not rs:
                continue
            d, lo, hi = ci(base, rs, k)
            y = len(rows) - 1 - i
            ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter([d], [y], s=46, color=col, edgecolor=SURF, lw=1.5, zorder=3)
        ax.axvline(0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_title(f"{NAME[ds]} — {lab}", fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.tick_params(axis="y", length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.set_xlabel("difference from TVDiag default", fontsize=8, color=INK2)
    axes[0].set_yticks(range(len(rows)), [r[1] for r in rows][::-1], fontsize=8.5)
    fig.text(0.5, 0.005, "Dot: mean paired difference over seeds (GAIA 3, Hotel 5), σ = 0.1. "
             "Bar: 95% confidence interval. Dashed line: no difference.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURF)
    plt.close(fig)


def main() -> None:
    loaded = {ds: load(ds) for ds in DS}
    data = {ds: v[0] for ds, v in loaded.items()}
    sigmas = {ds: v[1] for ds, v in loaded.items()}
    C = comparisons(data, sigmas)
    figure(data, OUT / "cl_variants.png")
    n_sig = {ref: sum(1 for k, v in C.items() if k[3] == ref and v[3] < 0.05) for ref, _ in REFS}
    n_tot = {ref: sum(1 for k in C if k[3] == ref) for ref, _ in REFS}
    md = [
        "# Five ways to weight the contrastive term (README §6.13)\n",
        "Each variant changes one thing on top of TVDiag's default, with one setting fixed "
        "before any run (`scripts/run_cl_variants.py`); types 1 and 2 are also run at "
        "β = 1/τ as a sensitivity check. Same seeds, 1 thread and 500-epoch cap as the "
        "references; each lane first re-trains TVDiag default and checks it bit-for-bit "
        "(`logs/`).\n",
        "## Summary — σ = 0.1, value (difference from TVDiag default)\n",
        summary_md(data, C),
        "## How much each weighting changes the term's gradient\n",
        "Cosine between the encoder gradient of each of TVDiag's three terms with and without "
        "the weighting, after 80 epochs of TVDiag (seed 0, one training batch; "
        "`scripts/cl_gradient_change.py`). 1 = no change.\n",
        gradient_md(),
        "## Types 3 and 4 on GAIA: the weighting or the training length?\n",
        control_md(data),
        "## The matrix type 4 sets from the diagnosis losses\n",
        align_md(data, sigmas),
        "## Full tables\n",
        table_md(data, sigmas, C),
        "Significant after correction: " + ", ".join(
            f"{n_sig[r]} of {n_tot[r]} against {lab}" for r, lab in REFS if n_tot[r]) + ".\n",
        "![five weightings](cl_variants.png)\n",
    ]
    (OUT / "CL_VARIANTS.md").write_text("\n".join(md))
    (OUT / "readme_table.md").write_text(summary_md(data, C))
    json.dump({"n_significant": n_sig, "n_comparisons": n_tot,
               "diff": {"|".join(map(str, k)): list(v) for k, v in C.items()}},
              open(OUT / "summary.json", "w"), indent=1)
    print((OUT / "CL_VARIANTS.md").read_text())


if __name__ == "__main__":
    main()
