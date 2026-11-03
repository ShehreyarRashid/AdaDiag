#!/usr/bin/env python3
"""README section 6.14: AdaDiag with trace's weight held fixed, metric and log learned.

Reads results/trace_fixed/runs_<ds>*.json (scripts/run_cl_variants.py, variants
trace_third / trace_tvdiag) and the paired references: TVDiag default, AdaDiag (the
section 6.8 ref_adadiag / D_ada_s0.5 records), TVDiag's rule at sigma_to 0.5
(results/cl_variants, variant `none`). Writes results/trace_fixed/TRACE_FIXED.md,
readme_table.md and summary.json.

    python scripts/make_trace_fixed_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_oracle_report as MO                                        # noqa: E402

OUT = ROOT / "results" / "trace_fixed"
DS = MO.DS
NAME = MO.NAME
METRICS = MO.METRICS
VARIANTS = (("trace_third", "trace fixed at 1/3 of both columns"),
            ("trace_tvdiag", "trace fixed at TVDiag's role (RCL 0.5, FTI 0)"))
REFS = (("tvdiag", "TVDiag default"), ("adadiag", "AdaDiag"))


def load(ds):
    base = MO.load(ds)
    rs = []
    for f in sorted(OUT.glob(f"runs_{ds}*.json")):
        rs += json.load(open(f))
    none = []
    for f in sorted((ROOT / "results" / "cl_variants").glob(f"runs_{ds}*.json")):
        none += [r for r in json.load(open(f)) if r["variant"] == "none"]
    sig = sorted({r["sigma_to"] for r in rs}) or [0.1]
    out = {("tvdiag", 0.1): base[("tvdiag", 0.1)]}
    for s in sig:
        out[("adadiag", s)] = base.get(("adadiag", s), [])
        out[("none", s)] = (base[("tvdiag", 0.1)] if s == 0.1 else
                            MO.by_seed([r for r in none if r["sigma_to"] == s]))
        for v, _ in VARIANTS:
            out[(v, s)] = MO.by_seed([r for r in rs if r["variant"] == v and r["sigma_to"] == s])
    return out, sig


def comparisons(data, sigmas):
    keys, raw = [], []
    for ds in DS:
        for s in sigmas[ds]:
            for v, _ in VARIANTS:
                b = data[ds].get((v, s), [])
                for ref, _ in REFS:
                    a = data[ds][("tvdiag", 0.1)] if ref == "tvdiag" else data[ds].get((ref, s), [])
                    for k, _ in METRICS:
                        if a and len(b) > 1:
                            keys.append((ds, v, s, ref, k))
                            raw.append(MO.paired(a, b, k))
    adj = MO.holm([p for _, p, _ in raw])
    return {k: (d, p, n, h) for k, (d, p, n), h in zip(keys, raw, adj)}


def summary_md(data, sigmas, C):
    L = ["| weighting | σ_to | GAIA HR@1 | GAIA F1 | Hotel HR@1 | Hotel F1 |",
         "|---|---|---|---|---|---|"]
    allsig = sorted(set(sigmas["gaia"]) | set(sigmas["hotel"]))
    for s in allsig:
        rows = [(("none", s), "TVDiag default" if s == 0.1 else "TVDiag's rule"),
                (("adadiag", s), "AdaDiag (current)")]
        rows += [((v, s), f"**AdaDiag, {lab}**") for v, lab in VARIANTS]
        for key, lab in rows:
            cells = []
            for ds in DS:
                rs = data[ds].get(key, [])
                for k in ("HR@1", "f1"):
                    if not rs:
                        cells.append("—")
                        continue
                    c = f"{MO.mean(rs, k):.3f}"
                    if key[0] in dict(VARIANTS):
                        d, p, n, h = C[(ds, key[0], s, "adadiag", k)]
                        b = "**" if h < 0.05 else ""
                        c += f" ({b}{d:+.3f}{b})"
                    cells.append(c)
            L.append(f"| {lab} | {s:g} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def weights_md(data, sigmas):
    L = ["| dataset | σ_to | weighting | RCL: metric / trace / log | FTI: metric / trace / log |",
         "|---|---|---|---|---|"]
    for ds in DS:
        for s in sigmas[ds]:
            ada = [r for r in data[ds].get(("adadiag", s), []) if r.get("weights")]
            if ada:
                W = np.mean([np.array(r["weights"]["w"]) for r in ada], 0)
                r_, f_ = MO.fmt_matrix(W)
                L.append(f"| {NAME[ds]} | {s:g} | AdaDiag (current) | {r_} | {f_} |")
            for v, lab in VARIANTS:
                rs = [r for r in data[ds].get((v, s), []) if r.get("w_final")]
                if rs:
                    W = np.mean([np.array(r["w_final"]) for r in rs], 0)
                    r_, f_ = MO.fmt_matrix(W)
                    # how far the learned metric/log split sits from a vertex: per seed and
                    # task column, the larger of the two as a share of what they hold together
                    def corner(c):
                        return np.mean([np.array(r["w_final"])[[0, 2], c].max()
                                        / max(np.array(r["w_final"])[[0, 2], c].sum(), 1e-12)
                                        for r in rs])
                    L.append(f"| | | {lab} | {r_} | {f_} |")
                    L.append(f"| | | ↳ larger of metric / log, share of the learned part "
                             f"(1 = all on one) | {corner(0):.3f} | {corner(1):.3f} |")
    return "\n".join(L) + "\n"


def full_md(data, sigmas, C):
    L = []
    for ds in DS:
        for s in sigmas[ds]:
            L += [f"**{NAME[ds]}, σ_to = {s:g}**\n",
                  "| weighting | seeds | " + " | ".join(l for _, l in METRICS) + " |",
                  "|---|---|" + "---|" * len(METRICS)]
            rows = [(("tvdiag", 0.1), "TVDiag default (σ = 0.1)")]
            if s != 0.1:
                rows.append((("none", s), f"TVDiag's rule at σ_to = {s:g}"))
            rows += [(("adadiag", s), "AdaDiag (current)")]
            rows += [((v, s), f"**{lab}**") for v, lab in VARIANTS]
            for key, lab in rows:
                rs = data[ds].get(key, [])
                if rs:
                    L.append(f"| {lab} | {len(rs)} | " + " | ".join(MO.ms(rs, k) for k, _ in METRICS) + " |")
            L += ["", "| difference | " + " | ".join(l for _, l in METRICS) + " |",
                  "|---|" + "---|" * len(METRICS)]
            for v, lab in VARIANTS:
                for ref, rlab in REFS:
                    if (ds, v, s, ref, "HR@1") not in C:
                        continue
                    cells = []
                    for k, _ in METRICS:
                        d, p, n, h = C[(ds, v, s, ref, k)]
                        b = "**" if h < 0.05 else ""
                        cells.append(f"{b}{d:+.3f}{b} (p {p:.2f}, Holm {h:.2f})")
                    L.append(f"| {v} − {rlab} | " + " | ".join(cells) + " |")
            L.append("")
    L.append(f"Paired t-test over seeds; Holm over all {len(C)} comparisons. Bold: significant "
             "after correction.\n")
    return "\n".join(L)


def main() -> None:
    loaded = {ds: load(ds) for ds in DS}
    data = {ds: v[0] for ds, v in loaded.items()}
    sigmas = {ds: v[1] for ds, v in loaded.items()}
    C = comparisons(data, sigmas)
    n_sig = sum(1 for v in C.values() if v[3] < 0.05)
    md = ["# AdaDiag with trace held fixed (README §6.14)\n",
          "Trace's weight is held at a fixed value in each task column and never learned; "
          "AdaDiag's module learns only how to split the rest between metric and log "
          "(`WeightingModule(fixed_trace=...)`). Same seeds, 1 thread and 500-epoch cap as "
          "the references; each lane first re-trains AdaDiag itself and checks it "
          "bit-for-bit (`logs/`).\n",
          "## Summary (difference from AdaDiag, same seeds)\n", summary_md(data, sigmas, C),
          "## The learned matrices\n", weights_md(data, sigmas),
          "## Full tables\n", full_md(data, sigmas, C),
          f"Significant after correction: {n_sig} of {len(C)}.\n"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "TRACE_FIXED.md").write_text("\n".join(md))
    (OUT / "readme_table.md").write_text(summary_md(data, sigmas, C))
    json.dump({"n_significant": n_sig, "n_comparisons": len(C),
               "diff": {"|".join(map(str, k)): list(v) for k, v in C.items()}},
              open(OUT / "summary.json", "w"), indent=1)
    print((OUT / "TRACE_FIXED.md").read_text())


if __name__ == "__main__":
    main()
