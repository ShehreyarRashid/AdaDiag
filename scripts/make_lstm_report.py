#!/usr/bin/env python3
"""README section 6.18: the LSTM aggregator the TVDiag paper describes, in TVDiag and AdaDiag.

Reads the fresh-seed runs (seeds 5-9) in results/selection/confirm_<ds>*.json, written by
scripts/select_adadiag.py (variants tvdiag, align, tvdiag_lstm, align_lstm), and the
training-length controls in results/selection/length_<ds>.json. Writes
results/selection/LSTM.md and results/selection/lstm_table.md.

    python scripts/make_lstm_report.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SEL = ROOT / "results" / "selection"
DS = (("gaia", "A · GAIA"), ("sockshop", "C · Sock Shop"), ("hotel", "D · Hotel"))
M = (("HR@1", "HR@1"), ("f1", "F1"), ("macro_f1", "macro-F1"))


def runs(ds):
    out = {}
    for f in sorted(glob.glob(str(SEL / f"confirm_{ds}*.json"))):
        for r in json.load(open(f)):
            if 5 <= r["seed"] <= 9:
                out.setdefault(r["variant"], {}).setdefault(r["seed"], r)
    lf = SEL / f"length_{ds}.json"
    if lf.exists():
        for r in json.load(open(lf)):
            out.setdefault(r["variant"], {}).setdefault(r["seed"], r)
    return out


def pair(R, a, b, k):
    s = sorted(set(R.get(a, {})) & set(R.get(b, {})))
    if len(s) < 2:
        return None
    x = np.array([R[a][i][k] for i in s]); y = np.array([R[b][i][k] for i in s])
    p = float(stats.ttest_rel(x, y).pvalue) if (x - y).std() > 0 else 1.0
    return float((x - y).mean()), p, len(s)


def cell(t):
    if t is None:
        return "—"
    d, p, n = t
    b = "**" if p < 0.05 else ""
    return f"{b}{d:+.3f}{b} (p {p:.2f})" + ("" if n == 5 else f" [{n}]")


def main() -> None:
    rows = []
    for ds, name in DS:
        R = runs(ds)
        m = lambda v, k: (f"{np.mean([r[k] for r in R[v].values()]):.3f}" if len(R.get(v, {})) else "—")
        ep = lambda v: (f"{np.mean([r['epochs'] for r in R[v].values()]):.0f}" if len(R.get(v, {})) else "—")
        rows.append((name, R, m, ep))
    L = ["## Values (mean over fresh seeds 5–9)\n",
         "| dataset | TVDiag, mean agg. | TVDiag-LSTM | AdaDiag-DL, mean agg. | AdaDiag-DL-LSTM | epochs (TVDiag-LSTM / AdaDiag-DL-LSTM) |",
         "|---|---|---|---|---|---|"]
    for name, R, m, ep in rows:
        L.append(f"| {name} | " + " | ".join(f"{m(v, 'HR@1')} / {m(v, 'f1')}" for v in
                                           ("tvdiag", "tvdiag_lstm", "align", "align_lstm"))
                 + f" | {ep('tvdiag_lstm')} / {ep('align_lstm')} |")
    L += ["\nHR@1 / F1 in each cell.\n", "## Paired differences (same seeds)\n",
          "| dataset | metric | LSTM vs mean, TVDiag | **AdaDiag-DL-LSTM − TVDiag-LSTM** | − TVDiag-LSTM stopped at the same epoch | AdaDiag-DL-LSTM − AdaDiag-DL (mean) |",
          "|---|---|---|---|---|---|"]
    for name, R, _, _ in rows:
        for k, lab in M:
            L.append(f"| {name} | {lab} | {cell(pair(R, 'tvdiag_lstm', 'tvdiag', k))} | "
                     f"{cell(pair(R, 'align_lstm', 'tvdiag_lstm', k))} | "
                     f"{cell(pair(R, 'align_lstm', 'capped:align_lstm', k))} | "
                     f"{cell(pair(R, 'align_lstm', 'align', k))} |")
    L.append("\nPaired t-test over seeds 5–9, uncorrected; bold: p < 0.05. [n]: fewer than 5 "
             "seeds. \"Stopped at the same epoch\": TVDiag-LSTM re-trained with each seed's cap set "
             "to the epoch AdaDiag-DL-LSTM stopped at (§6.13). On Hotel the two stop at the same "
             "epochs, so it is not run there.\n")
    md = ["# The LSTM aggregator (README §6.18)\n",
          "TVDiag's paper specifies GraphSAGE with an LSTM aggregator; its released code uses mean. "
          "`aggregator=\"lstm\"` reproduces dgl's LSTM reducer (`encoders.lstm_neighbours`). Both "
          "arms use it, with the same seeds, 1 thread, 500-epoch cap and aligned dropout masks as "
          "§6.17.\n", *L]
    (SEL / "LSTM.md").write_text("\n".join(md))
    (SEL / "lstm_table.md").write_text("\n".join(L))
    print((SEL / "LSTM.md").read_text())


if __name__ == "__main__":
    main()
