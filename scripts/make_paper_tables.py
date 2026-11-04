#!/usr/bin/env python3
"""Tables 4 and 5 of the TVDiag paper, in the paper's own format, with AdaDiag added.

Rows from the paper (every baseline and TVDiag) are copied exactly as printed
(`adadiag.paper_numbers`). Rows from this repository are added under each dataset:

    TVDiag (re-run)   this repository's TVDiag, same code as the AdaDiag rows
    AdaDiag-CL        AdaDiag as proposed: weights learned from the contrastive term
    AdaDiag-DL        AdaDiag selected in README section 6.17: weights from the diagnosis losses
    -LSTM             TVDiag and AdaDiag-DL with the paper's LSTM aggregator (section 6.18)
    -Spec             both with the six features the paper specifies and its code omits (6.19)
    -θ                both with Eq. 13's learnable weights only (6.19; Table 6 only)

All rows: fresh seeds 5-9 (never used to choose a variant), 1 thread, 500-epoch cap,
`isolate_weighting_rng` (results/selection/confirm_<ds>*.json). Dataset B (AIOps-22) is not
distributed with TVDiag, so it has no rows from this repository.

Columns follow the paper. In Tables 4 and 5, "⇑" is the paper's improvement rate of TVDiag
(as published) over the row, (TVDiag - row) / row; the p-value of a row from this repository
is a two-sided one-sample t-test of its seeds against TVDiag's published value (HR@1 in
Table 4, F1 in Table 5). Table 6 is the fair comparison: AdaDiag against the TVDiag re-run,
paired over the same seeds.

Writes results/paper_tables/: table4.tex, table5.tex, table6.tex, tables.tex (a standalone
document that \\input{}s them), tables.pdf, tables.csv.

    python scripts/make_paper_tables.py
"""
from __future__ import annotations

import csv
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adadiag import paper_numbers as P                                 # noqa: E402

OUT = ROOT / "results" / "paper_tables"
LETTERS = ("A", "B", "C", "D")
DS = {"A": "gaia", "C": "sockshop", "D": "hotel"}
OURS = (("tvdiag", "TVDiag (re-run)"), ("adadiag", "AdaDiag-CL"), ("align", "AdaDiag-DL"),
        # README section 6.18: both arms with the LSTM aggregator the paper describes
        ("tvdiag_lstm", "TVDiag-LSTM (re-run)"), ("align_lstm", "AdaDiag-DL-LSTM"),
        # README section 6.19: both arms with the six features the paper's text specifies and
        # its released code omits (config.PAPER)
        ("tvdiag_paper", "TVDiag-Spec (re-run)"), ("align_paper", "AdaDiag-DL-Spec"))
#: Table 6 only (Tables 4-5 are full): both arms with Eq. 13's learnable theta alone, the one
#: paper feature that did not hurt TVDiag (README section 6.19)
OURS6 = OURS + (("tvdiag+awl_learnable", "TVDiag-θ (re-run)"), ("align+awl_learnable", "AdaDiag-DL-θ"))
#: each row's paired reference in Table 6 (None: it is a reference)
REF = {"tvdiag": None, "adadiag": "tvdiag", "align": "tvdiag",
       "tvdiag_lstm": None, "align_lstm": "tvdiag_lstm",
       "tvdiag_paper": None, "align_paper": "tvdiag_paper",
       "tvdiag+awl_learnable": None, "align+awl_learnable": "tvdiag+awl_learnable"}
BOLD = ("align", "align_lstm")
SEEDS = range(5, 10)
GET = {"HR@1": lambda r: r["HR@1"], "HR@3": lambda r: r["HR@3"],
       "Avg@3": lambda r: (r["HR@1"] + r["HR@2"] + r["HR@3"]) / 3, "MRR@3": lambda r: r["MRR@3"],
       "Precision": lambda r: r["precision"], "Recall": lambda r: r["recall"],
       "F1-score": lambda r: r["f1"]}
T4 = ("HR@1", "HR@3", "Avg@3", "MRR@3")
T5 = ("Precision", "Recall", "F1-score")


# --------------------------------------------------------------------------- data
def runs(letter):
    """{variant: {seed: record}} for the fresh seeds, or {} when not run."""
    if letter not in DS:
        return {}
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "selection" / f"confirm_{DS[letter]}*.json"))):
        for r in json.load(open(f)):
            if r["seed"] in SEEDS:
                out.setdefault(r["variant"], {}).setdefault(r["seed"], r)
    return out


def paper_rows(letter, table):
    if table == 4:
        return [dict(modality=m.replace("Multimodal", "Multimodality"), method=n, p=p,
                     values=dict(zip(T4, v)), paper=True)
                for m, n, p, *v in P.TABLE4[letter]]
    return [dict(modality=m.replace("Multimodal", "Multimodality"), method=n, p=p,
                 values=dict(zip(T5, v)), paper=True)
            for m, n, p, *v in P.TABLE5[letter]]


def our_rows(letter, table):
    cols = T4 if table == 4 else T5
    key = "HR@1" if table == 4 else "F1-score"
    tv_paper = paper_rows(letter, table)[-1]["values"][key]
    R = runs(letter)
    rows = []
    for v, name in OURS:
        rs = R.get(v, {})
        if len(rs) < 2:
            continue
        vals = {c: float(np.mean([GET[c](r) for r in rs.values()])) for c in cols}
        sds = {c: float(np.std([GET[c](r) for r in rs.values()], ddof=1)) for c in cols}
        x = np.array([GET[key](r) for r in rs.values()])
        p = float(stats.ttest_1samp(x, tv_paper).pvalue) if x.std() > 0 else float("nan")
        rows.append(dict(modality="Multimodality", method=name, p=f"{p:.1E}", values=vals,
                         sd=sds, n=len(rs), paper=False, key=v))
    return rows


# --------------------------------------------------------------------------- LaTeX
def fmt(x):
    s = f"{x:.3f}"
    return s


def up(tv, x):
    """The paper's improvement rate of TVDiag over a row, in percent."""
    return (tv - x) / x * 100


def tex_table(table):
    cols = T4 if table == 4 else T5
    head = {"HR@1": r"\textit{HR}@1", "HR@3": r"\textit{HR}@3", "Avg@3": r"\textit{Avg}@3",
            "MRR@3": r"\textit{MRR}@3", "Precision": r"\textit{Precision}",
            "Recall": r"\textit{Recall}", "F1-score": "F1-score"}
    L = [r"\begin{table}[!ht]", r"\centering",
         (r"\caption{Root Cause Localization Comparison}\label{tab:rcl}" if table == 4 else
          r"\caption{Failure Type Identification Comparison}\label{tab:fti}"),
         r"\footnotesize\setlength{\tabcolsep}{3.2pt}",
         r"\begin{tabular}{c c l c " + " ".join("c c" for _ in cols) + "}", r"\toprule",
         r"Dataset & Modality & Method & p-value & "
         + " & ".join(f"{head[c]} & $\\Uparrow$ {head[c]}" for c in cols) + r" \\", r"\midrule"]
    csv_rows = []
    for li, letter in enumerate(LETTERS):
        rows = paper_rows(letter, table)
        ours = our_rows(letter, table)
        tv = rows[-1]["values"]
        allr = rows + ours
        best = {c: max(r["values"][c] for r in allr) for c in cols}
        n_rows = len(allr) + (1 if letter not in DS else 0)
        first = True
        for r in allr:
            if not r["paper"] and r is ours[0]:
                L.append(r"\cdashline{2-" + str(4 + 2 * len(cols)) + "}")
            ds_cell = (rf"\multirow{{{n_rows}}}{{*}}{{$\mathcal{{{letter}}}$}}" if first else "")
            first = False
            name = r["method"]
            if name == "TVDiag" and r["paper"]:
                name = r"\textbf{\textit{TVDiag}}"
            elif not r["paper"]:
                name = (r"\textbf{" + name + r"}$^\dagger$" if r["key"] in BOLD
                        else name + r"$^\dagger$")
            cells = []
            for c in cols:
                x = r["values"][c]
                v = fmt(x)
                if abs(x - best[c]) < 5e-4:
                    v = r"\textbf{" + v + "}"
                is_tv = r["paper"] and r["method"] == "TVDiag"
                u = "-" if is_tv else f"{up(tv[c], x):.3f}\\%".replace("-", "$-$")
                cells += [v, u]
            L.append(f"{ds_cell} & {r['modality']} & {name} & {r['p']} & " + " & ".join(cells) + r" \\")
            csv_rows.append(dict(table=table, dataset=letter, modality=r["modality"],
                                 method=r["method"], source="paper" if r["paper"] else "this repository",
                                 p_value=r["p"], **{c: round(r["values"][c], 6) for c in cols}))
        if letter not in DS:
            L.append(r" & \multicolumn{" + str(3 + 2 * len(cols)) + r"}{l}{\footnotesize\itshape "
                     r"AdaDiag: not run, dataset $\mathcal{B}$ (AIOps-22) is not distributed with "
                     r"TVDiag's code} \\")
        L.append(r"\bottomrule" if li == len(LETTERS) - 1 else r"\midrule")
    L += [r"\end{tabular}", r"\par\vspace{4pt}",
          r"\begin{minipage}{\linewidth}\footnotesize",
          (r"$\Uparrow$ indicates the improvement rate of \textit{TVDiag} compared to other methods "
           r"(as in the paper; negative: the row is above \textit{TVDiag}). For clarity, the best "
           r"result in each metric is shown in bold." if table == 4 else
           r"$\Uparrow$ indicates the improvement rate of \textit{TVDiag} compared to other methods "
           r"(as in the paper; negative: the row is above \textit{TVDiag}). The best performance for "
           r"each evaluation metric is indicated in bold."),
          r" Rows without $\dagger$ are copied from the paper (Table " + str(table) + r"). "
          r"$^\dagger$Rows below the dashed line are from this repository: mean over 5 fresh seeds "
          r"(5--9), same code for all three; p-value: two-sided one-sample $t$-test of the seeds "
          r"against \textit{TVDiag}'s published " + ("HR@1" if table == 4 else "F1-score") + r". "
          r"AdaDiag-CL: weights learned from the contrastive term (as proposed). "
          r"\textbf{AdaDiag-DL}: weights learned from the diagnosis losses (the variant selected "
          r"in README \S6.17). Rows marked -LSTM: the same two arms with the LSTM neighbour "
          r"aggregator the paper describes (its released code uses mean; README \S6.18). Rows "
          r"marked -Spec: the same two arms with the six features the paper's text specifies and "
          r"its released code omits (Eq.~3 per anchor, Eqs.~5--6 symmetric, Eq.~13 learnable, "
          r"Algorithm~1's per-layer normalisation, bidirectional edges, one AUG copy per sample; "
          r"README \S6.19). The paired comparison with the re-run is Table~\ref{tab:paired}.",
          r"\end{minipage}", r"\end{table}"]
    return "\n".join(L) + "\n", csv_rows


def tex_paired():
    """Table 6: AdaDiag against the TVDiag re-run, same code, same 5 seeds, paired."""
    cols = T4 + T5
    short = {"HR@1": "HR@1", "HR@3": "HR@3", "Avg@3": "Avg@3", "MRR@3": "MRR@3",
             "Precision": "P", "Recall": "R", "F1-score": "F1"}
    L = [r"\begin{table}[!ht]", r"\centering",
         r"\caption{AdaDiag against TVDiag under identical conditions (this repository; paired "
         r"over the same 5 seeds)}\label{tab:paired}",
         r"\small\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{c l " + "c " * len(cols) + "}", r"\toprule",
         r" & & \multicolumn{4}{c}{Root cause localization} & \multicolumn{3}{c}{Failure type identification} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}",
         r"Dataset & Method & " + " & ".join(short[c] for c in cols) + r" \\", r"\midrule"]
    letters = [l for l in LETTERS if l in DS and runs(l)]
    for li, letter in enumerate(letters):
        R = runs(letter)
        present = [(v, name.replace("θ", r"$\theta$")) for v, name in OURS6
                   if len(R.get(v, {})) > 1 and (REF[v] is None or len(R.get(REF[v], {})) > 1)]
        for vi, (v, name) in enumerate(present):
            rs = R[v]
            tv = R[REF[v]] if REF[v] else None
            if vi and REF[v] is None:                       # a new reference: the LSTM block
                L.append(r"\cdashline{2-" + str(2 + len(cols)) + "}")
            ds_cell = rf"\multirow{{{len(present)}}}{{*}}{{$\mathcal{{{letter}}}$}}" if vi == 0 else ""
            cells = []
            for c in cols:
                x = np.array([GET[c](rs[s]) for s in sorted(rs)])
                if REF[v] is None:
                    cells.append(f"{x.mean():.3f}" + r"{\scriptsize$\pm$" + f"{x.std(ddof=1):.3f}" + "}")
                    continue
                s_ = sorted(set(rs) & set(tv))
                a = np.array([GET[c](rs[s]) for s in s_])
                b = np.array([GET[c](tv[s]) for s in s_])
                d = float((a - b).mean())
                p = float(stats.ttest_rel(a, b).pvalue) if (a - b).std() > 0 else 1.0
                mark = "$^{*}$" if p < 0.05 else ""
                d = 0.0 if abs(d) < 5e-4 else d               # no "-0.000"
                ds_ = f"{d:+.3f}".replace("-", "$-$").replace("+", "$+$")
                cells.append(f"{a.mean():.3f}" + r"{\scriptsize(" + ds_ + ")}" + mark)
            nm = r"\textbf{" + name + "}" if v in BOLD else name
            L.append(f"{ds_cell} & {nm} & " + " & ".join(cells) + r" \\")
        L.append(r"\bottomrule" if li == len(letters) - 1 else r"\midrule")
    L += [r"\end{tabular}", r"\par\vspace{4pt}",
          r"\begin{minipage}{\linewidth}\footnotesize",
          r"TVDiag rows: mean {\scriptsize$\pm$ sd} over seeds 5--9. AdaDiag rows: mean "
          r"{\scriptsize(difference from the TVDiag re-run above it, same seeds and same "
          r"settings)}; $^{*}$ paired $t$-test $p<0.05$ (uncorrected). First block: TVDiag's "
          r"released configuration (mean aggregator). Below the first dashed line both arms use "
          r"the LSTM neighbour aggregator the paper describes; below the second (-Spec), both use "
          r"the six features the paper's text specifies and its code omits (README \S6.19); "
          r"below the third (-$\theta$), both use only one of them, Eq.~13's learnable weights. "
          r"All rows: 1 thread, 500-epoch cap, identical "
          r"dropout masks (\texttt{isolate\_weighting\_rng}). F1 is the paper's definition "
          r"(harmonic mean of weighted precision and weighted recall).",
          r"\end{minipage}", r"\end{table}"]
    return "\n".join(L) + "\n"


DOC = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=1.6cm]{geometry}
\usepackage{libertine}
\usepackage[libertine]{newtxmath}
\usepackage{booktabs,multirow,arydshln,graphicx}
\usepackage[labelfont=rm,textfont=rm,labelsep=period,skip=4pt]{caption}
\setlength\dashlinedash{2pt}\setlength\dashlinegap{2pt}
\makeatletter\setlength{\@fptop}{0pt}\makeatother
\setcounter{table}{3}
\begin{document}
\begin{center}
{\large\textbf{AdaDiag in the format of the TVDiag paper: Tables 4 and 5}}\\[2pt]
{\small Shehreyar Rashid (26I-9004) \quad Research Topics in Cloud Computing \quad
\url{https://github.com/ShehreyarRashid/AdaDiag}}
\end{center}
\noindent\small Baselines and \textit{TVDiag} are copied from Xie et al., \textit{TVDiag}, ACM TOSEM
35(2), Art.~40, 2026 (Tables 4--5). Rows marked $\dagger$ are this repository's re-implementation on
the released datasets $\mathcal{A}$ (GAIA), $\mathcal{C}$ (Sock Shop) and $\mathcal{D}$ (Hotel
Reservation). Table~\ref{tab:paired} compares AdaDiag with \textit{TVDiag} under identical code and
seeds, which is the comparison that isolates AdaDiag's effect; the gap between the re-run and the
published \textit{TVDiag} reflects the re-implementation (README \S6.1, \S6.9).
\normalsize
\input{table4.tex}
\clearpage
\input{table5.tex}
\clearpage
KEYRESULTS
\setcounter{table}{5}
\input{table6.tex}
\end{document}
"""


def key_results() -> str:
    """The three findings above Table 6, computed from the same runs."""
    d = {}
    for letter in DS:
        R = runs(letter)
        tv, dl = R.get("tvdiag", {}), R.get("align", {})
        s_ = sorted(set(tv) & set(dl))
        for k, c in (("HR@1", "HR@1"), ("F1", "F1-score")):
            a = np.array([GET[c](dl[i]) for i in s_]); b = np.array([GET[c](tv[i]) for i in s_])
            d[(letter, k)] = (float((a - b).mean()), float(stats.ttest_rel(a, b).pvalue))
    sig = [x for x, (m, p) in d.items() if p < 0.05]
    f = lambda x: f"{x:+.3f}".replace("-", "$-$").replace("+", "$+$")
    hr = ", ".join(f"$\\mathcal{{{l}}}$ {f(d[(l, 'HR@1')][0])}" for l in DS)
    f1 = ", ".join(f"$\\mathcal{{{l}}}$ {f(d[(l, 'F1')][0])}" for l in DS)
    gap = max(up(paper_rows(l, 4)[-1]["values"]["HR@1"],
                 next(r for r in our_rows(l, 4) if r["key"] == "align")["values"]["HR@1"])
              for l in ("A", "D"))
    # which metrics every AdaDiag row wins against every published baseline (TVDiag excluded)
    won, lost = [], []
    for table, cols in ((4, T4), (5, T5)):
        for c in cols:
            ok = True
            for l in DS:
                base = max(r["values"][c] for r in paper_rows(l, table) if r["method"] != "TVDiag")
                for r in our_rows(l, table):
                    if (r["key"] in ("adadiag", "align", "align_lstm")
                            and r["values"][c] <= base):
                        ok = False
                        lost.append((l, c, r["method"], base - r["values"][c]))
            (won if ok else []).append(c)
    c4 = {r["method"]: r["values"]["HR@1"] for r in paper_rows("C", 4) + our_rows("C", 4)}
    c5 = {r["method"]: r["values"]["F1-score"] for r in paper_rows("C", 5) + our_rows("C", 5)}
    return (r"\noindent\textbf{Key results.}" "\n" r"\begin{itemize}\setlength\itemsep{1pt}" "\n"
            r"\item \textbf{Against TVDiag under identical conditions} (Table~6), AdaDiag-DL changes "
            f"HR@1 by {hr} and F1 by {f1}; "
            + ("none of these differences is significant (paired $t$-test, $p\\geq 0.05$)."
               if not sig else f"{len(sig)} of them are significant (paired $t$-test, $p<0.05$).")
            + "\n" r"\item \textbf{Against the published TVDiag} (Tables 4--5), AdaDiag-DL is within "
            + f"{np.ceil(gap * 10) / 10:.1f}" + r"\% on HR@1 for $\mathcal{A}$ and $\mathcal{D}$, and it has the highest HR@1 on "
            f"$\\mathcal{{C}}$ ({c4['AdaDiag-DL']:.3f} against {c4['TVDiag']:.3f}). Its F1 on "
            f"$\\mathcal{{C}}$ ({c5['AdaDiag-DL']:.3f}) is below the paper's {c5['TVDiag']:.3f}, as is "
            f"the TVDiag re-run's ({c5['TVDiag (re-run)']:.3f}); that gap is the re-implementation's, "
            r"not AdaDiag's." "\n"
            r"\item \textbf{Every AdaDiag row beats every baseline in the paper} on "
            + ", ".join(w.replace("F1-score", "F1") for w in won)
            + r" on $\mathcal{A}$, $\mathcal{C}$ and $\mathcal{D}$"
            + ("" if not lost else
               ", and on every other metric except " + "; ".join(
                   " and ".join(sorted({c for l2, c, _, _ in lost if l2 == l}, key=(T4 + T5).index))
                   + f" on $\\mathcal{{{l}}}$" for l in sorted({l for l, _, _, _ in lost}))
               + ", where " + " and ".join(sorted({m for _, _, m, _ in lost}))
               + (" is" if len({m for _, _, m, _ in lost}) == 1 else " are")
               + f" at most {max(g for *_, g in lost) + 5e-4:.3f} below the best baseline")
            + r". Like TVDiag, AdaDiag clears the baselines. It keeps TVDiag's accuracy without adding to it."
            "\n" + lstm_bullet() + spec_bullet() + theta_bullet() + r"\end{itemize}" "\n")


def _pair(R, a, b, c):
    s_ = sorted(set(R.get(a, {})) & set(R.get(b, {})))
    if len(s_) < 2:
        return None
    x = np.array([GET[c](R[a][i]) for i in s_]); y = np.array([GET[c](R[b][i]) for i in s_])
    p = float(stats.ttest_rel(x, y).pvalue) if (x - y).std() > 0 else 1.0
    return float((x - y).mean()), p, len(s_)


def lstm_bullet() -> str:
    """README section 6.18, from the runs: what the LSTM aggregator does, and AdaDiag with it."""
    f = lambda x: f"{x:+.3f}".replace("-", "$-$").replace("+", "$+$")
    agg, ada, ps = [], [], []
    for l in DS:
        R = runs(l)
        h, f1 = _pair(R, "tvdiag_lstm", "tvdiag", "HR@1"), _pair(R, "tvdiag_lstm", "tvdiag", "F1-score")
        if h:
            agg.append(f"$\\mathcal{{{l}}}$ {f(h[0])} / {f(f1[0])}")
        h, f1 = _pair(R, "align_lstm", "tvdiag_lstm", "HR@1"), _pair(R, "align_lstm", "tvdiag_lstm", "F1-score")
        if h:
            ps += [(l, "HR@1", h), (l, "F1", f1)]
            star = lambda t: "$^{*}$" if t[1] < 0.05 else ""
            ada.append(f"$\\mathcal{{{l}}}$ {f(h[0])}{star(h)} / {f(f1[0])}{star(f1)}"
                       + ("" if h[2] == 5 else f" ({h[2]} seeds)"))
    if not agg:
        return ""
    # training-length control: TVDiag-LSTM stopped at AdaDiag-DL-LSTM's epoch, seed by seed
    parts = []
    for l in DS:
        lf = ROOT / "results" / "selection" / f"length_{DS[l]}.json"
        if not lf.exists():
            continue
        cap = {r["seed"]: r for r in json.load(open(lf)) if r["variant"] == "capped:align_lstm"}
        dl = runs(l).get("align_lstm", {})
        s_ = sorted(set(cap) & set(dl))
        if len(s_) < 2:
            continue
        cells = []
        for k in ("HR@1", "f1"):
            x = np.array([dl[i][k] for i in s_]); y = np.array([cap[i][k] for i in s_])
            pv = float(stats.ttest_rel(x, y).pvalue) if (x - y).std() > 0 else 1.0
            cells.append(f"{f(float((x - y).mean()))}" + ("$^{*}$" if pv < 0.05 else ""))
        parts.append(f"$\\mathcal{{{l}}}$ {cells[0]} / {cells[1]}"
                     + ("" if len(s_) == 5 else f" ({len(s_)} seeds)"))
    ctl = ("" if not parts else
           r" It stops earlier than TVDiag-LSTM on $\mathcal{A}$ and $\mathcal{C}$; against "
           r"TVDiag-LSTM stopped at the same epoch, HR@1 / F1: " + ", ".join(parts) + ".")
    # Holm over the AdaDiag-DL-LSTM - TVDiag-LSTM family (datasets x HR@1 / F1)
    order = sorted(range(len(ps)), key=lambda i: ps[i][2][1])
    adj, run = [0.0] * len(ps), 0.0
    for r_, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r_) * ps[i][2][1]))
        adj[i] = run
    surv = [(ps[i], adj[i]) for i in range(len(ps)) if adj[i] < 0.05]
    if surv:
        ctl += (" After Holm correction over the " + str(len(ps)) + " comparisons, "
                + ("only " if len(surv) == 1 else "")
                + ", ".join(f"$\\mathcal{{{l}}}$ {k} ({f(t[0])}, adjusted $p={a:.2f}$)"
                            for (l, k, t), a in surv)
                + (" remains" if len(surv) == 1 else " remain")
                + r" significant; the training-length control above shrinks it. With the mean "
                  r"aggregator AdaDiag-DL is at least as accurate and trains several times faster.")
    else:
        ctl += (r" No AdaDiag-DL-LSTM $-$ TVDiag-LSTM difference survives Holm correction over the "
                f"{len(ps)} comparisons (smallest adjusted $p={min(adj):.2f}$); with the mean "
                r"aggregator AdaDiag-DL is at least as accurate and trains several times faster.")
    return (r"\item \textbf{With the LSTM aggregator the paper describes} (rows -LSTM; Table~6, "
            r"lower blocks). The aggregator alone changes TVDiag's HR@1 / F1 by " + ", ".join(agg)
            + ". AdaDiag-DL-LSTM against TVDiag-LSTM (same seeds): " + ", ".join(ada)
            + r" ($^{*}$: paired $p<0.05$, uncorrected)." + ctl + "\n")


def spec_bullet() -> str:
    """README section 6.19, from the runs: TVDiag as its paper specifies it, and AdaDiag with it."""
    f = lambda x: ("0.000" if abs(x) < 5e-4 else f"{x:+.3f}").replace("-", "$-$").replace("+", "$+$")
    tv, ada, ps, further, cells = [], [], [], 0, 0
    for l in DS:
        R = runs(l)
        for c, table in (("HR@1", 4), ("F1-score", 5)):
            pub = paper_rows(l, table)[-1]["values"][c]
            m = {v: np.mean([GET[c](r) for r in R.get(v, {}).values()]) for v in ("tvdiag", "tvdiag_paper")
                 if R.get(v)}
            if len(m) == 2:
                cells += 1
                further += abs(m["tvdiag_paper"] - pub) > abs(m["tvdiag"] - pub)
        h, f1 = _pair(R, "tvdiag_paper", "tvdiag", "HR@1"), _pair(R, "tvdiag_paper", "tvdiag", "F1-score")
        if h:
            star = lambda t: "$^{*}$" if t[1] < 0.05 else ""
            tv.append(f"$\\mathcal{{{l}}}$ {f(h[0])}{star(h)} / {f(f1[0])}{star(f1)}")
        h, f1 = _pair(R, "align_paper", "tvdiag_paper", "HR@1"), _pair(R, "align_paper", "tvdiag_paper", "F1-score")
        if h:
            ps += [h, f1]
            ada.append(f"$\\mathcal{{{l}}}$ {f(h[0])} / {f(f1[0])}")
    if not tv:
        return ""
    order = sorted(range(len(ps)), key=lambda i: ps[i][1])
    adj, run = [0.0] * len(ps), 0.0
    for r_, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r_) * ps[i][1]))
        adj[i] = run
    return (r"\item \textbf{As the paper's text specifies it} (rows -Spec), TVDiag is worse than its "
            r"released code: HR@1 / F1 change by " + ", ".join(tv) + r" ($^{*}$: paired $p<0.05$). "
            r"Most of the loss comes from Algorithm~1's per-layer normalisation and from the single "
            r"AUG copy per sample (README \S6.19). It moves the re-run further from the "
            f"published HR@1 / F1 in {further} of {cells} cells. "
            r"AdaDiag-DL-Spec against TVDiag-Spec: " + ", ".join(ada)
            + ((f"; none is significant even before correction (smallest $p={min(t[1] for t in ps):.2f}$)."
                if min(t[1] for t in ps) >= 0.05 else
                f"; no difference survives Holm correction (smallest adjusted $p={min(adj):.2f}$).")
               if ps and min(adj) >= 0.05 else ".") + "\n")


def theta_bullet() -> str:
    """README section 6.19: Eq. 13's learnable theta alone, for TVDiag and AdaDiag-DL."""
    f = lambda x: ("0.000" if abs(x) < 5e-4 else f"{x:+.3f}").replace("-", "$-$").replace("+", "$+$")
    star = lambda t: "$^{*}$" if t[1] < 0.05 else ""
    tv, ada, ps, tot, own = [], [], [], [], []
    for l in DS:
        R = runs(l)
        g = _pair(R, "align+awl_learnable", "tvdiag", "F1-score")
        o = _pair(R, "tvdiag+awl_learnable", "tvdiag", "F1-score")
        if g and o:
            tot.append(f"$\\mathcal{{{l}}}$ {f(g[0])}")
            own.append(f(o[0]))
        h, f1 = (_pair(R, "tvdiag+awl_learnable", "tvdiag", c) for c in ("HR@1", "F1-score"))
        if h:
            tv.append(f"$\\mathcal{{{l}}}$ {f(h[0])}{star(h)} / {f(f1[0])}{star(f1)}")
        h, f1 = (_pair(R, "align+awl_learnable", "tvdiag+awl_learnable", c) for c in ("HR@1", "F1-score"))
        if h:
            ps += [h, f1]
            ada.append(f"$\\mathcal{{{l}}}$ {f(h[0])}{star(h)} / {f(f1[0])}{star(f1)}")
    if not ada:
        return ""
    order = sorted(range(len(ps)), key=lambda i: ps[i][1])
    adj, run = [0.0] * len(ps), 0.0
    for r_, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r_) * ps[i][1]))
        adj[i] = run
    return (r"\item \textbf{With only Eq.~13's learnable weights} (rows -$\theta$; Table~6, last "
            r"block), the one paper feature that does not hurt TVDiag: it changes TVDiag's HR@1 / F1 "
            r"by " + ", ".join(tv) + r". AdaDiag-DL-$\theta$ against TVDiag-$\theta$: " + ", ".join(ada)
            + r" ($^{*}$: paired $p<0.05$, uncorrected); "
            + ("no difference survives Holm correction over the " + str(len(ps))
               + f" comparisons (smallest adjusted $p={min(adj):.2f}$)" if min(adj) >= 0.05 else
               "some differences survive Holm correction")
            + r". Of AdaDiag-DL-$\theta$'s F1 gain over the released TVDiag (" + ", ".join(tot)
            + r"), $\theta$ alone gives TVDiag " + ", ".join(own) + "." "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t4, c4 = tex_table(4)
    t5, c5 = tex_table(5)
    (OUT / "table4.tex").write_text(t4)
    (OUT / "table5.tex").write_text(t5)
    (OUT / "table6.tex").write_text(tex_paired())
    (OUT / "tables.tex").write_text(DOC.replace(r"\url{https://github.com/ShehreyarRashid/AdaDiag}",
                                                r"github.com/ShehreyarRashid/AdaDiag")
                                    .replace("KEYRESULTS", key_results()))
    with open(OUT / "tables.csv", "w", newline="") as f:
        keys = ["table", "dataset", "modality", "method", "source", "p_value", *T4, *T5]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in c4 + c5:
            w.writerow(r)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "tables.tex"], cwd=OUT,
                           capture_output=True, text=True)
    for ext in ("aux", "log"):
        (OUT / f"tables.{ext}").unlink(missing_ok=True)
    print("pdflatex exit", r.returncode, "->", OUT / "tables.pdf")


if __name__ == "__main__":
    main()
