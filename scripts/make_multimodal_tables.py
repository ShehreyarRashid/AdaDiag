#!/usr/bin/env python3
"""The TVDiag paper's Tables 4 and 5, multimodal methods only, with the best AdaDiag added.

Every row except AdaDiag's is copied exactly as the paper prints it
(`adadiag.paper_numbers`): the multimodal baselines and TVDiag, with their p-values and the
paper's improvement-rate columns. Single-modality baselines are left out.

"AdaDiag (best)" is AdaDiag-DL with Eq. 13's learnable weights (README section 6.19), mean
over fresh seeds 5-9 (results/selection/confirm_<ds>*.json). It is the AdaDiag
configuration with the highest mean of HR@1 and F1 over A, C and D among those run on
fresh seeds (`CANDIDATES` below, chosen after the runs: a post-hoc choice, stated in the
table notes). Its p-value is a one-sample t-test of the five seeds against TVDiag's
published value (HR@1 in Table 4, F1 in Table 5). Dataset B is not released, so it has no
AdaDiag row.

Writes results/paper_tables/multimodal/: table4_mm.tex, table5_mm.tex, tables_mm.tex,
tables_mm.pdf, tables_mm.csv.

    python scripts/make_multimodal_tables.py
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

OUT = ROOT / "results" / "paper_tables" / "multimodal"
DS = {"A": "gaia", "C": "sockshop", "D": "hotel"}
LETTERS = ("A", "B", "C", "D")
T4 = ("HR@1", "HR@3", "Avg@3", "MRR@3")
T5 = ("Precision", "Recall", "F1-score")
GET = {"HR@1": lambda r: r["HR@1"], "HR@3": lambda r: r["HR@3"],
       "Avg@3": lambda r: (r["HR@1"] + r["HR@2"] + r["HR@3"]) / 3, "MRR@3": lambda r: r["MRR@3"],
       "Precision": lambda r: r["precision"], "Recall": lambda r: r["recall"],
       "F1-score": lambda r: r["f1"]}
#: AdaDiag configurations run on fresh seeds: (label, file pattern, variant, seeds)
CANDIDATES = (
    ("AdaDiag-DL", "selection/confirm_{}*.json", "align", range(5, 10)),
    ("AdaDiag-CL", "selection/confirm_{}*.json", "adadiag", range(5, 10)),
    ("AdaDiag-DL + learnable Eq. 13", "selection/confirm_{}*.json", "align+awl_learnable", range(5, 10)),
    ("AdaDiag-DL-LSTM", "selection/confirm_{}*.json", "align_lstm", range(5, 10)),
    ("AdaDiag-DL + neighbour contrast", "improve/confirm_{}.json", "align_improved", range(10, 15)),
)


def records(pattern, variant, seeds, ds):
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results" / pattern.format(ds)))):
        for r in json.load(open(f)):
            if r["variant"] == variant and r["seed"] in seeds:
                out.setdefault(r["seed"], r)
    return list(out.values())


def best() -> tuple:
    scored = []
    for lab, pat, v, seeds in CANDIDATES:
        rs = {L: records(pat, v, seeds, ds) for L, ds in DS.items()}
        if any(len(x) < 5 for x in rs.values()):
            continue
        s = np.mean([(np.mean([r["HR@1"] for r in x]) + np.mean([r["f1"] for r in x])) / 2
                     for x in rs.values()])
        scored.append((float(s), lab, v, seeds, rs))
    scored.sort(key=lambda t: -t[0])
    return scored[0], scored


def paper_rows(letter, table):
    rows = P.TABLE4[letter] if table == 4 else P.TABLE5[letter]
    cols = T4 if table == 4 else T5
    return [dict(method=n, p=p, values=dict(zip(cols, v)), paper=True)
            for m, n, p, *v in rows if m == "Multimodal"]


def ada_row(letter, table, rs):
    cols = T4 if table == 4 else T5
    key = "HR@1" if table == 4 else "F1-score"
    tv = paper_rows(letter, table)[-1]["values"][key]
    x = np.array([GET[key](r) for r in rs])
    p = float(stats.ttest_1samp(x, tv).pvalue)
    return dict(method="AdaDiag (best)", p=f"{p:.1E}", paper=False,
                values={c: float(np.mean([GET[c](r) for r in rs])) for c in cols},
                sd={c: float(np.std([GET[c](r) for r in rs], ddof=1)) for c in cols})


def tex_table(table, rs_by_letter, label):
    cols = T4 if table == 4 else T5
    head = {"HR@1": r"\textit{HR}@1", "HR@3": r"\textit{HR}@3", "Avg@3": r"\textit{Avg}@3",
            "MRR@3": r"\textit{MRR}@3", "Precision": r"\textit{Precision}",
            "Recall": r"\textit{Recall}", "F1-score": "F1-score"}
    cap = ("Root Cause Localization Comparison (multimodal methods)" if table == 4 else
           "Failure Type Identification Comparison (multimodal methods)")
    L = [r"\begin{table}[!ht]", r"\centering", rf"\caption{{{cap}}}\label{{tab:mm{table}}}",
         r"\small\setlength{\tabcolsep}{3.6pt}",
         r"\begin{tabular}{c l c " + " ".join("c c" for _ in cols) + "}", r"\toprule",
         r"Dataset & Method & p-value & "
         + " & ".join(f"{head[c]} & $\\Uparrow$ {head[c]}" for c in cols) + r" \\", r"\midrule"]
    csv_rows = []
    for li, letter in enumerate(LETTERS):
        rows = paper_rows(letter, table)
        tv = rows[-1]["values"]
        if letter in rs_by_letter:
            rows = rows + [ada_row(letter, table, rs_by_letter[letter])]
        best_v = {c: max(r["values"][c] for r in rows) for c in cols}
        n = len(rows) + (0 if letter in rs_by_letter else 1)
        for i, r in enumerate(rows):
            if not r["paper"]:
                L.append(r"\cdashline{2-" + str(3 + 2 * len(cols)) + "}")
            ds_cell = rf"\multirow{{{n}}}{{*}}{{$\mathcal{{{letter}}}$}}" if i == 0 else ""
            name = (r"\textbf{\textit{TVDiag}}" if r["method"] == "TVDiag" else
                    r"\textbf{AdaDiag (best)}$^\dagger$" if not r["paper"] else r["method"])
            cells = []
            for c in cols:
                x = r["values"][c]
                v = f"{x:.3f}"
                if abs(x - best_v[c]) < 5e-4:
                    v = r"\textbf{" + v + "}"
                u = ("-" if r["method"] == "TVDiag" else
                     f"{(tv[c] - x) / x * 100:.3f}\\%".replace("-", "$-$"))
                cells += [v, u]
            L.append(f"{ds_cell} & {name} & {r['p']} & " + " & ".join(cells) + r" \\")
            csv_rows.append(dict(table=table, dataset=letter, method=r["method"],
                                 source="paper" if r["paper"] else "this repository",
                                 p_value=r["p"], **{c: round(r["values"][c], 6) for c in cols}))
        if letter not in rs_by_letter:
            L.append(r" & \multicolumn{" + str(2 + 2 * len(cols)) + r"}{l}{\footnotesize\itshape "
                     r"AdaDiag: not run, dataset $\mathcal{B}$ (AIOps-22) is not released} \\")
        L.append(r"\bottomrule" if li == len(LETTERS) - 1 else r"\midrule")
    key = "HR@1" if table == 4 else "F1-score"
    L += [r"\end{tabular}", r"\par\vspace{4pt}", r"\begin{minipage}{\linewidth}\footnotesize",
          r"$\Uparrow$ indicates the improvement rate of \textit{TVDiag} compared to other methods, "
          r"as in the paper (negative: the row is above \textit{TVDiag}). The best result in each "
          r"metric is shown in bold. Rows without $\dagger$ are copied from the paper (Xie et al., "
          r"\textit{TVDiag}, ACM TOSEM 35(2), Art.~40, 2026, Table " + str(table) + r"); the "
          r"single-modality baselines are omitted. $^\dagger$\textbf{AdaDiag (best)}: " + label
          + r", mean over 5 seeds (5--9); p-value: two-sided one-sample $t$-test of the seeds "
          r"against \textit{TVDiag}'s published " + key.replace("F1-score", "F1-score") + r".",
          r"\end{minipage}", r"\end{table}"]
    return "\n".join(L) + "\n", csv_rows


DOC = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=1.8cm]{geometry}
\usepackage{libertine}
\usepackage[libertine]{newtxmath}
\usepackage{booktabs,multirow,arydshln}
\usepackage[labelfont=rm,textfont=rm,labelsep=period,skip=4pt]{caption}
\setlength\dashlinedash{2pt}\setlength\dashlinegap{2pt}
\makeatletter\setlength{\@fptop}{0pt}\makeatother
\setcounter{table}{3}
\begin{document}
\begin{center}
{\large\textbf{Multimodal methods and AdaDiag in the format of the TVDiag paper: Tables 4 and 5}}\\[2pt]
{\small Shehreyar Rashid (26I-9004) \quad Research Topics in Cloud Computing \quad
github.com/ShehreyarRashid/AdaDiag}
\end{center}
\noindent\small Baselines and \textit{TVDiag}: the paper's own numbers. AdaDiag: this
repository, on the released datasets $\mathcal{A}$ (GAIA), $\mathcal{C}$ (Sock Shop) and
$\mathcal{D}$ (Hotel Reservation). NOTE
\normalsize
\input{table4_mm.tex}
\input{table5_mm.tex}
\end{document}
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (score, lab, v, seeds, rs), scored = best()
    label = {"AdaDiag-DL + learnable Eq. 13":
             r"AdaDiag-DL (weights learned from the diagnosis losses, README \S6.17) with "
             r"Eq.~13's learnable loss weights (\S6.19)"}.get(lab, lab)
    t4, c4 = tex_table(4, rs, label)
    t5, c5 = tex_table(5, rs, label)
    (OUT / "table4_mm.tex").write_text(t4)
    (OUT / "table5_mm.tex").write_text(t5)
    note = (r"AdaDiag (best) is the AdaDiag configuration with the highest mean of HR@1 and F1 "
            r"over $\mathcal{A}$, $\mathcal{C}$ and $\mathcal{D}$ among the "
            + str(len(scored)) + r" run on fresh seeds; the choice was made after the runs. "
            r"Against \textit{TVDiag} re-run under identical conditions it is not significantly "
            r"different (README \S6.19), so the gaps to the published \textit{TVDiag} also reflect "
            r"the re-implementation (README \S6.1, \S6.9).")
    (OUT / "tables_mm.tex").write_text(DOC.replace("NOTE", note))
    with open(OUT / "tables_mm.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table", "dataset", "method", "source", "p_value",
                                          *T4, *T5])
        w.writeheader()
        for r in c4 + c5:
            w.writerow(r)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "tables_mm.tex"], cwd=OUT,
                       capture_output=True, text=True)
    for ext in ("aux", "log"):
        (OUT / f"tables_mm.{ext}").unlink(missing_ok=True)
    print(f"best: {lab} ({v}, seeds {seeds.start}-{seeds.stop - 1}), score {score:.4f}")
    for s, l, *_ in scored:
        print(f"  {s:.4f}  {l}")
    print("wrote", OUT / "tables_mm.pdf")


if __name__ == "__main__":
    main()
