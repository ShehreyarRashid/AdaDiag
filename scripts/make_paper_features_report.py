#!/usr/bin/env python3
"""README section 6.19: what the TVDiag paper specifies and its released code does not do.

Reads the fresh-seed runs (seeds 5-9) written by scripts/select_adadiag.py into
results/selection/confirm_<ds>*.json -- variants tvdiag, align (the code's version, from
section 6.17), tvdiag_paper, align_paper (all six paper switches, config.PAPER) and
tvdiag+<switch> (one switch at a time), align+awl_learnable (AdaDiag-DL with Eq. 13 only)
-- and the training-length controls in results/selection/length_<ds>.json
("capped:<variant>": the matched TVDiag stopped, seed by seed, at the epoch the variant
stopped at). Writes results/paper_features/:
PAPER_FEATURES.md, readme_table.md, summary.json, fig_paper_features.png.

    python scripts/make_paper_features_report.py
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
from adadiag import config, data, paper_numbers as P                   # noqa: E402

SEL = ROOT / "results" / "selection"
OUT = ROOT / "results" / "paper_features"
SEEDS = range(5, 10)
DS = (("gaia", "A · GAIA", "A"), ("sockshop", "C · Sock Shop", "C"), ("hotel", "D · Hotel", "D"))
GET = {"HR@1": lambda r: r["HR@1"],
       "Avg@3": lambda r: (r["HR@1"] + r["HR@2"] + r["HR@3"]) / 3,
       "MRR@3": lambda r: r["MRR@3"], "F1": lambda r: r["f1"],
       "macro-F1": lambda r: r["macro_f1"]}
PAIR_M = ("HR@1", "MRR@3", "F1", "macro-F1")
ARMS = (("tvdiag", "TVDiag, code"), ("tvdiag_noes", "TVDiag, code, no early stop"),
        ("tvdiag_paper", "TVDiag, paper"),
        ("align", "AdaDiag-DL, code"), ("align_paper", "AdaDiag-DL, paper"))
SWITCH = (("to_norm", "Eq. 3: TO averaged per anchor"),
          ("cm_mode", "Eqs. 5–6: CM symmetric + intra-modal negatives"),
          ("awl_learnable", "Eq. 13: θ learnable"),
          ("sage_alg1", "Alg. 1: σ and L2-normalise every layer"),
          ("bidirectional", "§4.2: bidirectional edges"),
          ("aug_mode", "§4.3: one AUG copy per incident"))
CATALOGUE = [
    ("1", "Eq. 3 (TO)", "each anchor's loss is the mean over its own positives (1/|Pᵢ|)",
     "one mean over every positive pair in the batch (`SupConLoss`: sum / non-zero count)",
     "`to_norm=\"paper\"`", "applied"),
    ("2", "Eqs. 5–6 (CM)", "symmetric, 1/2n over both directions; other samples of the same "
     "modality are negatives too", "one direction, metric → other; cross-modal negatives only "
     "(`UspConLoss`)", "`cm_mode=\"paper\"`", "applied"),
    ("3", "Eq. 13 (DW)", "θ₁…θ₄ are learnable", "`AutomaticWeightedLoss` is built after the "
     "optimizer, so θ stays 1: equal weights 0.5", "`awl_learnable=True`", "applied"),
    ("4", "Alg. 1, lines 6–7", "σ after every layer, then L2-normalise each node embedding",
     "ReLU after layer 1 only, no normalisation (`SAGEEncoder`)", "`sage_alg1=True`", "applied"),
    ("5", "§4.2", "edges are bidirectional", "edges used as given", "`bidirectional=True`",
     "applied (changes GAIA only)"),
    ("6", "§4.3 (AUG)", "one augmented graph per training sample, added to the training set",
     "10 copies per sample in a pool; each batch draws random copies, mostly of other incidents",
     "`aug_mode=\"paper\"`", "applied"),
    ("7", "§5.1", "LSTM aggregator", "mean", "`aggregator=\"lstm\"`", "applied in §6.18"),
    ("8", "§4.1", "alert extraction: 3-sigma, IForest, Drain, log Rules 1–2 (k = 0.5), fastText",
     "only the extracted 128-d embeddings are released", "—", "not possible: raw telemetry "
     "not released"),
    ("9", "§5.1", "dataset B (AIOps-22)", "not released", "—", "not possible"),
]


def runs(ds: str) -> dict:
    out: dict = {}
    for f in sorted(glob.glob(str(SEL / f"confirm_{ds}*.json"))):
        for r in json.load(open(f)):
            if r["seed"] in SEEDS:
                out.setdefault(r["variant"], {}).setdefault(r["seed"], r)
    lf = SEL / f"length_{ds}.json"
    if lf.exists():
        for r in json.load(open(lf)):
            if r["seed"] in SEEDS:
                out.setdefault(r["variant"], {}).setdefault(r["seed"], r)
    return out


def published(letter: str) -> dict:
    t4 = next(r for r in P.TABLE4[letter] if r[1] == "TVDiag")
    t5 = next(r for r in P.TABLE5[letter] if r[1] == "TVDiag")
    return {"HR@1": t4[3], "Avg@3": t4[5], "MRR@3": t4[6], "F1": t5[5]}


def one_way(ds) -> tuple[int, int]:
    A = ds.A.clone()
    A[:, torch_eye(A.shape[1])] = 0
    e = int((A > 0).sum())
    return int(((A > 0) & (A.transpose(1, 2) == 0)).sum()), e


def torch_eye(n):
    import torch
    return torch.eye(n, dtype=torch.bool)


def pair(R, a, b, k):
    s = sorted(set(R.get(a, {})) & set(R.get(b, {})))
    if len(s) < 2:
        return None
    x = np.array([GET[k](R[a][i]) for i in s])
    y = np.array([GET[k](R[b][i]) for i in s])
    d = x - y
    p = float(stats.ttest_rel(x, y).pvalue) if d.std() > 0 else 1.0
    h = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    return dict(d=float(d.mean()), p=p, n=len(s), ci=h)


def holm(cells: list) -> None:
    """Holm-adjust, in place, the 'p' of every non-None cell in the list (one family)."""
    live = [c for c in cells if c is not None]
    order = sorted(range(len(live)), key=lambda i: live[i]["p"])
    m, run = len(live), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * live[i]["p"]))
        live[i]["p_holm"] = run


def cell(c, holm_mark=True) -> str:
    if c is None:
        return "—"
    b = "**" if c["p"] < 0.05 else ""
    star = "†" if holm_mark and c.get("p_holm", 1.0) < 0.05 else ""
    return f"{b}{c['d']:+.3f}{b}{star} (p {c['p']:.2f})" + ("" if c["n"] == 5 else f" [{c['n']}]")


def mean_sd(R, v, k) -> str:
    if not R.get(v):
        return "—"
    x = np.array([GET[k](r) for r in R[v].values()])
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    R = {ds: runs(ds) for ds, _, _ in DS}
    loaded = {ds: data.load(ds) for ds, _, _ in DS}
    ow = {ds: one_way(loaded[ds]) for ds, _, _ in DS}
    summ: dict = {"seeds": list(SEEDS), "one_way_edges": {d: dict(one_way=a, edges=b)
                                                           for d, (a, b) in ow.items()}}

    # ---- values
    V = ["## Values (mean ± sd over fresh seeds 5–9)\n",
         "| dataset | arm | HR@1 | Avg@3 | MRR@3 | F1 | macro-F1 | epochs |",
         "|---|---|---|---|---|---|---|---|"]
    for ds, name, L in DS:
        pub = published(L)
        V.append(f"| {name} | TVDiag, published | {pub['HR@1']:.3f} | {pub['Avg@3']:.3f} | "
                 f"{pub['MRR@3']:.3f} | {pub['F1']:.3f} | — | — |")
        for v, lab in ARMS:
            if not R[ds].get(v):
                continue
            ep = np.mean([r["epochs"] for r in R[ds][v].values()])
            V.append(f"| {name} | {lab} | " + " | ".join(mean_sd(R[ds], v, k) for k in GET)
                     + f" | {ep:.0f} |")
        summ.setdefault("values", {})[ds] = {
            v: {k: float(np.mean([GET[k](r) for r in R[ds][v].values()])) for k in GET}
            | {"epochs": float(np.mean([r["epochs"] for r in R[ds][v].values()])),
               "train_s": float(np.mean([r["train_s"] for r in R[ds][v].values()])), "n": len(R[ds][v])}
            for v, _ in ARMS if R[ds].get(v)} | {"published": pub}

    # ---- paired families
    fams = {"paper_vs_code_tvdiag": ("tvdiag_paper", "tvdiag"),
            "dl_vs_tvdiag_paper": ("align_paper", "tvdiag_paper"),
            "dl_vs_tvdiag_paper_capped": ("align_paper", "capped:align_paper"),
            "dl_paper_vs_dl_code": ("align_paper", "align"),
            "dl_vs_tvdiag_code": ("align", "tvdiag"),
            "paper_vs_code_same_length": ("tvdiag_paper", "tvdiag_noes"),
            "awl_vs_same_length": ("tvdiag+awl_learnable", "tvdiag_noes"),
            "noes_vs_code": ("tvdiag_noes", "tvdiag")}
    F = {}
    for fam, (a, b) in fams.items():
        F[fam] = {(ds, k): pair(R[ds], a, b, k) for ds, _, _ in DS for k in PAIR_M}
        holm(list(F[fam].values()))
    summ["paired"] = {fam: {f"{ds}|{k}": c for (ds, k), c in cells.items()}
                      for fam, cells in F.items()}
    def ep_range(*vs):
        e = [r["epochs"] for ds, _, _ in DS for v in vs for r in R[ds].get(v, {}).values()]
        return f"{np.mean(e):.0f} (range {min(e)}–{max(e)})" if e else "—"

    PT = ["## Paired differences (same seeds 5–9)\n",
          "| dataset | metric | TVDiag: paper − code | TVDiag: paper − code trained to 500 epochs | "
          "AdaDiag-DL: paper − code | "
          "**paper versions: AdaDiag-DL − TVDiag** | − TVDiag-paper stopped at the same epoch | "
          "code versions: AdaDiag-DL − TVDiag (§6.17) |",
          "|---|---|---|---|---|---|---|---|"]
    for ds, name, _ in DS:
        for k in PAIR_M:
            PT.append(f"| {name} | {k} | {cell(F['paper_vs_code_tvdiag'][ds, k])} | "
                      f"{cell(F['paper_vs_code_same_length'][ds, k])} | "
                      f"{cell(F['dl_paper_vs_dl_code'][ds, k])} | "
                      f"{cell(F['dl_vs_tvdiag_paper'][ds, k])} | "
                      f"{cell(F['dl_vs_tvdiag_paper_capped'][ds, k])} | "
                      f"{cell(F['dl_vs_tvdiag_code'][ds, k])} |")
    PT.append("\nPaired t-test over the seeds; **bold**: p < 0.05 uncorrected; †: survives Holm "
              "across the 12 cells of its column (3 datasets × 4 metrics). [n]: fewer than 5 "
              "seeds. \"Trained to 500 epochs\": the code's TVDiag with early stopping off "
              "(`tvdiag_noes`); with θ learnable the total loss keeps falling for longer, so "
              f"the paper runs train for {ep_range('tvdiag_paper', 'align_paper')} epochs on "
              f"average, the code's for {ep_range('tvdiag', 'align')}. \"Stopped at the same "
              "epoch\": TVDiag-paper re-trained with each seed's cap set to the epoch "
              "AdaDiag-DL-paper stopped at (the §6.13 control); not run where both reach the "
              "cap.\n")

    # ---- one switch at a time
    S = {}
    for key, _ in SWITCH:
        for ds, _, _ in DS:
            for k in ("HR@1", "F1", "macro-F1"):
                S[key, ds, k] = pair(R[ds], f"tvdiag+{key}", "tvdiag", k)
    for ds, _, _ in DS:                       # same by construction when no edge is one-way
        if ow[ds][0] == 0:
            for k in ("HR@1", "F1", "macro-F1"):
                S["bidirectional", ds, k] = "same"
    holm([c for c in S.values() if isinstance(c, dict)])
    summ["one_at_a_time"] = {f"{a}|{b}|{c}": v for (a, b, c), v in S.items()}
    ST = ["## One paper feature at a time, in TVDiag (TVDiag + feature − TVDiag)\n",
          "| paper feature | " + " | ".join(f"{n} HR@1 | {n} F1" for _, n, _ in DS) + " |",
          "|---|" + "---|" * (2 * len(DS))]
    for key, lab in SWITCH:
        row = []
        for ds, _, _ in DS:
            for k in ("HR@1", "F1"):
                c = S[key, ds, k]
                row.append("identical (graphs already symmetric)" if c == "same" else cell(c))
        ST.append(f"| {lab} | " + " | ".join(row) + " |")
    row = []
    for ds, _, _ in DS:
        for k in ("HR@1", "F1"):
            row.append(cell(F["paper_vs_code_tvdiag"][ds, k], holm_mark=False))
    ST.append("| **all six together** | " + " | ".join(row) + " |")
    row = []
    for ds, _, _ in DS:
        for k in ("HR@1", "F1"):
            row.append(cell(F["awl_vs_same_length"][ds, k], holm_mark=False))
    ST.append("| Eq. 13, against TVDiag trained to the same 500 epochs | " + " | ".join(row) + " |")
    row = []
    for ds, _, _ in DS:
        for k in ("HR@1", "F1"):
            row.append(cell(F["noes_vs_code"][ds, k], holm_mark=False))
    ST.append("| (control) TVDiag trained to 500 epochs, no other change | " + " | ".join(row) + " |")
    ep = ["\nMean epochs trained: " + "; ".join(
        f"{name}: TVDiag {np.mean([r['epochs'] for r in R[ds]['tvdiag'].values()]):.0f}, "
        + ", ".join(f"+{key} {np.mean([r['epochs'] for r in R[ds][f'tvdiag+{key}'].values()]):.0f}"
                    for key, _ in SWITCH if R[ds].get(f"tvdiag+{key}"))
        for ds, name, _ in DS if R[ds].get("tvdiag")) + "."]
    ST += ep
    ST.append("\nSame test; †: survives Holm across every cell of the six single-feature rows "
              "(HR@1, F1 and macro-F1; macro-F1 is in summary.json). Bidirectional edges change "
              "nothing on Sock Shop and Hotel: "
              + "; ".join(f"{d}: {a:,} of {b:,} edges one-way" for d, (a, b) in ow.items())
              + ".\n")

    # ---- AdaDiag-DL with Eq. 13 only (the one paper feature that did not hurt TVDiag)
    AW = []
    if any(R[ds].get("align+awl_learnable") for ds, _, _ in DS):
        AV = ("tvdiag", "tvdiag+awl_learnable", "align", "align+awl_learnable")
        AL = {"tvdiag": "TVDiag, code", "tvdiag+awl_learnable": "TVDiag + θ learnable",
              "align": "AdaDiag-DL, code", "align+awl_learnable": "AdaDiag-DL + θ learnable"}
        AW = ["## AdaDiag-DL with Eq. 13 only (θ learnable, every other switch off)\n",
              "| dataset | arm | HR@1 | Avg@3 | MRR@3 | F1 | macro-F1 | epochs |",
              "|---|---|---|---|---|---|---|---|"]
        for ds, name, _ in DS:
            for v in AV:
                if R[ds].get(v):
                    ep = np.mean([r["epochs"] for r in R[ds][v].values()])
                    AW.append(f"| {name} | {AL[v]} | " + " | ".join(mean_sd(R[ds], v, k) for k in GET)
                              + f" | {ep:.0f} |")
        afam = {"awl_dl_vs_awl_tvdiag": ("align+awl_learnable", "tvdiag+awl_learnable"),
                "awl_dl_vs_awl_tvdiag_capped": ("align+awl_learnable", "capped:align+awl_learnable"),
                "awl_dl_vs_dl": ("align+awl_learnable", "align"),
                "awl_dl_vs_tvdiag": ("align+awl_learnable", "tvdiag")}
        for fam, (a, b) in afam.items():
            F[fam] = {(ds, k): pair(R[ds], a, b, k) for ds, _, _ in DS for k in PAIR_M}
            holm(list(F[fam].values()))
            summ["paired"][fam] = {f"{ds}|{k}": c for (ds, k), c in F[fam].items()}
        AW += ["\n| dataset | metric | **AdaDiag-DL + θ − TVDiag + θ** | − TVDiag + θ stopped at the "
               "same epoch | AdaDiag-DL + θ − AdaDiag-DL (code) | AdaDiag-DL + θ − TVDiag (code) |",
               "|---|---|---|---|---|---|"]
        for ds, name, _ in DS:
            for k in PAIR_M:
                AW.append(f"| {name} | {k} | " + " | ".join(cell(F[f][ds, k]) for f in afam) + " |")
        AW.append("\nPaired t-test over seeds 5–9; **bold**: p < 0.05 uncorrected; †: survives Holm "
                  "across the 12 cells of its column. The first column is the controlled comparison "
                  "(both arms with θ learnable); the last mixes AdaDiag-DL's effect with θ's own.\n")

    # ---- learned theta
    TH = ["## Eq. 13 with θ learnable: where the four weights end (0.5 / θ², mean over seeds)\n",
          "| dataset | run | RCL | FTI | ω·TO | ω·CM |", "|---|---|---|---|---|---|"]
    for ds, name, _ in DS:
        for v, lab in (("tvdiag+awl_learnable", "TVDiag + θ learnable"),
                       ("align+awl_learnable", "AdaDiag-DL + θ learnable"),
                       ("tvdiag_paper", "TVDiag, paper"), ("align_paper", "AdaDiag-DL, paper")):
            rs = [r for r in R[ds].get(v, {}).values() if "awl_weight" in r]
            if rs:
                w = np.mean([r["awl_weight"] for r in rs], 0)
                TH.append(f"| {name} | {lab} | " + " | ".join(f"{x:.2f}" for x in w) + " |")
                summ.setdefault("awl_weight", {}).setdefault(ds, {})[v] = w.tolist()
    TH.append("\nThe code's inert version leaves all four at 0.50.\n")

    # ---- closer to the published numbers?
    PB = ["## Distance from TVDiag's published numbers (re-run mean − published)\n",
          "| dataset | metric | published | TVDiag, code | TVDiag, paper |", "|---|---|---|---|---|"]
    for ds, name, L in DS:
        pub = published(L)
        for k in ("HR@1", "Avg@3", "MRR@3", "F1"):
            out = []
            for v in ("tvdiag", "tvdiag_paper"):
                x = np.array([GET[k](r) for r in R[ds].get(v, {}).values()])
                if len(x) < 2:
                    out.append("—")
                    continue
                p = float(stats.ttest_1samp(x, pub[k]).pvalue)
                out.append(f"{x.mean() - pub[k]:+.3f} (p {p:.2f})")
            PB.append(f"| {name} | {k} | {pub[k]:.3f} | {out[0]} | {out[1]} |")
    PB.append("\nOne-sample t-test of the five seeds against the published value.\n")

    cat = ["## What the paper specifies and the released code does not do\n",
           "| # | where in the paper | paper | released code | switch | status |",
           "|---|---|---|---|---|---|"] + [f"| {' | '.join(c)} |" for c in CATALOGUE]
    cat.append("\nEverything else checked matches: Algorithm 2's labels (metric and trace → root "
               "cause, log → failure type), Eqs. 2 and 7–11, the metrics of Eqs. 14–16 and "
               "F1 = 2PR/(P+R), the dimensions 128/64/32, Adam (lr 1e-3, weight decay 1e-4), "
               "τ = 0.3, early stopping on the loss. One reading is ours: Eq. 3 literally SUMS "
               "over anchors, which would multiply the TO term by the batch size (256–1,024) "
               "against ω = 0.1; `to_norm=\"paper\"` averages over anchors, matching the 1/2n "
               "of Eq. 5. AUG in paper mode takes 2 × batch-size graphs per step from the "
               "enlarged training set, as many as the code's batches hold.\n")

    md = ["# The TVDiag paper versus its released code (README §6.19)\n",
          "Settings throughout: fresh seeds 5–9, 1 thread, 500-epoch cap, aligned dropout masks "
          "(`isolate_weighting_rng`), as in §6.17. \"Code\" is the released behaviour (every "
          "switch off: the committed runs, whose loop departs from TVDiag's in the three small ways "
          "of §6.11 -- notably, AUG draws a fresh copy of each batch's own incidents every step "
          "rather than from a pool of 10 per incident); \"paper\" is `config.PAPER` "
          "(switches 1–6 on). "
          "The LSTM aggregator (7) is left out here; §6.18 tests it.\n",
          *cat, *V, "", *PT, *ST, *AW, *TH, *PB]
    (OUT / "PAPER_FEATURES.md").write_text("\n".join(md))
    (OUT / "readme_table.md").write_text("\n".join(V + [""] + PT))
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    figure(S, F, OUT / "fig_paper_features.png")
    print((OUT / "PAPER_FEATURES.md").read_text())


def figure(S, F, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
    COL = {"gaia": "#2a78d6", "sockshop": "#eb6834", "hotel": "#1baf7a"}
    MK = {"gaia": "o", "sockshop": "s", "hotel": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=SURF, sharey=True)
    for ax, k in zip(axes, ("HR@1", "F1")):
        ax.set_facecolor(SURF)
        items = [(lab, {ds: S[key, ds, k] for ds, _, _ in DS}) for key, lab in SWITCH]
        items.append(("All six (TVDiag, paper − code)",
                      {ds: F["paper_vs_code_tvdiag"][ds, k] for ds, _, _ in DS}))
        items.append(("AdaDiag-DL − TVDiag, paper versions",
                      {ds: F["dl_vs_tvdiag_paper"][ds, k] for ds, _, _ in DS}))
        n = len(items)
        for y, (lab, cells) in enumerate(items):
            yy = n - 1 - y
            for j, (ds, dname, _) in enumerate(DS):
                c = cells[ds]
                off = (1 - j) * 0.22
                if c == "same":
                    c = {"d": 0.0, "ci": 0.0}
                if not isinstance(c, dict):
                    continue
                ax.plot([c["d"] - c["ci"], c["d"] + c["ci"]], [yy + off] * 2, color=COL[ds],
                        lw=2, solid_capstyle="round", zorder=2)
                ax.scatter([c["d"]], [yy + off], s=48, marker=MK[ds], color=COL[ds],
                           edgecolor=SURF, lw=1.5, zorder=3,
                           label=dname if (y == 0) else None)
        ax.axhline(1.5, color=GRID, lw=1)
        ax.axvline(0, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_yticks(range(n))
        ax.set_yticklabels([lab for lab, _ in items][::-1], fontsize=8.5, color=INK)
        ax.set_xlabel(f"paired difference in {k} (mean, 95% CI over seeds 5–9)",
                      fontsize=8.5, color=INK2)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.set_title(k, fontsize=10, color=INK, loc="left")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=3, frameon=False,
                   fontsize=8.5, labelcolor=INK2)
    fig.suptitle("Each paper feature in TVDiag, against the released code's behaviour",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURF)


if __name__ == "__main__":
    main()
