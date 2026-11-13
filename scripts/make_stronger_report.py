#!/usr/bin/env python3
"""README section 6.24: a stronger AdaDiag -- the rule, the screen, the fresh-seed result.

Reads results/stronger/: plan.json and PLAN.md (written by `stronger.py` before any
confirmation run), screen_<ds>.json (dev seeds 0-2) and confirm_<ds>.json (fresh seeds
15-19). Writes results/stronger/STRONGER.md, readme_table.md, summary.json, fig_stronger.png.

    python scripts/make_stronger_report.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
import sys                                                             # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from adadiag import paper_numbers as P                                 # noqa: E402

OUT = ROOT / "results" / "stronger"
DS = (("gaia", "A · GAIA", "A"), ("sockshop", "C · Sock Shop", "C"), ("hotel", "D · Hotel", "D"))
GET = {"HR@1": lambda r: r["HR@1"], "HR@3": lambda r: r["HR@3"],
       "Avg@3": lambda r: (r["HR@1"] + r["HR@2"] + r["HR@3"]) / 3, "MRR@3": lambda r: r["MRR@3"],
       "F1": lambda r: r["f1"], "macro-F1": lambda r: r["macro_f1"]}
#: round -> (file prefix, seeds label, arms, paired comparisons, per-class pair)
ROUNDS = {
    1: ("confirm", "15–19",
        (("tvdiag", "TVDiag"), ("tvdiag_plus", "TVDiag + round-1 change"),
         ("align", "AdaDiag-DL"), ("align_plus", "AdaDiag+ round 1 (bag+root)")),
        {"dl": ("align_plus", "align", "**AdaDiag+ r1 − AdaDiag-DL**"),
         "dl_vs_tv_released": ("align_plus", "tvdiag", "AdaDiag+ r1 − TVDiag released"),
         "tvdiag": ("tvdiag_plus", "tvdiag", "TVDiag: + change − released"),
         "dl_vs_tv": ("align_plus", "tvdiag_plus", "AdaDiag+ r1 − TVDiag + change")},
        ("align", "align_plus")),
    2: ("confirm2", "20–24",
        (("tvdiag", "TVDiag"), ("align", "AdaDiag-DL"), ("align_plus", "AdaDiag+ round 1 (bag+root)"),
         ("align_plus2", "AdaDiag+ round 2")),
        {"dl": ("align_plus2", "align", "**AdaDiag+ r2 − AdaDiag-DL**"),
         "dl_vs_tv_released": ("align_plus2", "tvdiag", "AdaDiag+ r2 − TVDiag released"),
         "r1": ("align_plus", "align", "AdaDiag+ r1 − AdaDiag-DL"),
         "r2_vs_r1": ("align_plus2", "align_plus", "AdaDiag+ r2 − r1")},
        ("align", "align_plus2")),
    3: ("confirm3", "25–29",
        (("tvdiag", "TVDiag"), ("align", "AdaDiag-DL"), ("align_plus3", "AdaDiag+ round 3")),
        {"dl": ("align_plus3", "align", "**AdaDiag+ r3 − AdaDiag-DL**"),
         "dl_vs_tv_released": ("align_plus3", "tvdiag", "AdaDiag+ r3 − TVDiag released")},
        ("align", "align_plus3")),
}
PRIMARY = ("HR@1", "F1")


def runs(ds, prefix="confirm"):
    f = OUT / f"{prefix}_{ds}.json"
    R = {}
    for r in (json.load(open(f)) if f.exists() else []):
        R.setdefault(r["variant"], {})[r["seed"]] = r
    return R


def pair(R, a, b, k):
    s = sorted(set(R.get(a, {})) & set(R.get(b, {})))
    if len(s) < 2:
        return None
    x = np.array([GET[k](R[a][i]) for i in s]); y = np.array([GET[k](R[b][i]) for i in s])
    d = x - y
    p = float(stats.ttest_rel(x, y).pvalue) if d.std() > 0 else 1.0
    return dict(d=float(d.mean()), p=p, n=len(s), wins=int((d > 0).sum()), losses=int((d < 0).sum()))


def holm(cells):
    live = [c for c in cells if c]
    order = sorted(range(len(live)), key=lambda i: live[i]["p"])
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(live) - rank) * live[i]["p"]))
        live[i]["p_holm"] = run


def cell(c):
    if not c:
        return "—"
    b = "**" if c["p"] < 0.05 else ""
    star = "†" if c.get("p_holm", 1) < 0.05 else ""
    return f"{b}{c['d']:+.3f}{b}{star} (p {c['p']:.2f}, {c['wins']}/{c['n']})"


def published(letter):
    t4 = next(r for r in P.TABLE4[letter] if r[1] == "TVDiag")
    t5 = next(r for r in P.TABLE5[letter] if r[1] == "TVDiag")
    return {"HR@1": t4[3], "HR@3": t4[4], "Avg@3": t4[5], "MRR@3": t4[6], "F1": t5[5]}


def section(rnd, summ):
    prefix, seeds, arms, pairs, (pc_a, pc_b) = ROUNDS[rnd]
    R = {ds: runs(ds, prefix) for ds, _, _ in DS}
    if not any(R.values()):
        return [], None
    sm = summ.setdefault(f"round{rnd}", {"values": {}, "paired": {}, "per_class": {}})
    V = [f"## Round {rnd}: fresh seeds {seeds} (mean ± sd)\n",
         "| dataset | arm | HR@1 | HR@3 | Avg@3 | MRR@3 | F1 | macro-F1 | epochs |",
         "|---|---|---|---|---|---|---|---|---|"]
    for ds, name, L in DS:
        pub = published(L)
        V.append(f"| {name} | TVDiag, published | {pub['HR@1']:.3f} | {pub['HR@3']:.3f} | "
                 f"{pub['Avg@3']:.3f} | {pub['MRR@3']:.3f} | {pub['F1']:.3f} | — | — |")
        for v, lab in arms:
            rs = list(R[ds].get(v, {}).values())
            if not rs:
                continue
            c = {k: np.array([g(r) for r in rs]) for k, g in GET.items()}
            V.append(f"| {name} | {lab} | " + " | ".join(
                f"{c[k].mean():.3f} ± {c[k].std(ddof=1):.3f}" if len(rs) > 1 else f"{c[k].mean():.3f}"
                for k in GET) + f" | {np.mean([r['epochs'] for r in rs]):.0f} |")
            sm["values"].setdefault(ds, {})[v] = {k: float(c[k].mean()) for k in GET} | {
                "epochs": float(np.mean([r["epochs"] for r in rs])), "n": len(rs),
                "train_s": float(np.mean([r["train_s"] for r in rs]))}
    F = {}
    for fam, (a, b, _) in pairs.items():
        F[fam] = {(ds, k): pair(R[ds], a, b, k) for ds, _, _ in DS for k in GET}
        holm([F[fam][ds, k] for ds, _, _ in DS for k in PRIMARY])     # family: 3 datasets x 2
        sm["paired"][fam] = {f"{ds}|{k}": c for (ds, k), c in F[fam].items()}
    PT = [f"## Round {rnd}: paired differences (same fresh seeds {seeds})\n",
          "| dataset | metric | " + " | ".join(lab for _, _, lab in pairs.values()) + " |",
          "|---|---|" + "---|" * len(pairs)]
    for ds, name, _ in DS:
        for k in GET:
            PT.append(f"| {name} | {k} | " + " | ".join(cell(F[f][ds, k]) for f in pairs) + " |")
    PT.append(f"\nPaired t-test over seeds {seeds} (wins/seeds in brackets); **bold**: p < 0.05 "
              "uncorrected; †: survives Holm over the six primary cells of its column (HR@1 and "
              "F1 × 3 datasets).\n")
    PC = [f"## Round {rnd}: HR@1 and FTI recall per failure type ({pc_a} → {pc_b}; mean over seeds)\n",
          "| dataset | failure type | n | HR@1 | FTI recall |", "|---|---|---|---|---|"]
    for ds, name, _ in DS:
        a, b = R[ds].get(pc_a, {}), R[ds].get(pc_b, {})
        s = sorted(set(a) & set(b))
        if not s:
            continue
        for c in a[s[0]]["per_class"]:
            h0 = np.mean([a[i]["per_class"][c]["hr1"] for i in s])
            h1 = np.mean([b[i]["per_class"][c]["hr1"] for i in s])
            f0 = np.mean([a[i]["per_class"][c]["fti_recall"] for i in s])
            f1 = np.mean([b[i]["per_class"][c]["fti_recall"] for i in s])
            PC.append(f"| {name} | {c} | {a[s[0]]['per_class'][c]['n']} | {h0:.2f} → {h1:.2f} | "
                      f"{f0:.2f} → {f1:.2f} |")
            sm["per_class"].setdefault(ds, {})[c] = dict(hr1=(h0, h1), fti=(f0, f1))
    return V + [""] + PT + PC, (F, R, pairs, seeds)


def pooled_records(ds) -> dict:
    """{variant: {seed: record}} over every fresh seed: 5-9 (AdaDiag-DL and TVDiag from
    results/selection, the changes from confirm59), 15-19 (round 1), 20-24 (round 2)."""
    R = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "selection" / f"confirm_{ds}*.json"))):
        for r in json.load(open(f)):
            if r["seed"] in range(5, 10) and r["variant"] in ("align", "tvdiag"):
                R.setdefault(r["variant"], {}).setdefault(r["seed"], r)
    for prefix in ("confirm59", "confirm", "confirm2", "confirm3"):
        for v, rs in runs(ds, prefix).items():
            for sd, r in rs.items():
                R.setdefault(v, {}).setdefault(sd, r)
    return R


def pooled(summ):
    """Both changes against AdaDiag-DL and TVDiag (re-run) on every fresh seed they share."""
    r3 = "+".join(json.load(open(OUT / "plan.json")).get("final3", []))
    comps = (("align_plus", "align", "AdaDiag+ r1 (bag+root) − AdaDiag-DL"),
             ("align_plus2", "align", "AdaDiag+ r2 (bag+neigh) − AdaDiag-DL"),
             ("align_plus3", "align", f"AdaDiag+ r3 ({r3}) − AdaDiag-DL"),
             ("align_plus", "tvdiag", "AdaDiag+ r1 − TVDiag (re-run)"),
             ("align_plus2", "tvdiag", "AdaDiag+ r2 − TVDiag (re-run)"),
             ("align_plus3", "tvdiag", "AdaDiag+ r3 − TVDiag (re-run)"))
    R = {ds: pooled_records(ds) for ds, _, _ in DS}
    F = {}
    for i, (a, b, _) in enumerate(comps):
        F[i] = {(ds, k): pair(R[ds], a, b, k) for ds, _, _ in DS for k in GET}
        holm([F[i][ds, k] for ds, _, _ in DS for k in PRIMARY])
    L = ["## All fresh seeds pooled (5–9, 15–19, 20–24, 25–29)\n",
         "Each change against AdaDiag-DL and TVDiag on every fresh seed both have (round 1's "
         "change: 15 seeds; round 2's: 10; round 3's: 10). Seeds 5–9 are those of the "
         "paper-format tables (§6.17); no change was selected on them.\n",
         "| dataset | metric | " + " | ".join(lab for _, _, lab in comps) + " |",
         "|---|---|" + "---|" * len(comps)]
    for ds, name, _ in DS:
        for k in GET:
            L.append(f"| {name} | {k} | " + " | ".join(cell(F[i][ds, k]) for i in F) + " |")
    L.append("\nPaired t-test over the shared seeds (wins/seeds in brackets); **bold**: p < 0.05 "
             "uncorrected; †: survives Holm over the six primary cells of its column.\n")
    summ["pooled"] = {lab: {f"{ds}|{k}": F[i][ds, k] for ds, _, _ in DS for k in GET}
                      for i, (_, _, lab) in enumerate(comps)}
    pairs = {i: c for i, c in enumerate(comps) if i in (0, 1, 2)}
    return L, ({fam: F[fam] for fam in pairs}, R, pairs, "pooled fresh seeds")


def main() -> None:
    plan = json.load(open(OUT / "plan.json"))
    summ = {"plan": plan}
    body, last = [], None
    for rnd in ROUNDS:
        lines, _ = section(rnd, summ)
        body += lines
    lines, last = pooled(summ)
    body += lines
    final = plan.get("final3", plan.get("final2", plan.get("final", [])))
    md = ["# A stronger AdaDiag (README §6.24)\n",
          "Screened on development seeds 0–2 with AdaDiag-DL; each round's rule and final change "
          "were written to `PLAN.md` before its confirmation runs; round 1 is confirmed on fresh "
          "seeds 15–19, round 2 on 20–24 and round 3 on 25–29. Settings as §6.17.\n",
          f"**Final change:** `{'+'.join(final)}`.\n",
          (OUT / "PLAN.md").read_text().split("\n", 1)[1], *body]
    if last is not None:
        md.append("![paired differences](fig_stronger.png)\n")
        figure(last, OUT / "fig_stronger.png", "alert-type counts (+ root readout or neighbour contrast)")
    (OUT / "STRONGER.md").write_text("\n".join(md))
    (OUT / "readme_table.md").write_text("\n".join(body))
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    print("\n".join(md))


def figure(fig, path, final) -> None:
    """Paired differences with 95% CIs, HR@1 and F1, the round's comparisons x three datasets."""
    F, RR, pairs, seeds = fig
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
    COL = {"gaia": "#2a78d6", "sockshop": "#eb6834", "hotel": "#1baf7a"}
    MK = {"gaia": "o", "sockshop": "s", "hotel": "^"}
    rows = [(fam, lab.replace("**", "")) for fam, (_, _, lab) in pairs.items()][:3]
    rows = [(fam, lab.split(" (")[0] + " − AdaDiag-DL" if " (" in lab else lab) for fam, lab in rows]
    rows = [(fam, lab.replace(" (bag+root)", "").replace(" (bag+neigh)", "")) for fam, lab in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), facecolor=SURF, sharey=True)
    for ax, k in zip(axes, PRIMARY):
        ax.set_facecolor(SURF)
        for y, (fam, lab) in enumerate(rows):
            yy = len(rows) - 1 - y
            for j, (ds, dname, _) in enumerate(DS):
                a, b, _ = pairs[fam]
                R = RR[ds]
                s = sorted(set(R.get(a, {})) & set(R.get(b, {})))
                if len(s) < 2:
                    continue
                d = np.array([GET[k](R[a][i]) - GET[k](R[b][i]) for i in s])
                h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                off = (1 - j) * 0.22
                ax.plot([d.mean() - h, d.mean() + h], [yy + off] * 2, color=COL[ds], lw=2,
                        solid_capstyle="round", zorder=2)
                ax.scatter([d.mean()], [yy + off], s=48, marker=MK[ds], color=COL[ds],
                           edgecolor=SURF, lw=1.5, zorder=3, label=dname if y == 0 else None)
        ax.axvline(0, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([lab for _, lab in rows][::-1], fontsize=8.5, color=INK)
        ax.set_xlabel(f"paired difference in {k} (mean, 95% CI, fresh seeds {seeds})",
                      fontsize=8.5, color=INK2)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.set_title(k, fontsize=10, color=INK, loc="left")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, frameon=False,
                   fontsize=8.5, labelcolor=INK2)
    fig.suptitle("A stronger AdaDiag: " + final, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURF)


if __name__ == "__main__":
    main()
