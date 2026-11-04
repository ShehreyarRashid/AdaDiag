#!/usr/bin/env python3
"""README section 6.21: improving RCL and FTI -- the screen, the rule, the fresh-seed result.

Reads results/improve/: plan.json and PLAN.md (written by `improve.py --stage combine`,
before any confirmation run), screen_<ds>.json (dev seeds 0-2) and confirm_<ds>.json
(fresh seeds 10-14). Writes results/improve/IMPROVE.md, readme_table.md, summary.json.

    python scripts/make_improve_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
import sys                                                             # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from adadiag import paper_numbers as P                                 # noqa: E402

OUT = ROOT / "results" / "improve"
DS = (("gaia", "A · GAIA", "A"), ("sockshop", "C · Sock Shop", "C"), ("hotel", "D · Hotel", "D"))
GET = {"HR@1": lambda r: r["HR@1"], "HR@3": lambda r: r["HR@3"],
       "Avg@3": lambda r: (r["HR@1"] + r["HR@2"] + r["HR@3"]) / 3, "MRR@3": lambda r: r["MRR@3"],
       "F1": lambda r: r["f1"], "macro-F1": lambda r: r["macro_f1"]}
ARMS = (("tvdiag", "TVDiag"), ("tvdiag_improved", "TVDiag + change"),
        ("align", "AdaDiag-DL"), ("align_improved", "AdaDiag-DL + change"))
PAIRS = {"tvdiag": ("tvdiag_improved", "tvdiag"), "dl": ("align_improved", "align"),
         "dl_vs_tv": ("align_improved", "tvdiag_improved"), "dl_vs_tv_released": ("align_improved", "tvdiag")}
PRIMARY = ("HR@1", "F1")


def runs(ds):
    f = OUT / f"confirm_{ds}.json"
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


def main() -> None:
    plan = json.load(open(OUT / "plan.json"))
    R = {ds: runs(ds) for ds, _, _ in DS}
    summ = {"plan": plan, "values": {}, "paired": {}, "per_class": {}}

    V = ["## Fresh seeds 10–14 (mean ± sd)\n",
         "| dataset | arm | HR@1 | HR@3 | Avg@3 | MRR@3 | F1 | macro-F1 | epochs |",
         "|---|---|---|---|---|---|---|---|---|"]
    for ds, name, L in DS:
        pub = published(L)
        V.append(f"| {name} | TVDiag, published | {pub['HR@1']:.3f} | {pub['HR@3']:.3f} | "
                 f"{pub['Avg@3']:.3f} | {pub['MRR@3']:.3f} | {pub['F1']:.3f} | — | — |")
        for v, lab in ARMS:
            rs = list(R[ds].get(v, {}).values())
            if not rs:
                continue
            c = {k: np.array([g(r) for r in rs]) for k, g in GET.items()}
            V.append(f"| {name} | {lab} | " + " | ".join(
                f"{c[k].mean():.3f} ± {c[k].std(ddof=1):.3f}" if len(rs) > 1 else f"{c[k].mean():.3f}"
                for k in GET) + f" | {np.mean([r['epochs'] for r in rs]):.0f} |")
            summ["values"].setdefault(ds, {})[v] = {k: float(c[k].mean()) for k in GET} | {
                "epochs": float(np.mean([r["epochs"] for r in rs])), "n": len(rs)}

    F = {}
    for fam, (a, b) in PAIRS.items():
        F[fam] = {(ds, k): pair(R[ds], a, b, k) for ds, _, _ in DS for k in GET}
        holm([F[fam][ds, k] for ds, _, _ in DS for k in PRIMARY])     # family: 3 datasets x 2
        summ["paired"][fam] = {f"{ds}|{k}": c for (ds, k), c in F[fam].items()}
    PT = ["## Paired differences (same fresh seeds)\n",
          "| dataset | metric | **TVDiag: + change − released** | AdaDiag-DL: + change − released | "
          "AdaDiag-DL + change − TVDiag + change | AdaDiag-DL + change − TVDiag released |",
          "|---|---|---|---|---|---|"]
    for ds, name, _ in DS:
        for k in GET:
            PT.append(f"| {name} | {k} | " + " | ".join(cell(F[f][ds, k]) for f in PAIRS) + " |")
    PT.append("\nPaired t-test over seeds 10–14 (wins/seeds in brackets); **bold**: p < 0.05 "
              "uncorrected; †: survives Holm over the six primary cells of its column (HR@1 and "
              "F1 × 3 datasets).\n")

    # per failure type: where the change acts
    PC = ["## HR@1 and FTI recall per failure type (TVDiag, released → + change; mean over seeds)\n",
          "| dataset | failure type | n | HR@1 | FTI recall |", "|---|---|---|---|---|"]
    for ds, name, _ in DS:
        a, b = R[ds].get("tvdiag", {}), R[ds].get("tvdiag_improved", {})
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
            summ["per_class"].setdefault(ds, {})[c] = dict(hr1=(h0, h1), fti=(f0, f1))

    md = ["# Improving RCL and FTI (README §6.21)\n",
          "Screened on development seeds 0–2 with TVDiag; the rule and the final change were "
          "written to `PLAN.md` before any confirmation run; confirmed on fresh seeds 10–14 for "
          "TVDiag and AdaDiag-DL, each against itself as released. Settings as §6.17.\n",
          f"**Final change:** `{plan.get('decision', {}).get('final')}`.\n",
          (OUT / "PLAN.md").read_text().split("\n", 1)[1], *V, "", *PT, *PC]
    md.append("![paired differences](fig_improve.png)\n")
    figure(F, OUT / "fig_improve.png")
    (OUT / "IMPROVE.md").write_text("\n".join(md))
    (OUT / "readme_table.md").write_text("\n".join(V + [""] + PT))
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    print("\n".join(md))


def figure(F, path) -> None:
    """Paired differences with 95% CIs, HR@1 and F1, three comparisons x three datasets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
    COL = {"gaia": "#2a78d6", "sockshop": "#eb6834", "hotel": "#1baf7a"}
    MK = {"gaia": "o", "sockshop": "s", "hotel": "^"}
    rows = (("tvdiag", "TVDiag: + change − released"), ("dl", "AdaDiag-DL: + change − released"),
            ("dl_vs_tv_released", "AdaDiag-DL + change − TVDiag released"))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), facecolor=SURF, sharey=True)
    for ax, k in zip(axes, PRIMARY):
        ax.set_facecolor(SURF)
        for y, (fam, lab) in enumerate(rows):
            yy = len(rows) - 1 - y
            for j, (ds, dname, _) in enumerate(DS):
                a, b = PAIRS[fam]
                R = runs(ds)
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
        ax.set_xlabel(f"paired difference in {k} (mean, 95% CI, fresh seeds 10–14)",
                      fontsize=8.5, color=INK2)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.set_title(k, fontsize=10, color=INK, loc="left")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, frameon=False,
                   fontsize=8.5, labelcolor=INK2)
    fig.suptitle("Neighbour contrast in the RCL head (rcl_context='neighbour')", x=0.01,
                 ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURF)


if __name__ == "__main__":
    main()
