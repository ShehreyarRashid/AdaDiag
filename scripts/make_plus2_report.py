#!/usr/bin/env python3
"""v1.22 README section 6.25: improving GradFusion+ -- the rule, the screen, the fresh-seed result.

Reads results/plus2/ (plan.json, PLAN.md, runs/<ds>/*.json|npz) and, for GradFusion-DL and TVDiag
on seeds 5-9, results/baselines/seeds5-9_<ds>.json. Writes results/plus2/PLUS2.md,
readme_table.md, summary.json, confirm_<ds>.json (one record per arm and seed, ensembles
included) and fig_plus2.png.

    python scripts/make_plus2_report.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import plus2 as P2                                                     # noqa: E402
from gradfusion import paper_numbers as P                                 # noqa: E402

OUT = P2.OUT
DS = (("gaia", "A · GAIA", "A"), ("sockshop", "C · Sock Shop", "C"), ("hotel", "D · Hotel", "D"))
GET = {"HR@1": lambda r: r["HR@1"], "HR@3": lambda r: r["HR@3"],
       "Avg@3": lambda r: (r["HR@1"] + r["HR@2"] + r["HR@3"]) / 3, "MRR@3": lambda r: r["MRR@3"],
       "F1": lambda r: r["f1"], "macro-F1": lambda r: r["macro_f1"]}
PRIMARY = ("HR@1", "F1")


def selection(ds: str, name: str, seed: int):
    """GradFusion-DL / TVDiag on seeds 5-9 (same settings, run before round 4)."""
    f = ROOT / "results" / "baselines" / f"seeds5-9_{ds}.json"
    for r in json.load(open(f)):
        if r["seed"] == seed and r["variant"] == name:
            return r
    return None


def rec(ds, name, seed):
    return P2.get(ds, name, seed) or (selection(ds, name, seed) if name in ("align", "tvdiag") else None)


def pair(ds, a, b, k, seeds):
    s = [i for i in seeds if rec(ds, a, i) and rec(ds, b, i)]
    if len(s) < 2:
        return None
    x = np.array([GET[k](rec(ds, a, i)) for i in s]); y = np.array([GET[k](rec(ds, b, i)) for i in s])
    d = x - y
    p = float(stats.ttest_rel(x, y).pvalue) if d.std() > 0 else 1.0
    h = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
    return dict(d=float(d.mean()), p=p, n=len(s), wins=int((d > 0).sum()), losses=int((d < 0).sum()), ci=h)


def holm(cells):
    live = [c for c in cells if c]
    run = 0.0
    for rank, i in enumerate(sorted(range(len(live)), key=lambda i: live[i]["p"])):
        run = max(run, min(1.0, (len(live) - rank) * live[i]["p"]))
        live[i]["p_holm"] = run


def cell(c):
    if not c:
        return "—"
    b = "**" if c["p"] < 0.05 else ""
    return f"{b}{c['d']:+.3f}{b}{'†' if c.get('p_holm', 1) < 0.05 else ''} (p {c['p']:.2f}, {c['wins']}/{c['n']})"


def published(letter):
    t4 = next(r for r in P.TABLE4[letter] if r[1] == "TVDiag")
    t5 = next(r for r in P.TABLE5[letter] if r[1] == "TVDiag")
    return {"HR@1": t4[3], "HR@3": t4[4], "Avg@3": t4[5], "MRR@3": t4[6], "F1": t5[5]}


def main() -> None:
    plan = P2.plan()
    final, cfg = plan["final"], plan["final"].split("|")[0]
    lab = {"tvdiag": "TVDiag", "tvdiag|ens3": "TVDiag × 3 (control)", "align": "GradFusion-DL",
           "plus": "GradFusion+ (v1.21)", final: "**GradFusion** (`" + final + "`)"}
    arms = list(dict.fromkeys(["tvdiag", "tvdiag|ens3", "align", "plus", cfg, final]))
    lab.setdefault(cfg, f"`{cfg}`")
    summ = {"plan": plan, "values": {}, "paired": {}, "per_class": {}}

    def values(seeds, title):
        V = [f"## {title}\n", "| dataset | arm | HR@1 | HR@3 | Avg@3 | MRR@3 | F1 | macro-F1 | train s |",
             "|---|---|---|---|---|---|---|---|---|"]
        for ds, name, L in DS:
            pub = published(L)
            V.append(f"| {name} | TVDiag, published | {pub['HR@1']:.3f} | {pub['HR@3']:.3f} | "
                     f"{pub['Avg@3']:.3f} | {pub['MRR@3']:.3f} | {pub['F1']:.3f} | — | — |")
            for a in arms:
                rs = [rec(ds, a, s) for s in seeds]
                if any(r is None for r in rs):
                    continue
                c = {k: np.array([g(r) for r in rs]) for k, g in GET.items()}
                V.append(f"| {name} | {lab[a]} | " + " | ".join(
                    f"{c[k].mean():.3f} ± {c[k].std(ddof=1):.3f}" for k in GET)
                    + f" | {np.mean([r['train_s'] for r in rs]):.0f} |")
                summ["values"].setdefault(title, {}).setdefault(ds, {})[a] = {
                    k: float(c[k].mean()) for k in GET} | {
                    "train_s": float(np.mean([r["train_s"] for r in rs])), "n": len(rs)}
        return V

    comps = [(final, "plus", "**GradFusion − GradFusion+**"), (final, "align", "GradFusion − GradFusion-DL"),
             (final, "tvdiag", "GradFusion − TVDiag"), (final, "tvdiag|ens3", "GradFusion − TVDiag × 3 (30–34 only)"),
             ("tvdiag|ens3", "tvdiag", "TVDiag × 3 − TVDiag (30–34 only)")]
    if cfg != final and cfg != "plus":
        comps.insert(1, (cfg, "plus", f"`{cfg}` − GradFusion+"))

    def paired(seeds, title, key):
        F = {}
        for i, (a, b, _) in enumerate(comps):
            F[i] = {(ds, k): pair(ds, a, b, k, seeds) for ds, _, _ in DS for k in GET}
            holm([F[i][ds, k] for ds, _, _ in DS for k in PRIMARY])
        F = {i: f for i, f in F.items() if any(f.values())}
        T = [f"## {title}\n", "| dataset | metric | " + " | ".join(comps[i][2] for i in F) + " |",
             "|---|---|" + "---|" * len(F)]
        for ds, name, _ in DS:
            for k in GET:
                T.append(f"| {name} | {k} | " + " | ".join(cell(F[i][ds, k]) for i in F) + " |")
        T.append("\nPaired t-test over the shared seeds (wins/seeds in brackets); **bold**: p < 0.05 "
                 "uncorrected; †: survives Holm over the six primary cells of its column (HR@1 and "
                 "F1 × 3 datasets).\n")
        summ["paired"][key] = {comps[i][2]: {f"{ds}|{k}": c for (ds, k), c in F[i].items()} for i in F}
        return T, F

    body = values(P2.FRESH, "Fresh seeds 30–34 (mean ± sd)")
    t, F = paired(P2.FRESH, "Paired differences, fresh seeds 30–34", "fresh")
    body += [""] + t
    have59 = all(rec(ds, final, s) for ds, _, _ in DS for s in P2.TABLE)
    if have59:
        both = list(P2.TABLE) + list(P2.FRESH)
        body += values(P2.TABLE, "Seeds 5–9, the paper-format tables' seeds (mean ± sd)") + [""]
        t, F = paired(both, "Paired differences, seeds 5–9 and 30–34 pooled (10 seeds)", "pooled")
        body += t
    seeds_pc = (list(P2.TABLE) if have59 else []) + list(P2.FRESH)
    PC = ["## HR@1 and FTI recall per failure type (GradFusion+ → GradFusion; mean over "
          f"{len(seeds_pc)} seeds)\n", "| dataset | failure type | n | HR@1 | FTI recall |", "|---|---|---|---|---|"]
    for ds, name, _ in DS:
        a = [rec(ds, "plus", s) for s in seeds_pc]; b = [rec(ds, final, s) for s in seeds_pc]
        for c in a[0]["per_class"]:
            h0, h1 = (np.mean([r["per_class"][c]["hr1"] for r in x]) for x in (a, b))
            f0, f1 = (np.mean([r["per_class"][c]["fti_recall"] for r in x]) for x in (a, b))
            PC.append(f"| {name} | {c} | {a[0]['per_class'][c]['n']} | {h0:.2f} → {h1:.2f} | {f0:.2f} → {f1:.2f} |")
            summ["per_class"].setdefault(ds, {})[c] = dict(hr1=(float(h0), float(h1)), fti=(float(f0), float(f1)))
    body += PC
    K = ["\n## Ensemble size (exploratory, computed after the confirmation; fresh seeds 30–34)\n",
         "| members | A · GAIA HR@1 / F1 | C · Sock Shop | D · Hotel |", "|---|---|---|---|"]
    for offs in ((0,), (0, 100), (0, 100, 200)):
        cells = []
        for ds, _, _ in DS:
            rs = [P2.ensemble(ds, cfg, s, offs) for s in P2.FRESH]
            cells.append(f"{np.mean([r['HR@1'] for r in rs]):.3f} / {np.mean([r['f1'] for r in rs]):.3f}")
        K.append(f"| {len(offs)} | " + " | ".join(cells) + " |")
    body += K

    for ds, _, _ in DS:                                   # flat records for the table scripts
        recs = []
        for a, tag in ((final, "align_pp"), ("tvdiag|ens3", "tvdiag_ens3")):
            for s in list(P2.TABLE) + list(P2.FRESH):
                r = rec(ds, a, s)
                if r:
                    recs.append(dict(r, variant=tag, seed=s))
        json.dump(recs, open(OUT / f"confirm_{ds}.json", "w"))

    md = ["# GradFusion: results\n",
          "Screened on development seeds 0–2 against GradFusion+; the rule and every decision were "
          "written to `PLAN.md` before the confirmation runs; confirmed on fresh seeds 30–34. "
          "Settings: 1 thread, 500-epoch cap, `isolate_weighting_rng`.\n", f"**Final configuration:** `{final}`.\n",
          (OUT / "PLAN.md").read_text().split("\n", 1)[1], *body,
          "\n![paired differences](fig_plus2.png)\n"]
    figure(F, comps, OUT / "fig_plus2.png", "10 seeds" if have59 else "fresh seeds 30–34")
    (OUT / "PLUS2.md").write_text("\n".join(md))
    (OUT / "readme_table.md").write_text("\n".join(body))
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    print("\n".join(body))


def figure(F, comps, path, seeds) -> None:
    """Paired differences with 95% CIs for HR@1 and F1: three comparisons x three datasets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
    COL = {"gaia": "#2a78d6", "sockshop": "#eb6834", "hotel": "#1baf7a"}
    MK = {"gaia": "o", "sockshop": "s", "hotel": "^"}
    rows = [i for i in F if comps[i][2].replace("**", "") in
            ("GradFusion − GradFusion+", "GradFusion − GradFusion-DL", "GradFusion − TVDiag")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), facecolor=SURF, sharey=True)
    for ax, k in zip(axes, PRIMARY):
        ax.set_facecolor(SURF)
        for y, i in enumerate(rows):
            yy = len(rows) - 1 - y
            for j, (ds, dname, _) in enumerate(DS):
                c = F[i][ds, k]
                if not c:
                    continue
                off = (1 - j) * 0.22
                ax.plot([c["d"] - c["ci"], c["d"] + c["ci"]], [yy + off] * 2, color=COL[ds], lw=2,
                        solid_capstyle="round", zorder=2)
                ax.scatter([c["d"]], [yy + off], s=48, marker=MK[ds], color=COL[ds], edgecolor=SURF,
                           lw=1.5, zorder=3, label=dname if y == 0 else None)
        ax.axvline(0, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([comps[i][2].replace("**", "") for i in rows][::-1], fontsize=8.5, color=INK)
        ax.set_xlabel(f"paired difference in {k} (mean, 95% CI, {seeds})", fontsize=8.5, color=INK2)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.set_title(k, fontsize=10, color=INK, loc="left")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, frameon=False, fontsize=8.5,
                   labelcolor=INK2)
    fig.suptitle("GradFusion against GradFusion+, GradFusion-DL and TVDiag", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURF)


if __name__ == "__main__":
    main()
