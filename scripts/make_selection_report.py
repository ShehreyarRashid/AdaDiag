#!/usr/bin/env python3
"""README section 6.17: which AdaDiag is best? Renders the screening, the mechanism and cost
comparison, and the confirmation on fresh seeds (scripts/select_adadiag.py).

Writes results/selection/SELECTION.md, readme_table.md, summary.json, fig_selection.png.

    python scripts/make_selection_report.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import select_adadiag as S                                             # noqa: E402

RES = ROOT / "results"
OUT = RES / "selection"
DS = ("gaia", "hotel")
NAME = {"gaia": "GAIA", "hotel": "Hotel"}
CELLS = [(ds, k) for ds in DS for k in ("HR@1", "f1")]
LABEL = {"adadiag": "AdaDiag (learned from the contrastive term)",
         "adadiag_fid": "AdaDiag, TVDiag-fidelity loop",
         "v2_global": "V2-global (one learned matrix)",
         "warmup": "AdaDiag + 100-epoch weight warm-up",
         "trace_third": "AdaDiag, trace fixed at 1/3",
         "trace_tvdiag": "AdaDiag, trace fixed at TVDiag's role",
         "align": "Type 4: weights from the diagnosis losses"}


def _load(pattern):
    recs = []
    for f in sorted(glob.glob(str(RES / pattern))):
        d = json.load(open(f))
        recs += d if isinstance(d, list) else []
    return recs


def _by_seed(rs):
    out = {}
    for r in rs:
        out.setdefault(r["seed"], r)
    return out


def _f(pattern, **flt):
    return _by_seed([r for r in _load(pattern) if all(r.get(k) == v for k, v in flt.items())])


def screening_sources():
    """(candidate, dataset) -> (candidate runs by seed, matched TVDiag runs by seed, note)."""
    tv1 = {"gaia": _f("paper_comparison/sweep_gaia_tv*.json", arm="v0_raw", variant="default"),
           "hotel": _f("paper_comparison/ablate_hotel.json", arm="v0_raw", variant="default")}
    tv1 = {ds: {s: r for s, r in v.items() if not r.get("tag")} for ds, v in tv1.items()}
    fid_tv = {ds: _f(f"fidelity/fidelity_{ds}_*.json", arm="v0_raw") for ds in DS}
    scr = {ds: _load(f"selection/screen_{ds}.json") for ds in DS}
    sv = lambda ds, v: _by_seed([r for r in scr[ds] if r["variant"] == v])
    main = {ds: json.load(open(RES / f"{ds}.json")) for ds in DS}
    src = {
        "adadiag": {"gaia": (_f("ablation_gaia_s*.json", name="ref_adadiag"), tv1["gaia"], "1 thread"),
                    "hotel": (_by_seed(_load("controlled/rng_hotel.json")), tv1["hotel"], "1 thread, aligned")},
        "adadiag_fid": {"gaia": (_f("fidelity/fidelity_gaia_*.json", arm="v3_incident"), fid_tv["gaia"], "fidelity loop"),
                        "hotel": (sv("hotel", "adadiag_fid"), fid_tv["hotel"], "fidelity loop, aligned")},
        "v2_global": {ds: (_by_seed(main[ds]["v2_global"]), _by_seed(main[ds]["v0_raw"]), "2 threads (§6.2–6.3)")
                      for ds in DS},
        "warmup": {ds: (sv(ds, "warmup"), tv1[ds], "1 thread, aligned") for ds in DS},
        "trace_third": {"gaia": (_f("trace_fixed/runs_gaia*.json", variant="trace_third", sigma_to=0.1), tv1["gaia"], "1 thread"),
                        "hotel": (sv("hotel", "trace_third"), tv1["hotel"], "1 thread, aligned")},
        "trace_tvdiag": {"gaia": (_f("trace_fixed/runs_gaia*.json", variant="trace_tvdiag", sigma_to=0.1), tv1["gaia"], "1 thread"),
                         "hotel": (sv("hotel", "trace_tvdiag"), tv1["hotel"], "1 thread, aligned")},
        "align": {ds: (_f(f"cl_variants/runs_{ds}*.json", variant="align", sigma_to=0.1), tv1[ds], "1 thread")
                  for ds in DS},
    }
    return src


def paired(a, b, k):
    s = sorted(set(a) & set(b))
    d = np.array([a[i][k] - b[i][k] for i in s], float)
    if len(d) < 2:
        return float("nan"), float("nan"), len(d), d
    p = 1.0 if np.allclose(d, d[0]) and d[0] == 0 else (
        0.0 if np.allclose(d, d[0]) else float(stats.ttest_rel([a[i][k] for i in s], [b[i][k] for i in s]).pvalue))
    return float(d.mean()), p, len(d), d


def score_block(src):
    rows = {}
    for c, per in src.items():
        cells = {}
        for ds, k in CELLS:
            a, b, note = per[ds]
            cells[(ds, k)] = paired(a, b, k)
        means = [cells[x][0] for x in CELLS]
        rows[c] = dict(cells=cells, score=float(np.mean(means)), worst=float(np.min(means)),
                       notes={ds: per[ds][2] for ds in DS},
                       n={ds: cells[(ds, "HR@1")][2] for ds in DS})
    return rows


def screening_md(rows):
    L = ["| candidate | score | worst cell | GAIA HR@1 | GAIA F1 | Hotel HR@1 | Hotel F1 | seeds (GAIA / Hotel) | settings |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(rows, key=lambda c: -rows[c]["score"]):
        r = rows[c]
        cells = " | ".join(f"{r['cells'][x][0]:+.3f} (p {r['cells'][x][1]:.2f})" for x in CELLS)
        L.append(f"| {LABEL[c]} | **{r['score']:+.4f}** | {r['worst']:+.3f} | {cells} | "
                 f"{r['n']['gaia']} / {r['n']['hotel']} | {r['notes']['gaia']}; Hotel: {r['notes']['hotel']} |")
    return "\n".join(L) + "\n"


def confirm_rows():
    conf = {ds: _by_seed_v(_load(f"selection/confirm_{ds}*.json")) for ds in DS}
    if not all(conf.values()):
        return None
    out = {}
    for v, recs in conf["gaia"].items():
        ref = S.VARIANTS[v][2]
        if (v not in LABEL or ref is None or v not in conf["hotel"]
                or ref not in conf["gaia"]):          # the §6.17 finalists only (not §6.18 LSTM)
            continue
        cells, per_seed = {}, []
        for ds, k in CELLS:
            cells[(ds, k)] = paired(conf[ds][v], conf[ds][ref], k)
        # per-seed score: mean over the four cells for that seed (seeds shared by all four)
        seeds = sorted(set.intersection(*[set(conf[ds][v]) & set(conf[ds][ref]) for ds in DS]))
        for s in seeds:
            per_seed.append(np.mean([conf[ds][v][s][k] - conf[ds][ref][s][k] for ds, k in CELLS]))
        per_seed = np.array(per_seed)
        p_score = (float(stats.ttest_1samp(per_seed, 0).pvalue)
                   if len(per_seed) > 1 and per_seed.std() > 0 else float("nan"))
        mf = {ds: paired(conf[ds][v], conf[ds][ref], "macro_f1") for ds in DS}
        out[v] = dict(ref=ref, cells=cells, score=float(per_seed.mean()), p_score=p_score, macro=mf,
                      seeds=seeds, per_seed=per_seed.tolist(),
                      worse=[x for x in CELLS if cells[x][0] < 0 and cells[x][1] < 0.05])
    ps = [out[v]["p_score"] for v in out]
    for v, h in zip(out, S_holm(ps)):
        out[v]["holm_score"] = float(h)
    return out


def _by_seed_v(recs):
    out = {}
    for r in recs:
        out.setdefault(r["variant"], {}).setdefault(r["seed"], r)
    return out


def S_holm(ps):
    ps = np.asarray(ps, float)
    out = np.full_like(ps, np.nan)
    ok = ~np.isnan(ps)
    run = 0.0
    for r, i in enumerate(np.argsort(np.where(ok, ps, np.inf))[: ok.sum()]):
        run = max(run, min(1.0, (ok.sum() - r) * ps[i]))
        out[i] = run
    return out


def confirm_md(cr):
    L = ["| finalist | reference | score (fresh seeds) | p (score ≠ 0) | Holm | GAIA HR@1 | GAIA F1 | Hotel HR@1 | Hotel F1 | significantly worse cells | macro-F1 GAIA / Hotel (not scored) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in sorted(cr, key=lambda v: -cr[v]["score"]):
        r = cr[v]
        cells = " | ".join(f"{r['cells'][x][0]:+.3f} (p {r['cells'][x][1]:.2f})" for x in CELLS)
        L.append(f"| {LABEL[v]} | {'TVDiag, fidelity loop' if r['ref'] == 'tvdiag_fid' else 'TVDiag'} | "
                 f"**{r['score']:+.4f}** | {r['p_score']:.2f} | {r['holm_score']:.2f} | {cells} | "
                 f"{', '.join(f'{NAME[d]} {k}' for d, k in r['worse']) or 'none'} | "
                 + " / ".join(f"{r['macro'][d][0]:+.3f} (p {r['macro'][d][1]:.2f})" for d in DS) + " |")
    return "\n".join(L) + "\n"


def length_md():
    """Fresh seeds, GAIA: each finalist against TVDiag stopped at the same epoch, seed by seed."""
    cap = _by_seed_v(_load("selection/length_gaia.json"))
    conf = _by_seed_v(_load("selection/confirm_gaia*.json"))
    if not cap:
        return ""
    L = ["| finalist | epochs (finalist / TVDiag) | − TVDiag default: HR@1 / F1 | − TVDiag stopped at the same epoch: HR@1 / F1 |",
         "|---|---|---|---|"]
    tv = conf.get("tvdiag", {})
    for v in ("adadiag", "align", "trace_tvdiag", "trace_third"):
        c = cap.get(f"capped:{v}", {})
        f = conf.get(v, {})
        s_ = sorted(set(c) & set(f) & set(tv))
        if not s_:
            continue
        def m(a, b, k):
            d, p_, _, _ = paired({i: a[i] for i in s_}, {i: b[i] for i in s_}, k)
            return f"{d:+.3f} (p {p_:.2f})"
        ep = (f"{np.mean([f[i]['epochs'] for i in s_]):.0f} / {np.mean([tv[i]['epochs'] for i in s_]):.0f}")
        L.append(f"| {LABEL[v]} | {ep} | {m(f, tv, 'HR@1')} / {m(f, tv, 'f1')} | "
                 f"{m(f, c, 'HR@1')} / {m(f, c, 'f1')} |")
    return "\n".join(L) + "\n"


def per_class_md():
    """Fresh seeds: per failure type, each finalist's HR@1 and FTI recall minus TVDiag's."""
    conf = {ds: _by_seed_v(_load(f"selection/confirm_{ds}*.json")) for ds in DS}
    if not all(conf.values()):
        return ""
    fin = [v for v in ("adadiag", "align", "trace_tvdiag", "trace_third") if v in conf["gaia"]]
    L = ["| dataset | failure type | n | TVDiag HR@1 / FTI recall | "
         + " | ".join(f"{LABEL[v].split(':')[0].split(' (')[0]} − TVDiag" for v in fin) + " |",
         "|---|---|---|---|" + "---|" * len(fin)]
    for ds in DS:
        tv = conf[ds].get("tvdiag", {})
        if not tv:
            continue
        classes = list(next(iter(tv.values()))["per_class"])
        for c in classes:
            n = next(iter(tv.values()))["per_class"][c]["n"]
            th = np.mean([r["per_class"][c]["hr1"] for r in tv.values()])
            tf = np.mean([r["per_class"][c]["fti_recall"] for r in tv.values()])
            cells = []
            for v in fin:
                rs = conf[ds].get(v, {})
                s_ = sorted(set(rs) & set(tv))
                if not s_:
                    cells.append("—")
                    continue
                dh = np.mean([rs[i]["per_class"][c]["hr1"] - tv[i]["per_class"][c]["hr1"] for i in s_])
                df = np.mean([rs[i]["per_class"][c]["fti_recall"] - tv[i]["per_class"][c]["fti_recall"] for i in s_])
                cells.append(f"{dh:+.3f} / {df:+.3f}")
            L.append(f"| {NAME[ds]} | {c.strip('[]')} | {n} | {th:.3f} / {tf:.3f} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def mechanism_md():
    """Learning signal, weight shape, trace, robustness, cost: from the committed runs."""
    def L_(p):
        return _load(p)
    tv = {"gaia": [r for r in L_("paper_comparison/sweep_gaia_tv*.json") if r["arm"] == "v0_raw" and r.get("variant") == "default" and not r.get("tag")],
          "hotel": [r for r in L_("paper_comparison/ablate_hotel.json") if r["arm"] == "v0_raw" and r.get("variant") == "default" and not r.get("tag")]}
    spe = lambda rs: np.mean([r["train_s"] / r["epochs"] for r in rs]) if rs else float("nan")
    ada = {ds: [r for r in L_(f"ablation_{ds}_s*.json") if r["name"] == "ref_adadiag"] for ds in DS}
    al = {ds: [r for r in L_(f"cl_variants/runs_{ds}*.json") if r["variant"] == "align" and r["sigma_to"] == 0.1] for ds in DS}
    tf = {ds: [r for r in L_(f"trace_fixed/runs_{ds}*.json") if r["variant"] == "trace_third" and r["sigma_to"] == 0.1] for ds in DS}
    cost = lambda rs: "×" + "/".join(f"{spe(rs[ds]) / spe(tv[ds]):.1f}" for ds in DS)
    # robustness: Hotel F1 at sigma_to = 0.5 (TVDiag's rule there: the damage to recover)
    def f05(pattern, **flt):
        rs = [r for r in L_(pattern) if all(r.get(k) == v for k, v in flt.items())]
        return np.mean([r["f1"] for r in rs]) if rs else float("nan")
    tv05 = f05("cl_variants/runs_hotel*.json", variant="none", sigma_to=0.5)
    ada05 = np.mean([r["f1"] for r in L_("ablation_hotel_s*.json") if r["name"] == "D_ada_s0.5"])
    al05 = f05("cl_variants/runs_hotel*.json", variant="align", sigma_to=0.5)
    t305 = f05("trace_fixed/runs_hotel*.json", variant="trace_third", sigma_to=0.5)
    tt05 = f05("trace_fixed/runs_hotel*.json", variant="trace_tvdiag", sigma_to=0.5)
    rows = [
        ("AdaDiag", "contrastive term only (zero gradient from RCL/FTI, §6.6)", "vertex: 0.99–1.00 on one modality", "0.00",
         "FTI → log 0.73 (oracle: metric 0.77)", f"{ada05:.3f}", cost(ada)),
        ("AdaDiag, fidelity loop", "contrastive term only", "vertex", "0.00", "same as AdaDiag", "—", "≈ AdaDiag"),
        ("AdaDiag + warm-up", "contrastive term only, after 100 uniform epochs", "vertex after warm-up (§6.10)", "0.00 after warm-up", "—", "—", "≈ AdaDiag"),
        ("AdaDiag, trace fixed", "contrastive term only (metric / log)", "vertex inside the 1 − t", "fixed 1/3 or 0.5", "FTI → log 0.42 (1/3)", f"{t305:.3f} / {tt05:.3f}", cost(tf)),
        ("V2-global", "contrastive term only", "one matrix, vertex", "0.00", "—", "—", "≈ TVDiag"),
        ("**Type 4**", "**the diagnosis losses** (gradient alignment)", "**graded** (larger modality 0.50–0.84 of a column)", "**0.08–0.44**",
         "**FTI → metric 0.50** (oracle: metric 0.77)", f"{al05:.3f}", cost(al)),
    ]
    L = ["| candidate | learns from | weight shape | trace weight | Hotel FTI column | Hotel F1 at σ_to = 0.5 (TVDiag's rule: " + f"{tv05:.3f}" + ") | training time / epoch vs TVDiag (GAIA/Hotel) |",
         "|---|---|---|---|---|---|---|"]
    L += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(L) + "\n"


def figure(scr, cr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    order = sorted(scr, key=lambda c: scr[c]["score"])
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURF)
    ax.set_facecolor(SURF)
    for y, c in enumerate(order):
        ax.scatter([scr[c]["score"]], [y + 0.12], s=60, color=BLUE, edgecolor=SURF, lw=1.5, zorder=3,
                   label="screening (seeds 0–4 / 0–2)" if y == 0 else None)
        if cr and c in cr:
            r = cr[c]
            ps = np.array(r["per_seed"])
            h = stats.t.ppf(0.975, len(ps) - 1) * ps.std(ddof=1) / np.sqrt(len(ps)) if len(ps) > 1 else 0
            ax.plot([r["score"] - h, r["score"] + h], [y - 0.12] * 2, color=ORANGE, lw=2, zorder=2)
            ax.scatter([r["score"]], [y - 0.12], s=60, color=ORANGE, edgecolor=SURF, lw=1.5, zorder=3,
                       label="confirmation, fresh seeds 5–9 (95% CI)" if c == next(x for x in order if x in cr) else None)
    ax.axvline(0, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(range(len(order)), [LABEL[c] for c in order], fontsize=8.5)
    ax.set_xlabel("score: mean paired difference from matched TVDiag (GAIA and Hotel, HR@1 and F1)",
                  fontsize=8.5, color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURF)
    plt.close(fig)


def main() -> None:
    scr = score_block(screening_sources())
    cr = confirm_rows()
    figure(scr, cr, OUT / "fig_selection.png")
    md = ["# Which AdaDiag is best? (README §6.17)\n",
          "## 1. Screening: every variant against TVDiag under identical settings\n",
          "Score = mean of the four paired differences from the matched TVDiag (GAIA HR@1, GAIA "
          "F1, Hotel HR@1, Hotel F1). Hotel runs of the MLP variants are stream-aligned "
          "(§6.15).\n", screening_md(scr),
          "## 2. Mechanism, robustness and cost\n", mechanism_md(),
          "## 3. Selection rule, fixed before the confirmation runs\n", S.SELECTION_RULE + "\n"]
    if cr:
        md += ["## 4. Confirmation on fresh seeds 5–9 (never used to rank)\n", confirm_md(cr),
               "### Training length (GAIA, fresh seeds)\n",
               "TVDiag's stopping rule watches a loss that includes the TO term, so a finalist can "
               "gain on GAIA just by stopping earlier (§6.13). TVDiag stopped, seed by seed, at the "
               "finalist's epoch:\n", length_md(),
               "## 5. Per failure type, fresh seeds (HR@1 / FTI recall, minus TVDiag)\n",
               per_class_md()]
    if cr:
        sig = [v for v in cr if cr[v]["holm_score"] < 0.05 and not cr[v]["worse"]]
        if sig:
            best = max(sig, key=lambda v: cr[v]["score"])
            verdict = (f"**{LABEL[best]}**: its score is significantly above zero after Holm "
                       f"correction ({cr[best]['score']:+.4f}).")
        else:
            best = "align" if "align" in cr else max(cr, key=lambda v: cr[v]["score"])
            verdict = ("No finalist's score is significantly above zero (Holm p ≥ "
                       f"{min(cr[v]['holm_score'] for v in cr):.2f}), and none has a significantly "
                       "worse cell, so performance is a tie and the rule decides on mechanism: "
                       f"**{LABEL[best]}**, the only finalist that learns its weights from the "
                       "diagnosis losses and keeps them graded.")
        md += ["## 6. Verdict, by the rule fixed in advance\n", verdict + "\n",
               "Checks made after the confirmation (post hoc, uncorrected): on GAIA the selected "
               "variant's lead over TVDiag comes from stopping earlier (against TVDiag stopped at "
               "the same epoch: see the training-length table), and its GAIA macro-F1 is lower "
               "(rare failure types; see the confirmation table and §5). Both point the same way and "
               "are reported with the verdict, not used to change it.\n"]
    md += ["![selection](fig_selection.png)\n"]
    (OUT / "SELECTION.md").write_text("\n".join(md))
    (OUT / "readme_table.md").write_text(screening_md(scr) + ("\n" + confirm_md(cr) if cr else ""))
    json.dump({"screening": {c: dict(score=r["score"], worst=r["worst"],
                                     cells={f"{d}|{k}": list(v[:3]) for (d, k), v in r["cells"].items()})
                             for c, r in scr.items()},
               "confirmation": ({v: dict(score=r["score"], p=r["p_score"], holm=r["holm_score"],
                                         per_seed=r["per_seed"],
                                         cells={f"{d}|{k}": list(x[:3]) for (d, k), x in r["cells"].items()})
                                 for v, r in cr.items()} if cr else None)},
              open(OUT / "summary.json", "w"), indent=1)
    print((OUT / "SELECTION.md").read_text())


if __name__ == "__main__":
    main()
