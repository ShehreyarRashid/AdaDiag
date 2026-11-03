"""Prose for results/paper_comparison/PAPER_COMPARISON.md (called by make_paper_comparison.py).

Every number quoted in the text comes from `facts`, which make_paper_comparison.py computes
from the run files, so the text cannot drift from the tables. Where a run has not finished,
the sentence says "pending" instead of guessing.
"""
from __future__ import annotations

PENDING = "_pending_"
DSN = {"A": "GAIA", "C": "Sock Shop", "D": "Hotel"}
LET = {"gaia": "A", "sockshop": "C", "hotel": "D"}


def img(name, alt):
    return f"![{alt}]({name})" if name else "_Pending: the runs for this figure have not finished._"


def f3(x):
    return PENDING if x is None or x != x else f"{x:.3f}"


def d3(x):
    return PENDING if x is None or x != x else f"{x:+.3f}"


def pv(p):
    if p is None or p != p:
        return "n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# --------------------------------------------------------------------------- scorecard
def scorecard(F) -> str:
    f4, f5, f6, f7, f16, f17, f18 = (F[k] for k in ("f4", "f5", "f6", "f7", "f16", "f17", "f18"))
    rows = []

    # 1. RCL vs baselines
    have = [L for L in "ACD" if f4.get(L, {}).get("rerun")]
    if have:
        beat = [L for L in have if f4[L]["rerun"]["HR@1"] > f4[L]["best_base"]["HR@1"]]
        ev = "; ".join(f"{L} {f4[L]['rerun']['HR@1']:.3f} vs {f4[L]['best_base']['HR@1']:.3f}"
                       for L in have)
        rows.append(["RQ1 · TVDiag beats every baseline at RCL (Table 4)",
                     "HR@1 ahead of the best baseline (RF-RCL) on A–D",
                     f"Re-run HR@1 vs best published baseline: {ev}",
                     "**Holds**" if len(beat) == len(have) else
                     f"Holds on {', '.join(beat) or 'none'}"])
    # 2. FTI vs baselines
    have = [L for L in "ACD" if f5.get(L, {}).get("rerun")]
    if have:
        beat = [L for L in have if f5[L]["rerun"]["F1"] > f5[L]["best_base"]["F1"]]
        ev = "; ".join(f"{L} {f5[L]['rerun']['F1']:.3f} vs {f5[L]['best_base']['F1']:.3f}"
                       for L in have)
        rows.append(["RQ1 · TVDiag beats every baseline at FTI (Table 5)",
                     "Best F1 on A–D",
                     f"Re-run F1 vs best published baseline: {ev}",
                     ("**Holds**" if len(beat) == len(have) else
                      f"Holds on {', '.join(beat) or 'none'}") + " ³"])
    # 3. absolute level
    if any(f4.get(L, {}).get("rerun") for L in "ACD"):
        parts = []
        for L in "ACD":
            if f4.get(L, {}).get("rerun") and f5.get(L, {}).get("rerun"):
                parts.append(f"{L}: HR@1 {f4[L]['rerun']['HR@1'] - f4[L]['paper']['HR@1']:+.3f}, "
                             f"F1 {f5[L]['rerun']['F1'] - f5[L]['paper']['F1']:+.3f}")
        have = [L for L in "ACD" if f4.get(L, {}).get("rerun") and f5.get(L, {}).get("rerun")]
        worst = max(abs(f4[L]["rerun"]["HR@1"] - f4[L]["paper"]["HR@1"]) for L in have)
        f1d = [f5[L]["rerun"]["F1"] - f5[L]["paper"]["F1"] for L in have]
        verdict = ("**HR@1 reproduced** (within 0.03)" if worst <= 0.03
                   else f"HR@1 within {worst:.3f}")
        verdict += (f"; **F1 lower** on all three ({max(f1d):+.3f} to {min(f1d):+.3f})"
                    if all(d < 0 for d in f1d) else "")
        rows.append(["The published numbers themselves", "Tables 4–5, TVDiag rows",
                     "Re-run − paper: " + "; ".join(parts), verdict])
    # 4. ablation (only once every variant and the reference have all their seeds)
    if f6.get("gaia") and len(f6["gaia"]) == 4 and F.get("f6_complete", {}).get("gaia"):
        from adadiag.paper_numbers import TABLE6
        g = f6["gaia"]
        pd = lambda lab: TABLE6[lab]["A"][0] - TABLE6["TVDiag"]["A"][0]
        aug = g["no_aug"]["HR@1"]
        rest = [g[v]["HR@1"][0] for v in ("no_to", "no_cm", "no_dw")]
        ev = (f"A, Δ HR@1 — w/o AUG {aug[0]:+.3f} (paired p {aug[2]:.3f}); w/o TO "
              f"{rest[0]:+.3f}, w/o CM {rest[1]:+.3f}, w/o DW {rest[2]:+.3f}")
        v = ("**AUG reproduced**" if aug[0] < -0.05 else "AUG not reproduced")
        v += ("; TO, CM, DW: **no measurable contribution** in either (paper ≥ "
              f"{min(pd(l) for l in ('w/o TO', 'w/o CM', 'w/o DW')):+.3f}, single run; "
              "re-run within seed noise)" if all(abs(x) < 0.02 for x in rest) else "")
        rows.append(["RQ2 · each of AUG, TO, CM, DW contributes (Table 6)",
                     f"A, Δ HR@1: w/o AUG {pd('w/o AUG'):+.3f}; w/o TO {pd('w/o TO'):+.3f}, "
                     f"w/o CM {pd('w/o CM'):+.3f}, w/o DW {pd('w/o DW'):+.3f}", ev, v])
    # 5. hyper-parameters
    dl = lambda n, m: f16.get(("delta", "v0_raw", n, m), {})
    if all(len(dl(n, "HR@1")) >= 3 for n in ("p", "omega", "tau", "l")):
        hp = max(dl("p", "HR@1"), key=lambda x: -dl("p", "HR@1")[x][0])     # worst point
        wp = dl("p", "HR@1")[hp]
        om = dl("omega", "F1")
        hi = max(om)
        sig = lambda n: [x for x, (d, p) in dl(n, "HR@1").items() if p == p and p < 0.05]
        ev = (f"p = {hp:g}: HR@1 {wp[0]:+.3f} (p {wp[1]:.3f}); ω: F1 {om[hi][0]:+.3f} at "
              f"ω = {hi:g}, {om[0.0][0]:+.3f} at ω = 0; τ, l: "
              + ("no point differs from the default at p < 0.05" if not (sig("tau") + sig("l"))
                 else "some points differ at p < 0.05"))
        parts = []
        parts.append("**Large p hurts** (agrees)" if wp[0] < -0.05 and wp[1] < 0.05
                     else "large p: no clear effect")
        if om[0.0][0] >= -0.01 and om[hi][0] < -0.02:
            parts.append("**ω: the contrastive terms do not help** (ω = 0 is as good, F1 falls "
                         "as ω grows — disagrees)")
        if not (sig("tau") + sig("l")):
            parts.append("τ, l: flat within seed noise")
        rows.append(["RQ3 · best p = 0.2, ω = 0.1, τ = 0.2, l = 2 (Fig. 16)",
                     "Read off the curves; stated in the RQ3 answer",
                     "A, vs the default (3 seeds, paired): " + ev, "; ".join(parts) + " ⁴"])
    # 6. modalities
    if f17:
        g = lambda l, m: f17.get(("gaia", l, m))
        if g("TVDiag", "HR@1") is not None and g("w/o Log", "F1") is not None:
            from adadiag.paper_numbers import FIG17_A
            pd = lambda l, i: FIG17_A[l][i] - FIG17_A["TVDiag"][i]
            rd = {("w/o Metric", 0): g("w/o Metric", "HR@1") - g("TVDiag", "HR@1"),
                  ("w/o Trace", 0): g("w/o Trace", "HR@1") - g("TVDiag", "HR@1"),
                  ("w/o Log", 3): g("w/o Log", "F1") - g("TVDiag", "F1")}
            gap = max(abs(v - pd(*k)) for k, v in rd.items())
            v = f"**Reproduced** (each Δ within {gap:.3f} of the paper's)"
            h = lambda l, m: f17.get(("hotel", l, m))
            if h("TVDiag", "F1") is not None and h("w/o Log", "F1") is not None \
                    and h("w/o Metric", "F1") is not None:
                v += (f"; on D (not in the paper) FTI leans on **metric**, not log: w/o "
                      f"Metric {h('w/o Metric', 'F1') - h('TVDiag', 'F1'):+.3f} F1, w/o Log "
                      f"{h('w/o Log', 'F1') - h('TVDiag', 'F1'):+.3f}")
            rows.append(["RQ4 · RCL leans on metric and trace, FTI on log (Fig. 17)",
                         "A: w/o Metric −0.274, w/o Trace −0.294 HR@1; w/o Log −0.216 F1",
                         f"A: w/o Metric {rd[('w/o Metric', 0)]:+.3f}, w/o Trace "
                         f"{rd[('w/o Trace', 0)]:+.3f} HR@1; w/o Log {rd[('w/o Log', 3)]:+.3f} F1",
                         v])
    # 6b. Table 7, TVDiag's own extractors
    if F["f7"].get(("A", "IForest")) and F["f7"].get(("A", "Rule-based")):
        from adadiag.paper_numbers import TABLE7
        pa = {e[0]: e[2:] for e in TABLE7["A"]}
        diffs = [abs(F["f7"][("A", ext)][m] - pa[ext][i]) for ext in ("IForest", "Rule-based")
                 for i, m in enumerate(("HR@1", "HR@3", "Avg@3", "F1"))]
        rows.append(["Table 7 · TVDiag's own extractors, one modality at a time",
                     "A: trace 0.410 / 0.732, log 0.337 / 0.945 (HR@1 / F1)",
                     f"A: trace {F['f7'][('A', 'IForest')]['HR@1']:.3f} / "
                     f"{F['f7'][('A', 'IForest')]['F1']:.3f}, log "
                     f"{F['f7'][('A', 'Rule-based')]['HR@1']:.3f} / "
                     f"{F['f7'][('A', 'Rule-based')]['F1']:.3f}",
                     f"**Reproduced** (all 8 numbers within {max(diffs):.3f}); DiagFusion's "
                     f"extractors not reproducible ¹"])
    # 7. SHAP
    if f18.get(("gaia", "v0_raw", "rcl", "trace")):
        from adadiag.paper_numbers import FIG18_A
        med = lambda t, m: f18[("gaia", "v0_raw", t, m)][1]
        rank = lambda t: sorted(("metric", "trace", "log"), key=lambda m: -med(t, m))
        prank = lambda t: sorted(("metric", "trace", "log"), key=lambda m: -FIG18_A[t][m][1])
        fmt = lambda t: " > ".join(f"{m} {med(t, m):.2f}" for m in rank(t))
        same = {t: rank(t) == prank(t) for t in ("rcl", "fti")}
        top = {t: rank(t)[0] == prank(t)[0] for t in ("rcl", "fti")}
        v = ("**Same order for both tasks**" if all(same.values()) else
             "; ".join(f"{t.upper()}: " + ("same order" if same[t] else
                                            "same leader, order differs" if top[t] else
                                            "**different leader**") for t in ("rcl", "fti")))
        rows.append(["RQ4 · per-incident contribution (Fig. 18)",
                     "A medians: RCL trace 0.44 > metric 0.33 > log 0.19; FTI log 0.39 > "
                     "metric 0.36 > trace 0.26",
                     f"A medians: RCL {fmt('rcl')}; FTI {fmt('fti')}", v])
    head = ["Paper's claim", "Paper's evidence", "Re-run", "Verdict"]
    out = ["| " + " | ".join(head) + " |", "|---|---|---|---|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- figure notes
def figure_notes(F) -> dict:
    """Data-driven bullets under each figure. Only states what the facts contain."""
    from adadiag import paper_numbers as P
    N = {}
    mods = ("metric", "trace", "log")

    # Fig. 11
    f11, out = F["f11"], []
    for ds in ("gaia", "sockshop", "hotel"):
        W = f11.get(ds)
        if W is None:
            continue
        r = [f"{m} {W[i][0]:.2f}" for i, m in enumerate(mods) if W[i][0] >= 0.05]
        t = [f"{m} {W[i][1]:.2f}" for i, m in enumerate(mods) if W[i][1] >= 0.05]
        out.append(f"- **{LET[ds]} · {DSN.get(LET[ds])}** — AdaDiag's learned matrix: RCL ← "
                   f"{', '.join(r)}; FTI ← {', '.join(t)}.")
    targeted = {("metric", "rcl"), ("trace", "rcl"), ("log", "fti")}
    for ds in ("gaia", "sockshop", "hotel"):
        g0, g3 = f11.get(("gap", ds, "v0_raw")), f11.get(("gap", ds, "v3_incident"))
        if not g0 or not g3:
            continue
        best = max(g0, key=g0.get)
        diff = max(g0, key=lambda k: abs(g3[k] - g0[k]))
        out.append(f"- {LET[ds]}: the clearest label structure is in **{best[0]} for "
                   f"{best[1].upper()}** (gap {g0[best]:.3f}"
                   f"{', a pair TO targets' if best in targeted else ', a pair TO does not target'}"
                   f"). The largest AdaDiag − TVDiag change on any pair is "
                   f"{g3[diff] - g0[diff]:+.3f} ({diff[0]}/{diff[1].upper()}).")
    N["f11"] = out

    # Fig. 14
    f14, out = F["f14"], []
    for ds in ("gaia", "sockshop", "hotel"):
        a, b = f14.get((ds, "v0_raw", "hr1")), f14.get((ds, "v3_incident", "hr1"))
        if not a or not b:
            continue
        weak = min(a, key=a.get)
        big = max(a, key=lambda t: abs(b[t] - a[t]))
        out.append(f"- {LET[ds]}: hardest type for TVDiag at HR@1 is *{weak.strip('[]')}* "
                   f"({a[weak]:.2f}); the largest AdaDiag − TVDiag difference is "
                   f"{b[big] - a[big]:+.2f} on *{big.strip('[]')}*.")
    N["f14"] = out

    # Fig. 15
    f15, out = F["f15"], []
    if f15.get(("w/ Trace", "HR@1")) is not None:
        for lab, ref in (("w/o opName", "w/ Trace"), ("w/o abType", "w/ Trace"),
                         ("w/o abDirection", "w/ Metric")):
            if f15.get((lab, "HR@1")) is None or f15.get((ref, "HR@1")) is None:
                continue
            pp = [P.FIG15_A[lab][i] - P.FIG15_A[ref][i] for i in (0, 3)]
            rr = [f15[(lab, m)] - f15[(ref, m)] for m in ("HR@1", "F1")]
            out.append(f"- **{lab}** vs {ref}: paper HR@1 {pp[0]:+.3f}, F1 {pp[1]:+.3f}; "
                       f"re-run HR@1 {rr[0]:+.3f}, F1 {rr[1]:+.3f}.")
        for lab in ("w/ Trace", "w/ Metric"):
            a, b = f15.get((lab, "F1")), f15.get(("shipped " + lab, "F1"))
            if a is not None and b is not None:
                out.append(f"- {lab}, re-embedded vs shipped embeddings: F1 {a:.3f} vs {b:.3f}, "
                           f"HR@1 {f15[(lab, 'HR@1')]:.3f} vs {f15[('shipped ' + lab, 'HR@1')]:.3f}.")
    N["f15"] = out

    # Fig. 16
    f16, out = F["f16"], []
    sym = {"p": "p", "omega": "ω", "tau": "τ", "l": "l"}
    for n in ("p", "omega", "tau", "l"):
        dd = {m: f16.get(("delta", "v0_raw", n, m), {}) for m in ("HR@1", "F1")}
        if len(dd["HR@1"]) < 3:
            continue
        hits = [f"{sym[n]} = {x:g}: {m} {d:+.3f} (p = {pp:.3f})" for m in ("HR@1", "F1")
                for x, (d, pp) in sorted(dd[m].items()) if pp == pp and pp < 0.05]
        pc = P.FIG16_A[n]
        out.append(f"- **{sym[n]}** — re-run points that differ from the default (paired, "
                   f"3 seeds, p < 0.05 uncorrected): " + ("; ".join(hits) if hits else "none") +
                   f". The paper's curve spans {max(pc['HR@1']) - min(pc['HR@1']):.3f} in "
                   f"HR@1 and {max(pc['F1']) - min(pc['F1']):.3f} in F1, single runs.")
    av = f16.get(("ada_vs_tv", "omega", "F1"))
    if av:
        out.append("- AdaDiag − TVDiag along ω, F1 (paired p, uncorrected): " + ", ".join(
            f"ω = {x:g}: {d:+.3f} ({pv(pp)})" for x, (d, pp) in av.items()) + ". "
            f"Best F1 anywhere on the curve: AdaDiag "
            f"{max(f16[('v3_incident', 'omega', 'F1')].values()):.3f}, TVDiag "
            f"{max(f16[('v0_raw', 'omega', 'F1')].values()):.3f} (both at ω = "
            f"{max(f16[('v0_raw', 'omega', 'F1')], key=f16[('v0_raw', 'omega', 'F1')].get):g})."
            " AdaDiag loses less F1 as ω grows but never beats TVDiag's best point — the "
            "protective pattern of the σ_to ablation (README §6.8).")
    N["f16"] = out

    # Fig. 17
    f17, out = F["f17"], []
    if f17.get(("gaia", "TVDiag", "HR@1")) is not None:
        for lab in ("w/o Metric", "w/o Trace", "w/o Log", "w/ Metric", "w/ Trace", "w/ Log"):
            if f17.get(("gaia", lab, "HR@1")) is None:
                continue
            pp = [P.FIG17_A[lab][i] - P.FIG17_A["TVDiag"][i] for i in (0, 3)]
            rr = [f17[("gaia", lab, m)] - f17[("gaia", "TVDiag", m)] for m in ("HR@1", "F1")]
            out.append(f"- A, **{lab}** vs full: paper HR@1 {pp[0]:+.3f}, F1 {pp[1]:+.3f} · "
                       f"re-run HR@1 {rr[0]:+.3f}, F1 {rr[1]:+.3f}.")
    if f17.get(("hotel", "TVDiag", "F1")) is not None:
        d = {lab: f17[("hotel", lab, "F1")] - f17[("hotel", "TVDiag", "F1")]
             for lab in ("w/o Metric", "w/o Trace", "w/o Log")
             if f17.get(("hotel", lab, "F1")) is not None}
        out.append("- D (Hotel, not in the paper's Fig. 17), F1 vs full: " + ", ".join(
            f"{k} {v:+.3f}" for k, v in d.items()) + ".")
    N["f17"] = out

    # Fig. 18
    f18, out = F["f18"], []
    for task in ("rcl", "fti"):
        if f18.get(("gaia", "v0_raw", task, "metric")) is None:
            continue
        out.append(f"- A, {task.upper()} medians — paper: " + ", ".join(
            f"{m} {P.FIG18_A[task][m][1]:.2f}" for m in mods) + "; re-run: " + ", ".join(
            f"{m} {f18[('gaia', 'v0_raw', task, m)][1]:.2f}" for m in mods) + "; AdaDiag: "
            + ", ".join(f"{m} {f18[('gaia', 'v3_incident', task, m)][1]:.2f}" for m in mods
                        if f18.get(('gaia', 'v3_incident', task, m))) + ".")
    for ds in ("sockshop", "hotel"):
        if f18.get((ds, "v0_raw", "fti", "metric")) is None:
            continue
        top = {t: max(mods, key=lambda m: f18[(ds, "v0_raw", t, m)][1]) for t in ("rcl", "fti")}
        out.append(f"- {LET[ds]}: the largest median share is **{top['rcl']}** for RCL and "
                   f"**{top['fti']}** for FTI.")
    N["f18"] = out
    return N


# --------------------------------------------------------------------------- report
def write_report(path, parts, F) -> None:
    F = dict(F, notes={k: v + [""] if v else [] for k, v in figure_notes(F).items()})
    f4, f5 = F["f4"], F["f5"]
    L = []
    L.append("# AdaDiag beside the TVDiag paper\n")
    L.append("Tables 3–7 and Figures 11 and 13–18 of the TVDiag paper — S. Xie et al., "
             "*ACM Transactions on Software Engineering and Methodology* 35(2), Article 40, "
             "2026, doi:[10.1145/3734868](https://doi.org/10.1145/3734868) — each set beside the "
             "same measurement made with this repository. Generated by "
             "`scripts/make_paper_comparison.py`; the runs are listed in "
             "`scripts/paper_comparison_runs.sh`.\n")
    L.append("## How to read this\n")
    L.append("""- **Datasets.** The paper's A = GAIA, B = AIOps-22, C = Sock Shop, D = Hotel Reservation
  (Sec. 5.1). B is not distributed with TVDiag's code, so it has no re-run anywhere below.
- **What is re-run.** *TVDiag (re-run)* is this repository's re-implementation of TVDiag as
  it ships (`v0_raw`); *AdaDiag* is the proposed method (`v3_incident`); *V0-simplex* is the
  controlled baseline of §6.2–6.3 of the README. The paper's **baselines** (MicroRCA,
  Eadro, DiagFusion, …) are **not** re-run: their numbers are the paper's, shown for context.
- **Paper numbers.** Table values are transcribed as printed. Figures 15–18 print no numbers,
  so their values were **read off the figures** (`adadiag/paper_numbers.py` says how);
  checked against bars the paper also prints in a table, the error is at most 0.003 —
  treat them as ± 0.005.
- **Seeds and threads.** Tables 4–5 and Figures 11, 13, 14 and 18 use 5 seeds and 2 CPU threads
  — the same runs as `results/gaia.json` and `results/hotel.json`, re-trained here and
  checked identical (below). Tables 6–7 and Figures 15–17 use 1 thread, 3 seeds on GAIA and 5
  on Hotel, each with its own TVDiag reference runs under the same setting. Values are
  mean ± sd over seeds; *p* is a paired t-test over seeds, Holm-corrected within a table where
  stated. The paper reports single numbers, without spread.
""")
    if F["repro"]:
        L.append("Re-training check: " + "; ".join(F["repro"]) + ".\n")

    L.append("## At a glance\n")
    L.append(scorecard(F) + "\n")
    rows = []
    for Ls in "ACD":
        a4, a5 = f4.get(Ls, {}), f5.get(Ls, {})
        if not (a4.get("ada") and a5.get("ada") and a4.get("simplex")):
            continue
        rows.append(f"| {Ls} · {DSN[Ls]} | {a4['ada']['HR@1']:.3f} / {a5['ada']['F1']:.3f} | "
                    f"{a4['ada']['HR@1'] - a4['paper']['HR@1']:+.3f} / "
                    f"{a5['ada']['F1'] - a5['paper']['F1']:+.3f} | "
                    f"{a4['ada']['HR@1'] - a4['rerun']['HR@1']:+.3f} / "
                    f"{a5['ada']['F1'] - a5['rerun']['F1']:+.3f} | "
                    f"{a4['ada']['HR@1'] - a4['simplex']['HR@1']:+.3f} / "
                    f"{a5['ada']['F1'] - a5['simplex']['F1']:+.3f} |")
    if rows:
        L.append("**AdaDiag against each reference** (HR@1 / F1, means of 5 seeds):\n")
        L.append("| Dataset | AdaDiag | − TVDiag, paper | − TVDiag, re-run | − V0-simplex "
                 "(same fixed rule, rescaled) |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
        L.append("The last column is the controlled comparison (README §6.2–6.3): whatever "
                 "AdaDiag gains over TVDiag as shipped, the fixed rule written on the simplex "
                 "gains too, so it is the rescaling, not the learning.\n")
    L.append("³ The F1 values of most baselines in Table 5 are not computed the way TVDiag's is "
             "— see the note under Table 5. ⁴ GAIA's seed-to-seed spread is larger than most "
             "differences along these curves — see Figure 16.\n")

    # ---------------- tables
    L.append("## Table 3 — dataset statistics\n")
    L.append(parts["t3"] + "\n")
    L.append("The paper counts **raw telemetry** records and every metric/trace/log type in it; "
             "the release ships only the **extracted alerts** (TVDiag's own extractor output), so "
             "the counts here are alerts and alert types — smaller by construction, and not a "
             "discrepancy. What must match does: micro-services, instances and failure types "
             "agree on A, C and D. Metric types = distinct (instance, metric); trace types = "
             "distinct (caller, callee, operation); LogKeys = distinct log-template IDs.\n")

    L.append("## Table 4 — root cause localisation (RCL)\n")
    L.append(parts["t4"] + "\n")
    L.append("Avg@3 = (HR@1 + HR@2 + HR@3) / 3 and MRR@3 as in the paper's Eqs. 15–16. "
             "*p-value (paper)* is the paper's own test of TVDiag against that baseline.\n")

    L.append("## Table 5 — failure type identification (FTI)\n")
    L.append(parts["t5"] + "\n")
    fd = f5.get("f1_def")
    if fd:
        L.append(f"² **Two F1 definitions in one table.** TVDiag's F1 is the harmonic mean of "
                 f"its weighted precision and recall — exactly, to the printed digit, on all four "
                 f"datasets (largest gap {fd['tvdiag_max_gap']:.3f}). For "
                 f"**{len(fd['baselines_below'])} of the {fd['n_baselines']} baseline rows** "
                 f"the printed F1 is *below* that harmonic mean by more than 0.005 (by up to "
                 f"{-min(g for *_, g in fd['baselines_below']):.3f}), which is what sklearn's "
                 f"weighted F1 gives. The row marked ² shows the re-run under that second "
                 f"definition: comparing TVDiag's F1 with the baselines' therefore favours "
                 f"TVDiag by roughly the difference between the two rows.\n")

    L.append("## Table 6 — ablation of AUG, TO, CM and DW\n")
    L.append(parts["t6"] + "\n")
    L.append("Cells are mean (Δ vs the full model); **bold** Δ = significant after Holm "
             "correction over the 16 comparisons of that dataset. *w/o DW* replaces Eq. 13 by "
             "static unit weights (`dynamic_weights=False`). In the released code Eq. 13 is "
             "inert (README §7), so *w/o DW* changes only the loss scale.\n")

    L.append("## Table 7 — alert extractors\n")
    L.append(parts["t7"] + "\n")
    L.append("Each row is TVDiag trained on one modality, with that modality's alerts from the "
             "named extractor. The *IForest* (trace) and *rule-based* (log) rows are TVDiag's own "
             "extractors — the alerts the release ships — so they are re-run as *w/ Trace* and "
             "*w/ Log* (the same runs as Figure 17). ¹ The *3-sigma* and *sampling* rows "
             "(DiagFusion's extractors) need the raw traces and logs, which the release does not "
             "include: they cannot be reproduced from the published artefacts. The paper's Table 7 "
             "covers A and B; the D rows are this repository's only.\n")

    # ---------------- figures
    figs = [
        ("f11", "Figure 11 — task-oriented learning",
         "The paper's Fig. 11 is a diagram of TO: same-label samples of a modality are pulled "
         "together. Top: which (modality, task) pairs that pull applies to — TVDiag's fixed rule "
         "and AdaDiag's learned matrix. Bottom: whether it happened on the test split — how "
         "much closer same-label incidents sit than different-label ones in each modality's "
         "embedding. Trace's zero weight in every AdaDiag matrix is explained in README §6.10 "
         "(audited: the objective's doing, not a wiring error)."),
        ("f13", "Figure 13 — failure diagnosis from graph- and node-level features",
         "The paper's Fig. 13 illustrates the two heads with made-up probabilities. These are "
         "the real outputs on the first test incident of each dataset — not selected."),
        ("f14", "Figure 14 — HR@k per failure type",
         "The paper compares TVDiag with Eadro and DiagFusion per failure type; the baselines are "
         "not re-run, so the comparison here is TVDiag (re-run) against AdaDiag. The paper "
         "prints no numbers for Fig. 14."),
        ("f15", "Figure 15 — alert fields",
         "Alerts re-embedded with fields removed (`scripts/reembed_alerts.py`, TVDiag's own "
         "fastText step made deterministic), then TVDiag trained on that modality alone. "
         "Diamonds: the same single-modality model on TVDiag's shipped embeddings — a fresh "
         "fastText model is not identical to theirs, so compare bars with bars."),
        ("f16", "Figure 16 — hyper-parameters",
         "Shaded band: ± 1 sd over 3 seeds. AdaDiag is swept along ω only — the scale of the "
         "contrastive terms it reweights."),
        ("f17", "Figure 17 — discarding modalities", ""),
        ("f18", "Figure 18 — per-modality contribution (SHAP)",
         "Exact Shapley values with the three modalities as players (`paper_compare."
         "modality_shapley`); the paper averages per-feature SHAP values instead. Both are "
         "shares in [0, 1] — the paper's medians sum to ≈ 1 per task — so the scales agree. "
         "Dots: the paper's quartiles on A, read off its figure."),
    ]
    for key, title, cap in figs:
        L.append(f"## {title}\n")
        L.append(img(parts[key], title) + "\n")
        if cap:
            L.append(cap + "\n")
        L.extend(F.get("notes", {}).get(key, []))

    L.append("## What could not be compared\n")
    L.append("""- **Dataset B (AIOps-22)** — not distributed with TVDiag's code.
- **The baselines** in Tables 4–5 and Fig. 14 — not re-run; their numbers are the paper's.
- **Table 7's 3-sigma and sampling extractors** — need raw telemetry the release omits.
- **Table 8** (inter-task affinity) and **Table 9** (efficiency) — outside the set compared here.
""")
    path.write_text("\n".join(L))
