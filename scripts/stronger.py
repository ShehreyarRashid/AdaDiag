#!/usr/bin/env python3
"""README section 6.24: a stronger AdaDiag -- candidates screened, then confirmed.

Starting point: AdaDiag-DL (v5_align) as released. Candidates (all switches, off by default):

  bag      alert_bag=True          alert-type counts per instance next to the embeddings
                                   (src/adadiag/alerts.py)
  attn     aggregator="attn"       GATv2-style attention over neighbours instead of the mean
  selfres  self_res=True           a linear skip from each instance's own features
  ema      weight_ema=0.9          evaluate a moving average of the weights
  root     fti_readout="root_attn" FTI also reads the likely culprit (section 6.21)
  neigh    rcl_context="neighbour" RCL scores each instance against its neighbours (6.21)

Stages (development seeds 0-2, all three datasets, settings of section 6.17):
  A        base, and bag / attn / selfres / ema each alone
  plan-b   RULE picks the anchor; written to PLAN.md
  B        anchor + each other candidate, one at a time
  plan-c   RULE picks the additions; written to PLAN.md
  C        anchor + all kept additions (only when two or more are kept)
  decide   RULE fixes the final change; written to PLAN.md before any confirmation run
  confirm  fresh seeds 15-19, never used before: AdaDiag-DL and TVDiag, each as released
           and with the final change

    python scripts/stronger.py --stage A --dataset hotel
    python scripts/stronger.py --stage plan-b
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch                                                         # noqa: E402
from adadiag import config, data                                     # noqa: E402
from adadiag.alerts import with_alert_bag                            # noqa: E402
from adadiag.evaluate import per_class                               # noqa: E402
from adadiag.train import train                                      # noqa: E402

OUT = ROOT / "results" / "stronger"
DATASETS = ("gaia", "sockshop", "hotel")
DEV, FRESH = range(0, 3), range(15, 20)
ARM = "v5_align"
CANDIDATES = {
    "bag":     dict(alert_bag=True),
    "attn":    dict(aggregator="attn"),
    "selfres": dict(self_res=True),
    "ema":     dict(weight_ema=0.9),
    "root":    dict(fti_readout="root_attn"),
    "neigh":   dict(rcl_context="neighbour"),
    # round 2 only: the root readout with the FTI gradient into the RCL scores stopped
    "rootd":   dict(fti_readout="root_attn", root_detach=True),
    # round 3 only: Eq. 13 with learnable theta (section 6.19), the best AdaDiag on seeds 5-9
    "theta":   dict(awl_learnable=True),
}
STAGE_A = ("bag", "attn", "selfres", "ema")
RULE = (
    "Development seeds 0-2, AdaDiag-DL (v5_align), all three datasets. A variant's score is the "
    "mean over the three datasets of (dHR@1 + dF1) / 2, paired by seed against its reference. It "
    "is kept if its score is above +0.005 and no dataset loses more than 0.015 on HR@1 or on F1 "
    "against that reference. "
    "Stage A: bag, attn, selfres and ema each alone, reference AdaDiag-DL as released; the "
    "anchor is the kept candidate with the highest score (none kept: the anchor is AdaDiag-DL "
    "itself). "
    "Stage B: the anchor plus each other candidate (bag, attn, selfres, ema, root, neigh, minus "
    "the anchor), one at a time, reference the anchor. "
    "Stage C: with two or more kept additions, the anchor plus all of them is screened, reference "
    "the anchor; the final change is that combination if its score is at least the best single "
    "addition's, otherwise the anchor plus the best single addition. With one kept addition, the "
    "final change is the anchor plus it; with none, the anchor alone. If the final change is "
    "empty, nothing is confirmed. "
    "Confirmation: fresh seeds 15-19 for AdaDiag-DL and TVDiag, each as released and with the "
    "final change; paired t-tests, Holm over HR@1 and F1 x 3 datasets per comparison.")


FRESH2 = range(20, 25)
RULE2 = (
    "Round 2, added after round 1's confirmation (seeds 15-19) left two cells below AdaDiag-DL "
    "(GAIA F1 -0.009, Hotel HR@1 -0.008). The selection criterion changes from the mean score to "
    "balance, because the goal is to improve every dataset. One new candidate is screened on the "
    "development seeds 0-2: bag+rootd (the root readout with its softmax detached, so the FTI "
    "loss cannot reshape the RCL ranking). Eligible: every variant with bag screened on the "
    "development seeds (bag, bag+attn, bag+selfres, bag+ema, bag+root, bag+neigh, bag+root+neigh, "
    "bag+rootd). For each, the six development-seed differences against AdaDiag-DL (HR@1 and F1 x "
    "3 datasets) are computed; among those with a score of at least +0.005, the winner is the one "
    "whose smallest difference is largest. It is confirmed on fresh seeds 20-24, never used "
    "before, against AdaDiag-DL and TVDiag as released, with round 1's final change alongside.")


FRESH3 = range(25, 30)
SCREEN3 = (["theta"], ["bag", "theta"], ["bag", "root", "theta"], ["bag", "neigh", "theta"])
RULE3 = (
    "Round 3, the last, added after both rounds' changes were run on seeds 5-9: there AdaDiag-DL "
    "with Eq. 13's learnable theta (section 6.19) scores as high as round 1's change, and higher "
    "than AdaDiag-DL on all six cells, while theta acts on the loss weights rather than the "
    "features. Screened on the development seeds 0-2: theta alone, bag+theta, bag+root+theta and "
    "bag+neigh+theta. Eligible: these four and every variant eligible in round 2. Selection as "
    "round 2: among variants with a score of at least +0.005 against AdaDiag-DL, the one whose "
    "smallest of the six differences is largest. Confirmed on fresh seeds 25-29, never used "
    "before, against AdaDiag-DL and TVDiag as released. Whatever round 3 yields is reported; "
    "there is no round 4.")


def stamp() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def run(ds, arm, name, over, seeds, out):
    """Train `name` for each seed not yet in `out` (resumable) and append the records."""
    recs = json.load(open(out)) if out.exists() else []
    done = {(r["variant"], r["seed"]) for r in recs}
    d = with_alert_bag(ds) if over.get("alert_bag") else ds
    for seed in seeds:
        if (name, seed) in done:
            continue
        cfg = config.get(ds.name, isolate_weighting_rng=True, **over)
        res, model = train(arm, seed, d, cfg)
        res.update(variant=name, overrides=over, per_class=per_class(model, d)[0])
        recs = json.load(open(out)) if out.exists() else []
        recs.append(res)
        json.dump(recs, open(out, "w"))
        print(f"  {ds.name:8s} {name:28s} s{seed:<2d} HR@1 {res['HR@1']:.3f}  F1 {res['f1']:.3f}  "
              f"macro-F1 {res['macro_f1']:.3f}  ({res['epochs']} ep, {res['train_s']:.0f}s)", flush=True)


def load(stage="screen"):
    R = {}
    for ds in DATASETS:
        f = OUT / f"{stage}_{ds}.json"
        for r in (json.load(open(f)) if f.exists() else []):
            R.setdefault(ds, {}).setdefault(r["variant"], {})[r["seed"]] = r
    return R


def score(R, name, ref):
    per, worst = {}, 0.0
    for ds in DATASETS:
        a, b = R.get(ds, {}).get(name, {}), R.get(ds, {}).get(ref, {})
        s = sorted(set(a) & set(b) & set(DEV))
        if len(s) < len(DEV):
            return None
        dh = float(np.mean([a[i]["HR@1"] - b[i]["HR@1"] for i in s]))
        df = float(np.mean([a[i]["f1"] - b[i]["f1"] for i in s]))
        per[ds] = (dh, df)
        worst = min(worst, dh, df)
    return float(np.mean([(h + f) / 2 for h, f in per.values()])), worst, per


def kept(sc) -> bool:
    return sc is not None and sc[0] > 0.005 and sc[1] >= -0.015


def name_of(parts) -> str:
    return "+".join(parts) if parts else "base"


def over_of(parts) -> dict:
    o = {}
    for p in parts:
        o.update(CANDIDATES[p])
    return o


def plan() -> dict:
    f = OUT / "plan.json"
    return json.load(open(f)) if f.exists() else {}


def save_plan(p, md: str) -> None:
    json.dump(p, open(OUT / "plan.json", "w"), indent=1)
    with open(OUT / "PLAN.md", "a") as fh:
        fh.write(md + "\n")
    print(md)


def table(R, names, ref) -> str:
    L = ["| variant | GAIA dHR@1 / dF1 | Sock Shop | Hotel | score | worst | kept |",
         "|---|---|---|---|---|---|---|"]
    for n in names:
        sc = score(R, n, ref)
        if sc is None:
            raise SystemExit(f"screen incomplete for {n}")
        cells = " | ".join(f"{sc[2][d][0]:+.3f} / {sc[2][d][1]:+.3f}" for d in DATASETS)
        L.append(f"| {n} | {cells} | {sc[0]:+.4f} | {sc[1]:+.3f} | {'yes' if kept(sc) else 'no'} |")
    return "\n".join(L)


def plan_b() -> None:
    p = plan()
    if "anchor" in p:
        print("anchor already fixed:", p["anchor"]); return
    R = load()
    sc = {n: score(R, n, "base") for n in STAGE_A}
    ok = [n for n in STAGE_A if kept(sc[n])]
    anchor = [max(ok, key=lambda n: sc[n][0])] if ok else []
    p.update(anchor=anchor, stage_a={n: dict(score=s[0], worst=s[1], per=s[2]) for n, s in sc.items()},
             written_b=stamp())
    save_plan(p, f"\n## Stage A (reference: AdaDiag-DL as released), {p['written_b']}\n\n"
                 f"{table(R, STAGE_A, 'base')}\n\nAnchor: `{name_of(anchor)}`.")


def additions() -> list[str]:
    a = plan()["anchor"]
    return [n for n in CANDIDATES if n not in a]


def plan_c() -> None:
    p = plan()
    if "kept_b" in p:
        print("stage B already decided:", p["kept_b"]); return
    R = load()
    a = p["anchor"]
    names = [name_of(a + [n]) for n in additions()]
    sc = {n: score(R, name_of(a + [n]), name_of(a)) for n in additions()}
    ok = sorted([n for n in sc if kept(sc[n])], key=lambda n: -sc[n][0])
    p.update(kept_b=ok, stage_b={n: dict(score=s[0], worst=s[1], per=s[2]) for n, s in sc.items()},
             written_c=stamp())
    save_plan(p, f"\n## Stage B (reference: `{name_of(a)}`), {p['written_c']}\n\n"
                 f"{table(R, names, name_of(a))}\n\nKept additions, best first: "
                 f"{', '.join(ok) or 'none'}.")


def decide() -> list[str]:
    p = plan()
    if "final" in p:
        return p["final"]
    a, ok = p["anchor"], p["kept_b"]
    if len(ok) >= 2:
        R = load()
        comb = a + ok
        sc = score(R, name_of(comb), name_of(a))
        if sc is None:
            raise SystemExit("screen the combination first (--stage C)")
        best = p["stage_b"][ok[0]]["score"]
        final = comb if sc[0] >= best else a + [ok[0]]
        why = (f"Combination `{name_of(comb)}` against `{name_of(a)}`: "
               + " | ".join(f"{d}: {sc[2][d][0]:+.3f} / {sc[2][d][1]:+.3f}" for d in DATASETS)
               + f"; score {sc[0]:+.4f} (worst {sc[1]:+.3f}) against the best single addition, "
               f"{ok[0]}, {best:+.4f}.")
        p["stage_c"] = dict(score=sc[0], worst=sc[1], per=sc[2])
    else:
        final = a + ok
        why = f"{len(ok)} kept addition(s); no combination to screen."
    p.update(final=final, written_final=stamp())
    save_plan(p, f"\n## Decision, {p['written_final']} (before any confirmation run)\n\n{why} "
                 f"**Final change: `{name_of(final)}`** = `{over_of(final)}`.")
    return final


ELIGIBLE2 = (["bag"], ["bag", "attn"], ["bag", "selfres"], ["bag", "ema"], ["bag", "root"],
             ["bag", "neigh"], ["bag", "root", "neigh"], ["bag", "rootd"])


def decide2() -> list[str]:
    p = plan()
    if "final2" in p:
        return p["final2"]
    R = load()
    rows, best = [], None
    for parts in ELIGIBLE2:
        sc = score(R, name_of(parts), "base")
        if sc is None:
            raise SystemExit(f"screen incomplete for {name_of(parts)}")
        mn = min(v for d in DATASETS for v in sc[2][d])
        ok = sc[0] >= 0.005
        rows.append((parts, sc, mn, ok))
        if ok and (best is None or mn > best[1]):
            best = (parts, mn)
    L = ["| variant | GAIA dHR@1 / dF1 | Sock Shop | Hotel | score | smallest | eligible |",
         "|---|---|---|---|---|---|---|"]
    for parts, sc, mn, ok in rows:
        cells = " | ".join(f"{sc[2][d][0]:+.3f} / {sc[2][d][1]:+.3f}" for d in DATASETS)
        L.append(f"| {name_of(parts)} | {cells} | {sc[0]:+.4f} | {mn:+.3f} | {'yes' if ok else 'no'} |")
    final2 = best[0] if best else p["final"]
    p.update(final2=final2, written_final2=stamp())
    save_plan(p, f"\n## Round 2: decision, {p['written_final2']} (before its confirmation runs)\n\n"
                 f"Differences against AdaDiag-DL on the development seeds:\n\n" + "\n".join(L)
                 + f"\n\n**Round 2's final change: `{name_of(final2)}`** = `{over_of(final2)}`.")
    return final2


def decide3() -> list[str]:
    p = plan()
    if "final3" in p:
        return p["final3"]
    R = load()
    rows, best = [], None
    for parts in list(ELIGIBLE2) + list(SCREEN3):
        sc = score(R, name_of(parts), "base")
        if sc is None:
            raise SystemExit(f"screen incomplete for {name_of(parts)}")
        mn = min(v for d in DATASETS for v in sc[2][d])
        ok = sc[0] >= 0.005
        rows.append((parts, sc, mn, ok))
        if ok and (best is None or mn > best[1]):
            best = (parts, mn)
    L = ["| variant | GAIA dHR@1 / dF1 | Sock Shop | Hotel | score | smallest | eligible |",
         "|---|---|---|---|---|---|---|"]
    for parts, sc, mn, ok in rows:
        cells = " | ".join(f"{sc[2][d][0]:+.3f} / {sc[2][d][1]:+.3f}" for d in DATASETS)
        L.append(f"| {name_of(parts)} | {cells} | {sc[0]:+.4f} | {mn:+.3f} | {'yes' if ok else 'no'} |")
    final3 = best[0] if best else p["final2"]
    p.update(final3=final3, written_final3=stamp())
    save_plan(p, f"\n## Round 3: decision, {p['written_final3']} (before its confirmation runs)\n\n"
                 f"Differences against AdaDiag-DL on the development seeds:\n\n" + "\n".join(L)
                 + f"\n\n**Round 3's final change: `{name_of(final3)}`** = `{over_of(final3)}`.")
    return final3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["rule", "A", "plan-b", "B", "plan-c", "C", "decide", "confirm",
                             "rule2", "screen2", "decide2", "confirm2", "seeds5-9",
                             "rule3", "screen3", "decide3", "confirm3"])
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--variants", default=None, help="confirm: comma-separated subset")
    a = ap.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "rule":
        if not (OUT / "PLAN.md").exists():
            (OUT / "PLAN.md").write_text(
                f"# A stronger AdaDiag: the rule and the screen (README §6.24)\n\n"
                f"Rule written {stamp()}, before any run.\n\n**Rule.** {RULE}\n")
        print((OUT / "PLAN.md").read_text()); return
    if a.stage in ("plan-b", "plan-c", "decide"):
        {"plan-b": plan_b, "plan-c": plan_c, "decide": decide}[a.stage](); return
    if a.stage == "rule2":
        p = plan()
        if "rule2" not in p:
            p.update(rule2=RULE2, written_rule2=stamp())
            save_plan(p, f"\n## Round 2: rule, written {p['written_rule2']} (before its runs)\n\n"
                         f"**Rule.** {RULE2}")
        return
    if a.stage == "decide2":
        decide2(); return
    if a.stage == "rule3":
        p = plan()
        if "rule3" not in p:
            p.update(rule3=RULE3, written_rule3=stamp())
            save_plan(p, f"\n## Round 3: rule, written {p['written_rule3']} (before its runs)\n\n"
                         f"**Rule.** {RULE3}")
        return
    if a.stage == "decide3":
        decide3(); return
    if not (OUT / "PLAN.md").exists():
        raise SystemExit("write the rule first: --stage rule")
    ds = data.load(a.dataset)
    out = OUT / f"screen_{a.dataset}.json"
    if a.stage == "A":
        for parts in ([], *[[n] for n in STAGE_A]):
            run(ds, ARM, name_of(parts), over_of(parts), DEV, out)
    elif a.stage == "B":
        anc = plan()["anchor"]
        for n in additions():
            run(ds, ARM, name_of(anc + [n]), over_of(anc + [n]), DEV, out)
    elif a.stage == "screen2":
        if "rule2" not in plan():
            raise SystemExit("write round 2's rule first: --stage rule2")
        run(ds, ARM, "bag+rootd", over_of(["bag", "rootd"]), DEV, out)
    elif a.stage == "confirm2":
        fin2 = decide2()
        names = a.variants.split(",") if a.variants else ["align", "align_plus", "align_plus2",
                                                          "tvdiag"]
        for n in names:
            if n == "align_plus2" and fin2 == plan()["final"]:
                continue                                  # round 2 kept round 1's change
            arm = "v5_align" if n.startswith("align") else "v0_raw"
            over = {"align_plus": over_of(plan()["final"]), "align_plus2": over_of(fin2)}.get(n, {})
            run(ds, arm, n, over, FRESH2, OUT / f"confirm2_{a.dataset}.json")
    elif a.stage == "screen3":
        if "rule3" not in plan():
            raise SystemExit("write round 3's rule first: --stage rule3")
        for parts in SCREEN3:
            run(ds, ARM, name_of(parts), over_of(parts), DEV, out)
    elif a.stage == "confirm3":
        fin3 = decide3()
        for n in (a.variants.split(",") if a.variants else ["align", "align_plus3", "tvdiag"]):
            arm = "v5_align" if n.startswith("align") else "v0_raw"
            run(ds, arm, n, over_of(fin3) if n == "align_plus3" else {}, FRESH3,
                OUT / f"confirm3_{a.dataset}.json")
    elif a.stage == "seeds5-9":
        # both rounds' changes on seeds 5-9, the seeds of the paper-format tables (section 6.17),
        # where AdaDiag-DL and TVDiag's released code already have runs with these settings
        p = plan()
        for n, parts in (("align_plus", p["final"]), ("align_plus2", p["final2"]),
                         ("align_plus3", p.get("final3"))):
            if not parts or (n == "align_plus3" and parts in (p["final"], p["final2"], ["theta"])):
                continue                    # theta alone already has seeds 5-9 (section 6.17)
            run(ds, ARM, n, over_of(parts), range(5, 10), OUT / f"confirm59_{a.dataset}.json")
    elif a.stage == "C":
        p = plan()
        if len(p["kept_b"]) >= 2:
            parts = p["anchor"] + p["kept_b"]
            run(ds, ARM, name_of(parts), over_of(parts), DEV, out)
    else:
        names = a.variants.split(",") if a.variants else ["align", "align_plus", "tvdiag",
                                                          "tvdiag_plus"]
        fin = over_of(decide()) if any(n.endswith("plus") for n in names) else {}
        if any(n.endswith("plus") for n in names) and not fin:
            raise SystemExit("the final change is empty; nothing to confirm")
        for n in names:
            arm = "v5_align" if n.startswith("align") else "v0_raw"
            run(ds, arm, n, fin if n.endswith("plus") else {}, FRESH,
                OUT / f"confirm_{a.dataset}.json")
    print("done")


if __name__ == "__main__":
    main()
