#!/usr/bin/env python3
"""README section 6.21: changes aimed at improving RCL and FTI, screened then confirmed.

Stages:
  screen   TVDiag (v0_raw) with each candidate alone, development seeds 0-2, all three
           datasets -> results/improve/screen_<ds>.json
  combine  apply RULE (below) to the screen, write results/improve/PLAN.md (timestamped,
           before any confirmation run), and screen the combination on the dev seeds
  confirm  fresh seeds 10-14, never used before: TVDiag and AdaDiag-DL, each as released
           and with the final change -> results/improve/confirm_<ds>.json

Settings as section 6.17: 1 thread, 500-epoch cap, aligned dropout masks.

    python scripts/improve.py --stage screen  --dataset hotel
    python scripts/improve.py --stage combine
    python scripts/improve.py --stage confirm --dataset hotel
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
from adadiag.evaluate import per_class                               # noqa: E402
from adadiag.train import train                                      # noqa: E402

OUT = ROOT / "results" / "improve"
DATASETS = ("gaia", "sockshop", "hotel")
DEV, FRESH = range(0, 3), range(10, 15)
#: name -> config overrides, each tried alone in TVDiag
CANDIDATES = {
    "theta":      dict(awl_learnable=True),            # Eq. 13 as the paper states it (6.19)
    "es_diag":    dict(early_stop_on="diag"),          # stop on RCL + FTI, not the total loss
    "cw_sqrt":    dict(fti_class_weight="sqrt"),       # FTI loss weighted ~ n_c^-1/2
    "moddrop":    dict(aug_modality_drop=0.2),         # TVDiag's unused modality dropout
    "aug2":       dict(aug_copies=2),                  # two augmented copies per incident
    "rcl_global": dict(rcl_context="global"),          # RCL also sees the incident feature
    "rcl_neigh":  dict(rcl_context="neighbour"),       # RCL: node minus neighbours' mean
    "fti_root":   dict(fti_readout="root_attn"),       # FTI reads the likely culprit
}
RULE = (
    "On the development seeds 0-2, each candidate's score is the mean over the three datasets "
    "of (dHR@1 + dF1) / 2, paired against TVDiag on the same seeds. A candidate is kept if its "
    "score is above +0.005 and no dataset loses more than 0.015 on HR@1 or F1. The kept "
    "candidates are combined (if both RCL-context variants are kept, rcl_context='both'). The "
    "combination is screened on the same seeds; the final change is the combination if its "
    "score is at least the best kept candidate's, otherwise the best kept candidate alone. If "
    "no candidate is kept, nothing is confirmed. The final change is then confirmed on fresh "
    "seeds 10-14 for TVDiag and AdaDiag-DL, against each as released.")


def run(ds, arm, name, over, seeds, out):
    recs = json.load(open(out)) if out.exists() else []
    done = {(r["variant"], r["seed"]) for r in recs}
    for seed in seeds:
        if (name, seed) in done:
            continue
        cfg = config.get(ds.name, isolate_weighting_rng=True, **over)
        res, model = train(arm, seed, ds, cfg)
        res.update(variant=name, overrides=over, per_class=per_class(model, ds)[0])
        recs = json.load(open(out)) if out.exists() else []
        recs.append(res)
        json.dump(recs, open(out, "w"))
        print(f"  {ds.name:8s} {name:22s} s{seed:<2d} HR@1 {res['HR@1']:.3f}  F1 {res['f1']:.3f}  "
              f"macro-F1 {res['macro_f1']:.3f}  ({res['epochs']} ep, {res['train_s']:.0f}s)", flush=True)


def load(stage):
    R = {}
    for ds in DATASETS:
        f = OUT / f"{stage}_{ds}.json"
        for r in (json.load(open(f)) if f.exists() else []):
            R.setdefault(ds, {}).setdefault(r["variant"], {})[r["seed"]] = r
    return R


def score(R, name, ref="base"):
    """(score, worst single-dataset loss, per-dataset deltas) on the dev seeds."""
    per, worst = {}, 0.0
    for ds in DATASETS:
        a, b = R.get(ds, {}).get(name, {}), R.get(ds, {}).get(ref, {})
        s = sorted(set(a) & set(b))
        if not s:
            return None
        dh = float(np.mean([a[i]["HR@1"] - b[i]["HR@1"] for i in s]))
        df = float(np.mean([a[i]["f1"] - b[i]["f1"] for i in s]))
        per[ds] = (dh, df, len(s))
        worst = min(worst, dh, df)
    return float(np.mean([(h + f) / 2 for h, f, _ in per.values()])), worst, per


def combine() -> dict:
    R = load("screen")
    rows, kept = [], []
    for name in CANDIDATES:
        sc = score(R, name)
        if sc is None:
            raise SystemExit(f"screen incomplete for {name}")
        ok = sc[0] > 0.005 and sc[1] >= -0.015
        rows.append((name, sc, ok))
        if ok:
            kept.append(name)
    over = {}
    for name in kept:
        over.update(CANDIDATES[name])
    if "rcl_global" in kept and "rcl_neigh" in kept:
        over["rcl_context"] = "both"
    best = max(kept, key=lambda n: score(R, n)[0]) if kept else None
    plan = dict(rule=RULE, written=datetime.datetime.now().isoformat(timespec="seconds"),
                kept=kept, combination=over, best_single=best,
                screen={n: dict(score=sc[0], worst=sc[1],
                                per={d: dict(dHR1=v[0], dF1=v[1], n=v[2]) for d, v in sc[2].items()})
                        for n, sc, _ in rows})
    OUT.mkdir(parents=True, exist_ok=True)
    L = ["# Improving RCL and FTI: the rule and the screen (README §6.21)\n",
         f"Written {plan['written']}, before any confirmation run.\n", f"**Rule.** {RULE}\n",
         "| candidate | GAIA dHR@1 / dF1 | Sock Shop | Hotel | score | worst | kept |",
         "|---|---|---|---|---|---|---|"]
    for name, sc, ok in rows:
        cells = " | ".join(f"{sc[2][d][0]:+.3f} / {sc[2][d][1]:+.3f}" for d in DATASETS)
        L.append(f"| {name} | {cells} | {sc[0]:+.4f} | {sc[1]:+.3f} | {'yes' if ok else 'no'} |")
    L.append(f"\nKept: {', '.join(kept) or 'none'}. Combination: `{over}`. Best single: {best}.\n")
    (OUT / "PLAN.md").write_text("\n".join(L))
    json.dump(plan, open(OUT / "plan.json", "w"), indent=1)
    print("\n".join(L))
    return plan


def final_change() -> dict | None:
    """The change to confirm, decided by RULE from the screen and the combination's screen."""
    plan = json.load(open(OUT / "plan.json"))
    if not plan["kept"]:
        return None
    if len(plan["kept"]) == 1:
        return CANDIDATES[plan["kept"][0]]
    R = load("screen")
    sc = score(R, "combination")
    if sc is None:
        raise SystemExit("screen the combination first: --stage screen --variants combination")
    best = plan["screen"][plan["best_single"]]["score"]
    fin = plan["combination"] if sc[0] >= best else CANDIDATES[plan["best_single"]]
    if "decision" not in plan:          # recorded once, before the first confirmation run
        plan["decision"] = dict(combination_score=sc[0], combination_worst=sc[1],
                                best_single_score=best, final=fin,
                                written=datetime.datetime.now().isoformat(timespec="seconds"))
        json.dump(plan, open(OUT / "plan.json", "w"), indent=1)
        per = " | ".join(f"{d}: {sc[2][d][0]:+.3f} / {sc[2][d][1]:+.3f}" for d in DATASETS)
        with open(OUT / "PLAN.md", "a") as f:
            f.write(f"\n**Decision** ({plan['decision']['written']}, before the improved "
                    f"confirmation runs). Combination on the dev seeds: {per}; score "
                    f"{sc[0]:+.4f} (worst {sc[1]:+.3f}) against the best single, "
                    f"{plan['best_single']}, {best:+.4f}. Final change: `{fin}`.\n")
    return fin


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["screen", "combine", "confirm"])
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--variants", default=None, help="screen: comma-separated subset "
                                                     "(base, candidates, combination)")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "combine":
        combine()
        return
    ds = data.load(a.dataset)
    if a.stage == "screen":
        names = a.variants.split(",") if a.variants else ["base", *CANDIDATES]
        for n in names:
            over = ({} if n == "base" else json.load(open(OUT / "plan.json"))["combination"]
                    if n == "combination" else CANDIDATES[n])
            run(ds, "v0_raw", n, over, DEV, OUT / f"screen_{a.dataset}.json")
    else:
        out = OUT / f"confirm_{a.dataset}.json"
        names = a.variants.split(",") if a.variants else ["tvdiag", "tvdiag_improved",
                                                          "align", "align_improved"]
        # the released baselines do not depend on the rule, so they can run before it is applied
        fin = final_change() if any(n.endswith("improved") for n in names) else {}
        if fin is None:
            raise SystemExit("no candidate was kept; nothing to confirm")
        for n in names:
            arm = "v5_align" if n.startswith("align") else "v0_raw"
            run(ds, arm, n, dict(fin) if n.endswith("improved") else {}, FRESH, out)
    print("done")


if __name__ == "__main__":
    main()
