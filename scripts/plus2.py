#!/usr/bin/env python3
"""v1.22 README section 6.25: improving GradFusion+ -- round 4, screened then confirmed.

Starting point: GradFusion+ (section 6.24) = GradFusion-DL + alert-type counts + Eq. 13's learnable
theta. Candidates, all off by default, each at one setting fixed in advance:

  smooth    fti_smooth=0.1   label smoothing on the FTI loss (122-248 training incidents)
  bagdrop   bag_drop=0.2     dropout on the alert-type count columns during training
  baghead   bag_head=True    the heads also read the raw counts (a wide path beside the GNN)
  ens3      an ensemble of three models that differ only in their seed (s, s+100, s+200);
            their softmax outputs are averaged for RCL and for FTI

Every run stores its test logits (results/plus2/runs/<ds>/*.npz), so ensembles are computed
from finished runs. Stages: `rule` (writes the rule), `S` (singles, development seeds 0-2),
`decide-s`, `C` (the combination, only if two or more singles are kept), `E` (the two extra
members per development seed), `decide`, `confirm` (fresh seeds 30-34), `seeds5-9` (the
tables' seeds). Jobs are claimed through lock files, so several workers can share a stage:

    python scripts/plus2.py --stage rule
    python scripts/plus2.py --stage S &  python scripts/plus2.py --stage S
    python scripts/plus2.py --stage decide-s
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch                                                         # noqa: E402
from gradfusion import config, data                                     # noqa: E402
from gradfusion.alerts import with_alert_bag                            # noqa: E402
from gradfusion.evaluate import per_class                               # noqa: E402
from gradfusion.train import train                                      # noqa: E402

OUT = ROOT / "results" / "plus2"
DATASETS = ("gaia", "sockshop", "hotel")
COST = {"gaia": 180, "sockshop": 460, "hotel": 180}                  # seconds, for job order
DEV, FRESH, TABLE = range(0, 3), range(30, 35), range(5, 10)
MEMBERS = (0, 100, 200)                                              # seed offsets of ens3
PLUS = dict(alert_bag=True, awl_learnable=True)                      # GradFusion+ (section 6.24)
SINGLES = {"smooth": dict(fti_smooth=0.1), "bagdrop": dict(bag_drop=0.2),
           "baghead": dict(bag_head=True)}
RULE = (
    "Round 4. Reference: GradFusion+ (bag+theta) on the development seeds 0-2. Settings as section "
    "6.17. A variant's six differences are dHR@1 and dF1 on the three datasets, paired by seed; "
    "its score is their mean. "
    "Stage S: smooth, bagdrop and baghead, each added alone to GradFusion+. A single is kept if its "
    "score is at least +0.005 and its smallest difference is at least -0.010. With two or more "
    "kept, their combination is screened too (stage C). The single-model configuration is the "
    "eligible variant (kept singles, and the combination if its score is at least +0.005 and "
    "its smallest difference at least -0.010) whose smallest difference is largest; with none "
    "eligible it stays GradFusion+. "
    "Stage E: ens3 of that configuration (seeds s, s+100, s+200; softmax outputs averaged) "
    "against the configuration itself on seeds 0-2; the ensemble is adopted if its score is at "
    "least +0.005 and its smallest difference at least -0.010. "
    "Confirmation: fresh seeds 30-34, never used before, for TVDiag, GradFusion-DL, GradFusion+ and "
    "the final configuration (and, as a control, ens3 of TVDiag). Paired t-tests, Holm over "
    "HR@1 and F1 x 3 datasets per comparison. There is no round 5 on these seeds.")


def stamp() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def variant(name: str) -> tuple[str, dict]:
    """name -> (arm, overrides). 'tvdiag', 'align', 'plus', 'plus+smooth+baghead', ..."""
    if name == "tvdiag":
        return "v0_raw", {}
    if name == "align":
        return "v5_align", {}
    parts = name.split("+")
    assert parts[0] == "plus", name
    over = dict(PLUS)
    for p in parts[1:]:
        over.update(SINGLES[p])
    return "v5_align", over


def metrics(node_p: np.ndarray, type_p: np.ndarray, root: np.ndarray, y: np.ndarray) -> dict:
    """TVDiag's metrics (gradfusion.evaluate) from scores instead of a model."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    order = np.argsort(-node_p, axis=1, kind="stable")
    rank = (order == root[:, None]).argmax(1) + 1
    out = {f"HR@{k}": float((rank <= k).mean()) for k in range(1, 6)}
    out["MRR@3"] = float(np.where(rank <= 3, 1.0 / rank, 0.0).mean())
    yp = type_p.argmax(1)
    p = precision_score(y, yp, average="weighted", zero_division=0)
    r = recall_score(y, yp, average="weighted", zero_division=0)
    out.update(precision=float(p), recall=float(r), f1=float(2 * p * r / (p + r)) if p + r else 0.0,
               macro_f1=float(f1_score(y, yp, average="macro", zero_division=0)))
    return out


def _path(ds: str, name: str, seed: int) -> Path:
    return OUT / "runs" / ds / f"{name}__s{seed}"


def run_job(dsets: dict, ds_name: str, name: str, seed: int) -> None:
    f = _path(ds_name, name, seed)
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.with_suffix(".json").exists():
        return
    try:
        fd = os.open(str(f.with_suffix(".lock")), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return                                               # another worker has it
    if ds_name not in dsets:
        dsets[ds_name] = data.load(ds_name)
    ds = dsets[ds_name]
    arm, over = variant(name)
    d = with_alert_bag(ds) if over.get("alert_bag") else ds
    res, model = train(arm, seed, d, config.get(ds_name, isolate_weighting_rng=True, **over))
    model.eval()
    te = d.test_idx
    with torch.no_grad():
        keep = torch.ones(len(te), d.n_nodes, dtype=torch.bool)
        _, _, nl, tl = model(d.X[te], d.A[te], keep, d.cnt[te])
    m = metrics(torch.softmax(nl, 1).numpy(), torch.softmax(tl, 1).numpy(),
                d.root[te].numpy(), d.ytype[te].numpy())
    assert abs(m["HR@1"] - res["HR@1"]) < 1e-9 and abs(m["f1"] - res["f1"]) < 1e-9, (m, res)
    np.savez_compressed(f.with_suffix(".npz"), node=nl.numpy(), type=tl.numpy())
    res.update(variant=name, overrides=over, per_class=per_class(model, d)[0])
    json.dump(res, open(f.with_suffix(".json"), "w"))
    f.with_suffix(".lock").unlink()
    print(f"  {ds_name:8s} {name:24s} s{seed:<3d} HR@1 {res['HR@1']:.3f}  F1 {res['f1']:.3f}  "
          f"({res['epochs']} ep, {res['train_s']:.0f}s)", flush=True)


def single(ds: str, name: str, seed: int) -> dict | None:
    f = _path(ds, name, seed).with_suffix(".json")
    return json.load(open(f)) if f.exists() else None


_LABELS: dict = {}


def labels(ds_name: str):
    if ds_name not in _LABELS:
        d = data.load(ds_name)
        _LABELS[ds_name] = (d.root[d.test_idx].numpy(), d.ytype[d.test_idx].numpy(), d.types)
    return _LABELS[ds_name]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def ensemble(ds: str, name: str, seed: int, offsets=MEMBERS) -> dict | None:
    """Metrics of the ensemble of `name` trained with seeds seed + offsets (None if incomplete)."""
    fs = [_path(ds, name, seed + o).with_suffix(".npz") for o in offsets]
    if not all(f.exists() for f in fs):
        return None
    Z = [np.load(f) for f in fs]
    node = np.mean([softmax(z["node"]) for z in Z], 0)
    typ = np.mean([softmax(z["type"]) for z in Z], 0)
    root, y, types = labels(ds)
    m = metrics(node, typ, root, y)
    rank = (np.argsort(-node, axis=1, kind="stable") == root[:, None]).argmax(1) + 1
    m["per_class"] = {t: dict(n=int((y == c).sum()), hr1=float((rank[y == c] <= 1).mean()),
                              hr3=float((rank[y == c] <= 3).mean()),
                              fti_recall=float((typ[y == c].argmax(1) == c).mean()))
                      for c, t in enumerate(types) if (y == c).any()}
    m.update(variant=f"{name}|ens{len(offsets)}", seed=seed,
             epochs=float(np.mean([single(ds, name, seed + o)["epochs"] for o in offsets])),
             train_s=float(np.sum([single(ds, name, seed + o)["train_s"] for o in offsets])))
    return m


def get(ds: str, name: str, seed: int) -> dict | None:
    """A record by name; 'x|ens3' is the ensemble of x."""
    if name.endswith("|ens3"):
        return ensemble(ds, name[:-5], seed)
    return single(ds, name, seed)


def diffs(name: str, ref: str, seeds):
    """({ds: (dHR@1, dF1)}, score, smallest) or None if a run is missing."""
    per = {}
    for ds in DATASETS:
        a = [get(ds, name, s) for s in seeds]
        b = [get(ds, ref, s) for s in seeds]
        if any(x is None for x in a + b):
            return None
        per[ds] = (float(np.mean([x["HR@1"] - y["HR@1"] for x, y in zip(a, b)])),
                   float(np.mean([x["f1"] - y["f1"] for x, y in zip(a, b)])))
    flat = [v for d in DATASETS for v in per[d]]
    return per, float(np.mean(flat)), float(min(flat))


def ok(d) -> bool:
    return d is not None and d[1] >= 0.005 and d[2] >= -0.010


def plan() -> dict:
    f = OUT / "plan.json"
    return json.load(open(f)) if f.exists() else {}


def save_plan(p: dict, md: str) -> None:
    json.dump(p, open(OUT / "plan.json", "w"), indent=1)
    with open(OUT / "PLAN.md", "a") as fh:
        fh.write(md + "\n")
    print(md)


def table(names, ref, seeds=DEV) -> str:
    L = ["| variant | GAIA dHR@1 / dF1 | Sock Shop | Hotel | score | smallest | eligible |",
         "|---|---|---|---|---|---|---|"]
    for n in names:
        d = diffs(n, ref, seeds)
        if d is None:
            raise SystemExit(f"runs missing for {n}")
        cells = " | ".join(f"{d[0][k][0]:+.3f} / {d[0][k][1]:+.3f}" for k in DATASETS)
        L.append(f"| {n} | {cells} | {d[1]:+.4f} | {d[2]:+.3f} | {'yes' if ok(d) else 'no'} |")
    return "\n".join(L)


def decide_s() -> None:
    p = plan()
    if "kept" in p:
        print("stage S already decided:", p["kept"]); return
    names = [f"plus+{k}" for k in SINGLES]
    kept = [k for k in SINGLES if ok(diffs(f"plus+{k}", "plus", DEV))]
    p.update(kept=kept, written_s=stamp())
    save_plan(p, f"\n## Stage S (reference: GradFusion+), {p['written_s']}\n\n{table(names, 'plus')}\n\n"
                 f"Kept: {', '.join(kept) or 'none'}.")


def candidates_single() -> list[str]:
    kept = plan()["kept"]
    names = [f"plus+{k}" for k in kept]
    if len(kept) >= 2:
        names.append("plus+" + "+".join(kept))
    return names


def decide() -> None:
    p = plan()
    if "final" in p:
        print("already decided:", p["final"]); return
    names = candidates_single()
    elig = [(n, diffs(n, "plus", DEV)) for n in names]
    elig = [(n, d) for n, d in elig if ok(d)]
    cfg = max(elig, key=lambda t: t[1][2])[0] if elig else "plus"
    if "config" not in p:
        p.update(config=cfg, written_config=stamp())
        save_plan(p, f"\n## Single-model configuration, {p['written_config']}\n\n"
                     + (table(names, "plus") + "\n\n" if names else "No single was kept. ")
                     + f"Configuration: `{cfg}`.")
    cfg = p["config"]
    d = diffs(f"{cfg}|ens3", cfg, DEV)
    if d is None:
        raise SystemExit("run stage E first (the ensemble's extra members)")
    final = f"{cfg}|ens3" if ok(d) else cfg
    p.update(final=final, written_final=stamp())
    save_plan(p, f"\n## Stage E and decision, {p['written_final']} (before any confirmation run)\n\n"
                 f"Reference: `{cfg}`.\n\n{table([cfg + '|ens3'], cfg)}\n\n"
                 f"**Final configuration: `{final}`.**")


def jobs(stage: str) -> list[tuple[str, str, int]]:
    p = plan()
    J = []
    if stage == "S":
        J = [(n, s) for n in ["plus"] + [f"plus+{k}" for k in SINGLES] for s in DEV]
    elif stage == "C":
        J = [(n, s) for n in candidates_single() for s in DEV]
    elif stage == "E":
        J = [(p["config"], s + o) for s in DEV for o in MEMBERS]
    elif stage in ("confirm", "seeds5-9"):
        seeds = FRESH if stage == "confirm" else TABLE
        cfg, ens = p["final"].split("|")[0], p["final"].endswith("|ens3")
        J = [(n, s) for n in dict.fromkeys(["plus", cfg]) for s in seeds]
        if stage == "confirm":
            J += [(n, s) for n in ("align", "tvdiag") for s in seeds]
            J += [("tvdiag", s + o) for s in seeds for o in MEMBERS[1:]]          # the control
        if ens:
            J += [(cfg, s + o) for s in seeds for o in MEMBERS[1:]]
    J = list(dict.fromkeys(J))
    return sorted([(ds, n, s) for n, s in J for ds in DATASETS], key=lambda j: -COST[j[0]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["rule", "S", "decide-s", "C", "E", "decide", "confirm", "seeds5-9"])
    a = ap.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "rule":
        if not (OUT / "PLAN.md").exists():
            (OUT / "PLAN.md").write_text(
                f"# Improving GradFusion+: the rule and the screen (v1.22 README §6.25)\n\n"
                f"Rule written {stamp()}, before any run.\n\n**Rule.** {RULE}\n")
        print((OUT / "PLAN.md").read_text()); return
    if not (OUT / "PLAN.md").exists():
        raise SystemExit("write the rule first: --stage rule")
    if a.stage == "decide-s":
        decide_s(); return
    if a.stage == "decide":
        decide(); return
    dsets: dict = {}
    for ds, name, seed in jobs(a.stage):
        run_job(dsets, ds, name, seed)
    print("done")


if __name__ == "__main__":
    main()
