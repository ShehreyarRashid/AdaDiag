#!/usr/bin/env python3
"""README section 6.23: what TVDiag's encoder misses -- three diagnostics.

1. `--part evidence` (no training). Where the alert evidence sits. Per failure type: how
   often the root has an alert, how many instances have one, and how well a one-line rule on
   the DIRECTION of anomalous calls finds the root. Trace alerts are edge-level records
   (caller, callee, operation, PD/500/ERROR); TVDiag turns them into node features and, on
   Sock Shop and Hotel, symmetric edges, so its encoder cannot see who called whom. The rule:
   rank instances first by "receives anomalous calls but makes none to other instances"
   (the deepest anomalous callee), then by how many anomalous calls they receive. Ties are
   broken at random in expectation (a k-way tie at the top scores 1/k). Not tuned: one rule,
   reported on train and test.
2. `--part errors` (trains TVDiag, dev seeds 0-2, the section 6.21 settings, so seed s is
   bit-identical to "base" in results/improve/screen_<ds>.json -- asserted). Where its wrong
   top-1 guesses land: hop distance to the true root, and whether the guess is an instance
   that is never a root in the training split. Also the post-hoc HR@1 with those instances
   masked out (a prior read from training labels only).
3. Writes results/encoder/diagnose_<part>.json and prints the tables.

    python scripts/encoder_diagnose.py --part evidence
    python scripts/encoder_diagnose.py --part errors --dataset hotel
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "encoder"
DATASETS = ("gaia", "sockshop", "hotel")


def trace_rule(calls, names):
    """Scores per instance from one incident's trace alerts (higher = more likely root)."""
    idx = {n: k for k, n in enumerate(names)}
    inc, out = np.zeros(len(names)), np.zeros(len(names))
    for a in calls:
        src, dst = a[0], a[1]
        if dst not in idx:
            continue
        inc[idx[dst]] += 1                                  # self-spans count as received
        if src in idx and src != dst:
            out[idx[src]] += 1
    sink = (inc > 0) & (out == 0)
    return sink * 1000.0 + inc                              # sink first, then volume


def expected_hit(score, root, k=1):
    """P(root in top-k) when ties are broken uniformly at random."""
    s = score[root]
    above, tied = int((score > s).sum()), int((score == s).sum())
    if above >= k:
        return 0.0
    return min(1.0, (k - above) / tied)


def evidence():
    import csv
    res = {}
    for ds in DATASETS:
        d = ROOT / "data" / ds
        traces = json.load(open(d / "raw" / "traces.json"))
        nodes = json.load(open(d / "raw" / "nodes.json"))
        rows = [r for r in csv.DictReader(open(d / "label.csv"))
                if r["index"] in traces and r["instance"] in nodes.get(r["index"], [])]
        per = defaultdict(lambda: defaultdict(list))
        for r in rows:
            names = nodes[r["index"]]
            root = names.index(r["instance"])
            sc = trace_rule(traces[r["index"]], names)
            per[(r["data_type"], r["anomaly_type"])]["hr1"].append(expected_hit(sc, root, 1))
            per[(r["data_type"], r["anomaly_type"])]["hr3"].append(expected_hit(sc, root, 3))
            per[(r["data_type"], r["anomaly_type"])]["root_called"].append(float(sc[root] % 1000 > 0))
        res[ds] = {f"{sp}|{t}": {k: float(np.mean(v)) for k, v in m.items()} | {"n": len(m["hr1"])}
                   for (sp, t), m in sorted(per.items())}
        for sp in ("train", "test"):
            allr = [x for (s, _), m in per.items() if s == sp for x in m["hr1"]]
            res[ds][f"{sp}|ALL"] = {"hr1": float(np.mean(allr)), "n": len(allr)}
    return res


def _top_pairs(wrong, root, top, ds, te, k=8):
    """The k most frequent (failure type, true root, wrong guess) triples; one example index each."""
    from collections import Counter
    key = {}
    c = Counter()
    for i in wrong.tolist():
        t = (int(ds.ytype[te[i]]), int(root[i]), int(top[i]))
        c[t] += 1
        key.setdefault(t, i)
    return [(key[t], n) for t, n in c.most_common(k)]


def errors(ds_name, seeds):
    import torch
    from adadiag import config, data
    from adadiag.train import train
    torch.set_num_threads(1)
    ds = data.load(ds_name)
    base = {r["seed"]: r for r in json.load(open(ROOT / "results" / "improve" / f"screen_{ds_name}.json"))
            if r["variant"] == "base"}
    tr, te = ds.train_idx, ds.test_idx
    never = torch.bincount(ds.root[tr], minlength=ds.n_nodes) == 0
    U = ((ds.A + ds.A.transpose(1, 2)) > 0).float()
    out = {"never_root": [ds.instances[i] for i in range(ds.n_nodes) if never[i]], "seeds": {}}
    for seed in seeds:
        cfg = config.get(ds_name, isolate_weighting_rng=True)
        res, model = train("v0_raw", seed, ds, cfg)
        assert abs(res["HR@1"] - base[seed]["HR@1"]) < 1e-9, "not the screen's base run"
        model.eval()
        with torch.no_grad():
            keep = torch.ones(len(te), ds.n_nodes, dtype=torch.bool)
            _, _, logits, tl = model(ds.X[te], ds.A[te], keep, ds.cnt[te])
        root = ds.root[te]
        top = logits.argmax(1)
        wrong = (top != root).nonzero().flatten()
        hops = []
        for i in wrong.tolist():
            a = U[te[i]]
            reach, frontier, h = {int(root[i])}, {int(root[i])}, 0
            while int(top[i]) not in reach and frontier and h < 6:
                h += 1
                nxt = set(a[list(frontier)].nonzero()[:, 1].tolist()) - reach
                reach |= nxt; frontier = nxt
            hops.append(h if int(top[i]) in reach else -1)
        masked = logits.masked_fill(never[None, :], float("-inf"))
        per_type = {}
        for c, cname in enumerate(ds.types):
            sel = (ds.ytype[te] == c)
            if sel.sum() == 0:
                continue
            w = sel & (top != root)
            per_type[cname] = dict(
                n=int(sel.sum()), hr1=float((top[sel] == root[sel]).float().mean()),
                hr1_masked=float((masked[sel].argmax(1) == root[sel]).float().mean()),
                wrong_on_never_root=float(never[top[w]].float().mean()) if w.any() else 0.0)
        h = np.array(hops)
        # chance: a uniformly random wrong guess, i.e. the share of non-root instances one
        # hop from the root, averaged over the same wrong cases
        chance1 = float(np.mean([(U[te[i], int(root[i])].sum() - U[te[i], int(root[i]), int(root[i])])
                                 / (ds.n_nodes - 1) for i in wrong.tolist()])) if len(wrong) else 0.0
        out["seeds"][seed] = dict(
            hr1=float((top == root).float().mean()),
            hr1_masked=float((masked.argmax(1) == root).float().mean()),
            n_wrong=len(hops), hop1=float((h == 1).mean()) if len(h) else 0.0, hop1_chance=chance1,
            hop2=float((h == 2).mean()) if len(h) else 0.0,
            hop3plus=float((h >= 3).mean()) if len(h) else 0.0,
            unreachable=float((h == -1).mean()) if len(h) else 0.0,
            wrong_on_never_root=float(never[top[wrong]].float().mean()) if len(wrong) else 0.0,
            per_type=per_type,
            confusions=[[f"{ds.types[int(ds.ytype[te[i]])]}: {ds.instances[int(root[i])]} -> "
                         f"{ds.instances[int(top[i])]}", n] for i, n in _top_pairs(wrong, root, top, ds, te)])
        s = out["seeds"][seed]
        print(f"{ds_name} seed {seed}: HR@1 {s['hr1']:.3f} (masked {s['hr1_masked']:.3f}); "
              f"{s['n_wrong']} wrong: 1-hop {s['hop1']:.2f} (chance {s['hop1_chance']:.2f}), 2-hop {s['hop2']:.2f}, "
              f"3+ {s['hop3plus']:.2f}; on a never-root instance {s['wrong_on_never_root']:.2f}",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["evidence", "errors"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part == "evidence":
        r = evidence()
        json.dump(r, open(OUT / "diagnose_evidence.json", "w"), indent=1)
        for ds, m in r.items():
            print(f"\n{ds}")
            for k, v in m.items():
                print(f"  {k:40s} " + "  ".join(f"{kk} {vv:.2f}" if isinstance(vv, float) else f"{kk} {vv}"
                                                  for kk, vv in v.items()))
    else:
        f = OUT / f"diagnose_errors_{a.dataset}.json"
        r = errors(a.dataset, [int(s) for s in a.seeds.split(",")])
        if f.exists():                                       # keep seeds run earlier
            old = json.load(open(f))
            r["seeds"] = {**old["seeds"], **{str(k): v for k, v in r["seeds"].items()}}
        json.dump(r, open(f, "w"), indent=1)


if __name__ == "__main__":
    main()
