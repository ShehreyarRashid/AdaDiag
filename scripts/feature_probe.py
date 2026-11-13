#!/usr/bin/env python3
"""README section 6.24: do the alert-type counts carry information the embeddings lose?

Simple probes, no graph and no training of AdaDiag. Features per instance: TVDiag's three
128-d embeddings ("emb"), the alert-type counts of src/adadiag/alerts.py ("bag"), or both.
  FTI   logistic regression on the incident's max- and sum-pooled features -> failure type
  RCL   logistic regression per instance (root or not), ranked within the incident -> HR@1
  FTI at the true root (an upper bound for a perfect RCL): random forest on the root's bag
One fixed setting each (C = 0.1, standardised on the training split; 500 trees), not tuned.
Trained on the training split, scored on the test split, with TVDiag's metric definitions.
Writes results/stronger/probe.json.

    python scripts/feature_probe.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adadiag import data                                             # noqa: E402
from adadiag.alerts import alert_bags                                # noqa: E402

warnings.filterwarnings("ignore")


def f1_tvdiag(yt, yp):
    from sklearn.metrics import precision_score, recall_score
    p = precision_score(yt, yp, average="weighted", zero_division=0)
    r = recall_score(yt, yp, average="weighted", zero_division=0)
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    out = {}
    for name in ("gaia", "sockshop", "hotel"):
        ds = data.load(name)
        tr, te = ds.split == "train", ds.split == "test"
        y, root = ds.ytype.numpy(), ds.root.numpy()
        M, N = len(y), ds.n_nodes
        bag = alert_bags(ds)[0].numpy().transpose(0, 2, 1, 3).reshape(M, N, -1)
        emb = ds.X.numpy().transpose(0, 2, 1, 3).reshape(M, N, -1)
        both = np.concatenate([emb, bag], -1)

        def fti(F):
            G = np.concatenate([F.max(1), F.sum(1)], -1)
            G = (G - G[tr].mean(0)) / (G[tr].std(0) + 1e-6)
            clf = LogisticRegression(C=0.1, max_iter=3000).fit(G[tr], y[tr])
            return f1_tvdiag(y[te], clf.predict(G[te]))

        def rcl(F):
            D = F.shape[-1]
            mu, sd = F[tr].reshape(-1, D).mean(0), F[tr].reshape(-1, D).std(0) + 1e-6
            Z = (F - mu) / sd
            lab = np.zeros((M, N)); lab[np.arange(M), root] = 1
            clf = LogisticRegression(C=0.1, max_iter=3000).fit(Z[tr].reshape(-1, D), lab[tr].reshape(-1))
            s = clf.decision_function(Z[te].reshape(-1, D)).reshape(-1, N)
            return float((s.argmax(1) == root[te]).mean())

        rb = bag[np.arange(M), root]
        rf = RandomForestClassifier(500, random_state=0, n_jobs=1).fit(rb[tr], y[tr])
        out[name] = {"fti_f1": {k: fti(F) for k, F in (("emb", emb), ("bag", bag), ("both", both))},
                     "rcl_hr1": {k: rcl(F) for k, F in (("emb", emb), ("bag", bag), ("both", both))},
                     "fti_f1_root_bag_rf": f1_tvdiag(y[te], rf.predict(rb[te]))}
        o = out[name]
        print(f"{name:8s} FTI F1  emb {o['fti_f1']['emb']:.3f}  bag {o['fti_f1']['bag']:.3f}  "
              f"both {o['fti_f1']['both']:.3f} | at the true root {o['fti_f1_root_bag_rf']:.3f}")
        print(f"{name:8s} RCL HR@1 emb {o['rcl_hr1']['emb']:.3f}  bag {o['rcl_hr1']['bag']:.3f}  "
              f"both {o['rcl_hr1']['both']:.3f}", flush=True)
    (ROOT / "results" / "stronger").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "results" / "stronger" / "probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
