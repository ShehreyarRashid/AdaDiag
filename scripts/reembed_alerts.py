#!/usr/bin/env python3
"""Rebuild TVDiag's 128-d alert embeddings with chosen alert fields removed (paper Fig. 15).

The released data ship the alerts (data/<ds>/raw/{metrics,traces}.json) and TVDiag's
embeddings of them (data/<ds>/tmp/*.pkl). Fig. 15 removes alert fields BEFORE embedding,
so this script re-runs TVDiag's embedding step -- a supervised fastText model per modality,
trained on the training split with TVDiag's nearest-neighbour oversampling -- adapted from
TVDiag's process/EventProcess.py and process/events/fasttext_w2v.py (MIT, see
data/LICENSE-TVDiag). Two changes, both for reproducibility, neither in the method:

* fastText runs with thread=1 and a fixed seed, and Python's `random` is seeded, so the
  embedding is deterministic (TVDiag's default of 12 threads is not).
* "w/o abDirection" drops the LAST field of a metric alert. TVDiag drops field index 3,
  which is the direction on GAIA (4 fields) but does nothing on Hotel / Sock Shop, whose
  metric alerts have 3 fields.

Writes <out>/data/<ds>/{label.csv, raw -> original, tmp/<modality>.pkl} so the result
loads with `data.load(ds, <out>)`; the untouched modalities are copied from the release.

    python scripts/reembed_alerts.py --dataset gaia --variant trace_no_op --out /tmp/emb
    variants: trace_full trace_no_op trace_no_abtype metric_full metric_no_direction
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: variant -> (modality, fields kept). Trace alert = (src, dst, operation, abnormal type);
#: metric alert = (instance, [ip,] metric name, direction).
VARIANTS = {
    "trace_full":          ("trace", dict(op=True, ab_type=True)),
    "trace_no_op":         ("trace", dict(op=False, ab_type=True)),
    "trace_no_abtype":     ("trace", dict(op=True, ab_type=False)),
    "metric_full":         ("metric", dict(direction=True)),
    "metric_no_direction": ("metric", dict(direction=False)),
}


def doc_for(modality, alerts, node, opts):
    """One instance's alert 'sentence' -- TVDiag's EventProcess.build_embedding branches."""
    if modality == "trace":
        sel = [e for e in alerts if (node in e[0] or node in e[1])]
        if opts["op"] and opts["ab_type"]:
            return ["&".join(e) for e in sel]
        if not opts["op"] and opts["ab_type"]:
            return ["&".join(x for i, x in enumerate(e) if i != 2) for e in sel]
        if opts["op"] and not opts["ab_type"]:
            return ["&".join(x for i, x in enumerate(e) if i != 3) for e in sel]
        return ["&".join(e[:2]) for e in sel]
    if modality == "metric":
        if opts["direction"]:
            return ["&".join(e) for e in alerts if node in e[0]]
        return ["&".join(x for i, x in enumerate(e) if i != len(e) - 1)
                for e in alerts if (node in e[0] or node in e[1])]
    return ["&".join(e) for e in alerts if node in e[0]]


class FastTextEncoder:
    """TVDiag's FastTextEncoder, made deterministic (thread=1, seed)."""

    def __init__(self, modality, nodes, types, dim=128, epochs=5, seed=1, workdir="."):
        self.modality, self.nodes, self.types = modality, nodes, types
        self.dim, self.epochs, self.seed, self.workdir = dim, epochs, seed, workdir

    def _train(self, path):
        import fasttext
        return fasttext.train_supervised(path, dim=self.dim, minCount=1, minn=0, maxn=0,
                                         epoch=self.epochs, thread=1, seed=self.seed,
                                         verbose=0)

    def _oversample(self, data_set, labels, model):
        final = data_set.copy()
        for t in self.types:
            for node in self.nodes:
                tag = node + str(self.types.index(t))
                idxs = [i for i, lab in enumerate(labels) if lab.split("__label__")[-1] == tag]
                count = len(idxs)
                if count == 0:
                    continue
                texts = [data_set[i] for i in idxs]
                loop, target = 0, 1000
                while count < target:
                    loop += 1
                    if loop >= 10 * target:
                        break
                    text, _ = texts[random.randint(0, len(texts) - 1)].split("\t")
                    words = text.split()
                    if len(words) < 1:
                        continue
                    for j in random.sample(range(len(words)), 1):
                        words[j] = model.get_nearest_neighbors(words[j])[0][-1]
                    final.append(" ".join(words) + f"\t__label__{tag}\n")
                    count += 1
        return final

    def _save(self, rows, name):
        path = os.path.join(self.workdir, name)
        with open(path, "w") as f:
            f.writelines(rows)
        return path

    def fit(self, docs, labels):
        rows = [f"{' '.join(d)}\t{labels[i]}\n" for i, d in enumerate(docs)]
        model = self._train(self._save(rows, f"{self.modality}-train.txt"))
        rows = self._oversample(rows, labels, model)
        model = self._train(self._save(rows, f"{self.modality}-train-aug.txt"))
        self.vec = {w: model[w] for w in model.words}

    def embed(self, doc):
        out = np.zeros(self.dim, dtype="float32")
        text = " ".join(doc)
        if text != "":
            for w in sorted(set(text.split(" "))):   # sorted: a fixed summation order
                if w in self.vec:
                    out = out + np.asarray(self.vec[w])
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["gaia", "hotel", "sockshop"])
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--out", required=True, help="folder to write data/<dataset>/ into")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)

    modality, opts = VARIANTS[a.variant]
    src = ROOT / "data" / a.dataset
    dst = Path(a.out) / "data" / a.dataset
    (dst / "tmp").mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "label.csv", dst / "label.csv")
    if not (dst / "raw").exists():
        os.symlink((src / "raw").resolve(), dst / "raw")
    for m in ("metric", "trace", "log"):
        if m != modality:
            shutil.copy(src / "tmp" / f"{m}.pkl", dst / "tmp" / f"{m}.pkl")

    import csv
    rows = list(csv.DictReader(open(src / "label.csv")))
    alerts = json.load(open(src / "raw" / f"{modality}s.json"))
    nodes = json.load(open(src / "raw" / "nodes.json"))
    types = ["normal"] + list(dict.fromkeys(r["anomaly_type"] for r in rows))
    all_nodes = sorted({n for v in nodes.values() for n in v})

    train = [r for r in rows if r["data_type"] == "train"]
    docs, labels = [], []
    for r in train:
        for node in nodes[str(r["index"])]:
            docs.append(doc_for(modality, alerts.get(str(r["index"]), []), node, opts))
            k = types.index(r["anomaly_type"]) if node == r["instance"] else 0
            labels.append(f"__label__{node}{k}")

    with tempfile.TemporaryDirectory() as wd:
        enc = FastTextEncoder(modality, all_nodes, types, seed=a.seed, workdir=wd)
        enc.fit(docs, labels)

    shipped = pickle.load(open(src / "tmp" / f"{modality}.pkl", "rb"))
    embs = {}
    for idx in shipped:                       # same incidents as the release
        embs[idx] = [enc.embed(doc_for(modality, alerts.get(str(idx), []), node, opts))
                     for node in nodes[str(idx)]]
    pickle.dump(embs, open(dst / "tmp" / f"{modality}.pkl", "wb"))
    print(f"{a.dataset}/{a.variant}: {len(embs)} incidents, vocab {len(enc.vec)} -> {dst}")


if __name__ == "__main__":
    main()
