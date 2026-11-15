"""Methodology step 1 — data loading.

Builds dense tensors from TVDiag's released artefacts. Nothing here is re-derived: the
128-d event embeddings, the call graphs and the official splits all ship with the TVDiag
repository, and this repository bundles an unchanged copy in `data/` (see data/README.md),
so this module only reads and arranges them.

Per incident we produce
    X    (3, N, 128)  event embeddings, one plane per source (metric, trace, log)
    A    (N, N)       call-graph adjacency, A[d, s] = 1 when s sends to d
    root ()           index of the ground-truth culprit instance
    y    ()           index of the ground-truth failure type
    c    (3,)         alert counts per source -- the descriptor's non-embedding part

The per-dataset constants this module asserts against are in `EXPECTED` below.
"""
from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

MODALITIES = ("metric", "trace", "log")
_ALERT_FILE = {"metric": "metrics", "trace": "traces", "log": "logs"}

#: Expected shape of each dataset, asserted on load so a silent data change is caught early.
EXPECTED = {
    "gaia":     dict(incidents=1099, instances=10, types=5, train=160, test=939),
    "hotel":    dict(incidents=200,  instances=18, types=5, train=122, test=78),
    "sockshop": dict(incidents=489,  instances=21, types=7, train=248, test=241),
}


@dataclass
class Dataset:
    """Everything the model needs, as dense tensors."""
    X: torch.Tensor          # (M, 3, N, 128)
    A: torch.Tensor          # (M, N, N)
    root: torch.Tensor       # (M,)
    ytype: torch.Tensor      # (M,)
    cnt: torch.Tensor        # (M, 3)  standardised log1p alert counts
    split: np.ndarray        # (M,) of "train" / "test"
    types: list[str]
    instances: list[str]
    name: str
    keys: list[str] | None = None    # incident ids in label.csv, row order (v1.22 README section 6.24)
    root_dir: Path | None = None     # data/<name>/, for reading the raw alerts
    bag_vocab: dict | None = None    # alert-type vocabulary, set by alerts.with_alert_bag
    # alerts.with_alert_bag's cached result; not an init field, so dataclasses.replace drops it
    _with_bag: Dataset | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def n_nodes(self) -> int:
        return self.X.shape[2]

    @property
    def train_idx(self) -> np.ndarray:
        return np.where(self.split == "train")[0]

    @property
    def test_idx(self) -> np.ndarray:
        return np.where(self.split == "test")[0]

    def __repr__(self) -> str:
        return (f"<{self.name}: {len(self.split)} incidents, {self.n_nodes} instances, "
                f"{len(self.types)} types, {len(self.train_idx)}/{len(self.test_idx)} train/test>")


def read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_pickle(path: Path):
    # The pickles are TVDiag's released artefacts, checked against data/SHA256SUMS by
    # tests/test_data_integrity.py. Never point this loader at pickles you do not trust:
    # unpickling can run arbitrary code.
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _check(cond: bool, msg: str) -> None:
    """A data check that, unlike `assert`, still runs under `python -O`."""
    if not cond:
        raise ValueError(msg)


#: This repository's root, which ships `data/` (src/gradfusion/data.py -> parents[2]).
REPO_ROOT = Path(__file__).resolve().parents[2]


def load(name: str, repo: str | Path | None = None, strict: bool = True) -> Dataset:
    """Load one dataset.

    Args:
        name: one of "gaia", "hotel", "sockshop".
        repo: a folder containing `data/<name>/`. Defaults to this repository, which bundles
            all three; a clone of github.com/WHU-AISE/TVDiag works too.
        strict: assert the dataset matches EXPECTED[name].
    """
    root_dir = Path(repo if repo is not None else REPO_ROOT) / "data" / name
    if not root_dir.exists():
        raise FileNotFoundError(
            f"{root_dir} not found. This repository ships data/gaia, data/hotel and "
            f"data/sockshop -- if they are missing, re-download the repository, or pass "
            f"--repo pointing at a clone of https://github.com/WHU-AISE/TVDiag. AIOps-22 is "
            f"not distributed by TVDiag -- see README section 'Dataset'."
        )

    feats = {m: _read_pickle(root_dir / "tmp" / f"{m}.pkl") for m in MODALITIES}
    nodes = read_json(root_dir / "raw" / "nodes.json")
    edges = read_json(root_dir / "raw" / "edges.json")
    alerts = {m: read_json(root_dir / "raw" / f"{_ALERT_FILE[m]}.json") for m in MODALITIES}
    with open(root_dir / "label.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    types = sorted({r["anomaly_type"] for r in rows})
    type_index = {t: j for j, t in enumerate(types)}

    # Every incident must list the same instances in the same ORDER: `root` is an index into
    # that list and the RCL head has one output per position, so a per-incident reordering
    # would silently make index k mean a different service in different incidents. True for
    # all three released datasets (checked, 1 distinct ordering each) -- asserted, not assumed.
    orderings = {tuple(v) for v in nodes.values()}
    if len(orderings) != 1:
        raise ValueError(
            f"{name}: nodes.json has {len(orderings)} distinct instance orderings. "
            f"Root-cause indices are not comparable across incidents; this loader assumes one."
        )
    instance_names = list(next(iter(orderings)))
    n_nodes = len(instance_names)

    X, A, root, ytype, cnt, split, keys = [], [], [], [], [], [], []
    skipped = 0
    for r in rows:
        i = r["index"]
        if i not in feats["log"] or i not in edges:
            skipped += 1
            continue
        names = nodes[i]
        # A few Sock Shop rows name an instance that is absent from that incident's node list.
        if r["instance"] not in names:
            skipped += 1
            continue
        X.append(np.stack([np.asarray(feats[m][i], dtype=np.float32) for m in MODALITIES]))
        a = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        sd = np.asarray(edges[i], dtype=np.int64).reshape(-1, 2)     # rows (source, destination)
        a[sd[:, 1], sd[:, 0]] = 1.0
        A.append(a)
        root.append(names.index(r["instance"]))
        ytype.append(type_index[r["anomaly_type"]])
        cnt.append([len(alerts[m].get(i, [])) for m in MODALITIES])
        split.append(r["data_type"])
        keys.append(i)

    _check(len(X) > 0, f"{name}: no usable incidents in {root_dir} ({skipped} skipped)")
    split = np.asarray(split)
    counts = torch.tensor(np.asarray(cnt, dtype=np.float32))
    # log1p then standardise on the TRAIN split only: raw counts run 0-70 and would otherwise
    # dominate the 32-d embeddings they are concatenated with.
    counts = torch.log1p(counts)
    tr = split == "train"
    counts = (counts - counts[tr].mean(0)) / (counts[tr].std(0) + 1e-6)

    ds = Dataset(
        X=torch.tensor(np.stack(X)), A=torch.tensor(np.stack(A)),
        root=torch.tensor(root), ytype=torch.tensor(ytype), cnt=counts,
        split=split, types=types, instances=instance_names, name=name,
        keys=keys, root_dir=root_dir,
    )
    if strict and name in EXPECTED:
        e = EXPECTED[name]
        _check(ds.n_nodes == e["instances"], f"{name}: {ds.n_nodes} instances, expected {e['instances']}")
        _check(len(ds.types) == e["types"], f"{name}: {len(ds.types)} types, expected {e['types']}")
        _check(len(ds.train_idx) == e["train"] and len(ds.test_idx) == e["test"],
               f"{name}: {len(ds.train_idx)}/{len(ds.test_idx)} split, "
               f"expected {e['train']}/{e['test']}")
    return ds
