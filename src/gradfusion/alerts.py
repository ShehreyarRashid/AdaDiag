"""v1.22 README section 6.24: alert-type counts per instance, next to TVDiag's event embeddings.

TVDiag's node features are DiagFusion's: each instance's events are embedded with fastText
and AVERAGED (DiagFusion, IEEE TSC 2023, Eq. 2). An average keeps the mix of events but loses
how many there were and, once several alert types share an instance, which ones. The raw
alerts the embeddings were made from ship with TVDiag (data/<ds>/raw/), and they are short
structured records, so the counts can be read back exactly:

    metric  (instance, [host,] metric name, up|down)   -> one type per (name, direction)
    trace   (caller, callee, operation, PD|500|400|ERROR)
            -> the callee gets ("in", status); the caller, if a different instance,
               gets ("out", status); a span from an instance to itself gets ("self", status)
    log     (instance, template id)                     -> one type per template number

The vocabulary is built from the TRAINING incidents only; alert types never seen in training
are ignored. Each count c becomes log1p(c) / s, where s is that type's largest log1p count
over training instances, so values lie in [0, 1] for training data and 0 still means "no
alert" (AUG zeroes a dropped instance's features, which must read as "nothing here"). The
three planes are zero-padded to one width and appended to X, so each modality's encoder sees
its own types: X becomes (M, 3, N, 128 + V).
"""
from __future__ import annotations

import dataclasses
import re

import numpy as np
import torch

from .data import MODALITIES, read_json

_FILES = {"metric": "metrics", "trace": "traces", "log": "logs"}
_TEMPLATE_SUFFIX = re.compile(r"^.*-(\d+)$")


def _template(name: str, log_id: str) -> str:
    # Sock Shop and Hotel name templates "<instance>-<k>"; GAIA gives the number alone
    if name == "gaia":
        return log_id
    m = _TEMPLATE_SUFFIX.match(log_id)
    return m.group(1) if m else log_id


def _types(name: str, modality: str, a: list) -> list[tuple[str, tuple]]:
    """(instance, alert type) pairs one raw alert record contributes."""
    if modality == "metric":
        return [(a[0], (a[-2], a[-1]))]
    if modality == "log":
        return [(a[0], (_template(name, a[1]),))]
    src, dst, status = a[0], a[1], a[3]
    if src == dst:
        return [(dst, ("self", status))]
    return [(dst, ("in", status)), (src, ("out", status))]


def _raw_alerts(ds) -> dict[str, dict]:
    return {m: read_json(ds.root_dir / "raw" / f"{_FILES[m]}.json") for m in MODALITIES}


def vocabulary(ds, raw: dict | None = None) -> dict[str, list[tuple]]:
    """Sorted alert types per modality, from the TRAINING incidents only."""
    raw = _raw_alerts(ds) if raw is None else raw
    train = {ds.keys[i] for i in ds.train_idx}
    return {m: sorted({t for k in train for a in raw[m].get(k, []) for _, t in _types(ds.name, m, a)})
            for m in MODALITIES}


def alert_bags(ds) -> tuple[torch.Tensor, dict]:
    """(M, 3, N, V) scaled alert-type counts, and the vocabulary per modality."""
    if ds.keys is None or ds.root_dir is None:
        raise ValueError("alert_bags needs a Dataset from data.load (incident keys and raw alerts)")
    raw = _raw_alerts(ds)                  # read once, shared with vocabulary()
    vocab = vocabulary(ds, raw)
    pos = {m: {t: j for j, t in enumerate(v)} for m, v in vocab.items()}
    inst = {n: j for j, n in enumerate(ds.instances)}
    V = max(len(v) for v in vocab.values())
    B = np.zeros((len(ds.keys), len(MODALITIES), ds.n_nodes, V), dtype=np.float32)
    for q, k in enumerate(ds.keys):
        for mi, m in enumerate(MODALITIES):
            for a in raw[m].get(k, []):
                for n, t in _types(ds.name, m, a):
                    if n in inst and t in pos[m]:
                        B[q, mi, inst[n], pos[m][t]] += 1.0
    B = np.log1p(B)
    s = B[ds.train_idx].max(axis=(0, 2), keepdims=True)          # (1, 3, 1, V)
    B = B / np.where(s > 0, s, 1.0)
    return torch.tensor(B), vocab


def with_alert_bag(ds):
    """A copy of `ds` whose X carries the alert-type counts after the 128-d embeddings."""
    if ds._with_bag is not None:
        return ds._with_bag
    bag, vocab = alert_bags(ds)
    out = dataclasses.replace(ds, X=torch.cat([ds.X, bag], dim=-1), bag_vocab=vocab)
    out._with_bag = out                    # applying it twice is a no-op
    ds._with_bag = out
    return out
