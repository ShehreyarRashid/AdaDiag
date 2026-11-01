"""Hyperparameters, transcribed from TVDiag's config/exp_config.py.

Shared values first, then the per-dataset overrides. Nothing here is tuned for AdaDiag:
the whole point is that every arm runs under TVDiag's own settings.
"""
from __future__ import annotations

BASE = dict(
    alert_embedding_dim=128,
    graph_hidden_dim=64,
    graph_out=32,
    graph_layers=2,
    linear_hidden=[64],
    lr=1e-3,
    weight_decay=1e-4,
    temperature=0.3,
    contrastive_loss_scale=0.1,     # sigma
    epochs=500,
    aug_times=10,
    aggregator="mean",
)

PER_DATASET = {
    "gaia":     dict(feat_drop=0.0, batch_size=512, patience=10, ft_num=5, aug_percent=0.2),
    "hotel":    dict(feat_drop=0.3, batch_size=128, patience=10, ft_num=5, aug_percent=0.2),
    "sockshop": dict(feat_drop=0.0, batch_size=128, patience=10, ft_num=7, aug_percent=0.4),
    # AIOps-22 is configured in TVDiag but its data is NOT shipped -- see README, "Data".
    "aiops22":  dict(feat_drop=0.1, batch_size=128, patience=20, ft_num=9, aug_percent=0.2),
}


def get(dataset: str, **overrides) -> dict:
    if dataset not in PER_DATASET:
        raise KeyError(f"unknown dataset {dataset!r}; expected one of {list(PER_DATASET)}")
    cfg = dict(BASE)
    cfg.update(PER_DATASET[dataset])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
