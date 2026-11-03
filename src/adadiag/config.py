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

#: `ft_num` is TVDiag's failure-type count. The model does not read it -- `AdaDiag` takes
#: `n_types` from `len(ds.types)`, which is derived from the data. It is kept here so the
#: transcription stays faithful to `config/exp_config.py`, and `check_against` below asserts
#: the two agree, which catches a dataset whose label file has changed.
PER_DATASET = {
    "gaia":     dict(feat_drop=0.0, batch_size=512, patience=10, ft_num=5, aug_percent=0.2),
    "hotel":    dict(feat_drop=0.3, batch_size=128, patience=10, ft_num=5, aug_percent=0.2),
    "sockshop": dict(feat_drop=0.0, batch_size=128, patience=10, ft_num=7, aug_percent=0.4),
    # AIOps-22 is configured in TVDiag but its data is NOT shipped -- see README, "Dataset".
    "aiops22":  dict(feat_drop=0.1, batch_size=128, patience=20, ft_num=9, aug_percent=0.2),
}


#: README section 6.11. Three places where the committed runs depart from TVDiag's released
#: code, found in review; switching all three on reproduces TVDiag's training loop more
#: closely. The committed results use the defaults (all off) and stay reproducible.
FIDELITY = dict(
    self_loops=True,      # TVDiag adds a self-loop to every node with no in-edge
    aug_mode="pool",      # TVDiag samples the AUG half from a pre-built pool of copies
    restore_best=False,   # TVDiag evaluates the final model, not the best-loss epoch
    epochs=500,           # TVDiag's cap (the committed sweeps used 400)
)


def get(dataset: str, **overrides) -> dict:
    if dataset not in PER_DATASET:
        raise KeyError(f"unknown dataset {dataset!r}; expected one of {list(PER_DATASET)}")
    cfg = dict(BASE)
    cfg.update(PER_DATASET[dataset])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def check_against(cfg: dict, ds) -> None:
    """Assert the transcribed config agrees with the loaded data."""
    if cfg["ft_num"] != len(ds.types):
        raise ValueError(
            f"{ds.name}: config ft_num={cfg['ft_num']} but the data has {len(ds.types)} "
            f"failure types ({ds.types}). One of the two is out of date."
        )
