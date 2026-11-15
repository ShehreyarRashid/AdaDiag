"""Hyperparameters, transcribed from TVDiag's config/exp_config.py.

Shared values first, then the per-dataset overrides. Nothing here is tuned for GradFusion:
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

#: `ft_num` is TVDiag's failure-type count. The model does not read it -- `GradFusion` takes
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


#: v1.22 README section 6.11. Three places where the committed runs depart from TVDiag's released
#: code, found in review; switching all three on reproduces TVDiag's training loop more
#: closely. The committed results use the defaults (all off) and stay reproducible.
FIDELITY = dict(
    self_loops=True,      # TVDiag adds a self-loop to every node with no in-edge
    aug_mode="pool",      # TVDiag samples the AUG half from a pre-built pool of copies
    restore_best=False,   # TVDiag evaluates the final model, not the best-loss epoch
    epochs=500,           # TVDiag's cap (the committed sweeps used 400)
)


#: v1.22 README section 6.19. What the TVDiag PAPER specifies and its released code does not do.
#: Each switch alone, or all together, applies the paper's version; the defaults (the code's
#: version) are untouched, so every committed result reproduces bit for bit. The paper's
#: LSTM aggregator is the seventh (`aggregator="lstm"`, section 6.18) and is left out of this
#: preset: it was tested on its own there and costs 3.6-13x the training time.
PAPER = dict(
    to_norm="paper",       # Eq. 3: each anchor's mean over its own positives
    cm_mode="paper",       # Eqs. 5-6: symmetric, with intra-modal negatives
    awl_learnable=True,    # Eq. 13: theta learnable (the code never optimises it)
    sage_alg1=True,        # Algorithm 1, lines 6-7: sigma and L2-normalise at every layer
    bidirectional=True,    # Sec. 4.2: edges are bidirectional (changes GAIA only)
    aug_mode="paper",      # Sec. 4.3: one augmented copy per incident, added to the train set
)


#: GradFusion (README): the arm and options of one GradFusion+ member, and the seed offsets of
#: the three members whose softmax outputs `ensemble.predict` averages.
GRADFUSION_ARM = "v5_align"
GRADFUSION = dict(isolate_weighting_rng=True, alert_bag=True, awl_learnable=True)
MEMBER_SEED_OFFSETS = (0, 100, 200)

#: Every option the code reads besides BASE and PER_DATASET, all off (or at TVDiag's value) by
#: default. `get` rejects any other name, so a misspelt option fails loudly instead of being
#: silently ignored. Grouped by where each is documented (v1.22 README section).
OPTIONS = frozenset({
    # model and encoders (6.18-6.25)
    "aggregator", "sage_alg1", "bidirectional", "self_loops", "no_edges", "self_res",
    "rcl_context", "fti_readout", "root_detach", "embedding_cols", "bag_drop", "bag_head",
    "alert_bag",
    # weighting module and ablations (sections 1, 6.10-6.15)
    "drop_modalities", "detach", "descriptor", "w_constraint", "oracle_w", "oracle_class_w",
    "fixed_trace", "isolate_weighting_rng", "weight_warmup", "entropy_reg", "align_ema",
    # objective (6.13, 6.19, 6.21)
    "use_to", "use_cm", "to_norm", "cm_mode", "to_scale", "dynamic_weights", "awl_learnable",
    "cl_pos_beta", "cl_neg_beta", "cl_importance", "cl_curriculum", "cl_curriculum_start",
    "fti_smooth", "fti_class_weight", "fast_supcon",
    # augmentation and training loop (6.11, 6.20-6.24)
    "aug_mode", "aug_copies", "aug_modality_drop", "restore_best", "early_stop_on",
    "weight_ema", "measure_affinity",
})


def get(dataset: str, **overrides) -> dict:
    """TVDiag's configuration for `dataset`, with `overrides` applied (None means "keep")."""
    if dataset not in PER_DATASET:
        raise KeyError(f"unknown dataset {dataset!r}; expected one of {list(PER_DATASET)}")
    unknown = set(overrides) - set(BASE) - set(PER_DATASET[dataset]) - OPTIONS
    if unknown:
        raise KeyError(f"unknown config option(s) {sorted(unknown)}; see config.OPTIONS")
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
