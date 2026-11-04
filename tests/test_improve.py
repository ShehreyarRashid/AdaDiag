"""README section 6.21: switches aimed at improving RCL and FTI (no data needed)."""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adadiag import config                                         # noqa: E402
from adadiag.augment import drop_modality                          # noqa: E402
from adadiag.model import AdaDiag                                  # noqa: E402
from adadiag.train import fti_class_weights, make_batch            # noqa: E402


def _ds(M=12, N=7, T=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.tensor([0] * 6 + [1] * 3 + [2] * 2 + [3] * 1)
    return SimpleNamespace(
        X=torch.rand(M, 3, N, 128, generator=g) + 0.1, A=(torch.rand(M, N, N, generator=g) > 0.5).float(),
        root=torch.randint(0, N, (M,), generator=g), ytype=y, cnt=torch.zeros(M, 3),
        n_nodes=N, types=list(range(T)), train_idx=np.arange(M), split=np.array(["train"] * M))


def test_class_weights_have_mean_one_and_order_by_rarity():
    ds = _ds()
    for mode in ("sqrt", "balanced"):
        w = fti_class_weights(ds, mode)
        assert abs(float(w[ds.ytype].mean()) - 1.0) < 1e-5
        assert bool((w[1:] > w[:-1]).all())          # rarer type, larger weight
    assert fti_class_weights(ds, None) is None
    with pytest.raises(ValueError):
        fti_class_weights(ds, "x")


def test_aug_copies_and_modality_drop_touch_only_the_augmented_rows():
    ds = _ds()
    b = np.arange(4)
    cfg = config.get("hotel", aug_copies=2, aug_percent=0.0)
    X, A, keep, root, yt, cnt = make_batch(ds, b, cfg, random.Random(0))
    assert X.shape[0] == 12 and torch.equal(root, ds.root[b].repeat(3))
    cfg = config.get("hotel", aug_modality_drop=0.3, aug_percent=0.0)
    X1, *_ = make_batch(ds, b, cfg, random.Random(0))
    assert torch.equal(X1[:4], ds.X[b])               # raw rows untouched
    k = int(ds.n_nodes * 0.3)
    for i in range(4, 8):                              # each copy loses k (node, modality) planes
        zeroed = int((X1[i].abs().sum(-1) == 0).sum())
        assert 1 <= zeroed <= k


def test_default_batch_is_unchanged():
    ds = _ds()
    b = np.arange(5)
    cfg = config.get("hotel")
    a = make_batch(ds, b, cfg, random.Random(3))
    c = make_batch(ds, b, dict(cfg, aug_copies=1, aug_modality_drop=0.0), random.Random(3))
    assert all(torch.equal(x, y) for x, y in zip(a, c))


@pytest.mark.parametrize("ctx", ["none", "global", "neighbour", "both"])
@pytest.mark.parametrize("ro", ["max", "root_attn"])
def test_heads_with_context_and_root_readout_train(ctx, ro):
    torch.manual_seed(0)
    m = AdaDiag("v0_raw", 5, config.get("hotel", rcl_context=ctx, fti_readout=ro))
    X, A = torch.randn(3, 3, 6, 128), (torch.rand(3, 6, 6) > 0.5).float()
    keep = torch.ones(3, 6, dtype=torch.bool)
    keep[0, 2] = False
    fs, f, nl, tl = m(X, A, keep, torch.zeros(3, 3))
    assert nl.shape == (3, 6) and tl.shape == (3, 5)
    (nl.masked_fill(~keep, 0).sum() + tl.sum()).backward()
    assert m.locator.net[0].weight.grad is not None


def test_defaults_build_the_released_heads():
    torch.manual_seed(0); a = AdaDiag("v0_raw", 5, config.get("hotel"))
    torch.manual_seed(0); b = AdaDiag("v0_raw", 5, config.get("hotel", rcl_context="none", fti_readout="max"))
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    with pytest.raises(ValueError):
        AdaDiag("v0_raw", 5, config.get("hotel", rcl_context="x"))


def test_drop_modality_counts():
    X = torch.ones(4, 3, 10, 2)
    Y = drop_modality(X, 2, 0.3, random.Random(0))
    assert torch.equal(Y[:2], X[:2]) and torch.equal(X, torch.ones(4, 3, 10, 2))   # copy
    for i in (2, 3):
        assert int((Y[i].sum(-1) == 0).sum()) == 3
