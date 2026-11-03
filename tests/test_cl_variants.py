"""README section 6.13: the five weightings of the contrastive term (no data needed)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adadiag import config                                              # noqa: E402
from adadiag.losses import supcon_per_sample, supcon_weighted          # noqa: E402
from adadiag.model import AdaDiag                                       # noqa: E402
from adadiag.train import compute_terms, total_loss                    # noqa: E402
from adadiag.losses import AutomaticWeightedLoss                       # noqa: E402
from test_fidelity import _toy                                         # noqa: E402


def _xy(seed=0, n=24, d=8, k=4):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randint(0, k, (n,), generator=g)


def test_defaults_are_tvdiag_bit_for_bit():
    x, y = _xy()
    a, na = supcon_per_sample(x, y, 0.3)
    b, nb = supcon_weighted(x, y, 0.3)
    assert torch.equal(a, b) and na == nb


def test_the_rewritten_formula_is_tvdiags():
    """The weighted path, with weights that are all one, gives TVDiag's term."""
    x, y = _xy(1)
    a, na = supcon_per_sample(x, y, 0.3)
    b, nb = supcon_weighted(x, y, 0.3, pos_beta=1e-12, neg_beta=1e-12)
    assert na == nb and torch.allclose(a, b, atol=1e-5)


def test_hard_pair_weighting_raises_the_loss():
    """Weights of mean one, larger where the pair is harder: by Chebyshev's sum inequality
    the weighted term can only be larger (equal when every pair is equally hard)."""
    x, y = _xy(2)
    base = supcon_weighted(x, y, 0.3)[0].sum()
    for kw in (dict(pos_beta=1.0), dict(neg_beta=1.0)):
        assert supcon_weighted(x, y, 0.3, **kw)[0].sum() > base


def test_curriculum_keeps_the_easiest_anchors():
    x, y = _xy(3)
    full, _ = supcon_per_sample(x, y, 0.3)
    half, _ = supcon_weighted(x, y, 0.3, keep_frac=0.5)
    npos = torch.stack([((y == y[i]).sum() - 1) for i in range(len(y))]).float()
    kept = half > 0
    mean_loss = full / npos.clamp_min(1)
    valid = npos > 0
    assert kept.sum() < valid.sum()
    assert mean_loss[kept].max() <= mean_loss[valid & ~kept].min()
    # a mean over the included positive pairs, at the same scale as the full term
    assert torch.isclose(half.sum() / npos[valid].sum(), full[kept].sum() / npos[kept].sum())


def _batch(ds, n=6):
    b = ds.train_idx[:n]
    keep = torch.ones(len(b), ds.n_nodes, dtype=torch.bool)
    return (ds.X[b], ds.A[b], keep, ds.root[b], ds.ytype[b], ds.cnt[b])


def test_importance_weights_anchors_by_their_diagnosis_loss():
    ds = _toy()
    batch = _batch(ds)
    cfg0 = config.get("gaia", aug_times=0)
    cfg1 = config.get("gaia", aug_times=0, cl_importance=1.0)
    torch.manual_seed(0); m = AdaDiag("v0_raw", 3, cfg0).eval()
    with torch.no_grad():
        t0, _ = compute_terms(m, "v0_raw", batch, cfg0)
        t1, _ = compute_terms(m, "v0_raw", batch, cfg1)
    assert torch.equal(t0["l_rcl"], t1["l_rcl"]) and torch.equal(t0["l_fti"], t1["l_fti"])
    assert not torch.allclose(t0["l_to"], t1["l_to"])


def test_align_arm_sets_a_simplex_matrix_from_the_diagnosis_losses():
    ds = _toy()
    cfg = config.get("gaia", aug_times=0)
    torch.manual_seed(0)
    m = AdaDiag("v5_align", 3, cfg).train()
    assert torch.allclose(m.weighting.align_w, torch.full((3, 2), 1 / 3))
    terms, w = compute_terms(m, "v5_align", _batch(ds), cfg)
    W = m.weighting.align_w
    assert torch.allclose(W.sum(0), torch.ones(2)) and (W >= 0).all()
    assert torch.allclose(w[0], W) and float(m.weighting.align_ema.abs().sum()) > 0
    total_loss(AutomaticWeightedLoss(4), terms, cfg).backward()   # the graph survives
    assert m.weighting.n_parameters() == 0


def test_all_switches_off_leave_the_objective_unchanged():
    ds = _toy()
    batch = _batch(ds)
    base = config.get("gaia", aug_times=0)
    off = config.get("gaia", aug_times=0, cl_pos_beta=0.0, cl_neg_beta=0.0, cl_importance=0.0,
                     cl_curriculum=0)
    torch.manual_seed(0); m = AdaDiag("v0_raw", 3, base).eval()
    with torch.no_grad():
        a, _ = compute_terms(m, "v0_raw", batch, base, epoch=3)
        b, _ = compute_terms(m, "v0_raw", batch, off, epoch=3)
    assert all(torch.equal(a[k], b[k]) for k in a)
