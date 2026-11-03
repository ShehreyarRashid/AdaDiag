"""README section 6.11: the softened prior (v0_soft) and the TVDiag-fidelity switches."""
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adadiag import config                                              # noqa: E402
from adadiag.data import Dataset                                        # noqa: E402
from adadiag.model import AdaDiag                                       # noqa: E402
from adadiag.train import build_aug_pool, compute_terms, make_batch     # noqa: E402
from adadiag.weighting import PRIOR_SOFT, WeightingModule               # noqa: E402


def _toy(n=12, N=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(n, N, N, generator=g) > 0.6).float()
    A[:, :, 0] = 0
    A[:, 0, :] = 0                       # node 0: no in-edges and no out-edges
    return Dataset(X=torch.randn(n, 3, N, 128, generator=g), A=A,
                   root=torch.randint(1, N, (n,), generator=g), ytype=torch.arange(n) % 3,
                   cnt=torch.randn(n, 3, generator=g),
                   split=np.array(["train"] * 8 + ["test"] * 4), types=["a", "b", "c"],
                   instances=[f"i{k}" for k in range(N)], name="toy")


def test_soft_prior_is_the_proposed_matrix():
    w = WeightingModule("v0_soft", 96)(torch.zeros(2, 96), torch.zeros(2, 3))
    assert torch.allclose(w[0], torch.tensor(PRIOR_SOFT))
    assert torch.allclose(w[0].sum(0), torch.tensor([1.9, 1.1]))    # not on the simplex


def test_self_loops_go_only_where_there_is_no_in_edge():
    ds = _toy()
    cfg_on = config.get("gaia", self_loops=True)
    cfg_off = config.get("gaia")
    torch.manual_seed(0); m_on = AdaDiag("v0_raw", 3, cfg_on)
    torch.manual_seed(0); m_off = AdaDiag("v0_raw", 3, cfg_off)
    keep = torch.ones(4, ds.n_nodes, dtype=torch.bool)
    with torch.no_grad():
        f_on = m_on(ds.X[:4], ds.A[:4], keep, ds.cnt[:4])[0]["metric"]
        f_off = m_off(ds.X[:4], ds.A[:4], keep, ds.cnt[:4])[0]["metric"]
        A2 = ds.A[:4].clone()
        A2[:, 0, 0] = 1.0                 # what TVDiag does: a self-loop on node 0 only
        A2 = A2 + torch.diag_embed((A2.sum(-1) == 0).float())
        f_ref = m_off(ds.X[:4], A2, keep, ds.cnt[:4])[0]["metric"]
    assert torch.allclose(f_on, f_ref, atol=1e-6) and not torch.allclose(f_on, f_off)


def test_pool_batches_draw_other_incidents_and_keep_every_root():
    ds = _toy()
    cfg = config.get("gaia", aug_mode="pool", aug_times=3)
    rng = random.Random(0)
    pool = build_aug_pool(ds, cfg, rng)
    assert len(pool) == 3 * len(ds.train_idx)
    b = ds.train_idx[:4]
    X, A, keep, root, yt, cnt = make_batch(ds, b, cfg, rng, pool)
    assert X.shape[0] == 8 and keep[:4].all()
    assert keep[torch.arange(8), root].all()                        # the culprit is never dropped
    assert (~keep[4:]).any()                                        # copies are node-dropped


def test_paired_mode_is_the_default():
    ds = _toy()
    a = make_batch(ds, ds.train_idx[:4], config.get("gaia"), random.Random(1))
    b = make_batch(ds, ds.train_idx[:4], config.get("gaia", aug_mode="paired"), random.Random(1))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_soft_prior_loss_is_the_weighted_sum():
    ds = _toy()
    cfg = config.get("gaia")
    torch.manual_seed(0)
    m = AdaDiag("v0_soft", 3, cfg)
    batch = make_batch(ds, ds.train_idx[:8], cfg, random.Random(0))
    terms, w = compute_terms(m, "v0_soft", batch, cfg)
    from adadiag.losses import supcon_per_sample
    fs = m(*batch[:3], batch[5])[0]
    lab = {"rcl": batch[3], "fti": batch[4]}
    ref = sum(PRIOR_SOFT[i][j] * (lambda ps: ps[0].sum() / ps[1])(
        supcon_per_sample(fs[mod], lab[t], cfg["temperature"]))
        for i, mod in enumerate(("metric", "trace", "log")) for j, t in enumerate(("rcl", "fti")))
    assert torch.allclose(terms["l_to"], ref, atol=1e-5)
