"""README section 6.12: the oracle arm used by the headroom check (no data needed)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adadiag import config                                              # noqa: E402
from adadiag.model import AdaDiag                                       # noqa: E402
from adadiag.train import compute_terms, task_oriented_loss            # noqa: E402
from adadiag.weighting import PRIOR_SIMPLEX, WeightingModule            # noqa: E402
from test_fidelity import _toy                                         # noqa: E402

W_G = [[0.46, 0.0], [0.44, 0.12], [0.10, 0.88]]


def _make_oracle_weights():
    spec = importlib.util.spec_from_file_location("mow", ROOT / "scripts" / "make_oracle_weights.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_oracle_returns_its_fixed_matrix_for_every_incident():
    w = WeightingModule("v0_oracle", 96, fixed=W_G)(torch.zeros(4, 96), torch.zeros(4, 3))
    assert w.shape == (4, 3, 2)
    assert torch.allclose(w, torch.tensor(W_G).expand(4, 3, 2))


def test_oracle_with_the_prior_is_v0_simplex():
    z, c = torch.zeros(2, 96), torch.zeros(2, 3)
    a = WeightingModule("v0_oracle", 96, fixed=PRIOR_SIMPLEX)(z, c)
    b = WeightingModule("v0_simplex", 96)(z, c)
    assert torch.equal(a, b)
    # and without a matrix it falls back to uniform
    u = WeightingModule("v0_oracle", 96)(z, c)
    assert torch.allclose(u, torch.full((2, 3, 2), 1 / 3))


def test_oracle_renormalises_when_a_modality_is_removed():
    w = WeightingModule("v0_oracle", 96, fixed=W_G, active=(True, True, False))(
        torch.zeros(1, 96), torch.zeros(1, 3))[0]
    assert float(w[2].abs().sum()) == 0.0
    assert torch.allclose(w.sum(0), torch.ones(2))


def test_per_class_oracle_weights_each_incident_by_its_type():
    ds = _toy()
    W = [[[1, 0], [0, 0], [0, 1]],            # type a: metric -> RCL, log -> FTI
         [[0, 0], [1, 0], [0, 1]],            # type b: trace -> RCL
         [[0, 0], [0, 0], [1, 1]]]            # type c: log -> both
    cfg = config.get("gaia", aug_times=0, oracle_w=PRIOR_SIMPLEX, oracle_class_w=W)
    torch.manual_seed(0)
    model = AdaDiag("v0_oracle", 3, cfg)
    b = ds.train_idx[:6]
    keep = torch.ones(len(b), ds.n_nodes, dtype=torch.bool)
    batch = (ds.X[b], ds.A[b], keep, ds.root[b], ds.ytype[b], ds.cnt[b])
    with torch.no_grad():
        terms, _ = compute_terms(model, "v0_oracle", batch, cfg)
        fs = model(ds.X[b], ds.A[b], keep, ds.cnt[b])[0]
        w_ref = torch.tensor(W, dtype=torch.float32)[ds.ytype[b]]
        ref = task_oriented_loss(model, fs, {"rcl": ds.root[b], "fti": ds.ytype[b]}, w_ref, 0.3)
        glob = task_oriented_loss(model, fs, {"rcl": ds.root[b], "fti": ds.ytype[b]},
                                  torch.tensor(PRIOR_SIMPLEX).expand(len(b), 3, 2), 0.3)
    assert torch.allclose(terms["l_to"], ref)
    assert not torch.allclose(ref, glob)


def test_a_term_is_skipped_only_when_every_row_is_zero():
    """Row 0 zero but row 1 not: the term must still count (the per-class oracle varies
    across the batch, so checking row 0 alone would silently drop terms)."""
    ds = _toy()
    cfg = config.get("gaia", aug_times=0)
    torch.manual_seed(0)
    model = AdaDiag("v0_oracle", 3, cfg)
    b = ds.train_idx[:6]
    keep = torch.ones(len(b), ds.n_nodes, dtype=torch.bool)
    with torch.no_grad():
        fs = model(ds.X[b], ds.A[b], keep, ds.cnt[b])[0]
    labels = {"rcl": ds.root[b], "fti": ds.ytype[b]}
    w = torch.zeros(len(b), 3, 2)
    w[1:, 1, 0] = 1.0                          # trace -> RCL, every row but the first
    assert float(task_oriented_loss(model, fs, labels, w, 0.3)) != 0.0


def test_oracle_column_clips_and_normalises():
    col = _make_oracle_weights().column
    assert np.allclose(col([0.2, -0.1, 0.2]), [0.5, 0.0, 0.5])
    assert np.allclose(col([-0.1, 0.0, -0.3]), [1 / 3] * 3)        # nothing helps: uniform
    assert np.isclose(sum(col([0.05, 0.01, 0.3])), 1.0)
