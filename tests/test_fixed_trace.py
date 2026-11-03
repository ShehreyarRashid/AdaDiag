"""README section 6.14: AdaDiag with trace's weight held fixed (no data needed)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adadiag.weighting import MODALITIES, WeightingModule              # noqa: E402

TI = MODALITIES.index("trace")


@pytest.mark.parametrize("arm", ["v2_global", "v3_incident"])
@pytest.mark.parametrize("t", [[1 / 3, 1 / 3], [0.5, 0.0]])
def test_trace_is_held_and_columns_stay_on_the_simplex(arm, t):
    torch.manual_seed(0)
    m = WeightingModule(arm, 96, fixed_trace=t)
    w = m(torch.randn(5, 96), torch.randn(5, 3))
    assert torch.allclose(w[:, TI, :], torch.tensor(t).expand(5, 2))
    assert torch.allclose(w.sum(1), torch.ones(5, 2)) and (w >= 0).all()


def test_only_metric_and_log_are_learned():
    torch.manual_seed(0)
    m = WeightingModule("v3_incident", 96, fixed_trace=[1 / 3, 1 / 3])
    w = m(torch.randn(5, 96), torch.randn(5, 3))
    g_trace = torch.autograd.grad(w[:, TI].sum(), list(m.parameters()), allow_unused=True,
                                  retain_graph=True)
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in g_trace)
    g_ml = torch.autograd.grad(w[:, 0, 0].sum(), list(m.parameters()), allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0 for g in g_ml)
    assert m.n_parameters() == WeightingModule("v3_incident", 96).n_parameters()


def test_fixed_trace_is_refused_where_it_means_nothing():
    with pytest.raises(ValueError):
        WeightingModule("v0_raw", 96, fixed_trace=[0.5, 0.0])
    with pytest.raises(ValueError):
        WeightingModule("v3_incident", 96, fixed_trace=[1.5, 0.0])
