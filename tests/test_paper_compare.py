"""The comparison with the TVDiag paper: transcribed numbers and the new measurements."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adadiag import config, paper_compare as pc, paper_numbers as P   # noqa: E402
from adadiag.data import Dataset                                      # noqa: E402
from adadiag.model import AdaDiag                                     # noqa: E402
import numpy as np                                                    # noqa: E402


def test_transcribed_tvdiag_f1_is_the_harmonic_mean():
    """TVDiag's Table 5 F1 is the harmonic mean of its P and R (evaluate.py's definition)
    on all four datasets -- a check on the transcription and on the metric definition.
    (Most baseline rows are NOT, which the comparison report counts and discusses.)"""
    for L, rows in P.TABLE5.items():
        *_, p, r, f1 = rows[-1]
        assert rows[-1][1] == "TVDiag"
        assert abs(2 * p * r / (p + r) - f1) < 0.002, (L, p, r, f1)


def test_paper_tables_agree_where_they_overlap():
    """Fig. 17's w/ Trace / w/ Log are Table 7's TVDiag rows; Fig. 15's w/ Trace too;
    Table 4/5/6 give the same full-model numbers on A."""
    t7 = {e[0]: e[2:] for e in P.TABLE7["A"]}
    assert P.FIG17_A["w/ Trace"] == t7["IForest"] == P.FIG15_A["w/ Trace"]
    assert P.FIG17_A["w/ Log"] == t7["Rule-based"]
    tv4, tv5 = P.TABLE4["A"][-1], P.TABLE5["A"][-1]
    assert P.TABLE6["TVDiag"]["A"] == (tv4[3], tv4[4], tv4[5], tv5[5])


def test_digitised_values_are_within_tolerance_of_printed_ones():
    """The figure readings were checked against bars the paper also prints."""
    t7 = {e[0]: e[2:] for e in P.TABLE7["A"]}
    for a, b in zip(P.FIG15_A["w/ Trace"], t7["IForest"]):
        assert abs(a - b) <= 0.005
    fig16_default = [P.FIG16_A["p"][m][1] for m in P.METRICS4]          # p = 0.2
    for a, b in zip(fig16_default, P.TABLE6["TVDiag"]["A"]):
        assert abs(a - b) <= 0.005


def _toy(n=12, N=5, n_types=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 3, N, 128, generator=g)
    A = (torch.rand(n, N, N, generator=g) > 0.5).float()
    split = np.array(["train"] * 6 + ["test"] * 6)
    return Dataset(X=X, A=A, root=torch.randint(0, N, (n,), generator=g),
                   ytype=torch.arange(n) % n_types, cnt=torch.randn(n, 3, generator=g),
                   split=split, types=["a", "b", "c"], instances=[f"i{k}" for k in range(N)],
                   name="toy")


@pytest.mark.parametrize("arm", ["v0_raw", "v3_incident"])
def test_shapley_shares_are_a_distribution(arm):
    ds = _toy()
    torch.manual_seed(0)
    model = AdaDiag(arm, 3, config.get("gaia"))
    sh = pc.modality_shapley(model, ds)
    for task in ("fti", "rcl"):
        s = torch.tensor(sh[task]["share"])
        assert s.shape == (6, 3)
        assert torch.all(s >= 0) and torch.allclose(s.sum(1), torch.ones(6), atol=1e-3)


def test_shapley_is_exact_for_an_additive_game():
    """Efficiency and exactness: for v(S) = sum of player values, phi_m = value of m."""
    w = torch.tensor([[0.2, -0.5, 1.0]])
    v = {}
    for S in [frozenset(s) for s in ([], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2])]:
        v[S] = sum((w[:, k] for k in S), torch.zeros(1))
    assert torch.allclose(pc._shapley3(v), w)


def test_per_class_and_agreement_shapes():
    ds = _toy()
    torch.manual_seed(0)
    model = AdaDiag("v0_raw", 3, config.get("gaia"))
    pk = pc.per_class_k(model, ds)
    assert set(pk) <= set(ds.types)
    for v in pk.values():
        assert 0 <= v["hr1"] <= v["hr3"] <= v["hr5"] <= 1
    ag = pc.to_agreement(model, ds)
    assert set(ag) == {"metric", "trace", "log"}
    assert all(set(ag[m]) == {"rcl", "fti"} for m in ag)


def test_sweep_points_on_the_default_are_not_run_twice():
    cfg = config.get("gaia")
    assert pc.is_default("tau=0.3", cfg) and pc.is_default("l=2", cfg)
    assert not pc.is_default("tau=0.2", cfg) and not pc.is_default("no_dw", cfg)
