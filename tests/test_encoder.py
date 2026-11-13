"""README section 6.23: the encoder diagnostics (no data needed)."""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src")); sys.path.insert(0, str(HERE / "scripts"))
from adadiag import config                                         # noqa: E402
from adadiag.model import AdaDiag                                  # noqa: E402
from encoder_diagnose import expected_hit, trace_rule              # noqa: E402


def test_no_edges_ignores_the_graph_and_is_off_by_default():
    torch.manual_seed(0)
    X = torch.randn(2, 3, 6, 128)
    A = (torch.rand(2, 6, 6) > 0.4).float()
    keep = torch.ones(2, 6, dtype=torch.bool)
    c = torch.zeros(2, 3)
    torch.manual_seed(1); on = AdaDiag("v0_raw", 5, config.get("hotel", no_edges=True))
    torch.manual_seed(1); off = AdaDiag("v0_raw", 5, config.get("hotel"))
    on.eval(); off.eval()
    with torch.no_grad():
        Z = torch.zeros_like(A)
        assert torch.equal(on(X, A, keep, c)[2], on(X, Z, keep, c)[2])     # any graph = none
        assert torch.equal(on(X, A, keep, c)[2], off(X, Z, keep, c)[2])
        assert not torch.equal(off(X, A, keep, c)[2], off(X, Z, keep, c)[2])


def test_trace_rule_ranks_the_deepest_anomalous_callee_first():
    names = ["a", "b", "c", "d"]
    calls = [["a", "b", "op", "PD"], ["b", "c", "op", "PD"], ["c", "c", "db", "PD"],
             ["-1", "a", "op", "PD"]]
    s = trace_rule(calls, names)
    assert int(np.argmax(s)) == 2                     # c: receives, calls no other instance
    assert s[3] == 0                                  # d: no anomalous call at all


def test_expected_hit_breaks_ties_uniformly():
    s = np.array([5.0, 5.0, 1.0, 5.0])
    assert expected_hit(s, 0, 1) == 1 / 3
    assert expected_hit(s, 2, 1) == 0.0
    assert expected_hit(s, 2, 4) == 1.0
    assert expected_hit(s, 1, 3) == 1.0
