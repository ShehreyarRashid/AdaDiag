"""The two structural claims behind README section 6.6, as tests.

Both are self-contained: they build a model and synthetic batch directly, so they run
without data/.

  1. The weighting module receives EXACTLY zero gradient from both diagnosis losses. If a
     future change puts `w` on the prediction path, this test fails -- which is the point:
     that change would invalidate the "no inference cost" claim in the README.
  2. `l_to` is linear in `w`. A linear function on a simplex is minimised at a vertex, which
     is why the learned columns collapse to one-hot instead of settling on a graded
     allocation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adadiag import config                                        # noqa: E402
from adadiag.model import AdaDiag                                 # noqa: E402
from adadiag.train import task_oriented_loss                      # noqa: E402
from adadiag.weighting import MODALITIES, TASKS                   # noqa: E402

B, N, N_TYPES = 12, 8, 5


def _batch(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, len(MODALITIES), N, cfg["alert_embedding_dim"], generator=g)
    A = (torch.rand(B, N, N, generator=g) < 0.3).float()
    keep = torch.ones(B, N, dtype=torch.bool)
    cnt = torch.randn(B, len(MODALITIES), generator=g)
    root = torch.randint(0, N, (B,), generator=g)
    ytype = torch.randint(0, N_TYPES, (B,), generator=g)
    return X, A, keep, cnt, {"rcl": root, "fti": ytype}


@pytest.mark.parametrize("arm", ["v2_global", "v3_incident"])
def test_diagnosis_losses_give_the_weighting_module_no_gradient(arm):
    """Claim 1: d l_rcl / d theta_w == d l_fti / d theta_w == 0, exactly."""
    cfg = config.get("hotel")
    torch.manual_seed(0)
    model = AdaDiag(arm, N_TYPES, cfg)
    X, A, keep, cnt, labels = _batch(cfg)

    fs, f, node_logits, type_logits = model(X, A, keep, cnt)
    w = model.weights(f, cnt)
    params = list(model.weighting.parameters())
    assert params, f"{arm} should have learned weighting parameters"

    for name, term in (("l_rcl", F.cross_entropy(node_logits, labels["rcl"])),
                       ("l_fti", F.cross_entropy(type_logits, labels["fti"]))):
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        total = sum(0.0 if g is None else float(g.abs().sum()) for g in grads)
        assert total == 0.0, f"{arm}: {name} leaked gradient into the weighting module ({total})"

    # ...while l_to does supply one, so the module is not simply disconnected.
    l_to = task_oriented_loss(model, fs, labels, w, cfg["temperature"])
    grads = torch.autograd.grad(l_to, params, retain_graph=True, allow_unused=True)
    total = sum(0.0 if g is None else float(g.abs().sum()) for g in grads)
    assert total > 0.0, f"{arm}: l_to supplied no gradient either -- module is disconnected"


def test_task_oriented_loss_is_linear_in_w():
    """Claim 2: l_to(a*w1 + (1-a)*w2) == a*l_to(w1) + (1-a)*l_to(w2).

    Linear on the simplex => the minimiser is a vertex => one-hot columns.
    """
    cfg = config.get("hotel")
    torch.manual_seed(0)
    model = AdaDiag("v1_uniform", N_TYPES, cfg)       # fixed arm: w is an input, not learned
    X, A, keep, cnt, labels = _batch(cfg)
    with torch.no_grad():
        fs, _, _, _ = model(X, A, keep, cnt)

    nm, nt = len(MODALITIES), len(TASKS)
    g = torch.Generator().manual_seed(1)
    w1 = torch.rand(B, nm, nt, generator=g).softmax(1)
    w2 = torch.rand(B, nm, nt, generator=g).softmax(1)

    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        mix = alpha * w1 + (1 - alpha) * w2
        assert torch.allclose(mix.sum(1), torch.ones(B, nt), atol=1e-5), "mix left the simplex"
        with torch.no_grad():
            lhs = float(task_oriented_loss(model, fs, labels, mix, cfg["temperature"]))
            rhs = (alpha * float(task_oriented_loss(model, fs, labels, w1, cfg["temperature"]))
                   + (1 - alpha) * float(task_oriented_loss(model, fs, labels, w2, cfg["temperature"])))
        assert lhs == pytest.approx(rhs, rel=1e-5), (
            f"l_to is not linear in w at alpha={alpha}: {lhs} vs {rhs}")


def test_vertex_beats_interior_for_some_vertex():
    """The concrete consequence: some one-hot w has lower l_to than the uniform w."""
    cfg = config.get("hotel")
    torch.manual_seed(0)
    model = AdaDiag("v1_uniform", N_TYPES, cfg)
    X, A, keep, cnt, labels = _batch(cfg)
    with torch.no_grad():
        fs, _, _, _ = model(X, A, keep, cnt)

    nm, nt = len(MODALITIES), len(TASKS)
    uniform = torch.full((B, nm, nt), 1.0 / nm)
    with torch.no_grad():
        base = float(task_oriented_loss(model, fs, labels, uniform, cfg["temperature"]))
        best = min(
            float(task_oriented_loss(
                model, fs, labels,
                torch.zeros(B, nm, nt).index_fill_(1, torch.tensor([mi]), 1.0),
                cfg["temperature"]))
            for mi in range(nm)
        )
    assert best < base, (
        f"no vertex beat the uniform allocation ({best} vs {base}); the collapse argument "
        f"would not hold on this batch")
