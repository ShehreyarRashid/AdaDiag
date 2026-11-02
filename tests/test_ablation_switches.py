"""The ablation switches do what they claim, and do nothing at their defaults.

Self-contained (synthetic batch), so they run without data/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adadiag import ablation, config                              # noqa: E402
from adadiag.losses import AutomaticWeightedLoss                  # noqa: E402
from adadiag.model import AdaDiag                                 # noqa: E402
from adadiag.train import compute_terms, total_loss               # noqa: E402
from adadiag.weighting import MODALITIES                          # noqa: E402

B, N, N_TYPES = 12, 8, 5


def _batch(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, len(MODALITIES), N, cfg["alert_embedding_dim"], generator=g)
    A = (torch.rand(B, N, N, generator=g) < 0.3).float()
    keep = torch.ones(B, N, dtype=torch.bool)
    cnt = torch.randn(B, len(MODALITIES), generator=g)
    root = torch.randint(0, N, (B,), generator=g)
    yt = torch.randint(0, N_TYPES, (B,), generator=g)
    return X, A, keep, root, yt, cnt


def _model(arm, **over):
    cfg = config.get("hotel", **over)
    torch.manual_seed(0)
    return AdaDiag(arm, N_TYPES, cfg), cfg


@pytest.mark.parametrize("arm", ["v0_simplex", "v3_incident"])
def test_explicit_defaults_change_nothing(arm):
    """Spelling out every switch at its default gives the identical loss."""
    m1, c1 = _model(arm)
    m2, c2 = _model(arm, drop_modalities=(), w_constraint="simplex", detach=True,
                    descriptor="full", use_to=True, use_cm=True, entropy_reg=0.0,
                    dynamic_weights=True)
    m1.eval(); m2.eval()                    # hotel uses dropout; compare without it
    batch = _batch(c1)
    awl = AutomaticWeightedLoss(4)
    l1 = total_loss(awl, compute_terms(m1, arm, batch, c1)[0], c1)
    l2 = total_loss(awl, compute_terms(m2, arm, batch, c2)[0], c2)
    assert float(l1.detach()) == float(l2.detach())


@pytest.mark.parametrize("drop, rcl, fti", [
    (("log",),            [0.5, 0.5, 0.0], [0.5, 0.5, 0.0]),   # FTI's only source gone -> uniform
    (("metric",),         [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
    (("trace",),          [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    (("metric", "trace"), [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),   # log only
])
def test_prior_is_renormalised_over_remaining_modalities(drop, rcl, fti):
    m, cfg = _model("v0_simplex", drop_modalities=drop)
    X, A, keep, root, yt, cnt = _batch(cfg)
    fs, f, _, _ = m(X, A, keep, cnt)
    w = m.weights(f, cnt)[0]
    assert w[:, 0].tolist() == pytest.approx(rcl)
    assert w[:, 1].tolist() == pytest.approx(fti)
    for d in drop:
        assert torch.count_nonzero(fs[d]) == 0, f"{d} should contribute a zero embedding"


@pytest.mark.parametrize("arm", ["v1_uniform", "v2_global", "v3_incident"])
def test_removed_modality_gets_zero_weight_and_no_gradient(arm):
    m, cfg = _model(arm, drop_modalities=("trace",))
    batch = _batch(cfg)
    terms, w = compute_terms(m, arm, batch, cfg)
    ti = MODALITIES.index("trace")
    assert torch.all(w[:, ti, :] == 0)
    assert torch.allclose(w.sum(1), torch.ones(B, 2))
    total_loss(AutomaticWeightedLoss(4), terms, cfg).backward()
    assert all(p.grad is None for p in m.encoders["trace"].parameters()), \
        "a removed modality's encoder must not train"
    assert any(p.grad is not None for p in m.encoders["metric"].parameters())


def test_unconstrained_weights_shrink_under_l_to():
    """The identifiability argument (README section 1), checked directly.

    Without the simplex, every SupCon term is positive, so gradient descent on l_to alone
    lowers EVERY weight -- the optimiser silences the term instead of reallocating it.
    """
    m, cfg = _model("v3_incident", w_constraint="softplus")
    batch = _batch(cfg)
    opt = torch.optim.SGD(m.weighting.parameters(), lr=0.5)
    sums = []
    for _ in range(25):
        terms, w = compute_terms(m, "v3_incident", batch, cfg)
        sums.append(w.detach().sum(1).mean(0))
        opt.zero_grad(); terms["l_to"].backward(); opt.step()
    assert torch.all(sums[-1] < sums[0] * 0.9), f"column mass did not shrink: {sums[0]} -> {sums[-1]}"


def test_entropy_penalty_pushes_toward_uniform():
    m, cfg = _model("v3_incident", entropy_reg=1.0)
    batch = _batch(cfg)
    terms, w = compute_terms(m, "v3_incident", batch, cfg)
    assert "negent" in terms
    # negent is minimised (= -log 3 per column) by the uniform allocation
    assert float(terms["negent"].detach()) >= -2 * torch.log(torch.tensor(3.0)) - 1e-5


def test_registry_is_consistent():
    for name, (group, arm, over, desc) in ablation.ABLATIONS.items():
        ref = ablation.reference_for(name)
        assert (group == "ref") == (ref is None)
        if ref:
            assert ablation.ABLATIONS[ref][1] == arm, f"{name} is compared to a different arm"
    assert set(ablation.select(["B"])) >= {"ref_prior", "B_no_log"}
