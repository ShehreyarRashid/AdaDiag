"""v1.22 README section 6.25: the round-4 switches and the ensemble metrics (no data needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src")); sys.path.insert(0, str(HERE / "scripts"))
from gradfusion import config                                         # noqa: E402
from gradfusion.model import GradFusion                                  # noqa: E402
import plus2                                                       # noqa: E402


def _batch(v=7):
    torch.manual_seed(0)
    X = torch.rand(2, 3, 6, 128 + v)
    A = (torch.rand(2, 6, 6) > 0.5).float()
    return X, A, torch.ones(2, 6, dtype=torch.bool), torch.zeros(2, 3)


def test_bag_switches_need_the_alert_bag():
    for over in (dict(bag_drop=0.2), dict(bag_head=True)):
        with pytest.raises(ValueError):
            GradFusion("v0_raw", 5, config.get("hotel", **over))


def test_bag_drop_acts_in_training_only_and_spares_the_embeddings():
    X, A, keep, c = _batch()
    cfg = config.get("hotel", alert_bag=True, bag_drop=0.5, alert_embedding_dim=135, feat_drop=0.0)
    m = GradFusion("v0_raw", 5, cfg)
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(X, A, keep, c)[2], m(X, A, keep, c)[2])
    m.train()
    with torch.no_grad():
        assert not torch.equal(m(X, A, keep, c)[2], m(X, A, keep, c)[2])
    # with no count columns set, dropout has nothing to change
    X0 = X.clone(); X0[..., 128:] = 0
    with torch.no_grad():
        assert torch.equal(m(X0, A, keep, c)[2], m(X0, A, keep, c)[2])


def test_bag_head_widens_both_heads():
    X, A, keep, c = _batch()
    base = GradFusion("v0_raw", 5, config.get("hotel", alert_bag=True, alert_embedding_dim=135))
    wide = GradFusion("v0_raw", 5, config.get("hotel", alert_bag=True, alert_embedding_dim=135,
                                           bag_head=True))
    assert wide.locator.net[0].in_features == base.locator.net[0].in_features + 3 * 7
    assert wide.classifier.net[0].in_features == base.classifier.net[0].in_features + 2 * 3 * 7
    out = wide.eval()(X, A, keep, c)
    assert out[2].shape == (2, 6) and out[3].shape == (2, 5)


def test_metrics_match_the_definitions():
    node = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    typ = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    m = plus2.metrics(node, typ, np.array([1, 1, 0]), np.array([0, 1, 1]))
    assert m["HR@1"] == pytest.approx(1 / 3) and m["HR@2"] == pytest.approx(2 / 3)
    assert m["MRR@3"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)
    p, r = (0.5 + 1.0) * 1 / 3 + 0.0, 2 / 3              # weighted precision, recall
    p = (1 / 3) * 0.5 + (2 / 3) * 1.0
    assert m["f1"] == pytest.approx(2 * p * r / (p + r))


def test_variant_names_and_job_lists():
    assert plus2.variant("tvdiag") == ("v0_raw", {})
    arm, over = plus2.variant("plus+smooth+baghead")
    assert arm == "v5_align" and over == dict(alert_bag=True, awl_learnable=True,
                                              fti_smooth=0.1, bag_head=True)
    J = plus2.jobs("S")
    assert len(J) == 4 * 3 * 3 and J[0][0] == "sockshop"      # most expensive first
    assert np.allclose(plus2.softmax(np.array([[0.0, 0.0]])), [[0.5, 0.5]])


def test_ensemble_predict_averages_member_softmaxes():
    from types import SimpleNamespace
    from gradfusion.ensemble import predict
    X, A, keep, c = _batch()
    cfg = config.get("hotel", alert_bag=True, alert_embedding_dim=135)
    ms = []
    for s in (0, 1, 2):
        torch.manual_seed(s); ms.append(GradFusion("v0_raw", 5, cfg).eval())
    ds = SimpleNamespace(X=X, A=A, cnt=c, n_nodes=6)
    node, typ = predict(ms, ds, [0, 1])
    with torch.no_grad():
        outs = [m(X, A, keep, c) for m in ms]
    want = np.mean([plus2.softmax(o[2].numpy()) for o in outs], 0)
    assert np.allclose(node.numpy(), want, atol=1e-6)
    assert torch.allclose(node.sum(1), torch.ones(2), atol=1e-6)
    assert torch.allclose(typ.sum(1), torch.ones(2), atol=1e-6)
