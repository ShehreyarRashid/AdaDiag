"""v2.2 code review: input checks, the shared metric code, the SupCon reuse and `fast_supcon`.

Fast (about 3 s). The one slow test, a full training run checked against the committed
numbers, runs only with GRADFUSION_SLOW=1 (about 2.5 min on one core).
"""
import dataclasses
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from gradfusion import config, data                                    # noqa: E402
from gradfusion.alerts import with_alert_bag                           # noqa: E402
from gradfusion.augment import drop_non_root                           # noqa: E402
from gradfusion.ensemble import predict                                # noqa: E402
from gradfusion.evaluate import evaluate, metrics_from_scores          # noqa: E402
from gradfusion.losses import pairwise_cosine, supcon_per_sample       # noqa: E402
from gradfusion.model import GradFusion                                # noqa: E402
from gradfusion.train import task_oriented_loss, to_term, train        # noqa: E402
from gradfusion.weighting import MODALITIES, TASKS                     # noqa: E402


@pytest.fixture(scope="module")
def hotel():
    return data.load("hotel")


# ---------------------------------------------------------------- config
def test_config_rejects_a_misspelt_option():
    with pytest.raises(KeyError, match="awl_learnabel"):
        config.get("hotel", awl_learnabel=True)


def test_config_accepts_every_preset():
    for preset in (config.FIDELITY, config.PAPER, config.GRADFUSION):
        config.get("gaia", **preset)
    assert config.get("hotel", fast_supcon=None).get("fast_supcon") is None     # None = keep


# ---------------------------------------------------------------- losses
def test_fast_cosine_matches_the_reference():
    torch.manual_seed(0)
    e = torch.randn(300, 32)
    e[7] = 0.0                                     # an all-zero embedding stays at cosine 0
    assert torch.allclose(pairwise_cosine(e, fast=True), pairwise_cosine(e), atol=1e-6)


@pytest.mark.parametrize("n", [2, 64, 257])
def test_fast_supcon_matches_the_reference_in_value_and_gradient(n):
    torch.manual_seed(n)
    labels = torch.randint(0, 4, (n,))
    grads = []
    for fast in (False, True):
        e = torch.randn(n, 32, generator=torch.Generator().manual_seed(1), requires_grad=True)
        per, nz = supcon_per_sample(e, labels, 0.3, fast=fast)
        (per.sum() / nz).backward()
        grads.append((per.detach(), nz, e.grad))
    (p0, n0, g0), (p1, n1, g1) = grads
    assert n0 == n1
    assert torch.allclose(p0, p1, rtol=1e-5, atol=1e-5)
    assert torch.allclose(g0, g1, rtol=1e-4, atol=1e-6)


def test_reusing_precomputed_supcon_terms_is_bit_identical():
    torch.manual_seed(0)
    model = GradFusion("v5_align", 5, config.get("hotel"))
    fs = {m: torch.randn(40, 32, requires_grad=True) for m in MODALITIES}
    labels = {"rcl": torch.randint(0, 18, (40,)), "fti": torch.randint(0, 5, (40,))}
    w = torch.rand(40, 3, 2)
    fresh = task_oriented_loss(model, fs, labels, w, 0.3)
    terms = {(m, t): to_term(fs[m], labels[t], 0.3) for m in MODALITIES for t in TASKS}
    reused = task_oriented_loss(model, fs, labels, w, 0.3, terms=terms)
    assert torch.equal(fresh, reused)


# ---------------------------------------------------------------- augmentation
@pytest.mark.parametrize("n, pct", [(18, 0.2), (21, 0.4), (1, 0.5), (5, 0.0), (4, 1.0)])
def test_node_drop_keeps_the_root_and_drops_exactly_k(n, pct):
    B = 16
    X, A = torch.ones(B, 3, n, 4), torch.ones(B, n, n)
    root = torch.randint(0, n, (B,))
    Xa, Aa, keep = drop_non_root(X, A, root, pct, random.Random(0))
    assert keep[torch.arange(B), root].all()
    assert ((~keep).sum(1) == min(int(n * pct), n - 1)).all()
    assert (Xa.sum((1, 3)) == 0).eq(~keep).all()            # dropped = zeroed ...
    assert (Aa.sum(1) == 0)[~keep].all() and (Aa.sum(2) == 0)[~keep].all()   # ... and cut off


# ---------------------------------------------------------------- metrics
def test_score_metrics_on_a_hand_worked_example():
    node = np.array([[0.7, 0.2, 0.1],      # root 0, rank 1
                     [0.5, 0.3, 0.2],      # root 2, rank 3
                     [0.4, 0.4, 0.2]])     # root 1, tied with 0: stable order puts it 2nd
    root = np.array([0, 2, 1])
    y = np.array([0, 1, 1])
    m = metrics_from_scores(node, np.eye(2)[[0, 1, 1]], root, y)
    assert m["HR@1"] == pytest.approx(1 / 3) and m["HR@2"] == pytest.approx(2 / 3)
    assert m["HR@3"] == 1.0
    assert m["MRR@3"] == pytest.approx((1 + 1 / 3 + 1 / 2) / 3)
    assert m["f1"] == m["macro_f1"] == m["precision"] == 1.0


def test_metrics_and_predict_reject_empty_input():
    with pytest.raises(ValueError):
        metrics_from_scores(np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, int))
    with pytest.raises(ValueError):
        evaluate(None, None, np.array([], dtype=int))
    with pytest.raises(ValueError):
        predict([], None, [])


# ---------------------------------------------------------------- data
def test_adjacency_matches_edges_json(hotel):
    edges = json.loads((hotel.root_dir / "raw" / "edges.json").read_text())
    for q, key in enumerate(hotel.keys[:50]):
        a = torch.zeros(hotel.n_nodes, hotel.n_nodes)
        for s, d in edges[key]:
            a[d, s] = 1.0
        assert torch.equal(hotel.A[q], a)


def test_strict_load_raises_rather_than_asserts(monkeypatch):
    monkeypatch.setitem(data.EXPECTED, "hotel", dict(data.EXPECTED["hotel"], train=1))
    with pytest.raises(ValueError, match="split"):
        data.load("hotel")


def test_alert_bag_is_cached_and_idempotent(hotel):
    bag = with_alert_bag(hotel)
    assert with_alert_bag(hotel) is bag and with_alert_bag(bag) is bag
    assert bag.bag_vocab and hotel.bag_vocab is None
    assert torch.equal(bag.X[..., :128], hotel.X)


def test_train_rejects_an_empty_training_split(hotel):
    empty = dataclasses.replace(hotel, split=np.full(len(hotel.split), "test"))
    with pytest.raises(ValueError, match="empty"):
        train("v0_raw", 0, empty, config.get("hotel", epochs=1))


# ---------------------------------------------------------------- end to end (slow)
@pytest.mark.skipif(not os.environ.get("GRADFUSION_SLOW"), reason="set GRADFUSION_SLOW=1")
def test_one_member_reproduces_the_committed_hotel_run(hotel):
    torch.set_num_threads(1)
    cfg = config.get("hotel", **config.GRADFUSION)
    res, _ = train(config.GRADFUSION_ARM, 5, with_alert_bag(hotel), cfg)
    assert round(res["HR@1"], 3) == 0.833 and round(res["f1"], 3) == 0.821
