"""README section 6.19: what the TVDiag paper specifies and its released code does not do.

Each paper switch is checked against a literal transcription of the paper's equation, and
each default is checked to be the released code's behaviour.
"""
import math
import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adadiag import config, data                                         # noqa: E402
from adadiag.encoders import SAGEEncoder                                 # noqa: E402
from adadiag.losses import (AutomaticWeightedLoss, cm_paper,             # noqa: E402
                            supcon_anchor_mean, supcon_per_sample, uspcon)
from adadiag.model import AdaDiag                                        # noqa: E402
from adadiag.train import build_aug_pool, train                          # noqa: E402

T = 0.3


def _emb(seed=0, n=12, d=8):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randn(n, d, generator=g)


def _phi(a, b):
    return math.exp(float(F.cosine_similarity(a, b, dim=0)) / T)


def test_to_norm_paper_is_eq3_per_anchor_mean():
    x, _ = _emb()
    y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 4])     # 3 and 4 have no positive
    per, nz = supcon_anchor_mean(x, y, T)
    terms = []
    for i in range(len(y)):
        P = [j for j in range(len(y)) if j != i and y[j] == y[i]]
        N = [z for z in range(len(y)) if y[z] != y[i]]
        if not P:
            assert float(per[i]) == 0.0
            continue
        neg = sum(_phi(x[i], x[z]) for z in N)
        li = -sum(math.log(_phi(x[i], x[j]) / (_phi(x[i], x[j]) + neg)) for j in P) / len(P)
        assert abs(float(per[i]) - li) < 1e-4
        terms.append(li)
    assert nz == len(terms)
    assert abs(float(per.sum() / nz) - sum(terms) / len(terms)) < 1e-4


def test_to_norm_code_differs_only_in_normalisation():
    x, _ = _emb(1, n=10)
    y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2, 2, 2])
    code, nz = supcon_per_sample(x, y, T)
    paper, _ = supcon_anchor_mean(x, y, T)
    npos = torch.tensor([4] * 5 + [1] * 2 + [2] * 3, dtype=code.dtype)
    assert torch.allclose(paper, code / npos)
    assert nz == int(npos.sum())                     # the code divides by the pair count


def test_cm_paper_is_eqs_5_and_6():
    a, b = _emb(2)
    n = a.shape[0]

    def l(F1, F2, i):
        den = sum(_phi(F1[i], F1[k]) for k in range(n) if k != i) + \
              sum(_phi(F1[i], F2[k]) for k in range(n))
        return -math.log(_phi(F1[i], F2[i]) / den)

    ref = sum(l(a, b, i) + l(b, a, i) for i in range(n)) / (2 * n)
    assert abs(float(cm_paper(a, b, T)) - ref) < 1e-4
    # the released term is one direction, cross-modal negatives only
    one = sum(-math.log(_phi(a[i], b[i]) / sum(_phi(a[i], b[k]) for k in range(n)))
              for i in range(n)) / n
    assert abs(float(uspcon(a, b, T)) - one) < 1e-4


def test_alg1_normalises_every_layer_and_default_does_not():
    torch.manual_seed(0)
    enc = SAGEEncoder([8, 6, 4], alg1=True)
    torch.manual_seed(0)
    ref = SAGEEncoder([8, 6, 4])
    assert all(torch.equal(p, q) for p, q in zip(enc.parameters(), ref.parameters()))
    h = torch.randn(3, 7, 8)
    A = (torch.rand(3, 7, 7) > 0.5).float()
    with torch.no_grad():
        out = enc(h, A)
        deg = A.sum(-1, keepdim=True).clamp(min=1.0)
        x = F.normalize(F.relu(enc.self_fc[0](h) + enc.neigh_fc[0](torch.bmm(A, h) / deg)), dim=-1)
        y = F.normalize(F.relu(enc.self_fc[1](x) + enc.neigh_fc[1](torch.bmm(A, x) / deg)), dim=-1)
    assert torch.equal(out, y)
    norms = out.norm(dim=-1)
    assert bool(((norms - 1).abs() < 1e-5).logical_or(norms == 0).all()) and bool((out >= 0).all())
    assert not bool((ref(h, A) < 0).sum() == 0)       # default: no ReLU on the last layer


def test_bidirectional_symmetrises_and_is_a_no_op_on_symmetric_graphs():
    torch.manual_seed(0)
    X = torch.randn(2, 3, 6, 128)
    A = (torch.rand(2, 6, 6) > 0.6).float()
    keep = torch.ones(2, 6, dtype=torch.bool)
    c = torch.zeros(2, 3)
    torch.manual_seed(1); on = AdaDiag("v0_raw", 5, config.get("hotel", bidirectional=True))
    torch.manual_seed(1); off = AdaDiag("v0_raw", 5, config.get("hotel"))
    on.eval(); off.eval()
    with torch.no_grad():
        S = torch.maximum(A, A.transpose(1, 2))
        assert torch.equal(on(X, A, keep, c)[2], off(X, S, keep, c)[2])
        assert torch.equal(on(X, S, keep, c)[2], off(X, S, keep, c)[2])


def test_awl_inert_by_default_and_learns_when_switched_on():
    assert not AutomaticWeightedLoss(4).p.requires_grad
    ds = data.load("hotel")
    torch.set_num_threads(1)
    res, _ = train("v0_raw", 0, ds, config.get("hotel", epochs=3, awl_learnable=True))
    assert len(res["awl_theta"]) == 4 and any(abs(t - 1.0) > 1e-4 for t in res["awl_theta"])
    res0, _ = train("v0_raw", 0, ds, config.get("hotel", epochs=3))
    assert "awl_theta" not in res0


def test_paper_aug_is_one_copy_per_incident():
    ds = data.load("hotel")
    cfg = config.get("hotel", aug_times=1)
    one = build_aug_pool(ds, cfg, random.Random(0))
    assert [i for i, _ in one] == [int(i) for i in ds.train_idx]
    k = int(ds.n_nodes * cfg["aug_percent"])
    for i, keep in one:
        assert bool(keep[int(ds.root[i])]) and int((~keep).sum()) == k


@pytest.mark.parametrize("bad", [dict(to_norm="x"), dict(cm_mode="x"), dict(aug_mode="x")])
def test_unknown_modes_are_refused(bad):
    ds = data.load("hotel")
    with pytest.raises(ValueError):
        train("v0_raw", 0, ds, config.get("hotel", epochs=1, **bad))


def test_paper_preset_trains_both_arms():
    ds = data.load("hotel")
    torch.set_num_threads(1)
    for arm in ("v0_raw", "v5_align"):
        res, _ = train(arm, 0, ds, config.get("hotel", epochs=2, **config.PAPER))
        assert 0.0 <= res["HR@1"] <= 1.0 and "awl_theta" in res
