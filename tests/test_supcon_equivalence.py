"""The one test that must pass before any number is trusted.

AdaDiag needs a PER-SAMPLE supervised contrastive loss so that a per-incident weight can
multiply it. TVDiag's `SupConLoss` returns one scalar per batch. If the refactor is not
exactly equivalent, every arm -- including the baseline -- is silently a different model
and the whole comparison is void.

Run:  pytest -q
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adadiag.losses import supcon_per_sample  # noqa: E402


def tvdiag_supcon(embeddings, labels, temperature=0.3):
    """Verbatim from core/loss/SupervisedContrastiveLoss.py (device argument dropped)."""
    n = labels.shape[0]
    embeddings = F.normalize(embeddings, dim=1)
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    mask = torch.ones_like(similarity_matrix) * (labels.expand(n, n).eq(labels.expand(n, n).t()))
    mask_no_sim = torch.ones_like(mask) - mask
    mask_dj = torch.ones(n, n) - torch.eye(n, n)
    similarity_matrix = torch.exp(similarity_matrix / temperature) * mask_dj
    sim = mask * similarity_matrix
    no_sim = similarity_matrix - sim
    no_sim_sum_expend = torch.sum(no_sim, dim=1).repeat(n, 1).T
    sim_sum = sim + no_sim_sum_expend
    loss_partial = -torch.log(mask_no_sim + torch.div(sim, sim_sum) + torch.eye(n, n))
    nonzero_count = len(torch.nonzero(loss_partial)) or 1
    return torch.sum(torch.sum(loss_partial, dim=1)) / nonzero_count


@pytest.mark.parametrize("n,n_classes,dim", [(32, 5, 32), (64, 10, 32), (16, 3, 8), (128, 7, 64)])
def test_per_sample_sums_to_tvdiag_scalar(n, n_classes, dim):
    torch.manual_seed(0)
    emb = torch.randn(n, dim)
    labels = torch.randint(0, n_classes, (n,))
    reference = tvdiag_supcon(emb, labels, 0.3)
    per_sample, nz = supcon_per_sample(emb, labels, 0.3)
    assert torch.allclose(per_sample.sum() / nz, reference, atol=1e-6), (
        f"refactor changed the loss: {per_sample.sum()/nz:.6f} vs {reference:.6f}")


def test_awl_is_inert_as_tvdiag_ships_it():
    """TVDiag never steps the AutomaticWeightedLoss parameters, so it reduces to
    0.5*sum(losses) + n*log(2). Reproducing that exactly is part of the baseline."""
    import math
    from adadiag.losses import AutomaticWeightedLoss
    awl = AutomaticWeightedLoss(4, learnable=False)
    losses = [torch.tensor(float(x)) for x in (1, 2, 3, 4)]
    expected = 0.5 * sum(range(1, 5)) + 4 * math.log(2)
    assert abs(float(awl(*losses)) - expected) < 1e-4
    assert not any(p.requires_grad for p in awl.parameters())


def test_weight_columns_are_on_the_simplex():
    from adadiag.weighting import WeightingModule
    for arm in ("v0_simplex", "v1_uniform", "v2_global", "v3_incident"):
        wm = WeightingModule(arm, fuse_dim=96)
        w = wm(torch.randn(8, 96), torch.randn(8, 3))
        assert w.shape == (8, 3, 2)
        assert torch.allclose(w.sum(1), torch.ones(8, 2), atol=1e-5), f"{arm} columns must sum to 1"
