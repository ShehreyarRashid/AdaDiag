"""Why the trace weight goes to zero (README section 6.10): the checks behind the audit.

The zero is produced by the objective, not by a wiring error. These tests pin down the
three facts that claim rests on: the trace data sit in the trace slot, the weighting's only
gradient moves weight away from the modality with the higher contrastive loss, and the
diagnostics used to show it (the training monitor, the warm-up switch) change nothing they
are not meant to change.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adadiag import config, data                                        # noqa: E402
from adadiag.data import Dataset                                        # noqa: E402
from adadiag.identifiability import permute_modalities                 # noqa: E402
from adadiag.model import AdaDiag                                       # noqa: E402
from adadiag.train import compute_terms, make_batch, train              # noqa: E402
from adadiag.weighting import MODALITIES, WeightingModule               # noqa: E402

HAS_DATA = (ROOT / "data" / "hotel" / "tmp" / "trace.pkl").exists()


@pytest.mark.skipif(not HAS_DATA, reason="needs data/")
def test_trace_plane_holds_the_trace_file():
    """Slot 1 of X is tmp/trace.pkl, row 1 of w and the 'trace' encoder read slot 1."""
    ds = data.load("hotel")
    raw = pickle.load(open(ROOT / "data" / "hotel" / "tmp" / "trace.pkl", "rb"))
    first = str(next(iter(raw)))
    i = 0                                           # hotel keeps every incident, in order
    assert MODALITIES[1] == "trace"
    assert np.allclose(ds.X[i, 1].numpy(), np.asarray(raw[first], dtype=np.float32))


def _toy(n=16, N=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    split = np.array(["train"] * 10 + ["test"] * 6)
    return Dataset(X=torch.randn(n, 3, N, 128, generator=g),
                   A=(torch.rand(n, N, N, generator=g) > 0.5).float(),
                   root=torch.randint(0, N, (n,), generator=g),
                   ytype=torch.arange(n) % 3, cnt=torch.randn(n, 3, generator=g),
                   split=split, types=["a", "b", "c"],
                   instances=[f"i{k}" for k in range(N)], name="toy")


def test_permuting_slots_permutes_what_each_encoder_sees():
    ds = _toy()
    sw = permute_modalities(ds, ("trace", "metric", "log"))
    assert torch.equal(sw.X[:, 0], ds.X[:, 1]) and torch.equal(sw.X[:, 1], ds.X[:, 0])
    assert torch.equal(sw.cnt[:, 0], ds.cnt[:, 1])


def test_weighting_gradient_moves_weight_off_the_highest_loss():
    """The module's only signal is dl_to/dw = L, so a step lowers the weight of the
    modality whose contrastive loss is highest -- whatever that modality is useful for."""
    wm = WeightingModule("v2_global", 96)
    L = torch.tensor([[5.3, 5.2], [5.9, 5.5], [5.5, 5.1]])     # trace (row 1) highest
    opt = torch.optim.SGD(wm.parameters(), lr=1.0)
    w0 = wm(torch.zeros(1, 96), torch.zeros(1, 3))[0].detach()
    (wm(torch.zeros(1, 96), torch.zeros(1, 3))[0] * L).sum().backward()
    opt.step()
    w1 = wm(torch.zeros(1, 96), torch.zeros(1, 3))[0].detach()
    assert torch.all(w1[1] < w0[1])                             # trace loses in both columns
    assert w1[0, 0] > w0[0, 0] and w1[2, 1] > w0[2, 1]          # lowest-loss rows gain


def _cfg(**kw):
    return config.get("gaia", epochs=4, batch_size=8, patience=2, **kw)


def test_monitor_is_read_only():
    ds = _toy()
    seen = []
    a, _ = train("v3_incident", 0, ds, _cfg())
    b, _ = train("v3_incident", 0, ds, _cfg(), monitor=lambda e, c, w: seen.append(e))
    assert seen and all(a[k] == b[k] for k in ("HR@1", "f1", "epochs"))


def test_warmup_zero_is_the_method():
    ds = _toy()
    a, _ = train("v3_incident", 0, ds, _cfg())
    b, _ = train("v3_incident", 0, ds, _cfg(weight_warmup=0))
    assert all(a[k] == b[k] for k in ("HR@1", "f1", "epochs")) and a["w_final"] == b["w_final"]


def test_warmup_freezes_the_weights_and_trains_every_encoder_on_both_labels():
    ds = _toy()
    cfg = _cfg(weight_warmup=2)
    torch.manual_seed(0)
    model = AdaDiag("v3_incident", 3, cfg)
    import random
    batch = make_batch(ds, ds.train_idx[:8], cfg, random.Random(0))
    terms, w = compute_terms(model, "v3_incident", batch, cfg, warmup=True)
    params = list(model.weighting.parameters())
    g = torch.autograd.grad(terms["l_to"], params, allow_unused=True)
    assert all(x is None or float(x.abs().sum()) == 0.0 for x in g)       # module frozen
    g_tr = torch.autograd.grad(terms["l_to"], list(model.encoders["trace"].parameters()),
                               allow_unused=True)
    assert any(x is not None and float(x.abs().sum()) > 0 for x in g_tr)  # trace trained
    terms2, _ = compute_terms(model, "v3_incident", batch, cfg, warmup=False)
    g2 = torch.autograd.grad(terms2["l_to"], params, allow_unused=True)
    assert any(x is not None and float(x.abs().sum()) > 0 for x in g2)    # then it learns
