"""README section 6.18: the LSTM aggregator the TVDiag paper describes (no data needed)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adadiag import config                                              # noqa: E402
from adadiag.encoders import SAGEEncoder, lstm_neighbours               # noqa: E402
from adadiag.model import AdaDiag                                       # noqa: E402


def _graph(seed=0, B=3, N=7, D=8):
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(B, N, N, generator=g) > 0.55).float()
    A[0, 3] = 0                                  # node 3 of graph 0: no in-neighbour
    return torch.randn(B, N, D, generator=g), A


def test_dense_reducer_equals_a_node_by_node_loop():
    torch.manual_seed(0)
    h, A = _graph()
    lstm = torch.nn.LSTM(8, 8, batch_first=True)
    with torch.no_grad():
        out = lstm_neighbours(lstm, h, A)
    for b in range(A.shape[0]):
        for i in range(A.shape[1]):
            nb = [j for j in range(A.shape[2]) if A[b, i, j] > 0]
            if not nb:
                assert float(out[b, i].abs().sum()) == 0.0
                continue
            with torch.no_grad():
                _, (hn, _) = lstm(h[b, nb].unsqueeze(0))
            assert torch.allclose(out[b, i], hn[-1, 0], atol=1e-6)


def test_mean_aggregator_is_unchanged_by_default():
    torch.manual_seed(0)
    enc = SAGEEncoder([8, 6, 4])
    h, A = _graph(1)
    with torch.no_grad():
        deg = A.sum(-1, keepdim=True).clamp(min=1.0)
        x = torch.relu(enc.self_fc[0](h) + enc.neigh_fc[0](torch.bmm(A, h) / deg))
        ref = enc.self_fc[1](x) + enc.neigh_fc[1](torch.bmm(A, x) / deg)
        assert torch.equal(enc(h, A), ref)
    assert not hasattr(enc, "lstm")


def test_lstm_model_trains_end_to_end():
    cfg = config.get("hotel", aggregator="lstm")
    torch.manual_seed(0)
    m = AdaDiag("v3_incident", 5, cfg)
    assert all(e.aggregator == "lstm" for e in m.encoders.values())
    X = torch.randn(2, 3, 6, 128)
    A = (torch.rand(2, 6, 6) > 0.5).float()
    keep = torch.ones(2, 6, dtype=torch.bool)
    fs, f, nl, tl = m(X, A, keep, torch.zeros(2, 3))
    (nl.sum() + tl.sum()).backward()
    assert all(p.grad is not None for p in m.encoders["metric"].lstm.parameters())


def test_same_linear_initialisation_as_mean():
    """The LSTM is built after the linear layers, so they start from the same weights."""
    torch.manual_seed(0); a = SAGEEncoder([8, 6, 4])
    torch.manual_seed(0); b = SAGEEncoder([8, 6, 4], aggregator="lstm")
    assert all(torch.equal(x.weight, y.weight) for x, y in zip(a.self_fc, b.self_fc))
