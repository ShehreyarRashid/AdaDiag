"""README section 6.24: the switches of a stronger AdaDiag."""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src")); sys.path.insert(0, str(HERE / "scripts"))
from adadiag import config, data                                   # noqa: E402
from adadiag.alerts import alert_bags, vocabulary, with_alert_bag  # noqa: E402
from adadiag.encoders import SAGEEncoder                           # noqa: E402
from adadiag.model import AdaDiag                                  # noqa: E402
from adadiag.train import train                                    # noqa: E402

HAVE_DATA = (HERE / "data" / "hotel" / "raw" / "traces.json").exists()


def test_attention_weights_sum_to_one_and_isolated_nodes_get_nothing():
    torch.manual_seed(0)
    enc = SAGEEncoder([8, 6, 4], aggregator="attn")
    h = torch.randn(2, 5, 8)
    A = (torch.rand(2, 5, 5) > 0.5).float()
    A[:, 3, :] = 0                                   # node 3 has no in-neighbour
    msg = enc.attend(0, h, A)
    assert torch.allclose(msg[:, 3], torch.zeros(2, 8))
    # with every weight on one neighbour the message is that neighbour
    A1 = torch.zeros(1, 5, 5); A1[0, 0, 2] = 1.0
    assert torch.allclose(enc.attend(0, h[:1], A1)[0, 0], h[0, 2])


def test_attention_is_permutation_equivariant():
    torch.manual_seed(1)
    enc = SAGEEncoder([8, 6, 4], aggregator="attn").eval()
    h, A = torch.randn(1, 6, 8), (torch.rand(1, 6, 6) > 0.4).float()
    p = torch.randperm(6)
    out = enc(h, A)
    outp = enc(h[:, p], A[:, p][:, :, p])
    assert torch.allclose(out[:, p], outp, atol=1e-6)


def test_new_switches_are_off_by_default():
    torch.manual_seed(2)
    X, A = torch.randn(2, 3, 6, 128), (torch.rand(2, 6, 6) > 0.5).float()
    keep, c = torch.ones(2, 6, dtype=torch.bool), torch.zeros(2, 3)
    torch.manual_seed(3); m0 = AdaDiag("v0_raw", 5, config.get("hotel"))
    torch.manual_seed(3); m1 = AdaDiag("v0_raw", 5, config.get("hotel", self_res=False,
                                                                aggregator="mean"))
    m0.eval(); m1.eval()
    with torch.no_grad():
        assert torch.equal(m0(X, A, keep, c)[2], m1(X, A, keep, c)[2])
    assert not hasattr(m0, "res")


def test_self_res_adds_a_path_from_own_features():
    torch.manual_seed(4)
    X = torch.randn(1, 3, 6, 128)
    keep, c = torch.ones(1, 6, dtype=torch.bool), torch.zeros(1, 3)
    m = AdaDiag("v0_raw", 5, config.get("hotel", self_res=True, no_edges=True)).eval()
    with torch.no_grad():
        e = m(X, torch.zeros(1, 6, 6), keep, c)
    assert e[2].shape == (1, 6)


@pytest.mark.skipif(not HAVE_DATA, reason="needs data/")
def test_alert_bag_is_train_only_scaled_and_zero_means_no_alert():
    ds = data.load("hotel")
    bag, vocab = alert_bags(ds)
    assert bag.shape[:3] == (len(ds.keys), 3, ds.n_nodes)
    assert float(bag[ds.train_idx].max()) == pytest.approx(1.0)
    assert float(bag.min()) == 0.0
    assert vocab == vocabulary(ds)
    d2 = with_alert_bag(ds)
    assert torch.equal(d2.X[..., :128], ds.X) and d2.X.shape[-1] == 128 + bag.shape[-1]
    assert with_alert_bag(ds) is d2                       # cached
    # an instance with no raw alert of any kind has an all-zero bag
    import json
    raw = json.load(open(ds.root_dir / "raw" / "metrics.json"))
    k = ds.keys[0]
    silent = [j for j, n in enumerate(ds.instances) if all(a[0] != n for a in raw.get(k, []))]
    assert silent and float(bag[0, 0, silent].abs().sum()) == 0.0


@pytest.mark.skipif(not HAVE_DATA, reason="needs data/")
def test_alert_bag_needs_the_widened_dataset_and_trains():
    ds = data.load("hotel")
    with pytest.raises(ValueError):
        train("v0_raw", 0, ds, config.get("hotel", epochs=1, alert_bag=True))
    torch.set_num_threads(1)
    res, model = train("v0_raw", 0, with_alert_bag(ds),
                       config.get("hotel", epochs=2, alert_bag=True, aggregator="attn",
                                  self_res=True, weight_ema=0.9))
    assert 0.0 <= res["HR@1"] <= 1.0 and res["epochs"] == 2
