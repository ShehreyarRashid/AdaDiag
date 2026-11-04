"""README section 6.20: Eq. 17's inter-task affinity (the paper's Table 8)."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adadiag import config, data                                   # noqa: E402
from adadiag.affinity import task_affinity                         # noqa: E402
from adadiag.model import AdaDiag                                  # noqa: E402
from adadiag.train import train                                    # noqa: E402


def test_measuring_affinity_does_not_change_training():
    """Hotel has dropout, so this also checks the random stream is restored."""
    ds = data.load("hotel")
    torch.set_num_threads(1)
    a, _ = train("v0_raw", 0, ds, config.get("hotel", epochs=3))
    b, _ = train("v0_raw", 0, ds, config.get("hotel", epochs=3, measure_affinity=True))
    keys = ["HR@1", "HR@3", "MRR@3", "precision", "recall", "f1", "epochs"]
    assert all(a[k] == b[k] for k in keys)
    assert "affinity" not in a and set(b["affinity"]) == {"r2f", "f2r", "f2r_code"}


def test_eq17_and_the_released_computation():
    torch.manual_seed(0)
    cfg = config.get("gaia")                    # no dropout: every forward is deterministic
    m = AdaDiag("v0_raw", 5, cfg)
    X = torch.randn(8, 3, 6, 128)
    A = (torch.rand(8, 6, 6) > 0.5).float()
    keep = torch.ones(8, 6, dtype=torch.bool)
    batch = (X, A, keep, torch.randint(0, 6, (8,)), torch.randint(0, 5, (8,)), torch.zeros(8, 3))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    z = task_affinity(m, batch, cfg)
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())   # untouched

    def rcl(mm):
        _, _, nl, _ = mm(X, A, keep, batch[5])
        return float(F.cross_entropy(nl, batch[3]).detach())

    # the released f2r is the RCL loss's response to an RCL step
    import copy
    m1 = copy.deepcopy(m)
    opt = torch.optim.Adam(m1.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    l0 = F.cross_entropy(m1(X, A, keep, batch[5])[2], batch[3])
    l0.backward(); opt.step()
    assert abs(z["f2r_code"] - (1 - rcl(m1) / float(l0.detach()))) < 1e-6
    assert z["f2r"] != z["f2r_code"]
