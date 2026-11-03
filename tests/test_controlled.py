"""README section 6.15: TVDiag and AdaDiag differ only in the task-oriented block (no data)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adadiag import config                                              # noqa: E402
from adadiag.model import AdaDiag                                       # noqa: E402
from adadiag.train import train                                         # noqa: E402
from adadiag.weighting import ARMS                                      # noqa: E402
from test_fidelity import _toy                                          # noqa: E402


def _shared(m):
    return {k: v for k, v in m.state_dict().items() if not k.startswith("weighting")}


def test_every_arm_starts_from_the_same_encoders_and_heads():
    cfg = config.get("hotel")
    torch.manual_seed(0)
    ref = _shared(AdaDiag("v0_raw", 3, cfg))
    for arm in ARMS:
        torch.manual_seed(0)
        sd = _shared(AdaDiag(arm, 3, cfg))
        assert all(torch.equal(ref[k], sd[k]) for k in ref), arm


@pytest.mark.parametrize("iso", [False, True])
def test_the_random_stream_after_construction(iso):
    """A learned module's initialisation advances torch's stream (so the encoders' dropout
    masks shift) unless `isolate_weighting_rng` rewinds it; the module's own init is the same
    either way."""
    cfg = config.get("hotel", isolate_weighting_rng=iso)
    torch.manual_seed(0); AdaDiag("v0_raw", 3, cfg); s0 = torch.get_rng_state()
    torch.manual_seed(0); m = AdaDiag("v3_incident", 3, cfg); s1 = torch.get_rng_state()
    assert torch.equal(s0, s1) == iso
    torch.manual_seed(0); m_plain = AdaDiag("v3_incident", 3, config.get("hotel"))
    assert all(torch.equal(a, b) for a, b in zip(m.weighting.parameters(),
                                                 m_plain.weighting.parameters()))


@pytest.mark.parametrize("iso", [False, True])
def test_with_the_to_term_off_the_two_arms_train_identically(iso):
    """Hotel's settings, so the encoders use dropout. With TO off the arms can only differ
    through something outside the TO block; with the stream isolated they do not."""
    ds = _toy(n=24)
    cfg = config.get("hotel", use_to=False, epochs=4, aug_times=1,
                     isolate_weighting_rng=iso)
    r0, m0 = train("v0_raw", 0, ds, cfg)
    r1, m1 = train("v3_incident", 0, ds, cfg)
    p0, p1 = _shared(m0), _shared(m1)
    assert all(torch.equal(p0[k], p1[k]) for k in p0) == iso
