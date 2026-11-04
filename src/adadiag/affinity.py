"""Inter-task affinity, Eq. 17 of the TVDiag paper (Table 8; README section 6.20).

    Z_{T1 -> T2} = 1 - L_T2(f, theta_s|T1, theta_T2) / L_T2(f, theta_s, theta_T2)

i.e. take one optimiser step on task T1's loss alone and measure how much task T2's loss
falls. TVDiag computes it at every step of the first epoch (`core/ita.py`, called from
`core/TVDiag.py`) on deep copies of the model, each with a fresh Adam at the training lr and
weight decay, in training mode, on the same batch (raw + augmented graphs).

FIDELITY NOTE. In `core/ita.py` the FTI -> RCL half steps `model2` on the FTI loss and then
re-evaluates `model1` -- the copy that was stepped on the RCL loss:

    l_fti.backward(); opt2.step()
    _, _, root_logit2, _ = model1(batch_graphs)      # should be model2

so the published Z_FTI->RCL is in fact the RCL loss's response to an RCL step (RCL's
"self-affinity"), not FTI's effect on RCL. `task_affinity` returns both: `f2r` is Eq. 17 as
written, `f2r_code` reproduces the released computation.

The measurement never touches the model being trained, and the torch random stream (used
by dropout in the copies' forward passes) is restored afterwards, so a measured run trains
bit-identically to an unmeasured one.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F


def _losses(m, batch):
    X, A, keep, root, yt, cnt = batch
    _, _, nl, tl = m(X, A, keep, cnt)
    return (F.cross_entropy(nl.masked_fill(~keep, float("-inf")), root),
            F.cross_entropy(tl, yt))


def task_affinity(model, batch, cfg) -> dict:
    """Eq. 17 both ways on one batch. Returns {"r2f", "f2r", "f2r_code"}."""
    state = torch.get_rng_state()
    m1, m2 = copy.deepcopy(model), copy.deepcopy(model)
    m1.train(); m2.train()
    o1 = torch.optim.Adam(m1.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    o2 = torch.optim.Adam(m2.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # RCL -> FTI: step m1 on the RCL loss, measure the FTI loss
    l_rcl, l_fti = _losses(m1, batch)
    l_rcl.backward()
    o1.step()
    with torch.no_grad():
        _, l_fti_after = _losses(m1, batch)
    r2f = 1.0 - float(l_fti_after) / float(l_fti.detach())

    # FTI -> RCL: step m2 on the FTI loss, measure the RCL loss
    l_rcl0, l_fti0 = _losses(m2, batch)
    l_fti0.backward()
    o2.step()
    with torch.no_grad():
        l_rcl_after, _ = _losses(m2, batch)       # Eq. 17
        l_rcl_code, _ = _losses(m1, batch)        # TVDiag's core/ita.py: model1, not model2
    f2r = 1.0 - float(l_rcl_after) / float(l_rcl0.detach())
    f2r_code = 1.0 - float(l_rcl_code) / float(l_rcl0.detach())
    torch.set_rng_state(state)
    return {"r2f": r2f, "f2r": f2r, "f2r_code": f2r_code}
