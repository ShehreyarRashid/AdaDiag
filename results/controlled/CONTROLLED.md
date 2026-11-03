# Is everything but the TO block the same? (README §6.15)

## 1. Code review: every place the arm changes anything

| component | TVDiag vs AdaDiag | how checked |
|---|---|---|
| data, splits, seeds, hyper-parameters | same | `config.get`, one call for every arm |
| encoders, readout, heads, their initialisation | same | built before the weighting module; equal for every arm and seed (`tests/test_controlled.py`) |
| batch order (numpy) and AUG node drops (Python `random`) | same | neither stream is touched by the weighting module |
| encoder dropout masks (torch stream) | **different on Hotel** | AdaDiag's MLP initialisation advances the torch stream, so every dropout mask after it shifts; GAIA has no dropout. Fixed by `isolate_weighting_rng` (off in the committed runs) |
| cross-modal term, RCL and FTI losses, Eq. 13 (inert) | same | `compute_terms`, `total_loss`: no arm-dependent code |
| optimiser (Adam, lr, weight decay) | same | AdaDiag's extra parameters are per-parameter state only; no clipping, no scheduler |
| early stopping (training loss, patience 10) | same rule | but it watches the total loss, which includes the TO term, so a TO change also moves where training stops |
| best-epoch restore, evaluation, metrics | same | `evaluate` has no arm-dependent code; the module is not on the prediction path |
| **the TO block** | **different, by design** | TVDiag: 3 unit-weight terms (total weight 3). AdaDiag: 6 terms, columns on the simplex (total weight 2) — which is why the equal-mass reference V0-simplex exists |

## 2. The null test: TO switched off, the two arms must be identical

Seed 0, full training, `use_to=False`. Any difference here comes from outside the TO block (`scripts/check_controlled.py --part null`).

| dataset | `isolate_weighting_rng` | TVDiag, TO off: HR@1 / F1 (epochs) | AdaDiag, TO off | every parameter identical? |
|---|---|---|---|---|
| A · GAIA | off | 0.724 / 0.932 (189) | 0.724 / 0.932 (189) | **yes** |
| A · GAIA | on | 0.724 / 0.932 (189) | 0.724 / 0.932 (189) | **yes** |
| D · Hotel | off | 0.833 / 0.797 (117) | 0.846 / 0.753 (143) | no (max |Δ| 0.18) |
| D · Hotel | on | 0.833 / 0.797 (117) | 0.833 / 0.797 (117) | **yes** |

## 3. What the shifted stream did to the committed comparison

AdaDiag re-trained with `isolate_weighting_rng`, so its dropout masks are TVDiag's. Paired over seeds; Holm over the table.

| dataset | AdaDiag run | seeds | HR@1 | F1 | − TVDiag: HR@1 / F1 (Holm p) | − V0-simplex: HR@1 / F1 (Holm p) |
|---|---|---|---|---|---|---|
| A · GAIA | committed (stream shifted) | 3 | 0.740 ± 0.012 | 0.932 ± 0.015 | +0.013 / +0.007 (1.00 / 1.00) | +0.006 / -0.005 (1.00 / 1.00) |
| A · GAIA | **stream aligned** | 3 | 0.740 ± 0.012 | 0.932 ± 0.015 | +0.013 / +0.007 (1.00 / 1.00) | +0.006 / -0.005 (1.00 / 1.00) |
| | *identical to the committed run, bit for bit (no dropout on A · GAIA)* | | | | | |
| D · Hotel | committed (stream shifted) | 5 | 0.828 ± 0.015 | 0.781 ± 0.034 | -0.005 / -0.017 (1.00 / 1.00) | -0.010 / -0.019 (1.00 / 1.00) |
| D · Hotel | **stream aligned** | 5 | 0.836 ± 0.011 | 0.810 ± 0.007 | +0.003 / +0.012 (1.00 / 1.00) | -0.003 / +0.010 (1.00 / 1.00) |

## 4. Training length

TVDiag's stopping rule watches the total training loss, so changing the TO term also changes when training stops. TVDiag stopped, seed by seed, at AdaDiag's epoch:

| dataset | seed | AdaDiag (stream aligned) stopped at | TVDiag default: HR@1 / F1 (epochs) | TVDiag stopped there: HR@1 / F1 | AdaDiag: HR@1 / F1 |
|---|---|---|---|---|---|
| A · GAIA | 0 | 231 | 0.721 / 0.923 (292) | 0.732 / 0.923 | 0.743 / 0.936 |
| A · GAIA | 1 | 225 | 0.707 / 0.912 (292) | 0.720 / 0.916 | 0.726 / 0.916 |
| A · GAIA | 2 | 191 | 0.751 / 0.939 (191) | 0.751 / 0.939 | 0.750 / 0.944 |
| A · GAIA | mean | | 0.726 / 0.924 | 0.734 / 0.926 | 0.740 / 0.932 |
| D · Hotel | 0 | 180 | 0.846 / 0.809 (160) | 0.846 / 0.809 | 0.846 / 0.805 |
| D · Hotel | 1 | 112 | 0.821 / 0.771 (139) | 0.821 / 0.811 | 0.821 / 0.813 |
| D · Hotel | 2 | 163 | 0.833 / 0.791 (138) | 0.833 / 0.791 | 0.833 / 0.802 |
| D · Hotel | 3 | 153 | 0.846 / 0.819 (153) | 0.846 / 0.819 | 0.846 / 0.809 |
| D · Hotel | 4 | 125 | 0.821 / 0.798 (191) | 0.808 / 0.791 | 0.833 / 0.820 |
| D · Hotel | mean | | 0.833 / 0.798 | 0.831 / 0.804 | 0.836 / 0.810 |
