# AdaDiag with trace held fixed (README §6.14)

Trace's weight is held at a fixed value in each task column and never learned; AdaDiag's module learns only how to split the rest between metric and log (`WeightingModule(fixed_trace=...)`). Same seeds, 1 thread and 500-epoch cap as the references; each lane first re-trains AdaDiag itself and checks it bit-for-bit (`logs/`).

## Summary (difference from AdaDiag, same seeds)

| weighting | σ_to | GAIA HR@1 | GAIA F1 | Hotel HR@1 | Hotel F1 |
|---|---|---|---|---|---|
| TVDiag default | 0.1 | 0.726 | 0.924 | 0.833 | 0.798 |
| AdaDiag (current) | 0.1 | 0.740 | 0.932 | 0.828 | 0.781 |
| **AdaDiag, trace fixed at 1/3 of both columns** | 0.1 | 0.736 (-0.004) | 0.932 (-0.000) | 0.836 (+0.008) | 0.783 (+0.002) |
| **AdaDiag, trace fixed at TVDiag's role (RCL 0.5, FTI 0)** | 0.1 | 0.739 (-0.000) | 0.936 (+0.004) | 0.836 (+0.008) | 0.789 (+0.008) |
| TVDiag's rule | 0.5 | — | — | 0.836 | 0.728 |
| AdaDiag (current) | 0.5 | — | — | 0.823 | 0.771 |
| **AdaDiag, trace fixed at 1/3 of both columns** | 0.5 | — | — | 0.821 (-0.003) | 0.770 (-0.002) |
| **AdaDiag, trace fixed at TVDiag's role (RCL 0.5, FTI 0)** | 0.5 | — | — | 0.828 (+0.005) | 0.765 (-0.006) |

## The learned matrices

| dataset | σ_to | weighting | RCL: metric / trace / log | FTI: metric / trace / log |
|---|---|---|---|---|
| A · GAIA | 0.1 | AdaDiag (current) | 0.01 / 0.00 / 0.99 | 0.00 / 0.00 / 1.00 |
| | | trace fixed at 1/3 of both columns | 0.01 / 0.33 / 0.66 | 0.01 / 0.33 / 0.66 |
| | | ↳ larger of metric / log, share of the learned part (1 = all on one) | 0.986 | 0.989 |
| | | trace fixed at TVDiag's role (RCL 0.5, FTI 0) | 0.01 / 0.50 / 0.49 | 0.01 / 0.00 / 0.99 |
| | | ↳ larger of metric / log, share of the learned part (1 = all on one) | 0.981 | 0.990 |
| D · Hotel | 0.1 | AdaDiag (current) | 0.99 / 0.00 / 0.00 | 0.27 / 0.00 / 0.73 |
| | | trace fixed at 1/3 of both columns | 0.66 / 0.33 / 0.00 | 0.25 / 0.33 / 0.42 |
| | | ↳ larger of metric / log, share of the learned part (1 = all on one) | 0.996 | 0.630 |
| | | trace fixed at TVDiag's role (RCL 0.5, FTI 0) | 0.50 / 0.50 / 0.00 | 0.27 / 0.00 / 0.73 |
| | | ↳ larger of metric / log, share of the learned part (1 = all on one) | 0.996 | 0.732 |
| D · Hotel | 0.5 | AdaDiag (current) | 0.00 / 0.00 / 1.00 | 1.00 / 0.00 / 0.00 |
| | | trace fixed at 1/3 of both columns | 0.00 / 0.33 / 0.67 | 0.67 / 0.33 / 0.00 |
| | | ↳ larger of metric / log, share of the learned part (1 = all on one) | 1.000 | 0.999 |
| | | trace fixed at TVDiag's role (RCL 0.5, FTI 0) | 0.22 / 0.50 / 0.28 | 0.66 / 0.00 / 0.34 |
| | | ↳ larger of metric / log, share of the learned part (1 = all on one) | 0.832 | 0.736 |

## Full tables

**A · GAIA, σ_to = 0.1**

| weighting | seeds | HR@1 | HR@3 | MRR@3 | F1 | macro-F1 |
|---|---|---|---|---|---|---|
| TVDiag default (σ = 0.1) | 3 | 0.726 ± 0.022 | 0.889 ± 0.013 | 0.800 ± 0.018 | 0.924 ± 0.014 | 0.573 ± 0.006 |
| AdaDiag (current) | 3 | 0.740 ± 0.012 | 0.895 ± 0.011 | 0.810 ± 0.010 | 0.932 ± 0.015 | 0.564 ± 0.020 |
| **trace fixed at 1/3 of both columns** | 3 | 0.736 ± 0.010 | 0.899 ± 0.013 | 0.810 ± 0.011 | 0.932 ± 0.013 | 0.572 ± 0.013 |
| **trace fixed at TVDiag's role (RCL 0.5, FTI 0)** | 3 | 0.739 ± 0.011 | 0.896 ± 0.008 | 0.810 ± 0.009 | 0.936 ± 0.011 | 0.572 ± 0.009 |

| difference | HR@1 | HR@3 | MRR@3 | F1 | macro-F1 |
|---|---|---|---|---|---|
| trace_third − TVDiag default | +0.010 (p 0.29, Holm 1.00) | +0.010 (p 0.01, Holm 0.30) | +0.010 (p 0.12, Holm 1.00) | +0.007 (p 0.20, Holm 1.00) | -0.001 (p 0.86, Holm 1.00) |
| trace_third − AdaDiag | -0.004 (p 0.38, Holm 1.00) | +0.004 (p 0.56, Holm 1.00) | +0.000 (p 0.96, Holm 1.00) | -0.000 (p 0.86, Holm 1.00) | +0.007 (p 0.25, Holm 1.00) |
| trace_tvdiag − TVDiag default | +0.013 (p 0.25, Holm 1.00) | +0.006 (p 0.27, Holm 1.00) | +0.010 (p 0.27, Holm 1.00) | +0.011 (p 0.05, Holm 1.00) | -0.000 (p 0.98, Holm 1.00) |
| trace_tvdiag − AdaDiag | -0.000 (p 0.96, Holm 1.00) | +0.001 (p 0.93, Holm 1.00) | -0.000 (p 0.99, Holm 1.00) | +0.004 (p 0.21, Holm 1.00) | +0.008 (p 0.55, Holm 1.00) |

**D · Hotel, σ_to = 0.1**

| weighting | seeds | HR@1 | HR@3 | MRR@3 | F1 | macro-F1 |
|---|---|---|---|---|---|---|
| TVDiag default (σ = 0.1) | 5 | 0.833 ± 0.013 | 0.918 ± 0.015 | 0.871 ± 0.012 | 0.798 ± 0.018 | 0.784 ± 0.027 |
| AdaDiag (current) | 5 | 0.828 ± 0.015 | 0.921 ± 0.011 | 0.868 ± 0.011 | 0.781 ± 0.034 | 0.765 ± 0.043 |
| **trace fixed at 1/3 of both columns** | 5 | 0.836 ± 0.017 | 0.915 ± 0.007 | 0.870 ± 0.012 | 0.783 ± 0.028 | 0.768 ± 0.037 |
| **trace fixed at TVDiag's role (RCL 0.5, FTI 0)** | 5 | 0.836 ± 0.017 | 0.918 ± 0.007 | 0.871 ± 0.012 | 0.789 ± 0.036 | 0.773 ± 0.043 |

| difference | HR@1 | HR@3 | MRR@3 | F1 | macro-F1 |
|---|---|---|---|---|---|
| trace_third − TVDiag default | +0.003 (p 0.75, Holm 1.00) | -0.003 (p 0.75, Holm 1.00) | -0.001 (p 0.83, Holm 1.00) | -0.015 (p 0.42, Holm 1.00) | -0.016 (p 0.53, Holm 1.00) |
| trace_third − AdaDiag | +0.008 (p 0.07, Holm 1.00) | -0.005 (p 0.18, Holm 1.00) | +0.002 (p 0.14, Holm 1.00) | +0.002 (p 0.56, Holm 1.00) | +0.002 (p 0.54, Holm 1.00) |
| trace_tvdiag − TVDiag default | +0.003 (p 0.75, Holm 1.00) | +0.000 (p 1.00, Holm 1.00) | -0.000 (p 0.94, Holm 1.00) | -0.009 (p 0.66, Holm 1.00) | -0.011 (p 0.69, Holm 1.00) |
| trace_tvdiag − AdaDiag | +0.008 (p 0.07, Holm 1.00) | -0.003 (p 0.37, Holm 1.00) | +0.003 (p 0.11, Holm 1.00) | +0.008 (p 0.12, Holm 1.00) | +0.008 (p 0.10, Holm 1.00) |

**D · Hotel, σ_to = 0.5**

| weighting | seeds | HR@1 | HR@3 | MRR@3 | F1 | macro-F1 |
|---|---|---|---|---|---|---|
| TVDiag default (σ = 0.1) | 5 | 0.833 ± 0.013 | 0.918 ± 0.015 | 0.871 ± 0.012 | 0.798 ± 0.018 | 0.784 ± 0.027 |
| TVDiag's rule at σ_to = 0.5 | 5 | 0.836 ± 0.006 | 0.915 ± 0.007 | 0.870 ± 0.005 | 0.728 ± 0.022 | 0.711 ± 0.018 |
| AdaDiag (current) | 5 | 0.823 ± 0.006 | 0.915 ± 0.007 | 0.863 ± 0.003 | 0.771 ± 0.023 | 0.762 ± 0.026 |
| **trace fixed at 1/3 of both columns** | 5 | 0.821 ± 0.020 | 0.928 ± 0.007 | 0.868 ± 0.011 | 0.770 ± 0.045 | 0.750 ± 0.057 |
| **trace fixed at TVDiag's role (RCL 0.5, FTI 0)** | 5 | 0.828 ± 0.015 | 0.905 ± 0.007 | 0.862 ± 0.007 | 0.765 ± 0.037 | 0.750 ± 0.042 |

| difference | HR@1 | HR@3 | MRR@3 | F1 | macro-F1 |
|---|---|---|---|---|---|
| trace_third − TVDiag default | -0.013 (p 0.35, Holm 1.00) | +0.010 (p 0.24, Holm 1.00) | -0.004 (p 0.69, Holm 1.00) | -0.028 (p 0.19, Holm 1.00) | -0.034 (p 0.22, Holm 1.00) |
| trace_third − AdaDiag | -0.003 (p 0.75, Holm 1.00) | +0.013 (p 0.03, Holm 1.00) | +0.005 (p 0.41, Holm 1.00) | -0.002 (p 0.90, Holm 1.00) | -0.012 (p 0.47, Holm 1.00) |
| trace_tvdiag − TVDiag default | -0.005 (p 0.59, Holm 1.00) | -0.013 (p 0.14, Holm 1.00) | -0.010 (p 0.16, Holm 1.00) | -0.032 (p 0.14, Holm 1.00) | -0.034 (p 0.15, Holm 1.00) |
| trace_tvdiag − AdaDiag | +0.005 (p 0.54, Holm 1.00) | -0.010 (p 0.10, Holm 1.00) | -0.001 (p 0.76, Holm 1.00) | -0.006 (p 0.59, Holm 1.00) | -0.012 (p 0.39, Holm 1.00) |

Paired t-test over seeds; Holm over all 60 comparisons. Bold: significant after correction.

Significant after correction: 0 of 60.
