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
