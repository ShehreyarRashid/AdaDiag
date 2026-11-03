| method | A · GAIA HR@1 / F1 | C · Sock Shop HR@1 / F1 | D · Hotel HR@1 / F1 |
|---|---|---|---|
| **TVDiag (paper)** | **0.759 / 0.936** | **0.863 / 0.636** | **0.821 / 0.822** |
| AdaDiag | -0.019 / -0.004 | +0.008 † / -0.061 † | +0.007 / -0.041 |
| AdaDiag, random stream aligned (§6.15) | -0.019 / -0.004 | +0.008 † / -0.061 † | +0.015 † / -0.012 † |
| AdaDiag, TVDiag-fidelity loop (§6.11) | -0.019 † / -0.001 | +0.016 † / -0.050 † | +0.020 † / -0.019 |
| AdaDiag, trace fixed at 1/3 (§6.14) | -0.023 / -0.004 | — | +0.015 / -0.039 † |
| AdaDiag, trace fixed at TVDiag's role (§6.14) | -0.020 / -0.000 | — | +0.015 / -0.033 |
| Softened prior .9/.9/.1, .1/.1/.9 (§6.11) | -0.030 † / -0.021 † | +0.009 / -0.053 † | +0.017 † / -0.034 † |
| 1 · weight positive pairs, β = 1 (§6.13) | -0.030 / -0.011 | — | +0.015 † / -0.012 |
| 1 · weight positive pairs, β = 1/τ | -0.028 † / -0.009 | — | +0.012 † / -0.027 † |
| 2 · weight negative samples, β = 1 | -0.034 / -0.011 | — | +0.015 † / -0.022 |
| 2 · weight negative samples, β = 1/τ | -0.031 / -0.003 | — | +0.017 † / -0.030 † |
| 3 · weight by sample importance | -0.008 / -0.017 † | — | +0.017 / -0.021 |
| 4 · weights from the diagnosis losses | -0.014 / -0.002 | — | +0.007 / -0.018 |
| 5 · curriculum (easy → hard) | -0.033 / -0.008 | — | +0.012 / -0.015 † |
| Oracle, one matrix (§6.12, fitted on test) | -0.022 / -0.004 | — | +0.007 / -0.027 † |
| Oracle, per failure type (fitted on test) | -0.021 † / -0.005 | — | +0.007 / -0.022 |
