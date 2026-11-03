| candidate | score | worst cell | GAIA HR@1 | GAIA F1 | Hotel HR@1 | Hotel F1 | seeds (GAIA / Hotel) | settings |
|---|---|---|---|---|---|---|---|---|
| AdaDiag (learned from the contrastive term) | **+0.0090** | +0.003 | +0.013 (p 0.21) | +0.007 (p 0.14) | +0.003 (p 0.37) | +0.012 (p 0.26) | 3 / 5 | 1 thread; Hotel: 1 thread, aligned |
| AdaDiag, trace fixed at TVDiag's role | **+0.0080** | +0.000 | +0.013 (p 0.25) | +0.011 (p 0.05) | +0.000 (p 1.00) | +0.007 (p 0.48) | 3 / 5 | 1 thread; Hotel: 1 thread, aligned |
| Type 4: weights from the diagnosis losses | **+0.0072** | -0.005 | +0.018 (p 0.10) | +0.009 (p 0.39) | -0.005 (p 0.48) | +0.006 (p 0.62) | 3 / 5 | 1 thread; Hotel: 1 thread |
| AdaDiag, trace fixed at 1/3 | **+0.0065** | +0.003 | +0.010 (p 0.29) | +0.007 (p 0.20) | +0.003 (p 0.37) | +0.006 (p 0.61) | 3 / 5 | 1 thread; Hotel: 1 thread, aligned |
| AdaDiag, TVDiag-fidelity loop | **+0.0054** | -0.003 | +0.005 (p 0.44) | +0.007 (p 0.12) | -0.003 (p 0.37) | +0.013 (p 0.06) | 5 / 5 | fidelity loop; Hotel: fidelity loop, aligned |
| AdaDiag + 100-epoch weight warm-up | **+0.0044** | +0.000 | +0.009 (p 0.34) | +0.008 (p 0.16) | +0.000 (p 1.00) | +0.001 (p 0.97) | 3 / 5 | 1 thread, aligned; Hotel: 1 thread, aligned |
| V2-global (one learned matrix) | **-0.0006** | -0.005 | +0.002 (p 0.86) | +0.003 (p 0.57) | -0.003 (p 0.37) | -0.005 (p 0.80) | 5 / 5 | 2 threads (§6.2–6.3); Hotel: 2 threads (§6.2–6.3) |

| finalist | reference | score (fresh seeds) | p (score ≠ 0) | Holm | GAIA HR@1 | GAIA F1 | Hotel HR@1 | Hotel F1 | significantly worse cells | macro-F1 GAIA / Hotel (not scored) |
|---|---|---|---|---|---|---|---|---|---|---|
| Type 4: weights from the diagnosis losses | TVDiag | **+0.0062** | 0.20 | 0.60 | +0.011 (p 0.40) | +0.015 (p 0.06) | -0.008 (p 0.37) | +0.006 (p 0.40) | none | -0.019 (p 0.05) / +0.005 (p 0.57) |
| AdaDiag, trace fixed at TVDiag's role | TVDiag | **+0.0051** | 0.10 | 0.42 | +0.008 (p 0.47) | +0.010 (p 0.06) | -0.005 (p 0.37) | +0.008 (p 0.36) | none | +0.005 (p 0.15) / +0.008 (p 0.39) |
| AdaDiag (learned from the contrastive term) | TVDiag | **+0.0047** | 0.35 | 0.70 | +0.005 (p 0.43) | +0.014 (p 0.08) | +0.000 (p 1.00) | -0.000 (p 0.98) | none | +0.012 (p 0.21) / +0.000 (p 0.98) |
| AdaDiag, trace fixed at 1/3 | TVDiag | **+0.0022** | 0.41 | 0.70 | +0.011 (p 0.07) | +0.009 (p 0.02) | -0.010 (p 0.18) | -0.001 (p 0.96) | none | +0.002 (p 0.63) / -0.000 (p 0.99) |
