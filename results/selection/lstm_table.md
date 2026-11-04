## Values (mean over fresh seeds 5–9)

| dataset | TVDiag, mean agg. | TVDiag-LSTM | AdaDiag-DL, mean agg. | AdaDiag-DL-LSTM | epochs (TVDiag-LSTM / AdaDiag-DL-LSTM) |
|---|---|---|---|---|---|
| A · GAIA | 0.738 / 0.921 | 0.714 / 0.944 | 0.749 / 0.936 | 0.742 / 0.934 | 195 / 118 |
| C · Sock Shop | 0.869 / 0.554 | 0.851 / 0.510 | 0.872 / 0.564 | 0.829 / 0.567 | 199 / 136 |
| D · Hotel | 0.818 / 0.783 | 0.818 / 0.757 | 0.810 / 0.789 | 0.821 / 0.783 | 140 / 137 |

HR@1 / F1 in each cell.

## Paired differences (same seeds)

| dataset | metric | LSTM vs mean, TVDiag | **AdaDiag-DL-LSTM − TVDiag-LSTM** | − TVDiag-LSTM stopped at the same epoch | AdaDiag-DL-LSTM − AdaDiag-DL (mean) |
|---|---|---|---|---|---|
| A · GAIA | HR@1 | -0.024 (p 0.23) | +0.028 (p 0.15) | **+0.014** (p 0.02) | -0.008 (p 0.43) |
| A · GAIA | F1 | +0.023 (p 0.05) | **-0.010** (p 0.02) | **-0.004** (p 0.02) | -0.002 (p 0.75) |
| A · GAIA | macro-F1 | +0.004 (p 0.57) | **-0.051** (p 0.01) | -0.015 (p 0.48) | -0.028 (p 0.21) |
| C · Sock Shop | HR@1 | -0.018 (p 0.11) | -0.022 (p 0.08) | **-0.016** (p 0.04) | **-0.043** (p 0.01) |
| C · Sock Shop | F1 | **-0.043** (p 0.01) | **+0.057** (p 0.01) | +0.040 (p 0.11) | +0.003 (p 0.86) |
| C · Sock Shop | macro-F1 | **-0.042** (p 0.03) | **+0.053** (p 0.02) | +0.039 (p 0.13) | +0.003 (p 0.85) |
| D · Hotel | HR@1 | +0.000 (p 1.00) | +0.003 (p 0.70) | — | +0.010 (p 0.37) |
| D · Hotel | F1 | -0.026 (p 0.39) | +0.027 (p 0.09) | — | -0.006 (p 0.79) |
| D · Hotel | macro-F1 | -0.023 (p 0.44) | +0.028 (p 0.05) | — | -0.000 (p 1.00) |

Paired t-test over seeds 5–9, uncorrected; bold: p < 0.05. [n]: fewer than 5 seeds. "Stopped at the same epoch": TVDiag-LSTM re-trained with each seed's cap set to the epoch AdaDiag-DL-LSTM stopped at (§6.13). On Hotel the two stop at the same epochs, so it is not run there.
