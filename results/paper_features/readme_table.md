## Values (mean ± sd over fresh seeds 5–9)

| dataset | arm | HR@1 | Avg@3 | MRR@3 | F1 | macro-F1 | epochs |
|---|---|---|---|---|---|---|---|
| A · GAIA | TVDiag, published | 0.759 | 0.844 | 0.827 | 0.936 | — | — |
| A · GAIA | TVDiag, code | 0.738 ± 0.016 | 0.828 ± 0.008 | 0.809 ± 0.009 | 0.921 ± 0.017 | 0.573 ± 0.009 | 242 |
| A · GAIA | TVDiag, code, no early stop | 0.708 ± 0.012 | 0.810 ± 0.008 | 0.789 ± 0.007 | 0.911 ± 0.011 | 0.572 ± 0.007 | 500 |
| A · GAIA | TVDiag, paper | 0.684 ± 0.014 | 0.791 ± 0.019 | 0.769 ± 0.017 | 0.925 ± 0.007 | 0.516 ± 0.028 | 375 |
| A · GAIA | AdaDiag-DL, code | 0.749 ± 0.017 | 0.837 ± 0.010 | 0.818 ± 0.011 | 0.936 ± 0.006 | 0.554 ± 0.016 | 152 |
| A · GAIA | AdaDiag-DL, paper | 0.679 ± 0.016 | 0.787 ± 0.015 | 0.765 ± 0.015 | 0.921 ± 0.003 | 0.477 ± 0.019 | 298 |
| C · Sock Shop | TVDiag, published | 0.863 | 0.913 | 0.903 | 0.636 | — | — |
| C · Sock Shop | TVDiag, code | 0.869 ± 0.019 | 0.915 ± 0.016 | 0.905 ± 0.016 | 0.554 ± 0.018 | 0.535 ± 0.021 | 306 |
| C · Sock Shop | TVDiag, code, no early stop | 0.878 ± 0.012 | 0.918 ± 0.004 | 0.910 ± 0.006 | 0.562 ± 0.033 | 0.544 ± 0.034 | 500 |
| C · Sock Shop | TVDiag, paper | 0.810 ± 0.014 | 0.871 ± 0.012 | 0.858 ± 0.012 | 0.416 ± 0.018 | 0.393 ± 0.017 | 226 |
| C · Sock Shop | AdaDiag-DL, code | 0.872 ± 0.009 | 0.916 ± 0.004 | 0.906 ± 0.006 | 0.564 ± 0.028 | 0.543 ± 0.031 | 147 |
| C · Sock Shop | AdaDiag-DL, paper | 0.812 ± 0.017 | 0.874 ± 0.013 | 0.860 ± 0.013 | 0.437 ± 0.039 | 0.412 ± 0.036 | 244 |
| D · Hotel | TVDiag, published | 0.821 | 0.876 | 0.866 | 0.822 | — | — |
| D · Hotel | TVDiag, code | 0.818 ± 0.017 | 0.864 ± 0.006 | 0.856 ± 0.006 | 0.783 ± 0.057 | 0.772 ± 0.051 | 131 |
| D · Hotel | TVDiag, code, no early stop | 0.836 ± 0.006 | 0.877 ± 0.007 | 0.869 ± 0.004 | 0.776 ± 0.036 | 0.759 ± 0.048 | 500 |
| D · Hotel | TVDiag, paper | 0.826 ± 0.017 | 0.885 ± 0.012 | 0.872 ± 0.012 | 0.707 ± 0.034 | 0.703 ± 0.027 | 411 |
| D · Hotel | AdaDiag-DL, code | 0.810 ± 0.017 | 0.860 ± 0.010 | 0.852 ± 0.010 | 0.789 ± 0.047 | 0.777 ± 0.041 | 127 |
| D · Hotel | AdaDiag-DL, paper | 0.826 ± 0.023 | 0.880 ± 0.016 | 0.868 ± 0.017 | 0.717 ± 0.020 | 0.722 ± 0.018 | 460 |

## Paired differences (same seeds 5–9)

| dataset | metric | TVDiag: paper − code | TVDiag: paper − code trained to 500 epochs | AdaDiag-DL: paper − code | **paper versions: AdaDiag-DL − TVDiag** | − TVDiag-paper stopped at the same epoch | code versions: AdaDiag-DL − TVDiag (§6.17) |
|---|---|---|---|---|---|---|---|
| A · GAIA | HR@1 | **-0.054** (p 0.01) | -0.024 (p 0.06) | **-0.071**† (p 0.00) | -0.006 (p 0.63) | -0.006 (p 0.61) | +0.011 (p 0.40) |
| A · GAIA | MRR@3 | **-0.040** (p 0.01) | -0.021 (p 0.08) | **-0.054**† (p 0.01) | -0.004 (p 0.78) | -0.009 (p 0.50) | +0.009 (p 0.32) |
| A · GAIA | F1 | +0.004 (p 0.57) | **+0.014**† (p 0.00) | **-0.015**† (p 0.00) | -0.004 (p 0.29) | -0.004 (p 0.27) | +0.015 (p 0.06) |
| A · GAIA | macro-F1 | **-0.057** (p 0.01) | **-0.056**† (p 0.01) | **-0.078**† (p 0.00) | **-0.040** (p 0.05) | -0.019 (p 0.05) | -0.019 (p 0.05) |
| C · Sock Shop | HR@1 | **-0.059** (p 0.01) | **-0.068**† (p 0.00) | **-0.061**† (p 0.00) | +0.002 (p 0.86) | +0.002 (p 0.74) | +0.003 (p 0.72) |
| C · Sock Shop | MRR@3 | **-0.046** (p 0.01) | **-0.051**† (p 0.00) | **-0.046**† (p 0.00) | +0.002 (p 0.84) | +0.003 (p 0.67) | +0.002 (p 0.83) |
| C · Sock Shop | F1 | **-0.138**† (p 0.00) | **-0.147**† (p 0.00) | **-0.127**† (p 0.01) | +0.021 (p 0.29) | +0.027 (p 0.21) | +0.011 (p 0.62) |
| C · Sock Shop | macro-F1 | **-0.142**† (p 0.00) | **-0.151**† (p 0.00) | **-0.131**† (p 0.01) | +0.020 (p 0.32) | +0.024 (p 0.24) | +0.009 (p 0.69) |
| D · Hotel | HR@1 | +0.008 (p 0.43) | -0.010 (p 0.34) | +0.015 (p 0.07) | -0.000 (p 1.00) | -0.003 (p 0.75) | -0.008 (p 0.37) |
| D · Hotel | MRR@3 | +0.015 (p 0.06) | +0.003 (p 0.62) | +0.017 (p 0.08) | -0.003 (p 0.62) | -0.003 (p 0.62) | -0.005 (p 0.24) |
| D · Hotel | F1 | -0.076 (p 0.06) | **-0.069**† (p 0.01) | -0.072 (p 0.06) | +0.010 (p 0.47) | +0.009 (p 0.56) | +0.006 (p 0.40) |
| D · Hotel | macro-F1 | **-0.069** (p 0.03) | **-0.057** (p 0.04) | -0.055 (p 0.09) | +0.020 (p 0.22) | +0.019 (p 0.27) | +0.005 (p 0.57) |

Paired t-test over the seeds; **bold**: p < 0.05 uncorrected; †: survives Holm across the 12 cells of its column (3 datasets × 4 metrics). [n]: fewer than 5 seeds. "Trained to 500 epochs": the code's TVDiag with early stopping off (`tvdiag_noes`); with θ learnable the total loss keeps falling for longer, so the paper runs train for 336 (range 126–500) epochs on average, the code's for 184 (range 89–419). "Stopped at the same epoch": TVDiag-paper re-trained with each seed's cap set to the epoch AdaDiag-DL-paper stopped at (the §6.13 control); not run where both reach the cap.
