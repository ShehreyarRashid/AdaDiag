# The TVDiag paper's Tables 8 and 9 (README §6.20)

## Table 8: inter-task affinity, Eq. 17 (mean over the steps of epoch 0, then over seeds 5–9)

| dataset | loop | Z_RCL→FTI | Z_FTI→RCL, Eq. 17 | Z_FTI→RCL, as TVDiag computes it | steps per epoch | paper (Z_RCL→FTI / Z_FTI→RCL) |
|---|---|---|---|---|---|---|
| A · GAIA | TVDiag code (§6.11) | 2.67×10^-5 ± 2.95×10^-3 (p 0.98) | -2.03×10^-4 ± 1.62×10^-3 (p 0.79) | 2.48×10^-2 ± 4.39×10^-3 (p 0.00023) | 1 | 9.12×10^-3 / 8.77×10^-2 |
| A · GAIA | committed | 2.47×10^-4 ± 2.60×10^-3 (p 0.84) | -1.77×10^-4 ± 1.55×10^-3 (p 0.81) | 2.37×10^-2 ± 3.49×10^-3 (p 0.00011) | 1 | 9.12×10^-3 / 8.77×10^-2 |
| C · Sock Shop | TVDiag code (§6.11) | -2.58×10^-4 ± 2.40×10^-4 (p 0.075) | -3.37×10^-4 ± 4.48×10^-4 (p 0.17) | 2.85×10^-2 ± 2.13×10^-3 (p 7.4e-06) | 2 | not reported |
| C · Sock Shop | committed | -3.98×10^-4 ± 3.57×10^-4 (p 0.067) | -5.07×10^-4 ± 4.45×10^-4 (p 0.063) | 2.93×10^-2 ± 1.89×10^-3 (p 4.1e-06) | 2 | not reported |
| D · Hotel | TVDiag code (§6.11) | 3.10×10^-4 ± 2.67×10^-3 (p 0.81) | 1.58×10^-3 ± 1.38×10^-3 (p 0.062) | 2.26×10^-2 ± 3.75×10^-3 (p 0.00018) | 1 | not reported |
| D · Hotel | committed | 1.29×10^-5 ± 2.57×10^-3 (p 0.99) | 4.32×10^-4 ± 1.06×10^-3 (p 0.41) | 2.05×10^-2 ± 2.47×10^-3 (p 5e-05) | 1 | not reported |

mean ± sd over the five seeds; p: one-sample t-test against 0. "As TVDiag computes it": `core/ita.py` steps the second copy on the FTI loss, then re-evaluates the FIRST copy, the one stepped on the RCL loss, so it measures the RCL loss's response to an RCL step. Dataset B is not released.

## Table 9: efficiency (1 CPU thread; the paper used a GPU)

| dataset | method | training per epoch (s) | epochs | training total (s) | model inference per case (ms) | parameters | of which the weighting module |
|---|---|---|---|---|---|---|---|
| A · GAIA | TVDiag | 0.495 | 242 | 120 | 1.18 | 74,246 | 0 |
| A · GAIA | AdaDiag-CL | 0.742 (1.5×) | 243 | 180 | 1.17 (0.99×) | 81,036 | 6,790 |
| A · GAIA | AdaDiag-DL | 1.321 (2.7×) | 152 | 200 | 1.07 (0.91×) | 74,246 | 0 |
| A · GAIA | *TVDiag, paper (GPU)* | 0.654 | — | — | 3,075 (incl. alert extraction) | — | — |
| C · Sock Shop | TVDiag | 0.661 | 306 | 203 | 1.16 | 74,376 | 0 |
| C · Sock Shop | AdaDiag-CL | 1.072 (1.6×) | 250 | 269 | 1.25 (1.08×) | 81,166 | 6,790 |
| C · Sock Shop | AdaDiag-DL | 1.795 (2.7×) | 147 | 265 | 1.21 (1.04×) | 74,376 | 0 |
| D · Hotel | TVDiag | 0.394 | 131 | 52 | 1.13 | 74,246 | 0 |
| D · Hotel | AdaDiag-CL | 0.549 (1.4×) | 136 | 75 | 1.18 (1.04×) | 81,036 | 6,790 |
| D · Hotel | AdaDiag-DL | 0.908 (2.3×) | 127 | 116 | 1.15 (1.01×) | 74,246 | 0 |

Training: the fresh-seed runs of §6.17 (seeds 5–9, 500-epoch cap, early stopping); per epoch = run time / epochs. Inference: the model's forward pass for one incident, averaged over the test set, median of 3 passes; the paper's online time also includes alert extraction (the release ships extracted features, so it cannot be re-measured). AdaDiag-CL's weighting module is not on the prediction path, and AdaDiag-DL has no trainable weighting parameters, so inference is the same network for all three.
