# The TVDiag paper versus its released code (README §6.19)

Settings throughout: fresh seeds 5–9, 1 thread, 500-epoch cap, aligned dropout masks (`isolate_weighting_rng`), as in §6.17. "Code" is the released behaviour (every switch off: the committed runs, whose loop departs from TVDiag's in the three small ways of §6.11 -- notably, AUG draws a fresh copy of each batch's own incidents every step rather than from a pool of 10 per incident); "paper" is `config.PAPER` (switches 1–6 on). The LSTM aggregator (7) is left out here; §6.18 tests it.

## What the paper specifies and the released code does not do

| # | where in the paper | paper | released code | switch | status |
|---|---|---|---|---|---|
| 1 | Eq. 3 (TO) | each anchor's loss is the mean over its own positives (1/|Pᵢ|) | one mean over every positive pair in the batch (`SupConLoss`: sum / non-zero count) | `to_norm="paper"` | applied |
| 2 | Eqs. 5–6 (CM) | symmetric, 1/2n over both directions; other samples of the same modality are negatives too | one direction, metric → other; cross-modal negatives only (`UspConLoss`) | `cm_mode="paper"` | applied |
| 3 | Eq. 13 (DW) | θ₁…θ₄ are learnable | `AutomaticWeightedLoss` is built after the optimizer, so θ stays 1: equal weights 0.5 | `awl_learnable=True` | applied |
| 4 | Alg. 1, lines 6–7 | σ after every layer, then L2-normalise each node embedding | ReLU after layer 1 only, no normalisation (`SAGEEncoder`) | `sage_alg1=True` | applied |
| 5 | §4.2 | edges are bidirectional | edges used as given | `bidirectional=True` | applied (changes GAIA only) |
| 6 | §4.3 (AUG) | one augmented graph per training sample, added to the training set | 10 copies per sample in a pool; each batch draws random copies, mostly of other incidents | `aug_mode="paper"` | applied |
| 7 | §5.1 | LSTM aggregator | mean | `aggregator="lstm"` | applied in §6.18 |
| 8 | §4.1 | alert extraction: 3-sigma, IForest, Drain, log Rules 1–2 (k = 0.5), fastText | only the extracted 128-d embeddings are released | — | not possible: raw telemetry not released |
| 9 | §5.1 | dataset B (AIOps-22) | not released | — | not possible |

Everything else checked matches: Algorithm 2's labels (metric and trace → root cause, log → failure type), Eqs. 2 and 7–11, the metrics of Eqs. 14–16 and F1 = 2PR/(P+R), the dimensions 128/64/32, Adam (lr 1e-3, weight decay 1e-4), τ = 0.3, early stopping on the loss. One reading is ours: Eq. 3 literally SUMS over anchors, which would multiply the TO term by the batch size (256–1,024) against ω = 0.1; `to_norm="paper"` averages over anchors, matching the 1/2n of Eq. 5. AUG in paper mode takes 2 × batch-size graphs per step from the enlarged training set, as many as the code's batches hold.

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

## One paper feature at a time, in TVDiag (TVDiag + feature − TVDiag)

| paper feature | A · GAIA HR@1 | A · GAIA F1 | C · Sock Shop HR@1 | C · Sock Shop F1 | D · Hotel HR@1 | D · Hotel F1 |
|---|---|---|---|---|---|---|
| Eq. 3: TO averaged per anchor | +0.002 (p 0.41) | +0.000 (p 0.97) | +0.008 (p 0.27) | +0.013 (p 0.29) | -0.005 (p 0.37) | -0.003 (p 0.68) |
| Eqs. 5–6: CM symmetric + intra-modal negatives | +0.014 (p 0.09) | +0.004 (p 0.66) | +0.005 (p 0.28) | +0.016 (p 0.32) | -0.005 (p 0.67) | +0.016 (p 0.64) |
| Eq. 13: θ learnable | +0.007 (p 0.22) | +0.002 (p 0.75) | -0.005 (p 0.52) | **+0.043** (p 0.05) | +0.008 (p 0.43) | +0.031 (p 0.16) |
| Alg. 1: σ and L2-normalise every layer | **-0.088**† (p 0.00) | +0.005 (p 0.51) | **-0.040**† (p 0.00) | **-0.063** (p 0.01) | +0.005 (p 0.72) | -0.076 (p 0.07) |
| §4.2: bidirectional edges | -0.024 (p 0.06) | +0.004 (p 0.07) | identical (graphs already symmetric) | identical (graphs already symmetric) | identical (graphs already symmetric) | identical (graphs already symmetric) |
| §4.3: one AUG copy per incident | **-0.101** (p 0.01) | -0.023 (p 0.10) | -0.012 (p 0.26) | **-0.084** (p 0.04) | +0.010 (p 0.24) | -0.000 (p 1.00) |
| **all six together** | **-0.054** (p 0.01) | +0.004 (p 0.57) | **-0.059** (p 0.01) | **-0.138** (p 0.00) | +0.008 (p 0.43) | -0.076 (p 0.06) |
| Eq. 13, against TVDiag trained to the same 500 epochs | **+0.037** (p 0.00) | **+0.012** (p 0.03) | -0.014 (p 0.06) | +0.035 (p 0.10) | -0.010 (p 0.10) | +0.038 (p 0.08) |
| (control) TVDiag trained to 500 epochs, no other change | **-0.030** (p 0.00) | -0.010 (p 0.25) | +0.009 (p 0.27) | +0.009 (p 0.28) | +0.018 (p 0.11) | -0.007 (p 0.83) |

Mean epochs trained: A · GAIA: TVDiag 242, +to_norm 250, +cm_mode 259, +awl_learnable 428, +sage_alg1 164, +bidirectional 255, +aug_mode 463; C · Sock Shop: TVDiag 306, +to_norm 340, +cm_mode 226, +awl_learnable 368, +sage_alg1 166, +aug_mode 347; D · Hotel: TVDiag 131, +to_norm 131, +cm_mode 135, +awl_learnable 201, +sage_alg1 231, +aug_mode 155.

Same test; †: survives Holm across every cell of the six single-feature rows (HR@1, F1 and macro-F1; macro-F1 is in summary.json). Bidirectional edges change nothing on Sock Shop and Hotel: gaia: 19,462 of 38,614 edges one-way; sockshop: 0 of 52,142 edges one-way; hotel: 0 of 33,320 edges one-way.

## AdaDiag-DL with Eq. 13 only (θ learnable, every other switch off)

| dataset | arm | HR@1 | Avg@3 | MRR@3 | F1 | macro-F1 | epochs |
|---|---|---|---|---|---|---|---|
| A · GAIA | TVDiag, code | 0.738 ± 0.016 | 0.828 ± 0.008 | 0.809 ± 0.009 | 0.921 ± 0.017 | 0.573 ± 0.009 | 242 |
| A · GAIA | TVDiag + θ learnable | 0.745 ± 0.014 | 0.833 ± 0.007 | 0.815 ± 0.008 | 0.923 ± 0.011 | 0.575 ± 0.009 | 428 |
| A · GAIA | AdaDiag-DL, code | 0.749 ± 0.017 | 0.837 ± 0.010 | 0.818 ± 0.011 | 0.936 ± 0.006 | 0.554 ± 0.016 | 152 |
| A · GAIA | AdaDiag-DL + θ learnable | 0.749 ± 0.014 | 0.836 ± 0.005 | 0.818 ± 0.007 | 0.941 ± 0.009 | 0.569 ± 0.024 | 195 |
| C · Sock Shop | TVDiag, code | 0.869 ± 0.019 | 0.915 ± 0.016 | 0.905 ± 0.016 | 0.554 ± 0.018 | 0.535 ± 0.021 | 306 |
| C · Sock Shop | TVDiag + θ learnable | 0.864 ± 0.009 | 0.916 ± 0.005 | 0.905 ± 0.006 | 0.597 ± 0.037 | 0.573 ± 0.040 | 368 |
| C · Sock Shop | AdaDiag-DL, code | 0.872 ± 0.009 | 0.916 ± 0.004 | 0.906 ± 0.006 | 0.564 ± 0.028 | 0.543 ± 0.031 | 147 |
| C · Sock Shop | AdaDiag-DL + θ learnable | 0.877 ± 0.011 | 0.919 ± 0.008 | 0.910 ± 0.009 | 0.601 ± 0.018 | 0.572 ± 0.016 | 339 |
| D · Hotel | TVDiag, code | 0.818 ± 0.017 | 0.864 ± 0.006 | 0.856 ± 0.006 | 0.783 ± 0.057 | 0.772 ± 0.051 | 131 |
| D · Hotel | TVDiag + θ learnable | 0.826 ± 0.007 | 0.870 ± 0.006 | 0.862 ± 0.006 | 0.814 ± 0.038 | 0.798 ± 0.047 | 201 |
| D · Hotel | AdaDiag-DL, code | 0.810 ± 0.017 | 0.860 ± 0.010 | 0.852 ± 0.010 | 0.789 ± 0.047 | 0.777 ± 0.041 | 127 |
| D · Hotel | AdaDiag-DL + θ learnable | 0.826 ± 0.007 | 0.872 ± 0.010 | 0.864 ± 0.009 | 0.837 ± 0.021 | 0.829 ± 0.026 | 215 |

| dataset | metric | **AdaDiag-DL + θ − TVDiag + θ** | − TVDiag + θ stopped at the same epoch | AdaDiag-DL + θ − AdaDiag-DL (code) | AdaDiag-DL + θ − TVDiag (code) |
|---|---|---|---|---|---|
| A · GAIA | HR@1 | +0.004 (p 0.47) | -0.003 (p 0.59) | +0.000 (p 1.00) | +0.011 (p 0.26) |
| A · GAIA | MRR@3 | +0.002 (p 0.58) | -0.001 (p 0.82) | -0.001 (p 0.79) | +0.008 (p 0.21) |
| A · GAIA | F1 | +0.018 (p 0.07) | +0.006 (p 0.26) | **+0.005** (p 0.04) | **+0.020** (p 0.02) |
| A · GAIA | macro-F1 | -0.006 (p 0.68) | -0.007 (p 0.56) | **+0.015** (p 0.04) | -0.004 (p 0.76) |
| C · Sock Shop | HR@1 | **+0.013** (p 0.03) | +0.007 (p 0.21) | +0.005 (p 0.57) | +0.008 (p 0.50) |
| C · Sock Shop | MRR@3 | +0.005 (p 0.21) | +0.003 (p 0.46) | +0.004 (p 0.56) | +0.005 (p 0.59) |
| C · Sock Shop | F1 | +0.004 (p 0.84) | +0.017 (p 0.32) | +0.037 (p 0.10) | **+0.047** (p 0.02) |
| C · Sock Shop | macro-F1 | -0.001 (p 0.95) | +0.008 (p 0.71) | +0.029 (p 0.21) | **+0.037** (p 0.04) |
| D · Hotel | HR@1 | +0.000 (p 1.00) | -0.000 (p 1.00) | +0.015 (p 0.11) | +0.008 (p 0.30) |
| D · Hotel | MRR@3 | +0.002 (p 0.65) | +0.001 (p 0.81) | +0.012 (p 0.16) | +0.007 (p 0.23) |
| D · Hotel | F1 | +0.023 (p 0.39) | +0.028 (p 0.30) | +0.048 (p 0.17) | +0.054 (p 0.19) |
| D · Hotel | macro-F1 | +0.030 (p 0.37) | +0.035 (p 0.29) | +0.051 (p 0.13) | +0.057 (p 0.16) |

Paired t-test over seeds 5–9; **bold**: p < 0.05 uncorrected; †: survives Holm across the 12 cells of its column. The first column is the controlled comparison (both arms with θ learnable); the last mixes AdaDiag-DL's effect with θ's own.

## Eq. 13 with θ learnable: where the four weights end (0.5 / θ², mean over seeds)

| dataset | run | RCL | FTI | ω·TO | ω·CM |
|---|---|---|---|---|---|
| A · GAIA | TVDiag + θ learnable | 1.39 | 1.68 | 0.42 | 0.48 |
| A · GAIA | AdaDiag-DL + θ learnable | 0.69 | 0.79 | 0.48 | 0.46 |
| A · GAIA | TVDiag, paper | 1.05 | 1.43 | 0.41 | 0.44 |
| A · GAIA | AdaDiag-DL, paper | 0.87 | 1.14 | 0.51 | 0.44 |
| C · Sock Shop | TVDiag + θ learnable | 3.76 | 4.23 | 0.38 | 0.50 |
| C · Sock Shop | AdaDiag-DL + θ learnable | 3.48 | 3.83 | 0.48 | 0.50 |
| C · Sock Shop | TVDiag, paper | 1.13 | 1.26 | 0.38 | 0.46 |
| C · Sock Shop | AdaDiag-DL, paper | 1.32 | 1.50 | 0.48 | 0.46 |
| D · Hotel | TVDiag + θ learnable | 0.67 | 0.73 | 0.39 | 0.48 |
| D · Hotel | AdaDiag-DL + θ learnable | 0.69 | 0.76 | 0.49 | 0.48 |
| D · Hotel | TVDiag, paper | 1.04 | 1.45 | 0.37 | 0.44 |
| D · Hotel | AdaDiag-DL, paper | 1.18 | 1.73 | 0.49 | 0.44 |

The code's inert version leaves all four at 0.50.

## Distance from TVDiag's published numbers (re-run mean − published)

| dataset | metric | published | TVDiag, code | TVDiag, paper |
|---|---|---|---|---|
| A · GAIA | HR@1 | 0.759 | -0.021 (p 0.04) | -0.075 (p 0.00) |
| A · GAIA | Avg@3 | 0.844 | -0.016 (p 0.01) | -0.053 (p 0.00) |
| A · GAIA | MRR@3 | 0.827 | -0.018 (p 0.01) | -0.058 (p 0.00) |
| A · GAIA | F1 | 0.936 | -0.015 (p 0.11) | -0.011 (p 0.03) |
| C · Sock Shop | HR@1 | 0.863 | +0.006 (p 0.53) | -0.053 (p 0.00) |
| C · Sock Shop | Avg@3 | 0.913 | +0.002 (p 0.84) | -0.042 (p 0.00) |
| C · Sock Shop | MRR@3 | 0.903 | +0.002 (p 0.81) | -0.045 (p 0.00) |
| C · Sock Shop | F1 | 0.636 | -0.082 (p 0.00) | -0.220 (p 0.00) |
| D · Hotel | HR@1 | 0.821 | -0.003 (p 0.70) | +0.005 (p 0.58) |
| D · Hotel | Avg@3 | 0.876 | -0.012 (p 0.01) | +0.009 (p 0.16) |
| D · Hotel | MRR@3 | 0.866 | -0.010 (p 0.02) | +0.006 (p 0.33) |
| D · Hotel | F1 | 0.822 | -0.039 (p 0.20) | -0.115 (p 0.00) |

One-sample t-test of the five seeds against the published value.
