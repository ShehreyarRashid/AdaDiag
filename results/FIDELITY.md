# The softened prior, and three fixes to the training loop — README §6.11

Produced by `scripts/fidelity_runs.sh` and `scripts/make_fidelity_report.py`.

*Softened prior* = TVDiag's rule with 0.9 where it has 1 and 0.1 where it has 0 (RCL column metric / trace / log = .9 / .9 / .1; FTI = .1 / .1 / .9), as proposed in review. The *fidelity fixes* (`config.FIDELITY`) make the training loop match TVDiag's released code: self-loops on nodes with no in-edge, the augmented half of each batch drawn from a pre-built pool, the last epoch's model evaluated, a 500-epoch cap.

### 1. The committed settings (2 threads; the other three arms are the committed runs)

**A · GAIA**

| arm | seeds | HR@1 | HR@3 | MRR@3 | F1 | macro-F1 |
|---|---|---|---|---|---|---|
| TVDiag as shipped (1 / 1 / 0, 0 / 0 / 1) | 5 | 0.731 ± 0.021 | 0.892 ± 0.011 | 0.804 ± 0.015 | 0.918 ± 0.013 | 0.571 ± 0.005 |
| **softened prior (.9 / .9 / .1, .1 / .1 / .9)** | 5 | 0.729 ± 0.019 | 0.891 ± 0.013 | 0.803 ± 0.013 | 0.915 ± 0.013 | 0.576 ± 0.022 |
| TVDiag rule on the simplex (.5 / .5 / 0, 0 / 0 / 1) | 5 | 0.744 ± 0.010 | 0.897 ± 0.009 | 0.814 ± 0.008 | 0.932 ± 0.009 | 0.573 ± 0.006 |
| AdaDiag (learned) | 5 | 0.747 ± 0.013 | 0.896 ± 0.007 | 0.814 ± 0.008 | 0.935 ± 0.012 | 0.576 ± 0.024 |
| *softened − v0_raw* | | -0.002 (Holm 1.00) | -0.000 (Holm 1.00) | -0.001 (Holm 1.00) | -0.004 (Holm 1.00) | +0.005 (Holm 1.00) |
| *softened − v0_simplex* | | -0.015 (Holm 1.00) | -0.006 (Holm 1.00) | -0.010 (Holm 1.00) | -0.017 (Holm 0.95) | +0.004 (Holm 1.00) |
| *softened − v3_incident* | | -0.018 (Holm 1.00) | -0.005 (Holm 1.00) | -0.011 (Holm 1.00) | -0.020 (Holm 1.00) | +0.000 (Holm 1.00) |

**C · Sock Shop**

| arm | seeds | HR@1 | HR@3 | MRR@3 | F1 | macro-F1 |
|---|---|---|---|---|---|---|
| TVDiag as shipped (1 / 1 / 0, 0 / 0 / 1) | 5 | 0.868 ± 0.012 | 0.948 ± 0.006 | 0.904 ± 0.007 | 0.572 ± 0.014 | 0.550 ± 0.009 |
| **softened prior (.9 / .9 / .1, .1 / .1 / .9)** | 5 | 0.872 ± 0.016 | 0.953 ± 0.004 | 0.909 ± 0.010 | 0.583 ± 0.019 | 0.563 ± 0.018 |
| TVDiag rule on the simplex (.5 / .5 / 0, 0 / 0 / 1) | 5 | 0.872 ± 0.009 | 0.949 ± 0.005 | 0.907 ± 0.006 | 0.580 ± 0.013 | 0.556 ± 0.018 |
| AdaDiag (learned) | 5 | 0.871 ± 0.005 | 0.949 ± 0.008 | 0.907 ± 0.003 | 0.575 ± 0.024 | 0.552 ± 0.022 |
| *softened − v0_raw* | | +0.004 (Holm 1.00) | +0.005 (Holm 1.00) | +0.005 (Holm 1.00) | +0.011 (Holm 1.00) | +0.013 (Holm 1.00) |
| *softened − v0_simplex* | | +0.000 (Holm 1.00) | +0.003 (Holm 1.00) | +0.002 (Holm 1.00) | +0.003 (Holm 1.00) | +0.007 (Holm 1.00) |
| *softened − v3_incident* | | +0.001 (Holm 1.00) | +0.004 (Holm 1.00) | +0.002 (Holm 1.00) | +0.009 (Holm 1.00) | +0.011 (Holm 1.00) |

**D · Hotel**

| arm | seeds | HR@1 | HR@3 | MRR@3 | F1 | macro-F1 |
|---|---|---|---|---|---|---|
| TVDiag as shipped (1 / 1 / 0, 0 / 0 / 1) | 5 | 0.836 ± 0.011 | 0.921 ± 0.011 | 0.874 ± 0.010 | 0.796 ± 0.019 | 0.782 ± 0.027 |
| **softened prior (.9 / .9 / .1, .1 / .1 / .9)** | 5 | 0.838 ± 0.007 | 0.923 ± 0.013 | 0.876 ± 0.006 | 0.788 ± 0.011 | 0.773 ± 0.019 |
| TVDiag rule on the simplex (.5 / .5 / 0, 0 / 0 / 1) | 5 | 0.833 ± 0.013 | 0.918 ± 0.007 | 0.871 ± 0.008 | 0.804 ± 0.015 | 0.791 ± 0.022 |
| AdaDiag (learned) | 5 | 0.828 ± 0.015 | 0.923 ± 0.009 | 0.869 ± 0.012 | 0.781 ± 0.032 | 0.766 ± 0.042 |
| *softened − v0_raw* | | +0.003 (Holm 1.00) | +0.003 (Holm 1.00) | +0.002 (Holm 1.00) | -0.007 (Holm 1.00) | -0.009 (Holm 1.00) |
| *softened − v0_simplex* | | +0.005 (Holm 1.00) | +0.005 (Holm 1.00) | +0.005 (Holm 1.00) | -0.016 (Holm 1.00) | -0.019 (Holm 1.00) |
| *softened − v3_incident* | | +0.010 (Holm 1.00) | -0.000 (Holm 1.00) | +0.007 (Holm 1.00) | +0.007 (Holm 1.00) | +0.007 (Holm 1.00) |

Bold: significant after Holm correction over every comparison in this table (3 datasets × 3 references × 5 metrics). Paired t-test over seeds.

### 2. With the three fidelity fixes (1 thread, every arm re-run)

**A · GAIA**

| arm | seeds | HR@1 | HR@3 | MRR@3 | F1 | macro-F1 |
|---|---|---|---|---|---|---|
| TVDiag as shipped (1 / 1 / 0, 0 / 0 / 1) | 5 | 0.735 ± 0.021 | 0.895 ± 0.013 | 0.808 ± 0.015 | 0.928 ± 0.009 | 0.556 ± 0.053 |
| **softened prior (.9 / .9 / .1, .1 / .1 / .9)** | 5 | 0.734 ± 0.018 | 0.896 ± 0.012 | 0.807 ± 0.014 | 0.924 ± 0.009 | 0.551 ± 0.048 |
| TVDiag rule on the simplex (.5 / .5 / 0, 0 / 0 / 1) | 5 | 0.736 ± 0.024 | 0.900 ± 0.010 | 0.811 ± 0.016 | 0.934 ± 0.008 | 0.558 ± 0.049 |
| AdaDiag (learned) | 5 | 0.740 ± 0.015 | 0.898 ± 0.009 | 0.812 ± 0.011 | 0.935 ± 0.008 | 0.566 ± 0.054 |
| *softened − v0_raw* | | -0.001 (Holm 1.00) | +0.001 (Holm 1.00) | -0.000 (Holm 1.00) | -0.004 (Holm 1.00) | -0.005 (Holm 1.00) |
| *softened − v0_simplex* | | -0.003 (Holm 1.00) | -0.004 (Holm 1.00) | -0.003 (Holm 1.00) | -0.010 (Holm 1.00) | -0.007 (Holm 1.00) |
| *softened − v3_incident* | | -0.006 (Holm 1.00) | -0.002 (Holm 1.00) | -0.004 (Holm 1.00) | -0.011 (Holm 1.00) | -0.015 (Holm 1.00) |

**C · Sock Shop**

| arm | seeds | HR@1 | HR@3 | MRR@3 | F1 | macro-F1 |
|---|---|---|---|---|---|---|
| TVDiag as shipped (1 / 1 / 0, 0 / 0 / 1) | 5 | 0.867 ± 0.015 | 0.944 ± 0.008 | 0.903 ± 0.009 | 0.574 ± 0.006 | 0.555 ± 0.009 |
| **softened prior (.9 / .9 / .1, .1 / .1 / .9)** | 5 | 0.870 ± 0.012 | 0.944 ± 0.004 | 0.903 ± 0.007 | 0.584 ± 0.019 | 0.567 ± 0.017 |
| TVDiag rule on the simplex (.5 / .5 / 0, 0 / 0 / 1) | 5 | 0.874 ± 0.010 | 0.945 ± 0.006 | 0.906 ± 0.006 | 0.580 ± 0.012 | 0.558 ± 0.013 |
| AdaDiag (learned) | 5 | 0.879 ± 0.009 | 0.948 ± 0.004 | 0.910 ± 0.007 | 0.586 ± 0.019 | 0.566 ± 0.021 |
| *softened − v0_raw* | | +0.002 (Holm 1.00) | -0.001 (Holm 1.00) | +0.001 (Holm 1.00) | +0.010 (Holm 1.00) | +0.012 (Holm 1.00) |
| *softened − v0_simplex* | | -0.004 (Holm 1.00) | -0.002 (Holm 1.00) | -0.003 (Holm 1.00) | +0.004 (Holm 1.00) | +0.009 (Holm 1.00) |
| *softened − v3_incident* | | -0.009 (Holm 1.00) | -0.004 (Holm 1.00) | -0.007 (Holm 0.53) | -0.002 (Holm 1.00) | +0.001 (Holm 1.00) |

**D · Hotel**

| arm | seeds | HR@1 | HR@3 | MRR@3 | F1 | macro-F1 |
|---|---|---|---|---|---|---|
| TVDiag as shipped (1 / 1 / 0, 0 / 0 / 1) | 5 | 0.821 ± 0.020 | 0.921 ± 0.006 | 0.864 ± 0.014 | 0.785 ± 0.041 | 0.768 ± 0.048 |
| **softened prior (.9 / .9 / .1, .1 / .1 / .9)** | 5 | 0.821 ± 0.020 | 0.918 ± 0.011 | 0.863 ± 0.016 | 0.793 ± 0.045 | 0.779 ± 0.051 |
| TVDiag rule on the simplex (.5 / .5 / 0, 0 / 0 / 1) | 5 | 0.821 ± 0.020 | 0.923 ± 0.000 | 0.865 ± 0.013 | 0.780 ± 0.037 | 0.766 ± 0.043 |
| AdaDiag (learned) | 5 | 0.841 ± 0.011 | 0.921 ± 0.011 | 0.875 ± 0.010 | 0.803 ± 0.022 | 0.792 ± 0.021 |
| *softened − v0_raw* | | +0.000 (Holm 1.00) | -0.003 (Holm 1.00) | -0.001 (Holm 1.00) | +0.008 (Holm 1.00) | +0.010 (Holm 1.00) |
| *softened − v0_simplex* | | +0.000 (Holm 1.00) | -0.005 (Holm 1.00) | -0.002 (Holm 1.00) | +0.012 (Holm 1.00) | +0.013 (Holm 1.00) |
| *softened − v3_incident* | | -0.021 (Holm 1.00) | -0.003 (Holm 1.00) | -0.012 (Holm 1.00) | -0.011 (Holm 1.00) | -0.013 (Holm 1.00) |

Bold: significant after Holm correction over every comparison in this table (3 datasets × 3 references × 5 metrics). Paired t-test over seeds.

### 2b. AdaDiag against the fixed rules, with the fixes

| dataset | comparison | HR@1 | HR@3 | MRR@3 | F1 | macro-F1 |
|---|---|---|---|---|---|---|
| A · GAIA | AdaDiag − v0_raw | +0.005 (p 0.436, Holm 1.00) | +0.003 (p 0.315, Holm 1.00) | +0.004 (p 0.295, Holm 1.00) | +0.007 (p 0.119, Holm 1.00) | +0.010 (p 0.520, Holm 1.00) |
| A · GAIA | AdaDiag − v0_simplex | +0.003 (p 0.532, Holm 1.00) | -0.002 (p 0.338, Holm 1.00) | +0.001 (p 0.718, Holm 1.00) | +0.001 (p 0.403, Holm 1.00) | +0.008 (p 0.241, Holm 1.00) |
| C · Sock Shop | AdaDiag − v0_raw | +0.012 (p 0.031, Holm 0.94) | +0.003 (p 0.374, Holm 1.00) | +0.007 (p 0.041, Holm 1.00) | +0.012 (p 0.216, Holm 1.00) | +0.010 (p 0.299, Holm 1.00) |
| C · Sock Shop | AdaDiag − v0_simplex | +0.005 (p 0.284, Holm 1.00) | +0.002 (p 0.426, Holm 1.00) | +0.004 (p 0.164, Holm 1.00) | +0.006 (p 0.604, Holm 1.00) | +0.008 (p 0.499, Holm 1.00) |
| D · Hotel | AdaDiag − v0_raw | +0.021 (p 0.120, Holm 1.00) | -0.000 (p 1.000, Holm 1.00) | +0.011 (p 0.252, Holm 1.00) | +0.019 (p 0.441, Holm 1.00) | +0.023 (p 0.405, Holm 1.00) |
| D · Hotel | AdaDiag − v0_simplex | +0.021 (p 0.120, Holm 1.00) | -0.003 (p 0.621, Holm 1.00) | +0.010 (p 0.259, Holm 1.00) | +0.023 (p 0.348, Holm 1.00) | +0.026 (p 0.354, Holm 1.00) |

### 3. Does the fixed loop move the re-run closer to the paper?

| dataset | paper HR@1 / F1 | committed re-run (gap) | with the fixes (gap) |
|---|---|---|---|
| A · GAIA | 0.759 / 0.936 | 0.731 / 0.918 (-0.028 / -0.018) | 0.735 / 0.928 (-0.024 / -0.008) |
| C · Sock Shop | 0.863 / 0.636 | 0.868 / 0.572 (+0.005 / -0.064) | 0.867 / 0.574 (+0.004 / -0.062) |
| D · Hotel | 0.821 / 0.822 | 0.836 / 0.796 (+0.015 / -0.026) | 0.821 / 0.785 (-0.000 / -0.037) |

### 4. AdaDiag's learned matrix, before and after the fixes (5-seed means)

| dataset | committed: RCL m / t / l | committed: FTI m / t / l | fidelity: RCL m / t / l | fidelity: FTI m / t / l |
|---|---|---|---|---|
| A · GAIA | 0.01 / 0.00 / 0.99 | 0.00 / 0.00 / 1.00 | 0.04 / 0.00 / 0.96 | 0.01 / 0.00 / 0.99 |
| C · Sock Shop | 1.00 / 0.00 / 0.00 | 0.23 / 0.00 / 0.77 | 1.00 / 0.00 / 0.00 | 0.08 / 0.00 / 0.92 |
| D · Hotel | 1.00 / 0.00 / 0.00 | 0.25 / 0.00 / 0.75 | 1.00 / 0.00 / 0.00 | 0.32 / 0.00 / 0.68 |
