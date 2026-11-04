# Improving RCL and FTI: the rule and the screen (README §6.21)

Written 2026-11-04T07:29:22, before any confirmation run.

**Rule.** On the development seeds 0-2, each candidate's score is the mean over the three datasets of (dHR@1 + dF1) / 2, paired against TVDiag on the same seeds. A candidate is kept if its score is above +0.005 and no dataset loses more than 0.015 on HR@1 or F1. The kept candidates are combined (if both RCL-context variants are kept, rcl_context='both'). The combination is screened on the same seeds; the final change is the combination if its score is at least the best kept candidate's, otherwise the best kept candidate alone. If no candidate is kept, nothing is confirmed. The final change is then confirmed on fresh seeds 10-14 for TVDiag and AdaDiag-DL, against each as released.

| candidate | GAIA dHR@1 / dF1 | Sock Shop | Hotel | score | worst | kept |
|---|---|---|---|---|---|---|
| theta | +0.007 / -0.003 | +0.010 / +0.022 | -0.013 / -0.017 | +0.0011 | -0.017 | no |
| es_diag | +0.009 / +0.012 | +0.001 / -0.011 | +0.000 / +0.011 | +0.0037 | -0.011 | no |
| cw_sqrt | -0.007 / -0.002 | +0.015 / -0.020 | -0.004 / -0.016 | -0.0055 | -0.020 | no |
| moddrop | +0.006 / +0.013 | +0.018 / +0.004 | -0.004 / -0.001 | +0.0061 | -0.004 | yes |
| aug2 | -0.009 / -0.006 | +0.001 / +0.003 | +0.009 / -0.017 | -0.0031 | -0.017 | no |
| rcl_global | +0.012 / -0.003 | +0.011 / -0.005 | +0.000 / +0.013 | +0.0048 | -0.005 | no |
| rcl_neigh | +0.006 / +0.016 | +0.012 / +0.013 | -0.009 / +0.041 | +0.0133 | -0.009 | yes |
| fti_root | +0.001 / -0.024 | +0.019 / +0.013 | -0.009 / -0.002 | -0.0003 | -0.024 | no |

Kept: moddrop, rcl_neigh. Combination: `{'aug_modality_drop': 0.2, 'rcl_context': 'neighbour'}`. Best single: rcl_neigh.

**Decision** (2026-11-04T07:40:11, before the improved confirmation runs). Combination on the dev seeds: gaia: -0.020 / +0.013 | sockshop: +0.021 / +0.011 | hotel: -0.013 / +0.038; score +0.0085 (worst -0.020) against the best single, rcl_neigh, +0.0133. Final change: `{'rcl_context': 'neighbour'}`.
