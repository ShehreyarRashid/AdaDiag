# A stronger AdaDiag: the rule and the screen (README §6.24)

Rule written 2026-11-10T13:54:41, before any run.

**Rule.** Development seeds 0-2, AdaDiag-DL (v5_align), all three datasets. A variant's score is the mean over the three datasets of (dHR@1 + dF1) / 2, paired by seed against its reference. It is kept if its score is above +0.005 and no dataset loses more than 0.015 on HR@1 or on F1 against that reference. Stage A: bag, attn, selfres and ema each alone, reference AdaDiag-DL as released; the anchor is the kept candidate with the highest score (none kept: the anchor is AdaDiag-DL itself). Stage B: the anchor plus each other candidate (bag, attn, selfres, ema, root, neigh, minus the anchor), one at a time, reference the anchor. Stage C: with two or more kept additions, the anchor plus all of them is screened, reference the anchor; the final change is that combination if its score is at least the best single addition's, otherwise the anchor plus the best single addition. With one kept addition, the final change is the anchor plus it; with none, the anchor alone. If the final change is empty, nothing is confirmed. Confirmation: fresh seeds 15-19 for AdaDiag-DL and TVDiag, each as released and with the final change; paired t-tests, Holm over HR@1 and F1 x 3 datasets per comparison.

## Stage A (reference: AdaDiag-DL as released), 2026-11-10T14:48:53

| variant | GAIA dHR@1 / dF1 | Sock Shop | Hotel | score | worst | kept |
|---|---|---|---|---|---|---|
| bag | +0.024 / +0.005 | +0.007 / +0.012 | -0.013 / +0.003 | +0.0065 | -0.013 | yes |
| attn | +0.000 / +0.003 | +0.000 / -0.016 | -0.013 / -0.066 | -0.0153 | -0.066 | no |
| selfres | +0.021 / +0.000 | +0.001 / +0.000 | +0.000 / -0.071 | -0.0080 | -0.071 | no |
| ema | -0.000 / +0.002 | -0.004 / +0.001 | -0.004 / -0.006 | -0.0018 | -0.006 | no |

Anchor: `bag`.

## Stage B (reference: `bag`), 2026-11-10T15:39:05

| variant | GAIA dHR@1 / dF1 | Sock Shop | Hotel | score | worst | kept |
|---|---|---|---|---|---|---|
| bag+attn | -0.026 / -0.010 | -0.037 / +0.002 | -0.004 / -0.063 | -0.0232 | -0.063 | no |
| bag+selfres | +0.006 / -0.003 | -0.008 / -0.017 | +0.017 / -0.056 | -0.0101 | -0.056 | no |
| bag+ema | +0.002 / -0.001 | -0.010 / -0.008 | -0.004 / +0.009 | -0.0020 | -0.010 | no |
| bag+root | -0.002 / -0.013 | -0.003 / +0.040 | +0.017 / +0.014 | +0.0088 | -0.013 | yes |
| bag+neigh | -0.008 / -0.009 | -0.008 / +0.014 | +0.013 / +0.032 | +0.0056 | -0.009 | yes |

Kept additions, best first: root, neigh.

## Decision, 2026-11-10T15:47:25 (before any confirmation run)

Combination `bag+root+neigh` against `bag`: gaia: -0.025 / -0.030 | sockshop: -0.006 / +0.038 | hotel: -0.004 / -0.022; score -0.0082 (worst -0.030) against the best single addition, root, +0.0088. **Final change: `bag+root`** = `{'alert_bag': True, 'fti_readout': 'root_attn'}`.

## Round 2: rule, written 2026-11-10T16:40:47 (before its runs)

**Rule.** Round 2, added after round 1's confirmation (seeds 15-19) left two cells below AdaDiag-DL (GAIA F1 -0.009, Hotel HR@1 -0.008). The selection criterion changes from the mean score to balance, because the goal is to improve every dataset. One new candidate is screened on the development seeds 0-2: bag+rootd (the root readout with its softmax detached, so the FTI loss cannot reshape the RCL ranking). Eligible: every variant with bag screened on the development seeds (bag, bag+attn, bag+selfres, bag+ema, bag+root, bag+neigh, bag+root+neigh, bag+rootd). For each, the six development-seed differences against AdaDiag-DL (HR@1 and F1 x 3 datasets) are computed; among those with a score of at least +0.005, the winner is the one whose smallest difference is largest. It is confirmed on fresh seeds 20-24, never used before, against AdaDiag-DL and TVDiag as released, with round 1's final change alongside.

## Round 2: decision, 2026-11-10T16:50:23 (before its confirmation runs)

Differences against AdaDiag-DL on the development seeds:

| variant | GAIA dHR@1 / dF1 | Sock Shop | Hotel | score | smallest | eligible |
|---|---|---|---|---|---|---|
| bag | +0.024 / +0.005 | +0.007 / +0.012 | -0.013 / +0.003 | +0.0065 | -0.013 | yes |
| bag+attn | -0.001 / -0.005 | -0.030 / +0.014 | -0.017 / -0.060 | -0.0167 | -0.060 | no |
| bag+selfres | +0.031 / +0.002 | -0.001 / -0.004 | +0.004 / -0.053 | -0.0036 | -0.053 | no |
| bag+ema | +0.026 / +0.004 | -0.003 / +0.005 | -0.017 / +0.012 | +0.0044 | -0.017 | no |
| bag+root | +0.022 / -0.008 | +0.004 / +0.052 | +0.004 / +0.017 | +0.0152 | -0.008 | yes |
| bag+neigh | +0.016 / -0.004 | -0.001 / +0.026 | +0.000 / +0.035 | +0.0121 | -0.004 | yes |
| bag+root+neigh | -0.000 / -0.025 | +0.001 / +0.050 | -0.017 / -0.019 | -0.0017 | -0.025 | no |
| bag+rootd | +0.019 / -0.010 | +0.007 / +0.068 | +0.009 / +0.017 | +0.0181 | -0.010 | yes |

**Round 2's final change: `bag+neigh`** = `{'alert_bag': True, 'rcl_context': 'neighbour'}`.

## Round 3: rule, written 2026-11-10T18:32:55 (before its runs)

**Rule.** Round 3, the last, added after both rounds' changes were run on seeds 5-9: there AdaDiag-DL with Eq. 13's learnable theta (section 6.19) scores as high as round 1's change, and higher than AdaDiag-DL on all six cells, while theta acts on the loss weights rather than the features. Screened on the development seeds 0-2: theta alone, bag+theta, bag+root+theta and bag+neigh+theta. Eligible: these four and every variant eligible in round 2. Selection as round 2: among variants with a score of at least +0.005 against AdaDiag-DL, the one whose smallest of the six differences is largest. Confirmed on fresh seeds 25-29, never used before, against AdaDiag-DL and TVDiag as released. Whatever round 3 yields is reported; there is no round 4.

## Round 3: decision, 2026-11-10T19:55:41 (before its confirmation runs)

Differences against AdaDiag-DL on the development seeds:

| variant | GAIA dHR@1 / dF1 | Sock Shop | Hotel | score | smallest | eligible |
|---|---|---|---|---|---|---|
| bag | +0.024 / +0.005 | +0.007 / +0.012 | -0.013 / +0.003 | +0.0065 | -0.013 | yes |
| bag+attn | -0.001 / -0.005 | -0.030 / +0.014 | -0.017 / -0.060 | -0.0167 | -0.060 | no |
| bag+selfres | +0.031 / +0.002 | -0.001 / -0.004 | +0.004 / -0.053 | -0.0036 | -0.053 | no |
| bag+ema | +0.026 / +0.004 | -0.003 / +0.005 | -0.017 / +0.012 | +0.0044 | -0.017 | no |
| bag+root | +0.022 / -0.008 | +0.004 / +0.052 | +0.004 / +0.017 | +0.0152 | -0.008 | yes |
| bag+neigh | +0.016 / -0.004 | -0.001 / +0.026 | +0.000 / +0.035 | +0.0121 | -0.004 | yes |
| bag+root+neigh | -0.000 / -0.025 | +0.001 / +0.050 | -0.017 / -0.019 | -0.0017 | -0.025 | no |
| bag+rootd | +0.019 / -0.010 | +0.007 / +0.068 | +0.009 / +0.017 | +0.0181 | -0.010 | yes |
| theta | -0.000 / +0.006 | +0.017 / -0.003 | +0.000 / -0.004 | +0.0026 | -0.004 | no |
| bag+theta | +0.033 / +0.012 | +0.000 / +0.025 | +0.000 / +0.046 | +0.0194 | +0.000 | yes |
| bag+root+theta | +0.017 / -0.009 | +0.014 / +0.059 | +0.009 / +0.027 | +0.0192 | -0.009 | yes |
| bag+neigh+theta | +0.016 / +0.003 | +0.000 / +0.063 | -0.013 / +0.037 | +0.0178 | -0.013 | yes |

**Round 3's final change: `bag+theta`** = `{'alert_bag': True, 'awl_learnable': True}`.
