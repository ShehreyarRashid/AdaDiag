# Improving AdaDiag+: the rule and the screen (README §6.25)

Rule written 2026-11-12T13:29:22, before any run.

**Rule.** Round 4. Reference: AdaDiag+ (bag+theta) on the development seeds 0-2. Settings as section 6.17. A variant's six differences are dHR@1 and dF1 on the three datasets, paired by seed; its score is their mean. Stage S: smooth, bagdrop and baghead, each added alone to AdaDiag+. A single is kept if its score is at least +0.005 and its smallest difference is at least -0.010. With two or more kept, their combination is screened too (stage C). The single-model configuration is the eligible variant (kept singles, and the combination if its score is at least +0.005 and its smallest difference at least -0.010) whose smallest difference is largest; with none eligible it stays AdaDiag+. Stage E: ens3 of that configuration (seeds s, s+100, s+200; softmax outputs averaged) against the configuration itself on seeds 0-2; the ensemble is adopted if its score is at least +0.005 and its smallest difference at least -0.010. Confirmation: fresh seeds 30-34, never used before, for TVDiag, AdaDiag-DL, AdaDiag+ and the final configuration (and, as a control, ens3 of TVDiag). Paired t-tests, Holm over HR@1 and F1 x 3 datasets per comparison. There is no round 5 on these seeds.

## Stage S (reference: AdaDiag+), 2026-11-12T14:30:41

| variant | GAIA dHR@1 / dF1 | Sock Shop | Hotel | score | smallest | eligible |
|---|---|---|---|---|---|---|
| plus+smooth | -0.008 / +0.006 | +0.004 / +0.012 | +0.004 / -0.032 | -0.0024 | -0.032 | no |
| plus+bagdrop | -0.005 / -0.004 | +0.011 / +0.020 | +0.004 / +0.002 | +0.0048 | -0.005 | no |
| plus+baghead | -0.016 / -0.009 | -0.004 / +0.011 | +0.009 / +0.027 | +0.0029 | -0.016 | no |

Kept: none.

## Single-model configuration, 2026-11-12T14:30:59

No single was kept. Configuration: `plus`.

## Stage E and decision, 2026-11-12T14:59:18 (before any confirmation run)

Reference: `plus`.

| variant | GAIA dHR@1 / dF1 | Sock Shop | Hotel | score | smallest | eligible |
|---|---|---|---|---|---|---|
| plus|ens3 | -0.001 / +0.000 | +0.001 / +0.034 | +0.009 / -0.009 | +0.0057 | -0.009 | yes |

**Final configuration: `plus|ens3`.**
