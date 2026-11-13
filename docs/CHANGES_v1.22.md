# What changed in AdaDiag v1.22 (AdaDiag++)

Shehreyar Rashid (26I-9004) · 12 November 2026 · previous version: v1.21 (AdaDiag+)

## 1. Summary

- **New method: AdaDiag++.** Three AdaDiag+ models are trained with different seeds and
  their outputs are averaged at diagnosis time.
- **Result.** Over ten unused seeds, AdaDiag++ is above AdaDiag+ on HR@1 and F1 for all
  three datasets (A, C, D).
- **Nothing old changed.** All new switches are off by default. TVDiag, AdaDiag-DL and
  AdaDiag+ reproduce their earlier numbers exactly.

## 2. Why this change

AdaDiag+ is trained on only 122–248 incidents per dataset. Its results move from seed to
seed by about as much as the gains it made (F1 standard deviation about 0.02 on Sock Shop
and Hotel). It also lost 0.008 HR@1 on Sock Shop. The problem is variance, so the research
targeted variance.

## 3. What was tried (round 4)

The rule was written down before any run (`results/plus2/PLAN.md`). Each candidate was
screened against AdaDiag+ on development seeds 0–2. To pass, a candidate needed a mean gain
of at least +0.005 and no loss worse than −0.010 on any dataset.

| Candidate | What it does | Passed? |
|---|---|---|
| Label smoothing (`fti_smooth=0.1`) | Softer targets for the failure-type loss | No (Hotel F1 −0.032) |
| Count dropout (`bag_drop=0.2`) | Dropout on the alert-type counts | No (mean +0.0048, just short) |
| Wide head (`bag_head`) | Heads also read the raw counts | No (GAIA HR@1 −0.016) |
| **Ensemble of 3** | Average three AdaDiag+ models | **Yes** (mean +0.006) |

## 4. Results

Means over ten seeds (5–9 and 30–34), HR@1 / F1:

| Method | A · GAIA | C · Sock Shop | D · Hotel |
|---|---|---|---|
| TVDiag, published (one run) | 0.759 / 0.936 | 0.863 / 0.636 | 0.821 / 0.822 |
| TVDiag, re-run | 0.738 / 0.921 | 0.869 / 0.552 | 0.821 / 0.781 |
| AdaDiag-DL | 0.750 / 0.937 | 0.869 / 0.571 | 0.817 / 0.797 |
| AdaDiag+ (v1.21) | 0.759 / 0.944 | 0.860 / 0.624 | 0.836 / 0.833 |
| **AdaDiag++ (v1.22)** | **0.767 / 0.948** | **0.871 / 0.648** | **0.841 / 0.860** |

Gain of AdaDiag++ over AdaDiag+ (paired, same seeds):

| Dataset | HR@1 | F1 |
|---|---|---|
| A · GAIA | +0.008 (p 0.05) | +0.004 (p 0.01) † |
| C · Sock Shop | +0.011 (p 0.01) † | +0.024 (p 0.01) † |
| D · Hotel | +0.005 (p 0.34) | +0.027 (p < 0.01) † |

† still significant after Holm correction over the six cells.

## 5. Limits

- **Cost.** Training and diagnosis take three times as long as AdaDiag+.
- **Part of the gain is the ensemble, not AdaDiag.** Averaging three TVDiag models also
  helps TVDiag. Against that control (seeds 30–34), AdaDiag++ still leads on F1 for all
  three datasets (+0.022, +0.054, +0.025) and ties on HR@1.
- **Hotel HR@1 is a tie** (+0.005, not significant).
- **Published TVDiag numbers are single runs.** AdaDiag++'s ten-seed means are higher on
  all six cells, but the margins on GAIA HR@1 and Sock Shop are small.
- **Weak spots remain.** Hotel network-delay HR@1 is 0.37.
- **This is the fourth round on the same test split.** Each round used unused seeds, but
  the series is adaptive.

## 6. Files changed

| File | Change |
|---|---|
| `src/adadiag/ensemble.py` | New. `predict(models, ds, idx)` averages the members' outputs. |
| `src/adadiag/model.py` | New switches `bag_drop`, `bag_head` (off by default). |
| `src/adadiag/train.py` | New switch `fti_smooth` (off by default). |
| `scripts/plus2.py` | New. Round 4: rule, screen, decision, confirmation. |
| `scripts/make_plus2_report.py` | New. Builds `results/plus2/PLUS2.md` and the figure. |
| `scripts/make_paper_tables.py`, `make_multimodal_tables.py` | AdaDiag++ row added. |
| `tests/test_plus2.py` | New, 6 tests. Total is now 117. |
| `results/plus2/` | New. Rule, plan, 204 runs with test logits, report. |
| `results/paper_tables/` | Tables 4–6 rebuilt; AdaDiag++ is "AdaDiag (best)". |
| `README.md` | New §6.25, headline 20, scope note, commands. |
| `docs/architecture.md`, `results/README.md`, `UPLOAD_TO_GITHUB.md` | Updated. |
| `CITATION.cff`, `pyproject.toml`, `__init__.py` | Version 1.22.0. |

## 7. How to use AdaDiag++

```python
from adadiag import config, data
from adadiag.alerts import with_alert_bag
from adadiag.ensemble import predict
from adadiag.train import train

ds = with_alert_bag(data.load("hotel"))
cfg = config.get("hotel", isolate_weighting_rng=True, alert_bag=True, awl_learnable=True)
models = [train("v5_align", seed, ds, cfg)[1] for seed in (5, 105, 205)]
node_scores, type_scores = predict(models, ds, ds.test_idx)
```

## 8. How to push to GitHub

In your Codespace, after dragging `AdaDiag-v1.22.zip` into the file list:

```bash
git pull
unzip -q -o AdaDiag-v1.22.zip -d /tmp/new
cp -a /tmp/new/AdaDiag/. .
rm -rf /tmp/new AdaDiag-v1.22.zip
python -m pytest tests/ -q        # should print: 117 passed
git add -A
git commit -m "v1.22: AdaDiag++ (three AdaDiag+ models averaged, round 4), README 6.25"
git push
```

## 9. Reference for the technique

Deep ensembles: Lakshminarayanan, Pritzel and Blundell, NeurIPS 2017. It is a peer-reviewed
conference paper, not a journal. It is named here and in the code only; it has not been
added to any course report.
