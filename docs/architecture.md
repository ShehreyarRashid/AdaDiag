# Architecture

![TVDiag and the areas under consideration](figures/fig1_tvdiag_focus.png)

![GradFusion: what changed](figures/fig2_gradfusion_changes.png)

![GradFusion in detail](figures/fig3_gradfusion_detail.png)

Sources: `figures/fig1_tvdiag_focus.tex`, `fig2_gradfusion_changes.tex` and `fig3_gradfusion_detail.tex` (TikZ; each also as PDF and PNG). Figure 1 marks the four areas of TVDiag that this work looks at, Figure 2 shows what GradFusion changes, and Figure 3 gives the detail. In Figure 3 the colours mark what each step added: TVDiag (grey),
adaptive weights (blue), alert-type counts and learnable loss weights (green), ensemble
(orange).

## Forward pass (one model)

1. **Input.** For each incident and service instance: three 128-d alert embeddings (metric,
   trace, log; TVDiag's fastText averages) and the alert-type counts from `alerts.py`.
2. **Encoders.** One two-layer GraphSAGE encoder per modality over the service call graph
   (`encoders.py`).
3. **Fusion.** Node embeddings are concatenated across modalities; a max-pool gives the
   graph embedding (`model.py`).
4. **Heads.** The RCL head scores every instance; the FTI head scores every failure type
   (`heads.py`).

## Training objective (one model)

- `L_RCL` and `L_FTI`: cross-entropy on the two heads.
- `L_TO`: TVDiag's task-oriented contrastive loss. GradFusion weights each modality's term
  with a 3×2 matrix, set from how well that term's gradient agrees with `L_RCL` and `L_FTI`
  (arm `v5_align`, `weighting.py`, `train.py`).
- `L_CM`: TVDiag's cross-modal contrastive loss.
- The four losses are combined with learnable weights (TVDiag paper, Eq. 13;
  `awl_learnable=True`).
- Graph augmentation, optimiser and early stopping are TVDiag's (`config.py`).

## Ensemble

Three models are trained independently with seeds s, s + 100, s + 200.
`ensemble.predict` averages their softmax outputs, over instances for RCL and over failure
types for FTI.

## Configuration

```python
config.get(dataset, isolate_weighting_rng=True, alert_bag=True, awl_learnable=True)
```

with the dataset wrapped by `alerts.with_alert_bag(ds)` and the arm `"v5_align"`. Other
switches in the code are off by default and are not part of GradFusion (`HISTORY.md`).
