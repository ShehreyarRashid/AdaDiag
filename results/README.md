# Results

Every JSON here is produced by the scripts in `../scripts/` from the bundled `../data/`
(TVDiag's released data, unchanged), and every number in the top-level
README is rendered from these files by `scripts/make_report.py` — none is typed by hand.

| file | produced by |
|---|---|
| `gaia.json`, `hotel.json` | `run_sweep.py` — five arms × five seeds |
| `hotel_v4_fusion.json` | `run_sweep.py --arms v4_fusion` — the exploratory arm |
| `gaia_gradshare.json`, `hotel_gradshare.json` | `run_diagnostics.py` — gradient shares |
| `hotel_perclass.json` | `run_perclass.py` — per-class metrics and learned weights |
| `ablation_hotel_s0.json`, `ablation_hotel_s1.json` | `run_ablation.py` — all four ablation groups, 5 seeds, two shards (README §6.8) |
| `ablation_gaia_s0.json`, `ablation_gaia_s1.json` | `run_ablation.py --groups A B D` — 3 seeds, two shards |
| `trace_audit.json`, `TRACE_AUDIT.md`, `trace_audit.png` | `run_trace_audit.py` + `make_trace_report.py` — why AdaDiag gives trace zero weight (README §6.10): weights and the six contrastive losses per epoch, the slot-swap check, a warm-up test |
| `fidelity/`, `FIDELITY.md` | `fidelity_runs.sh` + `make_fidelity_report.py` — the softened prior (`v0_soft`) under the committed settings, and four arms re-run with the three TVDiag-fidelity fixes (README §6.11) |
| `oracle/` | `make_oracle_weights.py` → `oracle_weights.json` (the oracle matrices, fitted on the group-B ablations); `run_oracle.py` → `runs_gaia.json`, `runs_hotel.json`; `make_oracle_report.py` → `ORACLE.md`, `readme_table.md`, `summary.json`, `oracle_headroom.png` — the headroom check (README §6.12); `logs/` holds the run logs, including the check that the oracle arm given TVDiag's own matrix reproduces `ref_prior` bit-for-bit |
| `cl_variants/` | `run_cl_variants.py` → `runs_<dataset>_<lane>.json` (GAIA a + b and Hotel a: the five at σ = 0.1; Hotel b: the five at σ_to = 0.5; c: the training-length controls, `--capped-from`; d: types 1–2 at β = 1/τ; e: TVDiag at σ_to = 0.5); `cl_gradient_change.py` → `gradient_change.json`; `make_cl_report.py` → `CL_VARIANTS.md`, `readme_table.md`, `summary.json`, `cl_variants.png` — five ways to weight the contrastive term, against TVDiag and AdaDiag (README §6.13); `logs/` includes the check that TVDiag default re-trains bit-for-bit |
| `trace_fixed/` | `run_cl_variants.py --variants trace_third,trace_tvdiag --check-adadiag` → `runs_gaia.json`, `runs_hotel_a.json` (σ = 0.1), `runs_hotel_b.json` (σ_to = 0.5); `make_trace_fixed_report.py` → `TRACE_FIXED.md`, `readme_table.md`, `summary.json` — AdaDiag with trace's weight held fixed and only metric and log learned (README §6.14); `logs/` includes the check that AdaDiag re-trains bit-for-bit |
| `controlled/` | `check_controlled.py` → `null_<ds>.json` (TO off: the two arms must match), `rng_<ds>.json` (AdaDiag with the random stream aligned), `length_<ds>.json` / `length_rng_hotel.json` (TVDiag stopped at AdaDiag's epoch); `make_controlled_report.py` → `CONTROLLED.md`, `readme_table.md`, `summary.json` — is everything but the TO block the same (README §6.15) |
| `paper_vs_methods/` | `make_paper_vs_methods.py` (no training; reads the runs above) → `PAPER_VS_METHODS.md`, `table_A.tex` / `table_C.tex` / `table_D.tex` (LaTeX, booktabs), `all_methods.csv` (every method × dataset × metric with p-values), `readme_table.md`, `summary.json`, `fig_p1_values.png`, `fig_p2_gap.png` — every method against the TVDiag paper's Tables 4–5 only (README §6.16) |
| `selection/` | `select_adadiag.py` → `screen_<ds>.json` (stream-aligned re-runs for screening), `confirm_<ds>*.json` (fresh seeds 5–9: the finalists and TVDiag), `length_gaia.json` (TVDiag stopped at each finalist's epoch); `PLAN.md` (the rule and finalists, timestamped before the confirmation runs); `make_selection_report.py` → `SELECTION.md`, `readme_table.md`, `summary.json`, `fig_selection.png` — which AdaDiag is best (README §6.17) |
| `paper_tables/` | `make_paper_tables.py` → `table4.tex`, `table5.tex`, `table6.tex`, `tables.tex`, `tables.pdf`, `tables.csv` — the paper's Tables 4–5 in its own layout with TVDiag (re-run), AdaDiag-CL and AdaDiag-DL added, then the -LSTM and -Spec pairs (§6.18, §6.19); Table 6 also has the -θ pair (fresh seeds 5–9, from `selection/confirm_*.json`), and Table 6, the paired comparison (README §6.16) |
| `selection/LSTM.md`, `selection/lstm_table.md` | `make_lstm_report.py` — the LSTM aggregator for TVDiag and AdaDiag-DL on fresh seeds 5–9, with the training-length controls in `length_<ds>.json` (README §6.18); runs in `selection/confirm_<ds>_lstm_*.json` |
| `paper_features/` | `select_adadiag.py --variants tvdiag_paper,align_paper,tvdiag+<feature>,tvdiag_noes,align+awl_learnable` (runs in `selection/confirm_<ds>_paper_*.json`; the same-epoch control in `selection/length_<ds>.json`) → `make_paper_features_report.py` → `PAPER_FEATURES.md`, `readme_table.md`, `summary.json`, `fig_paper_features.png` — what the TVDiag paper specifies and its released code does not do, applied as switches (`config.PAPER`), for both arms and one feature at a time (README §6.19); `logs/check_defaults.log` is the check that the defaults re-train committed records bit-for-bit |
| `tables_8_9/` | `run_tables_8_9.py` → `affinity.json` (Eq. 17 at every step of epoch 0, both ways, and TVDiag's as-coded FTI → RCL; committed loop and TVDiag's loop; seeds 5–9), `efficiency.json` (training per epoch from the §6.17 runs, inference per case measured, parameter counts), `TABLES_8_9.md`, `summary.json` — the paper's Tables 8–9 (README §6.20) |
| `improve/` | `improve.py` → `screen_<ds>.json` (eight changes in TVDiag, dev seeds 0–2), `PLAN.md` / `plan.json` (the rule, the screen and the decision, written before any confirmation run), `confirm_<ds>.json` (fresh seeds 10–14: TVDiag and AdaDiag-DL, released and with the final change); `make_improve_report.py` → `IMPROVE.md`, `readme_table.md`, `summary.json`, `fig_improve.png` — improving RCL and FTI (README §6.21) |
| `paper_comparison/` | `paper_comparison_runs.sh` + `make_paper_comparison.py` — the comparison with the TVDiag paper's Tables 3–7 and Figures 11, 13–18 (README §6.9); its own README is `PAPER_COMPARISON.md` |

In `paper_comparison/`: `runs_<dataset>.json` are the default-setting runs behind Tables 4–5
and Figures 11/13/14/18 (2 threads; the GAIA and Hotel ones re-train records already in
`gaia.json` / `hotel.json` and are identical to them); `sweep_gaia_*.json` are Figure 16;
`ablate_*.json` are Tables 6–7 and Figure 17; `fig15_gaia.json` is Figure 15; `readme_table.md` is the table quoted in README §6.9. Each record
carries a `variant` (the configuration change, see `adadiag/paper_compare.VARIANTS`) and a
`tag` (`drop:<modalities>` or `emb:<re-embedding>`).

`run_identifiability.py` (README §6.6) is not committed as a file: part 1 takes seconds and
part 2's collapse figures are already in `gaia.json` / `hotel.json` as `w_final`. Pass
`--out` to save its output.

Each ablation record additionally carries `per_class` metrics, `grad_share` (the share of
the encoder gradient per loss term, measured on the trained model) and `weights` (the mean
learned matrix, its largest entry and its sum per task column).

Each sweep record carries the metrics, the arm, the seed, the epoch count, wall-clock
seconds, the weighting module's parameter count, and (for the learned arms) `w_final`, the
mean weight matrix at the end of training.

The environment that produced these files (OS, thread count, exact package versions) is in
[`ENVIRONMENT.md`](ENVIRONMENT.md). Records written from v1.2 on also carry an `env` field
with the PyTorch version and thread count. To re-derive any committed number:
`python scripts/verify_results.py --threads 2` (see its `--help`).

Reproduce with:

```bash
python scripts/run_sweep.py --dataset gaia
```

Results are CPU-deterministic given the seed **and** the thread count. A different PyTorch
version or thread count changes floating-point summation order, and over a full training run
that can move a single seed by a point or more (Hotel seed 1, V0-simplex: HR@1 0.821 with 2
threads, 0.833 with 1). The conclusions in the README rest on five-seed means with paired
tests, not on single runs.
