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
