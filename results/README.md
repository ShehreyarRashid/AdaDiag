# Results

Every JSON here is produced by the scripts in `../scripts/` against a checkout of
[WHU-AISE/TVDiag](https://github.com/WHU-AISE/TVDiag), and every number in the top-level
README is rendered from these files by `scripts/make_report.py` — none is typed by hand.

| file | produced by |
|---|---|
| `gaia.json`, `hotel.json` | `run_sweep.py` — five arms × five seeds |
| `hotel_v4_fusion.json` | `run_sweep.py --arms v4_fusion` — the exploratory arm |
| `gaia_gradshare.json`, `hotel_gradshare.json` | `run_diagnostics.py` — gradient shares |
| `hotel_perclass.json` | `run_perclass.py` — per-class metrics and learned weights |
| `run.log` | the full console transcript of the regeneration |

Each sweep record carries the metrics, the arm, the seed, the epoch count, wall-clock
seconds, the weighting module's parameter count, and (for the learned arms) `w_final`, the
mean weight matrix at the end of training.

Reproduce with:

```bash
python scripts/run_sweep.py --dataset gaia --repo /path/to/TVDiag
```

Results are CPU-deterministic given the seed, but a different PyTorch version or thread
count can shift the last decimal. The conclusions in the README rest on five-seed means
with paired tests, not on single runs.
