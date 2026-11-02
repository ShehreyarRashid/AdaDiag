# Environment that produced these results

| | |
|---|---|
| OS | Linux (x86-64), CPU only — no GPU, no CUDA kernels used |
| CPU threads | **2** (`torch.get_num_threads()`; main sweeps, diagnostics, per-class) · **1 per process** (ablations, two processes in parallel) |
| Python | 3.13.16 |
| PyTorch | 2.15.0 (PyPI build `+cu132`, run on CPU) |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| Data | `data/` — TVDiag commit `cf6f334`, verified by `data/SHA256SUMS` |
| Seeds | 0–4 (main sweeps, Hotel ablations) · 0–2 (GAIA ablations) |

Pinned versions: `requirements-lock.txt`.

## Why the thread count is listed

PyTorch's CPU kernels split reductions across threads, and the order of floating-point
additions changes with the thread count. The change is in the last bits of a single step,
but over ~150 training epochs it compounds into a different trajectory, so a single seed
can end a few points away. Same machine class, same versions and same thread count give
bit-identical results; `scripts/verify_results.py --threads 2` checks exactly that.

The README's claims rest on five-seed means and paired tests, which are what should be
compared across machines — not individual seeds.
