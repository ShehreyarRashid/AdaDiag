#!/usr/bin/env python3
"""Run the ablation studies (README section 6.8) and write results to JSON.

    python scripts/run_ablation.py --dataset hotel                  # all groups
    python scripts/run_ablation.py --dataset hotel --groups A B    # some groups
    python scripts/run_ablation.py --dataset hotel --only A_no_to  # one config

The file is written after every run and re-read on start, so an interrupted job resumes
where it stopped. `--shard i/n` splits the work across n processes (e.g. one per CPU core);
give each shard its own --out and pass all of them to make_ablation_report.py.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
ROOT = Path(__file__).resolve().parents[1]
import torch                                                      # noqa: E402
from adadiag import ablation, config, data                        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["gaia", "hotel", "sockshop"])
    ap.add_argument("--repo", default=str(ROOT), help="folder that contains data/ (default: this repository, which ships it; or a TVDiag checkout)")
    ap.add_argument("--groups", nargs="+", choices=ablation.GROUPS, default=None)
    ap.add_argument("--only", nargs="+", default=None, help="specific configuration names")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    ap.add_argument("--threads", type=int, default=None, help="torch threads per process")
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true", help="print the configurations and exit")
    a = ap.parse_args()

    if a.list:
        for n, (g, arm, ov, desc) in ablation.ABLATIONS.items():
            print(f"  {g:3s} {n:15s} {arm:12s} {desc}")
        return
    if not a.dataset:
        ap.error("--dataset is required (except with --list)")
    if a.threads:
        torch.set_num_threads(a.threads)

    names = ablation.select(a.groups, a.only)
    jobs = [(n, s) for n in names for s in range(a.seeds)]
    i, n = (int(x) for x in a.shard.split("/"))
    jobs = jobs[i::n]

    out = Path(a.out or f"results/ablation_{a.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else []
    have = {(r["name"], r["seed"]) for r in done}
    todo = [j for j in jobs if j not in have]

    ds = data.load(a.dataset, a.repo)
    cfg = config.get(a.dataset, epochs=a.epochs)
    config.check_against(cfg, ds)
    print(ds)
    print(f"{len(names)} configurations x {a.seeds} seeds; shard {a.shard}: "
          f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)

    for k, (name, seed) in enumerate(todo, 1):
        t = time.time()
        r = ablation.run_one(name, seed, ds, cfg)
        done.append(r)
        out.write_text(json.dumps(done, indent=1))
        print(f"  [{k}/{len(todo)}] {name:15s} seed {seed}  HR@1 {r['HR@1']:.3f}  "
              f"f1 {r['f1']:.3f}  macroF1 {r['macro_f1']:.3f}  "
              f"to-share {r['grad_share']['sigma*l_to']:5.1f}%  ({time.time() - t:.0f}s)",
              flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
