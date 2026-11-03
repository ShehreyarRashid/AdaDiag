#!/usr/bin/env python3
"""Headroom check (README section 6.12): train with the oracle matrices of
scripts/make_oracle_weights.py at three strengths of the task-oriented term.

Matches the settings of the section 6.8 ablations it is compared with -- 1 thread, the
500-epoch cap, sigma_to as `to_scale` -- so the prior's runs there (ref_prior, D_prior_*)
are paired references on the same seeds.

    python scripts/run_oracle.py --dataset gaia  --seeds 3 --threads 1
    python scripts/run_oracle.py --dataset hotel --seeds 5 --threads 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch                                                         # noqa: E402
from adadiag import config, data                                     # noqa: E402
from adadiag.evaluate import per_class                               # noqa: E402
from adadiag.weighting import PRIOR_SIMPLEX                          # noqa: E402
from adadiag.train import train                                     # noqa: E402

SIGMAS = (0.1, 0.5, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["gaia", "hotel"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-prior", action="store_true",
                    help="first train v0_oracle with the prior's own matrix (seed 0) and check "
                         "it reproduces the ablation's ref_prior record bit-for-bit")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    ow = json.load(open(ROOT / "results" / "oracle" / "oracle_weights.json"))[a.dataset]
    ds = data.load(a.dataset)
    if a.check_prior:
        cfg = config.get(a.dataset)
        cfg["oracle_w"] = PRIOR_SIMPLEX
        res, _ = train("v0_oracle", 0, ds, cfg)
        refs = []
        for f in sorted((ROOT / "results").glob(f"ablation_{a.dataset}_s*.json")):
            refs += [r for r in json.load(open(f)) if r["name"] == "ref_prior" and r["seed"] == 0]
        keys = ("HR@1", "HR@3", "f1", "epochs")
        same = all(res[k] == refs[0][k] for k in keys)
        print("prior-equivalence check:", {k: (res[k], refs[0][k]) for k in keys},
              "IDENTICAL" if same else "DIFFERENT", flush=True)
    out = Path(a.out or ROOT / "results" / "oracle" / f"runs_{a.dataset}.json")
    recs = json.load(open(out)) if out.exists() else []
    done = {(r["oracle"], r["sigma_to"], r["seed"]) for r in recs}

    for oracle in ("global", "class"):
        for s_to in SIGMAS:
            for seed in range(a.seeds):
                if (oracle, s_to, seed) in done:
                    continue
                cfg = config.get(a.dataset, to_scale=s_to)
                if oracle == "global":
                    cfg["oracle_w"] = ow["global"]
                else:
                    cfg["oracle_w"] = ow["global"]          # only what the module reports
                    cfg["oracle_class_w"] = [ow["class"][t] for t in ds.types]
                res, model = train("v0_oracle", seed, ds, cfg)
                res.pop("w_final", None)
                res.update(oracle=oracle, sigma_to=s_to, per_class=per_class(model, ds)[0])
                recs.append(res)
                json.dump(recs, open(out, "w"))
                print(f"  {oracle:6s} σ_to={s_to:<4} s{seed}  HR@1 {res['HR@1']:.3f}  "
                      f"F1 {res['f1']:.3f}  ({res['epochs']} ep)", flush=True)
    print("done", out)


if __name__ == "__main__":
    main()
