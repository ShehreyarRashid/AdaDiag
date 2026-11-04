#!/usr/bin/env python3
"""README section 6.20: the TVDiag paper's Table 8 (inter-task affinity) and Table 9
(efficiency), for TVDiag and AdaDiag.

Table 8. Eq. 17 at every step of the first epoch, averaged, as TVDiag measures it
(`adadiag.affinity`): fresh seeds 5-9, under the committed loop and under TVDiag's own loop
(config.FIDELITY: AUG from a pool, self-loops). Each record has Eq. 17 both ways and the
released code's FTI -> RCL, which re-evaluates the wrong model copy.

Table 9. Training time per epoch, from the fresh-seed runs of section 6.17 (1 CPU thread;
results/selection/confirm_<ds>*.json), and model inference per case, measured here: one
incident at a time, as TVDiag's `evaluate` does, 1 thread, median of 3 passes over the test
set. The paper's "offline" and "online" columns also include alert extraction from raw
telemetry, which the release omits, so only the model's share can be compared, and the
paper's times are on a GPU (GTX 1080).

Writes results/tables_8_9/: affinity.json, efficiency.json, TABLES_8_9.md, summary.json.

    python scripts/run_tables_8_9.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch                                                       # noqa: E402
from adadiag import config, data                                   # noqa: E402
from adadiag.model import AdaDiag                                  # noqa: E402
from adadiag.train import train                                    # noqa: E402

OUT = ROOT / "results" / "tables_8_9"
SEL = ROOT / "results" / "selection"
SEEDS = range(5, 10)
DS = (("gaia", "A · GAIA", "A"), ("sockshop", "C · Sock Shop", "C"), ("hotel", "D · Hotel", "D"))
#: TVDiag paper, Table 8 (p. 40:31): (Z_RCL->FTI, Z_FTI->RCL); D was not reported
PAPER_T8 = {"A": (9.12e-3, 8.77e-2), "B": (-9.00e-3, 1.08e-1)}
#: TVDiag paper, Table 9 (p. 40:32), TVDiag's row: offline (s), training per epoch (s),
#: online per case (s); GPU, and offline / online include alert extraction
PAPER_T9 = {"A": (433.738, 0.654, 3.075), "B": (1909.628, 1.718, 3.638)}
ARMS9 = (("tvdiag", "v0_raw", "TVDiag"), ("adadiag", "v3_incident", "AdaDiag-CL"),
         ("align", "v5_align", "AdaDiag-DL"))


def affinity_runs() -> list:
    out = OUT / "affinity.json"
    recs = json.load(open(out)) if out.exists() else []
    done = {(r["dataset"], r["loop"], r["seed"]) for r in recs}
    for ds, _, _ in DS:
        d = data.load(ds)
        for loop, over in (("committed", {}), ("tvdiag_code", dict(config.FIDELITY))):
            for seed in SEEDS:
                if (ds, loop, seed) in done:
                    continue
                cfg = config.get(ds, isolate_weighting_rng=True, measure_affinity=True, **over)
                cfg["epochs"] = 1
                res, _ = train("v0_raw", seed, d, cfg)
                recs.append(dict(dataset=ds, loop=loop, seed=seed, steps=res["affinity_steps"],
                                 **res["affinity"]))
                json.dump(recs, open(out, "w"), indent=1)
                a = res["affinity"]
                print(f"  {ds:8s} {loop:11s} s{seed}  RCL->FTI {a['r2f']:+.4f}  FTI->RCL "
                      f"{a['f2r']:+.4f}  as coded {a['f2r_code']:+.4f}", flush=True)
    return recs


@torch.no_grad()
def per_case(model, d) -> float:
    model.eval()
    times = []
    for i in d.test_idx:
        b = np.array([i])
        keep = torch.ones(1, d.n_nodes, dtype=torch.bool)
        t = time.perf_counter()
        model(d.X[b], d.A[b], keep, d.cnt[b])
        times.append(time.perf_counter() - t)
    return float(np.mean(times))


def efficiency() -> dict:
    out: dict = {}
    for ds, _, _ in DS:
        d = data.load(ds)
        recs = {}
        for f in glob.glob(str(SEL / f"confirm_{ds}*.json")):
            for r in json.load(open(f)):
                if r["seed"] in SEEDS:
                    recs.setdefault(r["variant"], {}).setdefault(r["seed"], r)
        for v, arm, _ in ARMS9:
            torch.manual_seed(0)
            m = AdaDiag(arm, len(d.types), config.get(ds))
            per_case(m, d)                                    # warm-up pass
            inf = float(np.median([per_case(m, d) for _ in range(3)]))
            rs = list(recs.get(v, {}).values())
            out.setdefault(ds, {})[v] = dict(
                train_per_epoch=float(np.mean([r["train_s"] / r["epochs"] for r in rs])),
                train_total=float(np.mean([r["train_s"] for r in rs])),
                epochs=float(np.mean([r["epochs"] for r in rs])),
                infer_per_case=inf,
                params=int(sum(p.numel() for p in m.parameters())),
                weighting_params=int(m.weighting.n_parameters()), n=len(rs))
            print(f"  {ds:8s} {v:8s} train/epoch {out[ds][v]['train_per_epoch']:.3f}s  "
                  f"infer/case {inf * 1e3:.2f} ms  params {out[ds][v]['params']:,}", flush=True)
    json.dump(out, open(OUT / "efficiency.json", "w"), indent=1)
    return out


def sci(x: float) -> str:
    m, e = f"{x:.2e}".split("e")
    return f"{m}×10^{int(e)}"


def report(aff: list, eff: dict) -> None:
    summ: dict = {"affinity": {}, "efficiency": eff}
    L = ["# The TVDiag paper's Tables 8 and 9 (README §6.20)\n",
         "## Table 8: inter-task affinity, Eq. 17 (mean over the steps of epoch 0, then over "
         "seeds 5–9)\n",
         "| dataset | loop | Z_RCL→FTI | Z_FTI→RCL, Eq. 17 | Z_FTI→RCL, as TVDiag computes it | "
         "steps per epoch | paper (Z_RCL→FTI / Z_FTI→RCL) |",
         "|---|---|---|---|---|---|---|"]
    for ds, name, letter in DS:
        for loop in ("tvdiag_code", "committed"):
            rs = [r for r in aff if r["dataset"] == ds and r["loop"] == loop]
            if not rs:
                continue
            cell = {}
            for k in ("r2f", "f2r", "f2r_code"):
                x = np.array([r[k] for r in rs])
                p = float(stats.ttest_1samp(x, 0.0).pvalue) if x.std() > 0 else float("nan")
                cell[k] = f"{sci(x.mean())} ± {sci(x.std(ddof=1))} (p {p:.2g})"
                summ["affinity"].setdefault(ds, {}).setdefault(loop, {})[k] = dict(
                    mean=float(x.mean()), sd=float(x.std(ddof=1)), p_vs_0=p, n=len(x))
            pp = PAPER_T8.get(letter)
            L.append(f"| {name} | {'TVDiag code (§6.11)' if loop == 'tvdiag_code' else 'committed'} | "
                     f"{cell['r2f']} | {cell['f2r']} | {cell['f2r_code']} | {rs[0]['steps']} | "
                     + (f"{sci(pp[0])} / {sci(pp[1])}" if pp else "not reported") + " |")
    L.append("\nmean ± sd over the five seeds; p: one-sample t-test against 0. \"As TVDiag "
             "computes it\": `core/ita.py` steps the second copy on the FTI loss, then "
             "re-evaluates the FIRST copy, the one stepped on the RCL loss, so it measures the "
             "RCL loss's response to an RCL step. Dataset B is not released.\n")
    L += ["## Table 9: efficiency (1 CPU thread; the paper used a GPU)\n",
          "| dataset | method | training per epoch (s) | epochs | training total (s) | model "
          "inference per case (ms) | parameters | of which the weighting module |",
          "|---|---|---|---|---|---|---|---|"]
    for ds, name, letter in DS:
        tv = eff[ds]["tvdiag"]
        for v, _, lab in ARMS9:
            e = eff[ds][v]
            rel = "" if v == "tvdiag" else f" ({e['train_per_epoch'] / tv['train_per_epoch']:.1f}×)"
            reli = "" if v == "tvdiag" else f" ({e['infer_per_case'] / tv['infer_per_case']:.2f}×)"
            L.append(f"| {name} | {lab} | {e['train_per_epoch']:.3f}{rel} | {e['epochs']:.0f} | "
                     f"{e['train_total']:.0f} | {e['infer_per_case'] * 1e3:.2f}{reli} | "
                     f"{e['params']:,} | {e['weighting_params']:,} |")
        if letter in PAPER_T9:
            o, t, c = PAPER_T9[letter]
            L.append(f"| {name} | *TVDiag, paper (GPU)* | {t:.3f} | — | — | {c * 1e3:,.0f} "
                     f"(incl. alert extraction) | — | — |")
    L.append("\nTraining: the fresh-seed runs of §6.17 (seeds 5–9, 500-epoch cap, early "
             "stopping); per epoch = run time / epochs. Inference: the model's forward pass for "
             "one incident, averaged over the test set, median of 3 passes; the paper's online "
             "time also includes alert extraction (the release ships extracted features, so it "
             "cannot be re-measured). AdaDiag-CL's weighting module is not on the prediction "
             "path, and AdaDiag-DL has no trainable weighting parameters, so inference is the "
             "same network for all three.\n")
    (OUT / "TABLES_8_9.md").write_text("\n".join(L))
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    print("\n".join(L))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    aff = affinity_runs()
    eff = efficiency()
    report(aff, eff)


if __name__ == "__main__":
    main()
