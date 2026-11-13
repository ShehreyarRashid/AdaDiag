#!/usr/bin/env python3
"""Run TVDiag's released code (github.com/WHU-AISE/TVDiag) unchanged and record its results.

README section 6.22. Nothing in the TVDiag checkout is edited. Three interventions, all from
outside its files:

* Missing modules. `core/model/Encoder.py` imports `core.model.backbone.tag` and
  `core.model.backbone.SGC`, which the repository does not contain (and never has), so the
  released code fails at import. Both are used only in commented-out code; empty placeholder
  modules are registered so the import succeeds.
* CPU instead of GPU. `core/loss/UnsupervisedContrastiveLoss.py` calls `.cuda()`, which
  fails on a machine without an NVIDIA GPU. `torch.Tensor.cuda` is redirected to return the
  tensor itself, at run time, so the code runs on the CPU exactly as written otherwise.
* A crash. `helper/eval.py`'s FTI_eval divides by precision + recall without a guard and
  raises ZeroDivisionError when both are 0 on a training batch (GAIA, seed 8). The failure
  is recorded (failures<tag>.json); `--guard` re-runs with a zero-guarded copy of that
  function, which feeds only TensorBoard during training, so the model is unchanged.
* More seeds and Hotel. `main.py` runs GAIA and Sock Shop with seed 2. `--mode main` runs
  that file as is; `--mode seeds` calls TVDiag's own `set_seed`, `build_dataloader`,
  `TVDiag.train` and `TVDiag.evaluate` in the order `train_and_evaluate` does, with the seed
  as a parameter, for any released dataset.

Environment (TVDiag's README: python 3.8.12, pytorch 2.1.1, dgl 2.1.0): python 3.11 (the
closest available), torch 2.1.1 and dgl 2.1.0 from PyPI, CPU, 1 thread per run. Run it with
that environment's interpreter, from anywhere:

    /path/to/venv/bin/python scripts/run_tvdiag_official.py --repo /path/to/TVDiag --mode main
    /path/to/venv/bin/python scripts/run_tvdiag_official.py --repo /path/to/TVDiag --mode seeds \\
        --datasets gaia,sockshop,hotel --seeds 2,5,6,7,8,9

Appends to results/tvdiag_official/runs<tag>.json (resumes) and copies each run's log.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "tvdiag_official"
TAG = ""                                     # --tag: one results file per parallel lane


def setup(repo: Path) -> None:
    import types
    import torch
    torch.set_num_threads(1)
    torch.Tensor.cuda = lambda self, *a, **k: self      # no GPU here: run .cuda() on the CPU
    os.chdir(repo)                                      # TVDiag reads data/ and writes logs/ relative to it
    sys.path.insert(0, str(repo))
    # core/model/Encoder.py imports core.model.backbone.tag (TAGEncoder) and .SGC (SGCEncoder),
    # which are not in the repository (never committed) and are used only in commented-out
    # lines. Empty placeholders let the import succeed; nothing calls them.
    for name, cls in (("core.model.backbone.tag", "TAGEncoder"),
                      ("core.model.backbone.SGC", "SGCEncoder")):
        if not (repo / (name.replace(".", "/") + ".py")).exists():
            mod = types.ModuleType(name)
            setattr(mod, cls, type(cls, (), {}))
            sys.modules[name] = mod


def commit(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return "unknown"


def parse_log(log_file) -> dict:
    """The test metrics TVDiag's `evaluate` logs (main.py keeps nothing else)."""
    text = Path(log_file).read_text()
    rcl = re.findall(r"\[Root localization\] HR@1: ([\d.]+)%, HR@2: ([\d.]+)%, HR@3: ([\d.]+)%, "
                     r"HR@4: ([\d.]+)%, HR@5: ([\d.]+)%, avg@3: ([\d.]+), MRR@3: ([\d.]+)", text)[-1]
    fti = re.findall(r"\[Failure type classification\] precision: ([\d.]+)%, recall: ([\d.]+)%, "
                     r"f1-score: ([\d.]+)%", text)[-1]
    h = [float(x) / 100 for x in rcl[:5]]
    return {"HR@1": h[0], "HR@2": h[1], "HR@3": h[2], "HR@4": h[3], "HR@5": h[4],
            "avg@3": float(rcl[5]), "MRR@3": float(rcl[6]), "pre": float(fti[0]) / 100,
            "rec": float(fti[1]) / 100, "f1": float(fti[2]) / 100,
            "train_time": None, "avg_train_time": None, "avg_inference_time": None}


def record(r: dict, dataset, seed, mode, log_file, repo, t0):
    import dgl
    import torch
    text = Path(log_file).read_text() if Path(log_file).exists() else ""
    m = re.findall(r"Early stop at epoch (\d+)", text)
    last = re.findall(r"Epoch (\d+) done", text)
    epochs = int(m[-1]) + 1 if m else (int(last[-1]) + 1 if last else None)
    rec = {"dataset": dataset, "seed": seed, "mode": mode, "variant": "tvdiag_official",
           "HR@1": r["HR@1"], "HR@2": r["HR@2"], "HR@3": r["HR@3"], "HR@4": r["HR@4"],
           "HR@5": r["HR@5"], "MRR@3": r["MRR@3"], "avg@3": r["avg@3"],
           "precision": r["pre"], "recall": r["rec"], "f1": r["f1"],
           "train_time": r["train_time"], "avg_train_time": r["avg_train_time"],
           "avg_inference_time": r["avg_inference_time"], "epochs": epochs,
           "guarded_fti_eval": bool(r.get("guarded", False)),
           "wall_s": round(time.time() - t0, 1),
           "env": {"commit": commit(repo), "python": platform.python_version(),
                   "torch": torch.__version__, "dgl": dgl.__version__, "threads": 1,
                   "device": "cpu (.cuda() redirected)"}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    if Path(log_file).exists():
        shutil.copy(log_file, OUT / "logs" / f"{dataset}_seed{seed}_{mode}.log")
    f = OUT / f"runs{TAG}.json"
    recs = json.load(open(f)) if f.exists() else []
    recs.append(rec)
    json.dump(recs, open(f, "w"), indent=1)
    print(f"  {dataset:8s} seed {seed} ({mode}): HR@1 {rec['HR@1']:.3f} HR@3 {rec['HR@3']:.3f} "
          f"avg@3 {rec['avg@3']:.3f} MRR@3 {rec['MRR@3']:.3f} | P {rec['precision']:.3f} "
          f"R {rec['recall']:.3f} F1 {rec['f1']:.3f} | {epochs} ep, {rec['wall_s']:.0f}s", flush=True)


def done() -> set:
    return {(r["dataset"], r["seed"], r["mode"])
            for f in OUT.glob("runs*.json") for r in json.load(open(f))}


def run_main(repo: Path) -> None:
    """main.py as released (gaia then sockshop, seed 2), executed as __main__; its results
    are read back from the logs it writes to ./logs/<dataset>/<dataset>.log."""
    if {("gaia", 2, "main"), ("sockshop", 2, "main")} <= done():
        return
    t0 = time.time()
    runpy.run_path(str(repo / "main.py"), run_name="__main__")
    for ds in ("gaia", "sockshop"):
        log = repo / "logs" / ds / f"{ds}.log"
        record(parse_log(log), ds, 2, "main", log, repo, t0)
    print(f"main.py finished in {time.time() - t0:.0f}s")


def guard_fti_eval() -> None:
    """--guard: TVDiag's helper/eval.py FTI_eval divides by precision + recall with no guard,
    and raises ZeroDivisionError when both are 0 on a training batch (GAIA seed 8). Those
    per-batch training metrics go only to TensorBoard, so a guarded copy (F1 = 0 when
    P + R = 0, as sklearn's zero_division=0) leaves training unchanged."""
    import core.TVDiag as tvcore
    import helper.eval as tveval
    orig = tveval.FTI_eval

    def guarded(output, target, k=5):
        try:
            return orig(output, target, k)
        except ZeroDivisionError:
            p, r = tveval.precision(output, target, k), tveval.recall(output, target, k)
            return {"pre": p, "rec": r, "f1": 0.0}

    tveval.FTI_eval = guarded
    tvcore.FTI_eval = guarded


def failed(ds, seed, err) -> None:
    f = OUT / f"failures{TAG}.json"
    recs = json.load(open(f)) if f.exists() else []
    recs.append(dict(dataset=ds, seed=seed, error=err))
    json.dump(recs, open(f, "w"), indent=1)
    print(f"  {ds:8s} seed {seed}: FAILED -- {err}", flush=True)


def run_seeds(repo: Path, datasets, seeds, guard=False) -> None:
    from config.exp_config import Config
    from core.TVDiag import TVDiag
    from helper.logger import get_logger
    from main import build_dataloader, set_seed
    if guard:
        guard_fti_eval()
    for ds in datasets:
        for seed in seeds:
            if (ds, seed, "seeds") in done():
                continue
            try:
                run_one(ds, seed, repo, guard, Config, TVDiag, get_logger, build_dataloader, set_seed)
            except Exception as e:                      # record the released code's failures
                import traceback
                tb = traceback.extract_tb(e.__traceback__)[-1]
                failed(ds, seed, f"{type(e).__name__}: {e} ({Path(tb.filename).name}:{tb.lineno})")


def run_one(ds, seed, repo, guard, Config, TVDiag, get_logger, build_dataloader, set_seed):
    t0 = time.time()
    config = Config(ds)
    config.reconstruct = False                    # as main.py: use the released features
    log_dir = f"./logs/seed{seed}/{ds}"
    set_seed(seed)                                # train_and_evaluate's order, seed varied
    logger = get_logger(log_dir, ds)
    logger.info("Load dataset")
    train_data, aug_data, test_data = build_dataloader(config, logger)
    logger.info("Training...")
    model = TVDiag(config, logger, log_dir)
    model.train(train_data, aug_data)
    res = model.evaluate(test_data)
    r = res.export_df(ds).iloc[0].to_dict()
    r["guarded"] = guard
    record(r, ds, seed, "seeds", os.path.join(log_dir, f"{ds}.log"), repo, t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, type=Path, help="a checkout of WHU-AISE/TVDiag")
    ap.add_argument("--mode", required=True, choices=["main", "seeds"])
    ap.add_argument("--datasets", default="gaia,sockshop,hotel")
    ap.add_argument("--seeds", default="2,5,6,7,8,9")
    ap.add_argument("--tag", default="", help="suffix for results/tvdiag_official/runs<tag>.json")
    ap.add_argument("--guard", action="store_true", help="zero-guard TVDiag's FTI_eval (see "
                                                          "guard_fti_eval); logging only")
    a = ap.parse_args()
    global TAG
    TAG = a.tag
    repo = a.repo.resolve()
    setup(repo)
    if a.mode == "main":
        run_main(repo)
    else:
        run_seeds(repo, a.datasets.split(","), [int(s) for s in a.seeds.split(",")], a.guard)


if __name__ == "__main__":
    main()
