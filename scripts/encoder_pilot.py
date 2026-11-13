#!/usr/bin/env python3
"""README section 6.23: how much does TVDiag's encoder use the service graph?

TVDiag (v0_raw) with every edge removed (`no_edges`), on the development seeds 0-2 of
section 6.21, so it pairs with the released TVDiag in results/improve/screen_<ds>.json
("base"). Writes results/encoder/pilot_<ds>.json.

    python scripts/encoder_pilot.py --dataset hotel
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch                                       # noqa: E402
from adadiag import data                           # noqa: E402
import improve                                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
a = ap.parse_args()
torch.set_num_threads(1)
ds = data.load(a.dataset)
improve.run(ds, "v0_raw", "no_edges", {"no_edges": True}, range(0, 3),
            ROOT / "results" / "encoder" / f"pilot_{a.dataset}.json")
