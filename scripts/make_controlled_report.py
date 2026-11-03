#!/usr/bin/env python3
"""README section 6.15: is everything but the TO block the same for TVDiag and AdaDiag?

Reads results/controlled/*.json (scripts/check_controlled.py) and the paired references
(TVDiag default, the section 6.8 ablation records). Writes results/controlled/CONTROLLED.md,
readme_table.md and summary.json.

    python scripts/make_controlled_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_oracle_report as MO                                        # noqa: E402

OUT = ROOT / "results" / "controlled"
NAME = MO.NAME
METRICS = MO.METRICS

REVIEW = [
    ("data, splits, seeds, hyper-parameters", "same", "`config.get`, one call for every arm"),
    ("encoders, readout, heads, their initialisation", "same", "built before the weighting module; equal for every arm and seed (`tests/test_controlled.py`)"),
    ("batch order (numpy) and AUG node drops (Python `random`)", "same", "neither stream is touched by the weighting module"),
    ("encoder dropout masks (torch stream)", "**different on Hotel**", "AdaDiag's MLP initialisation advances the torch stream, so every dropout mask after it shifts; GAIA has no dropout. Fixed by `isolate_weighting_rng` (off in the committed runs)"),
    ("cross-modal term, RCL and FTI losses, Eq. 13 (inert)", "same", "`compute_terms`, `total_loss`: no arm-dependent code"),
    ("optimiser (Adam, lr, weight decay)", "same", "AdaDiag's extra parameters are per-parameter state only; no clipping, no scheduler"),
    ("early stopping (training loss, patience 10)", "same rule", "but it watches the total loss, which includes the TO term, so a TO change also moves where training stops"),
    ("best-epoch restore, evaluation, metrics", "same", "`evaluate` has no arm-dependent code; the module is not on the prediction path"),
    ("**the TO block**", "**different, by design**", "TVDiag: 3 unit-weight terms (total weight 3). AdaDiag: 6 terms, columns on the simplex (total weight 2) — which is why the equal-mass reference V0-simplex exists"),
]


def load(name):
    f = OUT / name
    return json.load(open(f)) if f.exists() else []


def abl(ds, name):
    recs = []
    for f in sorted((ROOT / "results").glob(f"ablation_{ds}_s*.json")):
        recs += [r for r in json.load(open(f)) if r["name"] == name]
    return MO.by_seed(recs)


def null_md():
    L = ["| dataset | `isolate_weighting_rng` | TVDiag, TO off: HR@1 / F1 (epochs) | AdaDiag, TO off | every parameter identical? |",
         "|---|---|---|---|---|"]
    for ds in MO.DS:
        for r in load(f"null_{ds}.json"):
            t, a = r["tvdiag"], r["adadiag"]
            L.append(f"| {NAME[ds]} | {'on' if r['isolate_weighting_rng'] else 'off'} | "
                     f"{t['HR@1']:.3f} / {t['f1']:.3f} ({t['epochs']}) | {a['HR@1']:.3f} / "
                     f"{a['f1']:.3f} ({a['epochs']}) | "
                     f"{'**yes**' if r['identical'] else 'no (max |Δ| ' + format(r['max_param_diff'], '.2f') + ')'} |")
    return "\n".join(L) + "\n"


def rng_section():
    rows, facts = [], {}
    for ds in MO.DS:
        iso = MO.by_seed(load(f"rng_{ds}.json"))
        if not iso:
            continue
        com = abl(ds, "ref_adadiag")
        tv = MO.load(ds)[("tvdiag", 0.1)]
        prior = abl(ds, "ref_prior")
        seeds = [r["seed"] for r in iso]
        com = [r for r in com if r["seed"] in seeds]
        same = all(all(a[k] == b[k] for k in ("HR@1", "HR@3", "f1", "epochs"))
                   for a, b in zip(iso, com))
        out = {}
        for lab, a_rs in (("committed", com), ("isolated", iso)):
            for ref_lab, ref in (("tvdiag", tv), ("prior", prior)):
                for k in ("HR@1", "f1"):
                    out[(lab, ref_lab, k)] = MO.paired(ref, a_rs, k)
        facts[ds] = dict(n=len(iso), identical=same,
                         iso={k: MO.mean(iso, k) for k in ("HR@1", "f1")},
                         com={k: MO.mean(com, k) for k in ("HR@1", "f1")},
                         tv={k: MO.mean(tv, k) for k in ("HR@1", "f1")},
                         diff={"|".join(k): v for k, v in out.items()})
        rows.append((ds, iso, com, tv, out, same))
    ps = [v[1] for *_, out, _ in rows for v in out.values()]
    adj = MO.holm(ps)
    L = ["| dataset | AdaDiag run | seeds | HR@1 | F1 | − TVDiag: HR@1 / F1 (Holm p) | − V0-simplex: HR@1 / F1 (Holm p) |",
         "|---|---|---|---|---|---|---|"]
    i = 0
    for ds, iso, com, tv, out, same in rows:
        for lab, rs in (("committed (stream shifted)", com), ("**stream aligned**", iso)):
            key = "committed" if "committed" in lab else "isolated"
            cells = []
            for ref in ("tvdiag", "prior"):
                d1, p1, _ = out[(key, ref, "HR@1")]
                d2, p2, _ = out[(key, ref, "f1")]
                idx = list(out).index((key, ref, "HR@1"))
                h1, h2 = adj[i + idx], adj[i + list(out).index((key, ref, "f1"))]
                cells.append(f"{d1:+.3f} / {d2:+.3f} ({h1:.2f} / {h2:.2f})")
            L.append(f"| {NAME[ds]} | {lab} | {len(rs)} | {MO.ms(rs, 'HR@1')} | {MO.ms(rs, 'f1')} | "
                     + " | ".join(cells) + " |")
        i += len(out)
        if same:
            L.append(f"| | *identical to the committed run, bit for bit (no dropout on {NAME[ds]})* | | | | | |")
    return "\n".join(L) + "\n", facts


def length_md():
    L = ["| dataset | seed | AdaDiag (stream aligned) stopped at | TVDiag default: HR@1 / F1 (epochs) | TVDiag stopped there: HR@1 / F1 | AdaDiag: HR@1 / F1 |",
         "|---|---|---|---|---|---|"]
    facts = {}
    for ds in MO.DS:
        # Hotel: against the aligned-stream AdaDiag (the controlled comparison); GAIA's
        # aligned runs are the committed ones, bit for bit
        cap = MO.by_seed(load(f"length_rng_{ds}.json") or load(f"length_{ds}.json"))
        if not cap:
            continue
        tv = {r["seed"]: r for r in MO.load(ds)[("tvdiag", 0.1)]}
        for r in cap:
            t, a = tv[r["seed"]], r["adadiag"]
            L.append(f"| {NAME[ds]} | {r['seed']} | {r['cap']} | {t['HR@1']:.3f} / {t['f1']:.3f} "
                     f"({t['epochs']}) | {r['HR@1']:.3f} / {r['f1']:.3f} | {a['HR@1']:.3f} / {a['f1']:.3f} |")
        m = lambda xs, k: float(np.mean(xs))
        tvm = {k: np.mean([tv[r["seed"]][k] for r in cap]) for k in ("HR@1", "f1")}
        capm = {k: np.mean([r[k] for r in cap]) for k in ("HR@1", "f1")}
        adam = {k: np.mean([r["adadiag"][k] for r in cap]) for k in ("HR@1", "f1")}
        L.append(f"| {NAME[ds]} | mean | | {tvm['HR@1']:.3f} / {tvm['f1']:.3f} | {capm['HR@1']:.3f} / "
                 f"{capm['f1']:.3f} | {adam['HR@1']:.3f} / {adam['f1']:.3f} |")
        facts[ds] = dict(tv=tvm, cap=capm, ada=adam)
    return "\n".join(L) + "\n", facts


def main() -> None:
    rev = ["| component | TVDiag vs AdaDiag | how checked |", "|---|---|---|"]
    rev += [f"| {a} | {b} | {c} |" for a, b, c in REVIEW]
    rng_md, rng_facts = rng_section()
    len_md, len_facts = length_md()
    md = ["# Is everything but the TO block the same? (README §6.15)\n",
          "## 1. Code review: every place the arm changes anything\n", "\n".join(rev) + "\n",
          "## 2. The null test: TO switched off, the two arms must be identical\n",
          "Seed 0, full training, `use_to=False`. Any difference here comes from outside the "
          "TO block (`scripts/check_controlled.py --part null`).\n", null_md(),
          "## 3. What the shifted stream did to the committed comparison\n",
          "AdaDiag re-trained with `isolate_weighting_rng`, so its dropout masks are TVDiag's. "
          "Paired over seeds; Holm over the table.\n", rng_md,
          "## 4. Training length\n",
          "TVDiag's stopping rule watches the total training loss, so changing the TO term also "
          "changes when training stops. TVDiag stopped, seed by seed, at AdaDiag's epoch:\n", len_md]
    (OUT / "CONTROLLED.md").write_text("\n".join(md))
    (OUT / "readme_table.md").write_text(null_md() + "\n" + rng_md + "\n" + len_md)
    json.dump({"rng": rng_facts, "length": len_facts}, open(OUT / "summary.json", "w"), indent=1,
              default=lambda o: list(o) if isinstance(o, tuple) else float(o))
    print((OUT / "CONTROLLED.md").read_text())


if __name__ == "__main__":
    main()
