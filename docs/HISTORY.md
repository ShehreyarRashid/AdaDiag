# How AdaDiag++ came about

Each step was screened on development seeds 0–2 and confirmed on seeds not used before.
The full log of every step is in the git history of this repository (the versions before
v2.0).

| Step | What was added | Outcome |
|---|---|---|
| TVDiag | Base method, re-implemented without DGL | Matches the released code on seeds 5–9 |
| AdaDiag-DL | Modality weights from gradient alignment with the diagnosis losses | Ties TVDiag |
| AdaDiag+ (v1.21) | Alert-type counts and Eq. 13's learnable loss weights | F1 up on all three datasets; Sock Shop HR@1 −0.008 |
| **AdaDiag++ (v1.22)** | Three AdaDiag+ models, outputs averaged | Above AdaDiag+ on HR@1 and F1 for all three datasets |

## What was tried and dropped

- Attention over neighbours, a skip path from each instance's own features, weight averaging.
- A failure-type readout at the likely root, and a neighbour contrast in the RCL head.
- Label smoothing, dropout on the counts, heads that read the raw counts.
- An LSTM aggregator, and the TVDiag paper's exact specification where it differs from its code.

None of these is part of the final method: each either failed its screen or was
superseded on unused seeds. Their switches remain in the code, off by default.

## Notes for readers of the code

- Comments that cite "v1.22 README section 6.x" refer to the full research log in the git
  history (the README of the last version before v2.0).
- In `scripts/plus2.py`, `plus` is AdaDiag+ and `plus|ens3` is AdaDiag++.
- `results/plus2/PLAN.md` is the record written before the confirmation runs; it is kept
  as written.
