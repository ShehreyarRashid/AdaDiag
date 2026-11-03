# Selection plan (written 2026-11-03T11:16, before any confirmation run)

Finalists: AdaDiag (the method as proposed) and Type 4 (the design that learns from the
diagnosis losses), plus the two highest screening scores among the remaining candidates
(adadiag_fid, v2_global, warmup, trace_third, trace_tvdiag) once the Hotel stream-aligned
re-runs are in. Each finalist runs on fresh seeds 5-9 on GAIA and Hotel, 1 thread,
500-epoch cap, `isolate_weighting_rng=True`, next to its matched TVDiag (the fidelity-loop
TVDiag for the fidelity-loop AdaDiag).

Rule: Score = mean of the four paired differences from the matched TVDiag (GAIA HR@1, GAIA F1, Hotel HR@1, Hotel F1) on the fresh seeds 5-9. The best AdaDiag is the highest score, provided no cell is significantly worse than TVDiag (paired t, p < 0.05). If no candidate's score is significantly above zero (one-sample t over the 10 per-seed scores, Holm over the finalists), the selection is reported as a tie on performance and decided by mechanism (learns from the diagnosis losses, graded weights), then by training cost.

## Finalists (fixed 2026-11-03T11:36+05:00, after screening, before any confirmation result)

Screening scores after the Hotel stream-aligned re-runs: AdaDiag +0.0090, trace fixed at TVDiag's role +0.0080, Type 4 +0.0072, trace fixed at 1/3 +0.0065, fidelity loop +0.0054, warm-up +0.0044, V2-global -0.0006.
Finalists: adadiag, align (Type 4), trace_tvdiag, trace_third. References: tvdiag.
