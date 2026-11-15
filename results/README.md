# results/

| Folder | Contents |
|---|---|
| `plus2/` | The experiment behind the README tables. `PLAN.md`, `plan.json`: the rule and decisions, written before the confirmation runs. `runs/<ds>/<variant>__s<seed>.json` and `.npz`: one record and the test logits per run (`plus` = GradFusion+, `align` = GradFusion-DL, `tvdiag`; seeds 0–2 development, 30–34 and 5–9 confirmation, +100 and +200 the ensemble members). `PLUS2.md`, `summary.json`, `confirm_<ds>.json`, `fig_plus2.png`: the report. `probe.json`: the feature probe. |
| `fast_supcon/` | v2.2: GradFusion with the optional `fast_supcon=True` on seeds 5–9 and 30–34, against the exact runs above (docs/CODE_REVIEW.md, section 5). |
| `baselines/` | TVDiag and GradFusion-DL on seeds 5–9 (same settings), run before `plus2/`. |
| `tvdiag_official/` | TVDiag's released code run unchanged (seeds 2, 5–9), with logs and the authors' own logs. |
| `paper_tables/multimodal/` | Tables 4–5 in the TVDiag paper's format, with GradFusion. |
