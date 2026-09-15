# Documentation index

**Start here when resuming: `../STATUS.md`** (what is done, what is not, where we paused, how to resume).

| File | Content | Status |
|---|---|---|
| 01_challenge_status.md | Verified challenge timeline, closure, winners, publication routes | final |
| 02_task_and_rules.md | Task definition from the API code, data flow, metrics, rules | final |
| 03_dataset.md | Dataset facts; dataset-wide statistics pending pipeline | partial |
| 04_literature_review.md | InterField, GOLF, JSSR, related representations, open problems | final |
| 05_research_ideas.md | 12 directions with evidence, ranking, selection, seven questions | final |
| 06_multi_dataset_strategy.md | External datasets, licenses, HOT3D compatibility (verified), HOT3D-IF protocol | final |
| 07_method_formulation.md | Notation, equations (1)-(15), derived labels, inference, decomposition, metrics | final |
| 08_experiment_plan.md | Splits, baselines, ablations, tracked fields, tables, figures, schedule | final |
| 09_figure_study.md | Figure conventions from ten papers and consequences for ours | final |
| research_log.md | Chronological hypotheses, actions, results, decisions | ongoing |
| 03_dataset_stats_autogen.md | Measured dataset statistics (generated) | final |
| ../STATUS.md | Handover: completed / not completed / pause point / resume commands | current |

Code map: `fif/geometry.py` (transforms, rays, meshes, soft NN, Procrustes), `fif/labels.py`
(derived labels), `fif/metrics.py` (official + stratified + geometric), `fif/data_cache.py`,
`fif/models.py`, `fif/losses.py`, `fif/train_fif.py`, `fif/train_interfield.py`,
`fif/umetrack_fk.py`, `fif/fisheye.py`, `fif/viz.py`; `scripts/` for pipeline, HOT3D build,
experiments, tables and figures; `paper/` LaTeX (CVPR kit) with `refs.bib`.
