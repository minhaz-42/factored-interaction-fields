# 08. Experiment, ablation, figure and table plan

## Splits (subject level; automated check in scripts/check_splits.py)

* Fold A (official dev split, also used by GOLF): train ASC023, SPI102, LWA828,
  YZH016, XXI103, PCW023, MHA016, MMO925; validate XYZ109, LYA722.
* Fold B (robustness): train the other eight; validate MHA016, MMO925 (both cover
  several objects). Main comparisons are reported on both folds where compute
  allows; ablations on Fold A.
* Never mix recordings across splits; caches are built per subject.
* Official test subjects (BBL925, KHE522, SHE109) are not downloaded for training.

## Sampling and inputs

* Frames extracted at 10 fps for the InterField baseline (parity with the official
  numbers) and 5 fps for cached-feature models (compute budget), both views.
* Cached DINOv3 ViT-B/16 patch tokens at 480 x 384, last block, PCA 768 -> 256 and
  int8 per-token scaling (validated against fp16 on a two-subject subset before
  adoption). ViT-L/16 only for the final comparison if time allows.

## Baselines

| Id | Model | Views | Notes |
|---|---|---|---|
| B0 | predict zeros; predict train-mean field | - | sanity anchors |
| B1 | InterField ResNet-50, 224^2 full frame (official code, retrained on Fold A/B on MPS) | mono | official baseline |
| B2 | DINO-Direct: frozen DINOv3-B tokens, 42 joint queries, direct regression | mono | consensus architecture without adaptation |
| B3 | B2 + second view + Plücker ray embeddings | stereo | strong image-encoder + stereo baseline (GOLF components) |
| B4 | B3 trained on SHOW3D + HOT3D-Quest3 subset | stereo | multi-dataset baseline (if HOT3D pipeline works) |
| Ours | B3 + factored geometric head + precision-weighted fusion + auxiliary losses | stereo (and mono) | FIF |

## Ablations (Fold A)

1. Heads: direct only / geometric only / mean fusion / learned gate / precision fusion.
2. Auxiliary losses: without L_joint; without L_obj; without L_vis; without NLL terms.
3. Soft nearest-vertex temperature: fixed 25, 100, annealed; hard argmin at train.
4. Object pose parametrisation: control points + Procrustes vs 6D rotation + t.
5. Object category: given vs predicted (21-way head) vs none (direct head only).
6. Views: mono vs stereo; with and without Plücker rays.
7. Backbone: ViT-B vs ViT-L (if feasible); feature cache PCA vs full.
8. Temporal: +-1 neighbouring cached frames (JSSR-style) as a check.
9. Training data: SHOW3D only vs + HOT3D; branch-wise HOT3D pretraining.
10. Seeds: 2 seeds for the main rows; report mean and spread.

## Tracked fields per run (experiments/<run_id>/config.yaml + metrics.json)

dataset(s), train split, val split, backbone, image resolution, views, lr, batch
size, epochs, optimizer, scheduler, augmentation, parameter count (trainable /
total), peak GPU (MPS) memory, training time, inference time per frame, ADE
(mean/left/right), acc@10/50/100, recall, and the geometric metrics of docs/07 Sec.
7.9, plus stratified ADE tables.

## Tables

1. Main comparison on held-out subjects (Fold A and B): B0-B4, Ours; ADE, acc@k,
   recall, SD, direction error, penetration, params, time.
2. Ablation study (Fold A).
3. Cross-dataset: SHOW3D -> HOT3D, HOT3D -> SHOW3D, joint; B3 vs Ours.
4. Single-view vs multi-view for B2/B3/Ours, with along-ray vs lateral error.
5. Difficulty analysis: magnitude strata, visibility, observability, joint group,
   object, subject.
6. Efficiency and model size.

## Figures (all qualitative content from real frames and real predictions)

1. Teaser: real SHOW3D stereo frame -> hand/object localisation -> camera-frame
   geometry -> factored representation -> predicted field with GT overlay.
2. Architecture (vector diagram).
3. Geometry of the interaction field: 21 joints, posed mesh, nearest-vertex
   vectors, soft assignment, symmetry invariance.
4. Qualitative comparison B1 / B3 / Ours on held-out frames: overlays + 3D views.
5. Failure cases: out-of-view object, full occlusion, wrong association.
6. Cross-dataset transfer: SHOW3D and HOT3D frames with predictions; transfer bars.
7. Ablation / accuracy-efficiency trade-off and error-vs-uncertainty curves.
8. Dataset analysis: field-magnitude histogram, observability strata, per-object
   counts (docs/03).

## Compute schedule (local M5, sequential)

Feature extraction ~2-3 h; B1 training 5-6 h per fold; each head run 30-90 min;
HOT3D subset download/rectification ~2 h; total ~3 machine-days including
ablations. Anything that does not fit is reported as not run.

## HOT3D-IF (cross-dataset) plan, added after compatibility verification

* Source: HOT3D-Clips `train_quest3` (ground truth available), 7 participants.
  Participant split: HOT3D-train = {P0002, P0003, P0010, P0013}; HOT3D-eval =
  {P0017, P0018, P0021}. Subset budget: ~40 training clips + ~24 evaluation clips
  (10 fps frames, both streams rectified), ~6-7 GB of downloads.
* Frames and labels are produced by `scripts/build_hot3d_if.py` in SHOW3D's format;
  tokens are cached with `scripts/cache_features.py --source-fps 30 --extract-fps 10`.
* Experiments (fold-A checkpoints): (1) zero-shot SHOW3D -> HOT3D-eval for B3 and FIF
  (`scripts/eval_checkpoint.py`); (2) HOT3D-train -> HOT3D-eval and HOT3D-train ->
  SHOW3D fold-A validation (config with `eval:` block); (3) SHOW3D + HOT3D-train ->
  SHOW3D fold-A validation (`extra_train:` block), for B3 and FIF.
* Leakage check: HOT3D participant ids never coincide with SHOW3D subject ids
  (different id formats); objects overlap by design and are declared.
