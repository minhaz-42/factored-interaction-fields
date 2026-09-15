# Research log

Format per entry: date, hypothesis, experiment/action, result, interpretation, next step.
Failed experiments are recorded as well.

## 2026-09-12 (day 1)

**Action.** Verified challenge status from primary sources (Codabench API, HANDS
pages, workshop page, arXiv). Result: challenge closed 2026-08-31; workshop held
2026-09-08; no September 18 deadline exists in any official source. Framing
switched to "research on the SHOW3D interaction-field benchmark".

**Action.** Read the official API implementation end to end (dataset.py,
camera.py, interaction_field/__init__.py, extract_images.py, baseline/*). Verified
on one downloaded scene that labels build (298 sampled frames -> 227 tracking-valid
-> 70 labelled; both hands), that videos are 1024x1280 grayscale pinhole, that the
stereo baseline is 63.9 mm, and that projections land inside the image.

**Action.** Located and read the 1st (GOLF) and 3rd (JSSR) place reports; the 2nd
place report is not publicly accessible.

**Observation 1 (from GOLF Tab. 3).** Backbone adaptation gives -5.7 mm, RoI tokens
-2.7 mm, stereo fusion -0.19 mm, Plücker -0.6 mm on the hidden test set.
**Observation 2 (from JSSR Tab. 1).** Temporal context -0.08 mm, explicit stereo
endpoint search -0.45 mm on a clip-held-out split.
**Observation 3 (geometry).** With fx = 454 px, baseline 64 mm, depth 200-400 mm,
disparity is 73-145 px and one pixel of disparity error is 1.4-5.5 mm of depth.
Stereo is geometrically informative, so its small empirical gain points to a
modelling gap or to an error budget not dominated by depth.
**Observation 4 (label pipeline).** The target is a deterministic function of
(hand joints, object pose, known mesh); all three are released for training and
the object category is released at test time. No top solution uses the mesh.

**Hypothesis (to be tested first, before any method work).** The residual error of
direct-regression models is dominated by (a) large-magnitude far-field targets and
(b) targets whose object or hand is outside the egocentric field of view or fully
occluded, rather than by near-contact depth error. Test: error decomposition of
the InterField baseline and a frozen-DINOv3 direct-regression model on the held-out
subjects, stratified by magnitude, observability, joint, and along-ray vs lateral
component.

**Compute constraint.** Only a 16 GB Apple M5 (MPS) is available. Measured:
DINOv2 ViT-B/14 @518: 6.9 img/s fp16; ViT-L/14: 2.5 img/s. DINOv3 ViT-B/16 and
ViT-L/16 (timm, ungated) benchmark pending. Consequence: frozen backbones with
cached features; heads trained on cache; InterField baseline retrained end to end.

**Data.** Background download of the 468 training recordings (both headset views,
calibration, metadata, hand_pose v2, object_pose v1, hand profiles) started
13:40 local; ~43 GB expected.

**Next.** Finish idea ranking (docs/05), multi-dataset plan (docs/06); when data
lands: extract 10 fps frames + labels, compute dataset statistics
(docs/03), train the InterField baseline on the 8/2 subject split, run the error
decomposition.

## 2026-09-12 (day 1, continued): implementation of the pipeline

**Done.** `fif/geometry.py` (transforms, Plücker rays, GLB mesh loading with faces,
exact and soft nearest-vertex, differentiable weighted Procrustes with CPU SVD on
MPS), `fif/labels.py` (derived labels; verified identical to the official field to
1e-14 mm on the sample scene), `fif/metrics.py` (official metrics reproduce the
organisers' evaluator exactly on a half-missing test; stratified and geometric
metrics return zero on GT), `scripts/check_splits.py` (passes), `scripts/
cache_features.py` (ViT-B/16 480x384 PCA-256 int8, 7.3 img/s, 186 KB/img; PCA keeps
96.4% variance on 43k tokens), `fif/train_interfield.py` (official baseline on
MPS), `fif/models.py`, `fif/losses.py`, `fif/data_cache.py`, `fif/train_fif.py`
(config-driven, 10-15M trainable parameters, 17.5 samples/s stereo training at
batch 32). End-to-end smoke test on one scene passes.

**Observation 5 (sample scene, zero predictor = field magnitude).** Mean magnitude
by joint group: wrist 92 mm, palm 37, proximal 47, fingertips 17, distal 19. Joints
occluded by the object have smaller magnitudes (18 mm) than visible ones (34 mm):
occluded joints are typically in contact. Along-ray vs lateral split of the GT
vector itself: 13.5 vs 26.0 mm.

**HOT3D.** BOP-format zips contain object poses only; webdataset *test* clips have
no hand/object GT; *train* clips contain `hands.json` with UmeTrack
`T_world_from_wrist` + 22 joint angles (no landmarks) and `objects.json`. Landmark
recovery needs UmeTrack forward kinematics with the per-clip hand shape
(`__hand_shapes.json__`). Quest 3 camera model: FISHEYE624, 1280x1024 landscape.
Decision: HOT3D transfer is kept as the secondary experiment; implement FK +
rectification only after the SHOW3D results are in.

**Throughput (measured).** InterField ResNet-50 on MPS: 47 img/s training (batch 32),
125 img/s evaluation, 4.6 GB. Fold A at 10 fps (~75k frames) -> ~27 min/epoch, 20
epochs ~9 h. FIF hybrid heads on cached tokens: 17.5 samples/s (stereo, 1440 tokens,
batch 32), i.e. ~40 min/epoch at 5 fps.

**Figure pipeline validated.** `scripts/fig_field_geometry.py` renders a real frame
(ASC023/aria_inspecting_f5e9:000864) with projected skeletons, the posed aria mesh
wireframe and GT vectors; overlays align with the visible hands and glasses.
Panel (c) shows the soft nearest-vertex weights at tau = 400 vs 25 mm^2.

**Paper tooling.** CVPR author kit fetched into paper/; pdflatex compiles a
two-column test document with the figure.

**UmeTrack FK re-implemented (fif/umetrack_fk.py, numpy).** Reproduces SHOW3D's
stored `landmarks_3d_mm` from `joint_angles` + wrist pose + subject profile with
0.0000 mm error over 20 frames; the profile is a left-hand model and the right hand
uses the mirrored model. This unlocks HOT3D landmark recovery (HOT3D clips store
UmeTrack angles + wrist pose, not landmarks). Fisheye624 forward model implemented
(fif/fisheye.py) and sanity-checked. HOT3D objects are keyed by BOP id with names
(e.g. bottle_mustard) and metre units; alias mapping must use names, not ids.

**HOT3D-IF pipeline validated end to end on one clip.** Rectification of the
Quest 3 fisheye stream to a 1024x1280 virtual pinhole (f = 454.45, roll -90 deg)
yields an upright SHOW3D-like frame; FK landmarks and SHOW3D meshes under HOT3D
poses overlay correctly (visual check and box containment 100%). HOT3D Quest 3
training split: 1,288 clips from 7 participants (P0002 175, P0003 473, P0010 190,
P0013 197, P0017 84, P0018 49, P0021 120), enabling a participant-level split.
`scripts/build_hot3d_if.py` writes SHOW3D-format frames, labels and aux rows;
the full build (~60-80 clips) is scheduled after the SHOW3D download.

**Late day 1: tooling completed while the download runs.**
* Bug fixed: the token positional embedding was created lazily in `forward`, hence
  absent from the optimizer and incompatible with checkpoint loading; it is now a
  proper parameter sized from the token grid. All earlier smoke runs are void (they
  were smoke tests only).
* Trainer generalised: optional `extra_train` sources (ConcatDataset) and an `eval`
  block for cross-dataset validation; `scripts/eval_checkpoint.py` evaluates any
  checkpoint on any (cache, aux, subjects) set; per-joint uncertainties and errors
  are saved at evaluation (`uncertainty_val.npz`) for calibration analysis.
* Throughput (isolated step, batch 32, stereo 1440 tokens): hybrid L6/d384 48.9
  samples/s; L4/d384 69.2; L6/d256 71.9; direct L6/d384 53.4; eval 170 samples/s.
  Budget set to 10 epochs at 5 fps (~2.5 h per run).
* Feature cache: aux-label filter added (only frames with a field, a confident hand
  or a confident object are cached); frame-stride logic generalised to 30 fps HOT3D.
* Figures: scripts for Fig. 1 (teaser), Fig. 3 (field geometry), Fig. 4/5
  (qualitative, failure cases), Fig. 6b (transfer), Fig. 7a (accuracy vs cost),
  Fig. 7b/c (uncertainty calibration, sparsification), Fig. 8 (dataset statistics),
  plus a TikZ architecture figure; all tested on real sample data.
* Paper: CVPR kit, macros, verified refs.bib (27 entries), drafted Sections 1-5 and
  Limitations with result-dependent statements marked TBD; compiles without
  overfull boxes.
* Download: link-limited at ~4 MB/s; ~1.5-2 h remaining at 14:45.

## 2026-09-12 (evening): dataset-wide statistics (docs/03_dataset_stats_autogen.md)

**Result.** 87,366 labelled frames at 10 fps, identical to the organisers' count.
Field magnitude: mean 82.8 / median 30.6 / p90 243.6 mm; 30.2% of joints > 60 mm;
wrist median 96.5 mm, fingertips 13.8 mm. Joints occluded by the object: 24.0%
(median magnitude 11.8 mm, i.e. in contact). Object out of view in 1.3% of
labelled frames, partially visible in 4.8%; hands out of view in 3.0%.
Stereo: median disparity 94 px, 3.26 mm depth per pixel.

**Interpretation.** Hypothesis (b) of day 1 -- that unobservable targets drive a
large share of the residual error -- is not supported: out-of-view cases are rare.
Hypothesis (a) remains: the mean is dominated by the far-field tail (30% of joints
beyond 60 mm, wrist and proximal joints, non-interacting hands). Observability is
kept as an evaluation stratum, not as a central claim. The 24% occluded-in-contact
joints are a second stratum of interest: for them the answer is essentially "zero
vector", and the question is whether models recognise contact.

**HOT3D-IF built.** 63 Quest 3 training clips (9 per participant), 10 fps, both
streams rectified; counts in the log below.

**HOT3D-IF v1 counts.** 63 clips (9 per participant), 3,150 frames at 10 fps,
2,381 labelled (left+right fields), 1.1 GB of rectified frames. Labelled frames:
HOT3D-train {P0002 295, P0003 281, P0010 347, P0013 382} = 1,305; HOT3D-eval
{P0017 368, P0018 332, P0021 376} = 1,076. Objects: keyboard 459, bowl 307, OJ 260,
vase 246, mug 160, mouse 134, ... (21 aliases). Field magnitude median 35.0 mm, mean
50.8, p90 125 (lighter tail than SHOW3D); 81.6% joints visible; median joint depth
378 mm. Decision: enlarge HOT3D-train with 120 more clips (train participants only)
so that the HOT3D -> SHOW3D direction is not dominated by data scarcity.

**Cache restart.** The token cache ran at ~3 files/min while sharing the GPU with the
InterField baseline; the baseline was paused (SIGSTOP, auto-resume when the cache
finishes) and the cache script was restarted with a 3-batch prefetch thread (skips
finished scenes). 124,954 images pass the aux filter (frames with a field, a
confident hand or a confident object).

## 2026-09-12 (end of day): work paused

**Paused at 18:05 with all compute stopped.** See `STATUS.md` for the full handover.

**Where we stopped.** DINOv3 feature cache at 15,995 of 124,954 images (12.8%);
HOT3D extra build at clip 31 of 120 with no labels written; zero valid experiments.

**Incidents (root causes recorded so they are not repeated).**

1. A shell pipeline of the form `python ... | grep | tee log && echo "pipeline done"`
   reports success when the Python is killed, because the pipeline's exit status is
   `tee`'s. A false "pipeline done" therefore fired the downstream waiters and started
   training on a 10% cache. Two fold-B runs produced this way were deleted as void.
   Fixed with `set -o pipefail` and a `CACHE_COMPLETE` marker emitted by the cache
   script itself.
2. Three GPU processes ran concurrently (two caches and a trainer), putting two into
   uninterruptible wait and cutting throughput from about 12 to 5 img/s. This is the
   documented 16 GB single-model failure mode. Rule reaffirmed: one GPU process at a
   time on this machine.
3. Cache batch size 32 measured slower than 16 (6.6 vs 9.1 img/s); reverted.
4. The token positional embedding was built lazily inside `forward`, so it never
   entered the optimizer and broke checkpoint loading. Fixed; pre-fix checkpoints void.

**Decisions taken.**

* The InterField baseline will not be retrained for fold A. Our fold-A validation
  split was verified byte-identical to the organisers' published dev split (11,948
  frames, 20,163 hand-fields), so their 60.47 mm is the authoritative B1 number. It is
  wired into the table generator via `tables/b1_official_reference.json`. Saves 9 h.
* Early stopping (patience 3, minimum improvement 0.05 mm) added to the trainer.
* Scope and hardware choice left open for the user: full plan ~48 h, trimmed
  ~16-20 h, rented GPU ~3 h.

**Next action on resume.** Restart the feature cache as a single process
(`--batch 16`), wait for `CACHE_COMPLETE`, then run `scripts/check_splits.py` before
any training.
