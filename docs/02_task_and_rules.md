# 02. Task definition, data flow, metrics and rules

Everything here was read from the official API implementation
(`third_party/SHOW3D-dataset-api`, commit c2064dc) and the Codabench pages, not
from summaries. File references point to that vendored copy.

## Input

* Two synchronized egocentric Meta Quest 3 monochrome cameras per frame:
  `headset0`, `headset1`. Released videos are H.264 MP4, 60 fps, **1024 wide x
  1280 tall (portrait)**, faces blurred, fisheye-undistorted to a plain pinhole
  (`DistortionModel: PinholePlane`). Decoded frames are 3-channel with identical
  channels (verified on `ASC023/aria_inspecting_f5e9`).
* Calibration per view per scene: shared intrinsics `fx, fy, cx, cy, ImageSizeX,
  ImageSizeY` (e.g. fx=fy=454.45, cx=512.9, cy=633.8 for the sample scene) and a
  per-frame 4x4 `T_WorldFromCamera` under `T_WorldFromCamera_by_index[i]` with an
  `is_synthesized` flag. Stereo baseline measured on the sample scene: 63.9 mm,
  relative rotation 0.77 deg.
* "World" is the **back-mounted rig frame**, which moves with the subject. It is
  consistent across the 10 cameras and all annotations within a frame, but not
  across frames (scenes/README.md). The field is a difference of two world points,
  so translation cancels; only the rotation matters when moving between frames.
* Object category (`object_alias`) is present in both manifests, including the
  test manifest. GOLF (1st place) conditions its object locator on this category.
* Single-view setting = `headset0` only (`multiview=False`); multi-view =
  both headsets. Targets are view-independent; the choice must be declared.

## Target

* Hands are 21-landmark UmeTrack skeletons (HOT3D convention: 0-4 fingertips
  thumb..pinky, 5 wrist, 6-7 thumb intermediate/distal, 8-10 index, 11-13 middle,
  14-16 ring, 17-19 pinky proximal/intermediate/distal, 20 palm center).
* Object geometry: canonical HOT3D mesh vertices bundled as `.glb`
  (`show3d/assets/objects/<alias>.glb`, 1.7k-5.3k vertices), posed by the per-frame
  world-from-object `R, t` (`ObjectPoseFrame.pose_vertices`).
* Field: for each joint `p_j`, `v_j = argmin_{q in V_obj} ||q - p_j|| - p_j`
  (`nearest_neighbor_vectors`, exact nearest **vertex**, not nearest surface point
  on a triangle). Two fields per frame: `left_to_object`, `right_to_object`, each
  `(21, 3)` in world millimetres.
* Validity (`build_interaction_labels`, `Show3DInteractionFieldDataset.__getitem__`):
  a frame yields labels only if headset tracking is valid for **all selected views**
  (no synthesized extrinsic), the object pose exists with `confidence > 0.5`, and a
  hand has `confidence > 0.5` with landmarks. Each hand is validated separately; a
  frame may have one or two fields. Invalid frames still appear in the test
  manifest and may be predicted; they are simply not scored.
* Hand pose version `v2` (v1 landmarks are systematically undersized; see
  hand_pose/README.md). Object pose version `v1`.

## Data flow (train)

1. Manifest `train_manifest_202607.jsonl`: 468 rows `(subject_id, scene_id,
   object_alias)`, 10 subjects, 21 objects, 57 action verbs.
2. `extract_images.py --fps k --save-labels` decodes each MP4 once, keeps every
   60/k-th frame as grayscale JPEG, writes `index.jsonl` (one row per frame per
   view) and `labels.jsonl` (one row per labelled frame, keyed by
   `sample_id = "<subject>/<scene>:<frame:06d>"`). At 10 fps headset0 the official
   baseline reports 87,366 labelled frames.
3. The baseline expresses targets in the camera frame (`V_cam = V_world @ R`) and
   rotates predictions back with `R_world_from_camera` (exact; ADE is invariant).

## Data flow (test)

* `test_manifest_5fps_202607.jsonl`: 20,042 frames from 139 recordings, subjects
  `BBL925` (4,768), `KHE522` (8,867), `SHE109` (6,407); 16 objects; frame stride 12
  (5 fps); 16-150 frames per recording. Five training objects never appear in the
  test set: `bbq, ranch, mustard, vegetables, waffles`. Test-subject hand/object
  labels are withheld (HF index: `has_hand_pose=False, has_object_pose=False`).
* Submission: zip with `predictions.jsonl` at root; one JSON per line with
  `sample_id`, `left_to_object`, `right_to_object` as `(21,3)` lists or `null`.
  `validate_submission.py` rejects unknown sample ids and non-(21,3) shapes.

## Metrics (`evaluate_prediction_records`, `_MetricAccumulator`)

Per directed field, over valid targets:
* `ade_mm = sum_j ||pred_j - gt_j||_2 / (21 * #predicted targets)` (mean per-joint
  endpoint error over predicted targets only).
* `acc@t = fraction of joints with error <= t`, t in {10, 50, 100} mm, over predicted
  targets only.
* `recall = (#valid targets - #missing predictions) / #valid targets`.
* `mean_ade_mm`, `mean_recall`: unweighted mean of the two hand fields.
* Leaderboard primary key `official_score` (lower better) combines accuracy and
  coverage with a withheld formula; zero-prediction scores 80.35. GOLF's numbers
  (official 27.47 vs ADE 27.82 at recall 1) show that at full recall the score is
  within ~0.4 of mean ADE, so it is roughly ADE plus a coverage penalty (INFERRED).
* ACC (m/s^2), from the SHOW3D paper, is implemented in `baseline/evaluate.py` as
  the discrete second difference of the field over consecutive sampled frames; it
  is not a leaderboard column.

## Rules (verbatim sources: Codabench Terms, Overview; HANDS general rules)

* "Use the SHOW3D dataset only under ... CC BY-NC 4.0."
* "Do not attempt to recover hidden labels or tune submissions manually against the
  evaluation server."
* "Submissions must be generated by a model or deterministic algorithm that can be
  described by the participant."
* "Participants must declare their input setting and any external training data
  in their submission description. The live leaderboard is combined; organizers
  publish separate final tables for methods with and without external training data."
* HANDS general rules: "Teams may use any publicly available and appropriately
  licensed data (if allowed by the track)"; "Any supervised/unsupervised training on
  the validation/testing set is not allowed"; winners must provide reproducible code.
* Temporal information: the task is per-frame; nothing forbids using neighbouring
  frames of the released test videos (JSSR used seven stereo pairs and was ranked).
  The README states participants "may choose their own temporal sampling strategy
  for training and inference". UNCERTAIN only in the sense that no rule addresses
  it explicitly; the precedent (JSSR, 3rd place) makes it de facto allowed.
* Auxiliary supervision from released training annotations (hand landmarks, object
  poses, calibration) is allowed; GOLF trains a hand/object locator from them.
* Licensing: dataset, code and bundled meshes are CC BY-NC 4.0; HOT3D-derived
  meshes require attribution per `assets/objects/ATTRIBUTION.md`. The arXiv paper
  itself is CC BY 4.0 (paper license, not dataset license).

## Reproducibility requirements (for our paper, not the closed challenge)

Subject-level splits only; never mix frames of one recording across train/val;
record every config; report on held-out training subjects because test labels
are withheld and the server is closed.
