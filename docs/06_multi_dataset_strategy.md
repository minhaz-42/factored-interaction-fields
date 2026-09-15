# 06. Multi-dataset strategy

## Candidate external datasets

| Dataset | License | Modality / cameras | Hand labels | Object labels / meshes | Calibration | Contact | Joint-level field derivable? | Role for us | Conflict with SHOW3D rules |
|---|---|---|---|---|---|---|---|---|---|
| HOT3D (Banerjee et al., CVPR 2025) | Sequence data CC BY-SA; hand data CC BY-NC-SA; object models CC BY-SA with no-sale clause | Aria (1 RGB + 2 mono) and Quest 3 (2 x 1280x1024 mono fisheye, 30 fps); 19 subjects, 33 objects, 833 min; HOT3D-Clips on HF as 150-frame webdataset tars (~85-100 MB each) | UmeTrack (same 21-landmark skeleton and per-subject profiles as SHOW3D) and MANO | Mocap 6-DoF poses; GLB meshes; 22 objects identical to SHOW3D's | Fisheye intrinsics per stream + per-frame T_world_from_camera | derivable (posed mesh) | **Yes, exactly the SHOW3D definition** | transfer evaluation both directions; joint training; branch pretraining | none (public, licensed; must be declared as external data) |
| ARCTIC (Fan et al., CVPR 2023) | MPI non-commercial research license; no redistribution; derived models need permission | 8 exocentric RGB + 1 egocentric RGB (Aria-style headset), 10 subjects, 11 articulated objects, 2.1M frames | MANO meshes | articulated meshes (two rigid parts), mocap | yes | dense vertex contact and fields | approximately: MANO joints differ from UmeTrack landmarks (no palm centre; different tip definitions); articulated objects break the single-rigid-object assumption | evaluation-only or hand-branch pretraining; low priority | none for research, but weight release restrictions |
| AssemblyHands (Ohkawa et al., CVPR 2023) | CC BY-NC 4.0 | 490k egocentric + exocentric images (custom headset), 3D hand keypoints | 3D keypoints (4.2 mm error) | none | yes | none | **No** (no object geometry) | optional hand-branch pretraining only | none |
| HOI4D (Liu et al., CVPR 2022) | CC BY-NC 4.0 | 2.4M egocentric RGB-D frames, 9 subjects, 800 instances / 16 categories | MANO | category-level object poses with reconstructed meshes | yes | derivable but noisy | approximately (MANO joints; reconstructed meshes) | secondary transfer target; low priority | none |
| Ego4D / Ego-Exo4D | Ego4D license agreement (no redistribution); hand pose annotations exist in Ego-Exo4D | egocentric video | Ego-Exo4D: 3D hand poses (68k manual, 4.3M auto) | none | yes (Aria) | none | **No** | not used | n/a |
| H2O (Kwon et al., ICCV 2021) | research license (site registration) | 4 RGB-D + egocentric | MANO | 6-DoF poses, meshes | yes | derivable | approximately (MANO) | possible small transfer target | none |
| TACO (Liu et al., CVPR 2024) | research | third-person + egocentric | MANO | tool/object meshes | yes | interaction fields (scalar) | approximately | not used initially | none |
| HO-Cap (2024), OakInk2 (2024) | CC BY 4.0 / CC BY-SA 4.0 | multi-view RGB-D, exocentric | MANO | meshes, poses | yes | derivable | approximately | not used initially | none |

## Compatibility analysis

Only HOT3D is label-compatible without approximation: identical 21-landmark
UmeTrack skeleton (the SHOW3D hand_pose README states the convention "follows the
HOT3D / UmeTrack convention"), identical canonical meshes for 22 objects (SHOW3D's
`object_pose/README.md` maps aliases to HOT3D LIDs), the same Quest 3 stereo
camera pair, and world-frame poses per frame. The joint-level field can therefore
be generated with the *same* code path (`build_interaction_labels`). Differences
to handle: (1) HOT3D Quest 3 images are fisheye and 30 fps; SHOW3D releases
pinhole-undistorted 60 fps video, so HOT3D frames must be rectified to a virtual
pinhole camera with SHOW3D-like intrinsics; (2) HOT3D "world" is a static mocap
frame, SHOW3D "world" is the moving rig frame -- irrelevant for a difference of
two points; (3) HOT3D hands are annotated by mocap-driven fitting, SHOW3D by
ego-exo triangulation; label noise differs.

All MANO-based datasets require a joint-convention mapping (MANO 21 joints vs
UmeTrack 21 landmarks with palm centre and different tip points); forcing this
mapping introduces systematic offsets of several millimetres, comparable to the
GT noise, so they are not used for field supervision.

## Decision

1. **Primary: SHOW3D** for training and held-out-subject evaluation.
2. **Secondary: HOT3D-Quest3 clips** (a subset: e.g. 60 training clips and all
   test-split clips containing SHOW3D-shared objects, ~10-15 GB) for
   (a) zero-shot transfer SHOW3D -> HOT3D, (b) HOT3D -> SHOW3D, (c) joint training,
   (d) pretraining the object-pose branch of the factored model. Hypothesis: the
   factored model's transfer gap is smaller than the direct model's because the
   geometry it relies on is shared while appearance is not.
3. **Not used for supervision**: ARCTIC (license, articulation, MANO), AssemblyHands
   (no objects), HOI4D (noisy category-level meshes), Ego4D (no 3D). AssemblyHands
   is kept as an optional hand-branch pretraining ablation if time permits.

## Leakage checks (to be automated in `scripts/check_splits.py`)

* SHOW3D: subject-level split; assert no `(subject, scene)` appears in two splits;
  assert validation subjects are absent from any feature cache used for training.
* HOT3D: split by participant id as in the official `clip_splits.json`; assert no
  HOT3D participant overlaps with SHOW3D subjects (different ids; different capture
  campaigns; verified by id sets). Objects overlap by design (that is the point of
  the protocol) and is declared.
* Test subjects `BBL925, KHE522, SHE109` are never downloaded for training; their
  labels are not released.

## Data budget (local)

SHOW3D train subset ~43 GB (downloading), extracted frames 18-37 GB depending on
fps/views, feature cache 24-48 GB depending on compression, HOT3D subset 10-15 GB.
Total fits in the ~176 GB free at start, with margin for checkpoints.

## HOT3D compatibility: verified facts (2026-09-12)

* HOT3D-Clips training tars (`train_quest3/clip-*.tar`, ~100 MB, 150 frames at 30 fps,
  two Quest 3 monochrome fisheye streams 1201-1/1201-2 at 1280x1024) contain
  `hands.json` (UmeTrack `T_world_from_wrist` + 22 joint angles, MANO params, amodal
  boxes, visibility), `objects.json` (BOP id, `object_name`, `T_world_from_object`,
  boxes, masks, visibility; **metres**), `cameras.json` (FISHEYE624 params,
  `T_world_from_camera`) and `__hand_shapes.json__` (per-clip UmeTrack hand model).
  Test-split clips carry no hand/object ground truth.
* Our numpy UmeTrack FK reproduces SHOW3D landmarks exactly and places HOT3D
  landmarks inside HOT3D's own amodal hand boxes (left = direct model, right =
  mirrored) in 100% of tested cases.
* SHOW3D's bundled GLB meshes posed with HOT3D `T_world_from_object` (converted to
  mm) project to HOT3D's annotated object boxes within 1 px through our Fisheye624
  model: the canonical object frames are identical, so labels can be generated with
  the same `nearest_neighbor_vectors` code.
* Object identity must be mapped by `object_name` (e.g. `bottle_mustard` ->
  `mustard`), not by BOP id (BOP ids are renumbered; SHOW3D's LID table refers to the
  original HOT3D library ids).

## HOT3D-IF protocol (proposed)

1. Rectify each fisheye stream to a virtual portrait pinhole with SHOW3D-like
   intrinsics (1024x1280, f = 454.45, principal point centred) and SHOW3D's
   orientation (hands below, left hand on the left).
2. For each frame and hand with UmeTrack pose, define the target object as the
   SHOW3D-shared object (21 aliases) whose posed mesh is nearest to any joint of that
   hand; discard hands whose nearest distance exceeds 150 mm (no interaction) and
   frames without any shared object.
3. Fields, camera-frame joints, object pose, visibility and observability follow the
   SHOW3D derivation exactly (`fif.labels` schema), with the alias given as input.
4. Splits by HOT3D participant id (from `info.json`); evaluation uses a disjoint set
   of training-split participants because test-split clips have no labels.
