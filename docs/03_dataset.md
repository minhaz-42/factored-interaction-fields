# 03. Dataset analysis (SHOW3D, interaction-field subset)

Facts below come from the HuggingFace repository metadata and index parquets,
the dataset READMEs, the official API, and direct inspection of downloaded files.
Statistics marked [PENDING] are computed by `scripts/dataset_stats.py` once the
training subset is on disk and are filled in from `tables/dataset_stats.json`.

## Whole dataset (HF `facebook/show3d-dataset`, revision 2026-07-31)

| Item | Value | Source |
|---|---|---|
| Recordings | 2,137 (1,689 train + 448 test scenes) | dataset_index_{train,test}.parquet |
| Subjects | 38 in the paper; 32 train + 6 test in the release (`AAN828, BBL925, BCO829, KHE522, NCH828, SHE109` are test) | README, parquet |
| Cameras | 2 egocentric Quest 3 (released) + 8 exocentric rig cameras (not released: `has_rig*` False for every scene) | parquet |
| Video | H.264 MP4, 60 fps, 1024 x 1280 portrait, monochrome (3 identical channels), face-blurred, fisheye-undistorted (PinholePlane) | recording_info.json, calibration, decoded frame |
| Mean length | 33.5 s (about 2,012 frames); sample scene 1,786 frames | README, recording_info |
| Annotations | hand pose v2 (1,689 train scenes, 32 subjects, ~3.5 M frames; 77.4% / 82.8% of frames with left/right conf > 0.5), object pose v1 (468 scenes, 10 subjects, 21 aliases, 830,597 frames, 74.5% with a pose), captions | hand_pose/README, object_pose/README |
| Objects | 27 aliases in the dataset; 22 shared with HOT3D (identical meshes); 21 in the interaction-field task | object_pose/README |
| Size | 227 GB total; 124 GB video (4,274 MP4, median 26 MB); hand_pose v2 66.6 GB | HF file metadata |
| License | CC BY-NC 4.0 (dataset, API, meshes) | LICENSE, README |

## Interaction-field subset (train manifest)

| Item | Value |
|---|---|
| Recordings | 468, 10 subjects (ASC023 62, SPI102 62, LWA828 58, YZH016 58, XXI103 45, PCW023 43, MHA016 42, MMO925 40, XYZ109 32, LYA722 26) |
| Objects (recordings) | birdhousetoy 47, dinotoy 42, keyboard 42, mug 41, vase 37, dumbbell 36, milk 33, brushholder 30, balandabowl 26, orangejuice 26, aria 23, bbq 16, mustard 18, ranch 16, mouse 8, canparmesan 5, cansoup 5, cantomatosauce 5, vegetables 5, waffles 5, mug2 2 |
| Actions | 57 verbs (inspecting 67, shaking 37, washing 29, pouring-out 26, tapping 23, pick-up-put-down 22, ...) |
| Subject x object | sparse: each subject handles 4-9 objects; several objects are handled by 1-3 subjects only (bbq/mustard/ranch: LWA828, MMO925, SPI102; cans/vegetables/waffles: XXI103 only; mouse: XYZ109 only) |
| Source frames | 830,597 per view at 60 fps; 87,366 labelled frames at 10 fps headset0 (official baseline) |
| Download used here | headset0/1 videos + calibration + metadata for the 468 recordings (~27 GB), hand_pose v2 (~16 GB), object_pose v1 (0.3 GB) |

## Hidden test set

20,042 frames at 5 fps (stride 12) from 139 recordings of `BBL925` (4,768), `KHE522`
(8,867), `SHE109` (6,407); 16 objects; five training objects never appear
(bbq, ranch, mustard, vegetables, waffles); objects mouse, cans, mug2, vegetables
and waffles are handled by a single training subject each, so the test set probes
subject generalisation on thinly covered objects. Labels withheld.

## Calibration and stereo geometry

Pinhole intrinsics shared by both headset cameras within a scene (sample: fx = fy =
454.45 px, cx = 512.9, cy = 633.8); per-frame `T_WorldFromCamera` with
`is_synthesized` flags (sample scene: 433 of 1,786 frames synthesized = 24%).
Stereo baseline 63.9 mm (dataset mean; sample relative rotation 0.77 degrees).
Joint depths in camera 0: median 308 mm, p90 441 mm, giving a median disparity of
94 px and a median depth resolution of 3.26 mm per pixel of disparity.

## Validity funnel (what becomes a label)

frame -> headset tracking valid in all selected views -> object confidence > 0.5 ->
hand confidence > 0.5 with landmarks -> (21,3) nearest-vertex field per hand.
Dataset-wide funnel: see below.

## Field statistics (dataset-wide, 10 fps, 87,366 labelled frames; see 03_dataset_stats_autogen.md)

Our derived labels reproduce the organisers' count exactly (87,366 labelled
frames at 10 fps). Over 3.10 M joint vectors: mean 82.8 mm, median 30.6 mm, p90
243.6 mm; 18.6% of joints within 10 mm, 28.9% within 15 mm, 30.2% beyond 60 mm.
The distribution is strongly joint dependent: fingertips median 13.8 mm (52% within
15 mm), distal 17.4, intermediate 29.6, thumb 30.7, palm 37.4, proximal 47.6,
wrist 96.5 mm (87% beyond 60 mm). Per object, medians range from 22 mm (cans) to
101 mm (mouse, whose recordings involve typing with the other hand far away); mug2
and mouse have the heaviest tails. The mean ADE of any predictor is therefore
dominated by far-field vectors of the wrist, proximal joints and non-interacting
hands.

## Visibility and observability (dataset-wide, view 0)

* 95.1% of labelled joints project inside the reference image; 76.0% are visible
  (inside and not occluded by the posed object mesh). Occluded joints are mostly in
  contact: median magnitude 11.8 mm vs 39.8 mm for visible joints.
* The object is at least 90% inside view 0 in 93.9% of labelled frames, partially
  visible in 4.8%, and essentially out of view (< 10% of vertices) in 1.3% (1.3% in
  view 1 as well). Hands are out of view in 3.0% of hand targets.
* Consequence: the hypothesis that unobservable targets account for a large share
  of the residual error is **not supported** at the frame level; out-of-view cases
  are rare and will be reported as a stratum, but the error budget must be sought in
  the heavy magnitude tail and in occluded-in-contact joints.

## Validity funnel (dataset-wide, 10 fps)

138,579 sampled frames -> 126,321 headset-tracking valid (91.2%) -> 100,010 with a
confident object pose (72.2%) -> 87,366 with at least one confident hand and the
object (63.0%); 60,176 frames carry both hands; 68,270 left and 79,272 right
fields. Per-subject labelled frames range from 4,490 (PCW023) to 16,562 (SPI102);
fold A holds out 11,948 frames (XYZ109 5,921 + LYA722 6,027), fold B 17,087
(MHA016 8,904 + MMO925 8,183).

## What is available when

| | Train | Test |
|---|---|---|
| Both headset videos, calibration, frame metadata | yes | yes |
| Object category (alias) | yes | yes (manifest) |
| Hand landmarks, object pose, meshes | yes | no |
| Field labels | derivable | withheld |

## Known label limitations (from the SHOW3D paper and READMEs)

Hand GT from mono-tracking solver fitted to ego-exo triangulation: median 6.4-7.9 mm
MPJPE, P90 12.5-16.5 mm; failure modes: interpenetration with the object, missed
hand under severe occlusion, contamination by face-blur boxes. Object GT from
FoundPose + GoTrack: P50 translation 1.5-3.5 mm, yield 53.5-92.6% depending on
object; rotational ambiguity for symmetric objects (irrelevant to the field, which
depends on the surface only). Confidence 0.5 filters most failures.
