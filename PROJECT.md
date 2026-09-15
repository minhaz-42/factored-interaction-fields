# Project record

A single document covering what this project is, what it found, and where it was submitted.
For how to run the code, see [README.md](README.md).

---

## 1. What the project is

The SHOW3D benchmark defines a **joint-anchored interaction field**: for each of the 21 UmeTrack
landmarks of each hand, the three-dimensional vector to the nearest vertex of the object being
manipulated, estimated from the two monochrome cameras of a Meta Quest 3 headset. Every published
solution regresses those 126 numbers per frame directly from pixels.

This project started from an observation about the target rather than the architecture. The field is
a deterministic function of three quantities that are all supervised during training:

1. the hand joints,
2. the 6-DoF pose of the object,
3. the object mesh, which is a fixed asset shipped with the benchmark.

So instead of learning `image -> field`, we estimate the arguments and apply the function. That is
**FIF, factored interaction fields**: a transformer decoder over frozen DINOv3 tokens predicts
camera-frame joints and object control points, weighted Procrustes alignment turns the control
points into a pose in closed form, a differentiable soft nearest-vertex operator evaluates the field
on the posed mesh, and the result is fused with a direct regression head by predicted precision.

**The question was falsifiable and the criterion was fixed before any model was trained.** Three
predictions: the factored model beats a matched direct head at a shared frozen representation; the
gain concentrates on far-field and occluded joints; and it transfers better to a second dataset.

---

## 2. What was found

The first two predictions held. The third failed, and opening the trained models showed the
mechanism was not the one the architecture implies. The paper reports all of this plainly.

### Headline numbers, mean ADE in mm on held-out subjects, lower is better

| Model | Fold A | Fold B |
|---|---|---|
| InterField ResNet-50, official baseline | 60.47 | -- |
| Direct regression, single view | 49.58 | -- |
| Direct regression, stereo, no ray encoding | 47.47 | -- |
| Direct regression, stereo + Plücker rays | 46.15 / 45.73 | 35.68 |
| FIF, single view | 45.19 | -- |
| **FIF, stereo, full** | **44.43 / 43.82** | **34.08** |

Two numbers are the two seeds. They are a range, not a variance estimate, and the paper says so.

### The three findings that matter more than the headline

**The gain is carried by the direct head, not by the fusion.** The direct head *inside* the factored
model scores 44.55 on its own, 1.6 mm better than the same head trained without a geometric branch.
The fused output improves on it by 0.12 mm. The geometric branch is therefore a structured auxiliary
task that shapes the shared representation, and it can be dropped at deployment. The geometric head
alone is 55.11, worse than a single-view direct regressor, and its predicted scale saturated at its
clamp in four of the five precision-fusion runs.

**The object pose is the bottleneck of the geometric route.** Replacing the predicted object pose
with ground truth lowers the geometric head from 55.11 to 46.46; replacing the predicted joints
lowers it only to 51.35. The joints are already within the noise of SHOW3D's own hand annotation.

**The advantage does not transfer.** Across four cross-dataset settings on HOT3D-IF the factored
model is never ahead, and adding 2,889 HOT3D frames to training removes its SHOW3D margin exactly.
The failure traces to the object-pose branch, which is the least transferable component.

---

## 3. What was built

- **HOT3D-IF**, a joint-level version of the task on HOT3D, the only public dataset sharing SHOW3D's
  object meshes, hand skeleton and headset. Fisheye624 rectification to SHOW3D's virtual pinhole
  camera, landmark recovery by UmeTrack forward kinematics, object alias mapping, and the same field
  derivation. The forward kinematics reproduces SHOW3D's stored landmarks to 1.2e-5 mm.
- **A controlled protocol.** Subject-level folds, one frozen DINOv3 ViT-B/16 cache shared by every
  token model, one decoder, one optimiser, one schedule, so differences are attributable to the
  factorisation alone. Fold A holds out XYZ109 and LYA722, the organisers' development split; fold B
  holds out MHA016 and MMO925.
- **18 training runs** covering the baselines, the full model, every ablation, a second seed on the
  main comparison, and the four cross-dataset settings.
- **A fully generated results pipeline.** No number in the paper is typed by hand; every table comes
  from `scripts/make_paper_tables.py` and every figure from `scripts/fig_*.py`, both reading the run
  artefacts directly.

All of it ran on **one Apple M5 laptop with 16 GB of unified memory**, about 35 hours of compute in
total, with no cloud and no funding.

---

## 4. The manuscript

20 pages in the Elsevier two-column layout. 7,307 words of body text excluding references, captions,
tables and equations. 11 figures, 6 tables, 62 references, every one of them cited and verified
against its arXiv abstract page or Crossref DOI.

Before submission the whole paper was audited twice: a full proofread, and a pass that re-derived
every numeric claim from the run artefacts. **26 defects were found and fixed**, among them a PCA
variance and token count that disagreed with the cache log, figure-selection percentiles that
disagreed with the selection script, a monotonicity claim that fold B breaks, an epoch budget the
HOT3D-only runs do not share, a shared-object count that exceeded the number of objects in the
dataset, and two equation labels sitting on the wrong row of their `align` block, which had made
four references point one equation too far.

Final build: zero overfull lines, zero undefined references, all fonts embedded, no raster image
below 301 dpi.

---

## 5. Submission

| | |
|---|---|
| **Journal** | Image and Vision Computing (Elsevier), ISSN 0262-8856 |
| **Article type** | Full Length Article |
| **Manuscript number** | IMAVIS-D-26-04847 |
| **Submitted** | 15 September 2026, 12:31, receipt confirmed 12:37 |
| **Portal** | https://submit.elsevier.com/IMAVIS |
| **Title** | Factored Interaction Fields: Learning Joint-Anchored Hand-Object Proximity Through Its Geometric Arguments |
| **Corresponding author** | Tanvir Ahmed, North South University, Dhaka, Bangladesh |

**Files sent:** `paper/main.pdf`, `paper/abstract.pdf`, `paper/highlights.pdf`,
`figures/graphical_abstract.pdf`.

**Choices made at submission:** no competing interests; no funding; **subscription** rather than open
access, because a hybrid-journal APC is not payable without funding; five classifications covering
3D object pose estimation, 3D from multiple views, human-body pose estimation, 3D from a single
image and 3D point cloud understanding; research data linked to this repository; and **opted in to
the free SSRN preprint**, which posts once the manuscript clears initial desk review.

**Tracking.** Author login is at https://www.editorialmanager.com/IMAVIS/ . That site carries a
banner saying not to use it for live submission; the banner refers to submitting, not to tracking.
A regional editor is assigned before the paper gets its reference number and goes out for review.

**Why this venue.** The work is a controlled measurement of one design decision with an honest
negative component, not a leaderboard entry. It needs a venue that reviews on rigour rather than on
rank, has no mandatory article charge, and publishes egocentric and geometric vision. Image and
Vision Computing fits all three.

---

## 6. What is in this repository, and what is not

Committed here, about 14 MB: the model and training code, one configuration per run, the per-run
metrics every table is built from, the run logs, every figure script, the LaTeX source and the built
manuscript.

Not committed, and why:

| Artefact | Size | Reason |
|---|---|---|
| SHOW3D and HOT3D raw data | ~100 GB | redistributed by their own authors, not here |
| DINOv3 feature cache | 22 GB | regenerate with `scripts/cache_features.py` |
| Derived interaction-field labels | 774 MB | too large for a code host, archived separately |
| Trained checkpoints, 36 files | 2.0 GB | same |
| Per-frame predictions | 299 MB | same |

---

## 7. Outstanding

1. **Deposit the 3 GB of derived labels, checkpoints and predictions** and add the link to
   `README.md`. The abstract, conclusion and Data availability statement all promise them. This has
   a deadline: the SSRN preprint makes that promise public as soon as desk review clears. Mendeley
   Data is Elsevier's own repository, is free, issues a DOI and appears in the submission system's
   repository list, so it links to the article more cleanly than Zenodo.
2. **Sign in to the journal account** from the confirmation email, which is required to track the
   submission and file revisions.
3. **Consider arXiv cs.CV.** SSRN gives the DOI and the timestamp, but computer vision researchers
   read arXiv.

---

## 8. Licence

Code under MIT, see [LICENSE](LICENSE). Data artefacts derived from SHOW3D are adaptations of a
CC BY-NC 4.0 dataset and carry the same non-commercial terms with attribution; HOT3D-IF labels carry
HOT3D's share-alike terms. Neither dataset is redistributed here.
