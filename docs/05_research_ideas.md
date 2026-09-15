# 05. Research directions: generation, evidence, ranking, selection

Each direction is scored against the evidence collected in docs/04 and the
constraints in docs/01-02. "Expected improvement" is relative to a frozen-DINOv3
direct-regression baseline at matched backbone (our reproducible reference), not
to GOLF's ViT-H+ ensemble, which is out of reach on local compute.

## A. Temporal interaction-field reasoning
* Hypothesis: aggregating features or predictions over neighbouring frames (60 fps
  video is available at test time) reduces error via smoothness and disocclusion.
* Novelty: low-moderate. ARCTIC InterField-LSTM (2023) and JSSR's Temporal7 (2026)
  already do this. Prior overlap: high.
* Feasibility: high on cached per-frame features (temporal transformer over T
  frames). Cost: linear in T.
* Expected improvement: small. Evidence: JSSR -0.08 mm from seven frames; ARCTIC
  -0.6 mm of 9.6. The GT field is already smooth (ACC 0.083 m/s^2).
* Datasets/annotations: SHOW3D only. Failure modes: motion blur, head motion in the
  moving rig frame, association changes when the nearest surface point jumps.
* Ablations: window size, feature- vs prediction-level fusion, causal vs centred.
* Paper potential alone: weak (a section, not a paper).

## B. Cross-dataset transfer (SHOW3D <-> HOT3D) at joint level
* Hypothesis: SHOW3D-trained joint-field models transfer to HOT3D-Quest3 (same 22
  objects, same UmeTrack skeleton, same headset) better than the reverse; models
  that expose object geometry explicitly transfer better than direct regression.
* Novelty: moderate. The SHOW3D paper studied transfer only for the scalar,
  canonical-mesh-input InterField task; no joint-level image-only protocol exists.
* Feasibility: medium. HOT3D-Clips (BOP format) are ~85-100 MB per 150-frame clip
  on HuggingFace, ungated; hand data are CC BY-NC-SA (research use fine). Quest 3
  images are fisheye and need rectification to SHOW3D's pinhole convention (the
  Aria camera library has no wheel for this Python; a numpy fisheye model or a
  separate Python 3.11 env is needed). Undistorted image statistics differ
  (30 fps, lab lighting).
* Expected improvement: not an accuracy gain on SHOW3D per se (the SHOW3D paper
  found +2% from adding HOT3D); its value is scientific: a transfer table.
* Failure modes: fisheye rectification errors, hand-pose label convention
  mismatch (verify landmark order), domain gap dominating everything.
* Ablations: train on A test on B, both directions; joint training; branch-wise
  pretraining.
* Paper potential: strong as a *component* (Table: cross-dataset; Figure: transfer).

## C. Kinematic graph reasoning over the 21 joints
* Hypothesis: explicit skeleton constraints (bone lengths, GNN message passing)
  improve the coherence of the 21 vectors.
* Novelty: low (skeleton GNNs are standard in hand pose). Joint-query transformers
  already exchange information across joints via self-attention.
* Feasibility: high; cost negligible. Expected improvement: small.
* Paper potential: weak.

## D. Uncertainty-aware interaction fields
* Hypothesis: per-joint heteroscedastic uncertainty (i) improves accuracy on a
  heavy-tailed, noisily labelled target by down-weighting ambiguous samples, (ii)
  is calibrated enough to gate the fusion of complementary predictors, and (iii)
  identifies unobservable targets.
* Novelty: moderate. Aleatoric uncertainty exists for hand pose (BMVC 2025) but
  not for interaction fields; using it as a principled fusion weight replaces the
  learned scalar gates (JSSR) and equal-weight ensembles (GOLF).
* Feasibility: high; one extra output per head; NLL training.
* Expected improvement: 1-3% ADE typical; large gains in analysis value
  (error-vs-uncertainty curves, calibration).
* Failure modes: NLL instability early in training (mitigate: warm-up with ADE,
  clamp log-variance), over-confidence on occluded joints.
* Ablations: Gaussian vs Laplace NLL; isotropic vs full covariance; fusion by
  precision weighting vs learned gate vs mean.
* Paper potential: moderate; strongest when combined with E.

## E. Object-aware geometric (factored) interaction fields
* Hypothesis: predicting the *causes* of the field (camera-frame hand joints and
  the 6-DoF pose of the known object mesh) and computing the field with a
  differentiable nearest-surface operator yields fields that are geometrically
  consistent by construction, exploits supervision the direct models ignore (hand
  landmarks, object poses, meshes), decomposes the error into interpretable parts,
  and transfers better across datasets because object geometry is shared.
* Novelty: moderate-high for this task. The target's analytic definition has not
  been used as a structured output layer; top solutions regress vectors directly
  and either ignore the mesh (JSSR, deliberately) or use only the category (GOLF).
  Prior overlap: pose-then-contact pipelines (CPF, HOISDF, HOPformer) and contact
  losses; our contribution is the end-to-end field-level training through the
  soft nearest-point operator, the hybrid with a direct head, and the analysis.
* Feasibility: medium-high. Heads on frozen tokens; nearest-point over 1.7k-5.3k
  vertices is a (21 x M) distance matrix per hand, trivial. Object 6-DoF regression
  of known objects from features is the hard part, but the field is invariant to
  the object's symmetry group (surface-defined), so symmetric objects (cans, mugs
  up to the handle, bottles) do not need disambiguation; a symmetry-aware ADD-S
  loss handles the auxiliary pose term.
* Expected improvement: uncertain in raw ADE (the direct head is strong); very
  likely large gains in surface consistency and penetration metrics; likely gains
  in far-field targets where the vector depends on global object localisation.
* Datasets/annotations: SHOW3D hand landmarks, object R,t, bundled meshes (all
  released); HOT3D for transfer.
* Failure modes: (1) compounding errors of hand and object estimates exceed the
  direct head's error -> mitigated by the hybrid fusion; (2) object out of view ->
  pose unobservable; (3) partially wrong mesh alignment -> wrong association.
* Ablations: direct only / geometric only / hybrid; soft-min temperature; with and
  without auxiliary joint and pose losses; category known vs predicted; mono vs
  stereo; backbone size; symmetric vs asymmetric objects.
* Paper potential: strong; supports Figures 1-3 (representation), consistency
  metrics, decomposition, and transfer.

## F. Occlusion- and observability-aware prediction and evaluation
* Hypothesis: a sizeable share of the residual error comes from targets that are
  not observable from the egocentric views (object or hand outside the field of
  view or fully occluded); these labels exist because GT comes from exocentric
  cameras. Visibility labels can be derived per joint (depth test against the posed
  mesh, image bounds) and used as auxiliary supervision and as evaluation strata.
* Novelty: moderate (benchmark analysis contribution; new derived labels).
* Feasibility: high (labels + calibration only). Expected improvement: analysis and
  a modest accuracy gain from visibility-conditioned heads.
* Paper potential: strong as analysis; weak as sole method.

## G. View-consistent (explicit stereo) fields
* Hypothesis: explicit correspondence (cost volume / epipolar attention) on hand
  joints resolves depth better than token concatenation with ray embeddings.
* Novelty: low-moderate (Epipolar Transformers, UmeTrack, JSSR). Evidence against:
  GOLF -0.19 mm, JSSR -0.45 mm. Worth testing *within* E (stereo for the joint
  branch) rather than as the paper's thesis.

## H. Adaptive computation: not pursued (no evidence of benefit for this task).

## I. Joint-level confidence: subsumed by D.

## J. Physical / geometric consistency: subsumed by E (endpoints on the surface by
construction; non-penetration as an auxiliary constraint and a metric).

## K. Observability-stratified benchmark analysis: subsumed by F.

## L. Label-noise-aware training (confidence weighting, robust losses): cheap
add-on to D; not a paper.

## Ranking

Scores 1 (weak) to 5 (strong). Fit = fit to a small-compute, geometry-oriented
research setup; Vis = ability to produce meaningful figures.

| Dir. | Novelty | Importance | Feasibility | Beats baselines | Publication | Fit | Vis | Total |
|---|---|---|---|---|---|---|---|---|
| E factored fields | 4 | 5 | 4 | 3 | 5 | 5 | 5 | 31 |
| D uncertainty | 3 | 4 | 5 | 3 | 3 | 5 | 4 | 27 |
| F observability | 3 | 5 | 5 | 2 | 3 | 5 | 4 | 27 |
| B cross-dataset | 3 | 4 | 3 | n/a (2) | 4 | 4 | 4 | 24 |
| G explicit stereo | 2 | 3 | 3 | 2 | 2 | 4 | 3 | 19 |
| A temporal | 2 | 3 | 5 | 2 | 2 | 4 | 2 | 20 |
| C kinematic GNN | 1 | 2 | 5 | 2 | 1 | 4 | 2 | 17 |

## Selection

Primary idea: **E**, with **D** as the fusion mechanism (precision-weighted hybrid
of a direct head and a geometry-derived head), **F** as the analysis backbone
(observability and visibility strata, derived labels), and **B** as the
multi-dataset component (HOT3D-Quest3 joint-level transfer protocol). G and A are
tested only as ablations inside the joint branch.

Why this is not incremental: the three published solutions treat the field as an
unstructured 126-vector; we treat it as a *derived* quantity of hand geometry and
a known rigid surface, which (i) changes what the network must learn, (ii) makes
geometric consistency measurable and enforceable, (iii) admits an error
decomposition nobody has published, and (iv) yields a testable transfer claim.

Falsification: if, at matched frozen backbone, the hybrid does not beat the direct
head on ADE *and* the geometric head is not better on far-field or occluded
strata *and* transfer is not improved, the hypothesis is rejected; the
decomposition and observability results are still reportable as an analysis paper.

## Answers to the seven pre-coding questions (Section 21)

1. InterField fails to model spatial detail (224^2 full frame), depth (monocular,
   no geometry), object identity/shape, and any structure among the 126 outputs.
2. GOLF solves dense features (DINOv3 + LoRA), local detail (RoI tokens), weak
   stereo and ray conditioning, and joint-query decoding; it reaches 27.8 mm.
3. Unsolved: the origin of the remaining error; the tiny stereo gain; the unused
   mesh/pose structure; uncertainty; observability; cross-dataset behaviour.
4. Our idea is necessary because the target is analytically determined by
   quantities that are supervised and partly known at test time; ignoring them
   discards constraints and interpretability.
5. Multiple datasets help by testing the claim that geometry-factored predictors
   transfer (shared meshes, shared skeleton) where appearance-driven regressors do
   not; HOT3D also supplies lab data to probe whether it complements in-the-wild data.
6. Falsifying experiment: matched-backbone comparison direct vs hybrid on held-out
   subjects, stratified; and the two-direction transfer table.
7. Convincing evidence: consistent gains across subjects and strata with error
   bars over seeds, geometric-consistency metrics that separate the methods,
   a decomposition showing where error lives, transfer results, and qualitative
   overlays on real frames including failures.
