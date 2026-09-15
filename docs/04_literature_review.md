# 04. Literature review (state as of 2026-09-12)

Sources were read in full text (arXiv PDFs converted locally) unless marked
"abstract only". Every entry lists what the method solves and what it leaves open
for the SHOW3D joint-level interaction-field task.

## 4.1 The interaction-field representation

**ARCTIC (Fan et al., CVPR 2023, arXiv:2204.13662).** Introduces interaction-field
estimation: for every vertex of mesh a, the distance to the closest vertex of mesh
b (F^{a->b} in R^{V_a}); four directed fields (l->o, r->o, o->l, o->r) on
articulated objects. Baseline **InterField-SF**: ResNet image feature (2048-d ->
512-d) concatenated to every sub-sampled *canonical-pose* hand/object vertex,
PointNet, regression head, upsampled to the full mesh. **InterField-LSTM** adds a
bidirectional LSTM (window 11) over image features. Reported Average Distance
Error 8-10 mm (hand->object) on ARCTIC, and temporal modelling improves ADE by
~0.6 mm and smoothness (ACC). Inputs include the canonical meshes, i.e. object
identity and shape are given; output is a scalar distance per vertex, not a 3D
vector. Open: 3D vectors, unknown pose, egocentric in-the-wild imagery.

**TACO (Liu et al., CVPR 2024, arXiv:2401.08399).** Extends the benchmark to six
fields including tool->target object, again scalar distances from InterField-SF
variants (mean distance errors 10-35 mm depending on field). Confirms that the
representation is used as a *distance* field in prior work.

**SHOW3D (Rim et al., CVPR 2026, arXiv:2603.28760).** Section 5.1 re-uses ARCTIC's
vertex-level definition and trains InterField for cross-dataset transfer between
SHOW3D and HOT3D (Table 2: SHOW3D->HOT3D 14.70 mm, HOT3D->SHOW3D 22.57 mm,
SHOW3D->SHOW3D 13.82 mm, HOT3D+SHOW3D->SHOW3D 13.50 mm; ACC 2.2-5.6 m/s^2). The
paper's ADE numbers are for the scalar vertex field with canonical meshes as input
and are **not comparable** to the challenge's 3D joint-anchored vectors (baseline
60.5 mm). The paper does not define the (21,3) joint field; that definition lives
only in the challenge API. The paper also reports UmeTrack multi-view hand pose
transfer (MKPE 14.3-22.2 mm on SHOW3D) and text-conditioned object pose forecasting.

**Challenge API (Rim et al., 2026).** Defines the joint-anchored 3D vector field,
the (21,3) fixed-size target, validity rules, ADE/recall/acc@k. The InterField
baseline is a ResNet-50 on the full 224^2 grayscale frame regressing 126 numbers in
the camera frame; 60.47 mm on held-out XYZ109/LYA722 (recall 1.0, acc@10 0.150,
acc@50 0.682, acc@100 0.861). The oracle hand-crop variant reaches 54.70 mm. The
README notes the GT magnitude is heavy-tailed (median ~32 mm, mean ~83 mm) and
attributes the residual to monocular depth ambiguity.

## 4.2 Challenge solutions (the current state of the art)

**GOLF (Zou et al., JIIOV, arXiv:2609.08607; 1st).** Shared DINOv3 ViT-H+/16
encoder (frozen weights; LoRA rank 128, alpha 256 on QKV/output and SwiGLU
projections in every block; LayerNorm affine trainable), input 560x448 (35x28
tokens), features fused from blocks {10,20,26,31}. A locator with three region
queries predicts square RoIs for left hand, right hand, object (object query
conditioned on the *provided object category*), trained with Smooth-L1 on
center/log-size, BCE presence, and a coverage loss using GT-derived boxes.
16x16 GridSample RoI tokens per region per view, plus the full global grid, plus
origin-aware Plücker ray embeddings [d, o x d] of every token in a reference camera
frame (reference view swapped during training). A 20-layer, width-512 decoder with
42 joint queries (Joint Transformer formulation, Abou Zeid 2023) outputs 2x21x3
vectors in the reference camera frame, rotated to world. Training: three stages,
batch 96, 40+20 epochs, then 2 epochs on all 10 subjects; HFlip TTA; ensemble with
a directly fine-tuned (last 24 blocks) variant.
Hidden-test progression (Tab. 3, mean ADE mm): InterField 59.38 -> ViT-L/16 mono
400x320 43.50 -> both-view training + HFlip 560x448 39.10 -> local RoI tokens 36.36
-> stereo fusion 36.17 -> Plücker 35.57 -> ViT-H+ LoRA 64/16 29.86 -> LoRA 128 + LN
28.91 -> full-train FT 27.96 -> ensemble 27.82 (official 27.47). Dev split (XYZ109 +
LYA722) mean ADE 28.31 raw / 28.04 TTA, i.e. the dev split tracks the hidden test.
Key evidence: **backbone adaptation dominates (-5.7 mm), local detail -2.7 mm,
stereo fusion only -0.19 mm, ray encoding -0.6 mm.** Feature-space RoI sampling
matches image-space re-cropping (37.77 vs 37.93 mm).

**JSSR (Jin et al., Ant Group/CASIA, arXiv:2609.06955; 3rd).** Trainable DINOv3
encoder at 512x512 on seven synchronized stereo pairs; hand-aware temporal
aggregation; 42 joint queries. Heads: camera-0 3D joints p_j, a direct field
f_j^dir, and per-view 2D endpoints with depth. Endpoint formulation e_j = p_j + f_j
is projectable, so a 32-depth epipolar candidate search scores candidates by
joint-conditioned image compatibility plus stereo feature agreement (cosine),
then a hand-shared candidate set (21x5x9 = 945) lets joints select a common local
"surface" support; a zero-initialised gate applies the geometric correction as a
residual. Training 8 epochs, batch 16, lr 1e-5/1e-4, after removing 535
"geometrically inconsistent" annotations; 72,307 train / 12,201 eval targets on a
clip-held-out split. Ablation (mean ADE mm): ResNet-50 29.17; DINO-Direct 16.28;
+Temporal7 16.19; +two-view endpoints 15.93; +full JSSR 15.74 (-2.8%). LB: single
33.59, ensemble 32.61. Note their clip-held-out split leaks subjects (same subjects
in train and eval), which is why its ADE is half of the test ADE. Explicitly avoids
CAD models and object pose. Key evidence: **temporal context (-0.08 mm) and explicit
stereo endpoint search (-0.45 mm) give small gains; joint-conditioned surface
sharing is the most useful geometric idea.**

**Joint-Query Spatial Attention (Zakour et al., TUM; 2nd).** Only the title is
public (HANDS 2026 Board 303). The same group's UA-Fit (Board 289) is an
uncertainty-weighted analytical solver for multi-view hand mesh recovery, which
suggests a per-joint uncertainty component; this is UNVERIFIED.

**Consensus architecture of the top three:** DINOv3 tokens (both views) ->
transformer decoder with 42 joint queries -> direct 3D vector regression trained
with masked ADE. Object geometry is used only as a category embedding (GOLF) or not
at all (JSSR); no method predicts hand joints or object pose as an intermediate
except JSSR's auxiliary joint head; none reports an error decomposition; none models
uncertainty; none evaluates cross-dataset.

## 4.3 Related representations and constraints

* **CPF (Yang et al., ICCV 2021)**: contact potential field, spring-mass energy
  between contacting hand/object vertices, used for fitting. **HOISDF (Qi et al.,
  CVPR 2024)**: global SDFs of hand and object guide pose regression (DexYCB,
  HO3D). **gSDF (Chen et al., CVPR 2023)**: geometry-driven SDFs for hand-object
  reconstruction. **DECO (Tripathi et al., ICCV 2023)**: dense vertex contact for
  bodies. **EgoPHI (Ilic et al., ECCV 2026, arXiv:2608.13014)**: dense contact and
  force on hand/object meshes from a single egocentric RGB image with known object
  geometry. Common thread: proximity/contact is derived from *posed geometry*; the
  interaction field is exactly such a derived quantity, yet the challenge solutions
  regress it directly.
* **Hand-object pose (egocentric)**: HOPformer / EPIC-Contact (Bansal et al., ECCV
  2026, arXiv:2606.30598; ARCTIC and EPIC-Contact, not SHOW3D); HANDS23 challenge
  report (Fan, Ohkawa et al., ECCV 2024, arXiv:2403.16428): egocentric distortion
  handling, high-capacity transformers and multi-view fusion help; fast motion,
  narrow-FOV object views and close hand-hand-object contact remain unsolved.
  HaMeR (Pavlakos et al., 2024) established large ViTs for hand mesh recovery.
* **Multi-view geometry in networks**: Epipolar Transformers (He et al., CVPR
  2020) fuse features along epipolar lines; Learnable Triangulation (Iskakov et
  al., ICCV 2019) volumetric aggregation; Geometry-Biased Transformer (Moliner et
  al., FG 2024) biases attention by view geometry; UmeTrack (Han et al., SIGGRAPH
  Asia 2022) is the multi-view, world-space egocentric hand tracker whose skeleton
  SHOW3D uses; Plücker ray embeddings for camera-aware tokens (LVSM, ICLR 2025;
  PlückeRF 2025; Light Field Networks, Sitzmann et al., NeurIPS 2021, cited by
  GOLF). D4RT (Zhang et al., 2025) is a feed-forward 4D reconstruction model used by
  JSSR as a descriptor.
* **Uncertainty for 3D hands**: correlation-aware aleatoric uncertainty via a
  linear layer parametrising a joint Gaussian (Lee, Nam, Oh, BMVC 2025,
  arXiv:2509.01242); UST-Hand (2026) normalising-flow hypotheses. No interaction-
  field method models uncertainty.
* **Datasets with joint-compatible labels**: HOT3D (Banerjee et al., CVPR 2025):
  19 subjects, 33 objects, 833 min, Aria + Quest 3 (two 1280x1024 monochrome
  fisheye streams at 30 fps), UmeTrack and MANO hands, mocap object poses; 22 SHOW3D
  objects are HOT3D objects with identical meshes. Licenses: sequence data CC BY-SA,
  hand data CC BY-NC-SA, models CC BY-SA with no-sale clause.
* **Survey**: "Hand-Object Interaction in the Age of Large Foundation Models"
  (arXiv:2607.28394, 2026) lists five uncertainties (shape, spatial, physical,
  semantic, dynamic) and distinguishes contact maps, affordance maps, field
  representations and dense correspondences; it flags the gap between geometric
  completeness and interaction correctness.

## 4.4 What is solved and what is open

Solved by prior work: strong dense features for hands and objects (DINOv3 + LoRA),
joint-query decoding, per-region detail (RoI tokens), calibrated ray encoding,
weak stereo fusion, ensembling. These bring the hidden-test ADE from 59 to 28 mm.

Open after GOLF/JSSR:
1. **Where the remaining 28 mm comes from.** No published decomposition by field
   magnitude, joint, hand, visibility/occlusion, object, depth, along-ray vs lateral
   component, or observability (object or hand outside the egocentric field of view).
2. **Why stereo adds < 1 mm** despite a 64 mm baseline and 200-400 mm working depth
   (about 3 mm depth per pixel of disparity at 300 mm).
3. **Unused structure**: the target is a deterministic function of hand joints,
   object pose and a *known* object mesh; the 21 endpoints of a hand lie on one
   rigid surface; hand landmarks and object poses are available as auxiliary
   supervision; none of this is exploited beyond GOLF's locator boxes.
4. **Uncertainty and label noise**: GT hand error median 6-8 mm (P90 12-16 mm),
   object translation P50 1.5-3.5 mm; frames are filtered at confidence 0.5; models
   are trained with plain ADE.
5. **Cross-dataset behaviour at joint level**: unknown; the SHOW3D paper's transfer
   numbers concern a different (scalar, canonical-mesh-input) task.
6. **Temporal context** at test time (60 fps video) is barely used (JSSR: -0.08 mm).
