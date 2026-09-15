# 07. Method: factored interaction fields with uncertainty-weighted fusion

Working name: **FIF** (Factored Interaction Fields). All equations are written
before implementation; implementation must follow them or update this file.

## 7.1 Notation

* Views v in {0, 1} (headset0, headset1), reference camera c = headset0.
  Intrinsics K_v (pinhole), extrinsics T_{w<-v} = [R_{w<-v} | t_{w<-v}] (world =
  rig frame, per frame).
* Hands h in {L, R}; joints j = 1..J, J = 21 (UmeTrack landmarks).
  World joints p^w_{h,j}; camera-frame joints
  p_{h,j} = R_{w<-c}^T (p^w_{h,j} - t_{w<-c}).                          (1)
* Object alias a (given in the manifest), canonical vertices Q_a = {q_m}, m = 1..M_a,
  and faces F_a (from the bundled GLB). World pose (R^w_o, t^w_o); camera-frame pose
  R_o = R_{w<-c}^T R^w_o,  t_o = R_{w<-c}^T (t^w_o - t_{w<-c}).          (2)
  Posed vertices q_m(theta_o) = R_o q_m + t_o, theta_o = (R_o, t_o).
* Ground-truth field (camera frame):
  m*(h,j) = argmin_m || q_m(theta_o) - p_{h,j} ||_2,
  v_{h,j} = q_{m*(h,j)}(theta_o) - p_{h,j}.                             (3)
  The submitted world-frame field is R_{w<-c} v_{h,j}; ADE is invariant to this
  rotation. (Matches `nearest_neighbor_vectors` in the official API: nearest
  *vertex*, not nearest point on a triangle.)

The field is therefore a deterministic function G of (p_h, theta_o, Q_a):
v_h = G(p_h, theta_o; Q_a). Direct regressors learn G∘(image -> v) end to end;
we learn the arguments and apply G.

## 7.2 Encoder and camera-aware tokens (shared with baselines)

* Frozen DINOv3 ViT-B/16 (ViT-L/16 if compute allows), input H x W = 480 x 384
  (portrait), patch tokens X_v in R^{N x D}, N = 30 x 24 = 720 per view.
* Ray embedding of token at pixel centre u in view v, expressed in the reference
  camera: d = normalize( R_{c<-v} K_v^{-1} [u; 1] ), o = R_{w<-c}^T (t_{w<-v} - t_{w<-c})
  in metres, Plücker pi(u, v) = [d, o x d] in R^6; token input
  X'_v = P X_v + MLP(pi) + e_view(v).                                   (4)
  (This is the calibration-aware component of GOLF, used as a baseline component.)
* Memory M = concat_v X'_v (1440 tokens for stereo; 720 for mono).

## 7.3 Decoder and heads

Learned queries: 2J = 42 joint queries z_{h,j} and N_o = 16 object queries z^o_n,
one per canonical control point c_n in Q_a (farthest-point sampled per object;
object identity enters as an embedding e_obj(a) added to object queries).
A transformer decoder (L layers, width d_model) with self-attention over all
queries and cross-attention to M outputs refined queries.

Heads (all linear or 2-layer MLP):
* Direct field: v^dir_{h,j} = W_v z_{h,j} in R^3 (mm), with log-scale
  s^dir_{h,j} = w_s^T z_{h,j} in R.                                       (5)
* Hand joints: p^hat_{h,j} = W_p z_{h,j} in R^3 (camera frame, mm); hand presence
  logit a_h from the mean of the hand's queries.                          (6)
* Object control points: c^hat_n = W_c z^o_n in R^3 (camera frame, mm).
  Pose by weighted Procrustes (Kabsch) between {c_n} and {c^hat_n}:
  (R^hat_o, t^hat_o) = argmin_{R in SO(3), t} sum_n w_n || R c_n + t - c^hat_n ||^2,   (7)
  solved in closed form with SVD (differentiable); w_n = softmax of a per-query
  confidence logit. Ablation: direct 6D-rotation + translation regression.
* Geometric field via a soft nearest-vertex operator on the posed mesh:
  q^hat_m = R^hat_o q_m + t^hat_o,
  w_{h,j,m} = softmax_m( - || q^hat_m - p^hat_{h,j} ||^2 / tau ),
  q^hat*_{h,j} = sum_m w_{h,j,m} q^hat_m,
  v^geo_{h,j} = q^hat*_{h,j} - p^hat_{h,j}.                                (8)
  tau in mm^2 (annealed from 400 to 25 during training; hard argmin at test).
  Because (8) depends on the posed *surface* only, it is invariant to the object's
  symmetry group; symmetric objects need no rotation disambiguation.
  Geometric log-scale s^geo_{h,j} = MLP([z_{h,j}, z^o_bar, H(w_{h,j,.}), log||v^geo||]),
  where H is the entropy of the soft assignment.                             (9)
* Per-joint visibility logit b_{h,j} (auxiliary; label derived, Sec. 7.6).

## 7.4 Uncertainty-weighted fusion

With precisions lambda^k_{h,j} = exp(-2 s^k_{h,j}), k in {dir, geo}:
v^hat_{h,j} = ( lambda^dir v^dir + lambda^geo v^geo ) / ( lambda^dir + lambda^geo ).   (10)
This is the maximum-likelihood combination of two independent isotropic Gaussian
estimates and replaces the learned scalar gate of JSSR and equal-weight ensembles.
Ablations: mean (lambda equal), learned scalar gate, direct only, geometric only.

## 7.5 Training objective

Masks: m_h = 1 if hand h has a valid GT field (object confident, hand confident).
* Field loss on the fused output (the metric): 
  L_field = (1/sum_h m_h) sum_h m_h (1/J) sum_j || v^hat_{h,j} - v_{h,j} ||_2.   (11)
* Heteroscedastic losses on each head (isotropic Gaussian on the residual):
  L_k = mean_{h,j valid} [ || v^k_{h,j} - v_{h,j} ||^2 / (2 exp(2 s^k)) + 3 s^k ], k in {dir, geo}.  (12)
  (Robust variant for ablation: Laplace, || . ||_1 / b + 3 log b.)
* Hand joints (camera frame, includes absolute depth):
  L_joint = mean_{h,j valid} || p^hat_{h,j} - p_{h,j} ||_1 .                (13)
  Ground truth p from `landmarks_3d_mm` via (1); available whenever the hand is
  confident, even when the object is not, so this term uses more frames than (11).
* Object pose, symmetry-aware (ADD-S on posed vertices, plus translation):
  L_obj = (1/M) sum_m min_{m'} || q^hat_m - q_{m'}(theta_o) ||_2 + || t^hat_o - t_o ||_1 .   (14)
  Available whenever the object is confident, even when no hand is.
* Presence and visibility: L_pres = BCE(a_h, m_h^any), L_vis = BCE(b_{h,j}, vis_{h,j}).
* Total:
  L = L_field + lambda_dir L_dir + lambda_geo L_geo + lambda_p L_joint + lambda_o L_obj
      + lambda_pres L_pres + lambda_vis L_vis,                                   (15)
  with initial weights (1, 0.1, 0.1, 0.02, 0.02, 0.1, 0.1) chosen so that the
  auxiliary terms are of the same order as L_field in millimetres; tuned on the
  validation subjects only.

## 7.6 Derived auxiliary labels (from released training annotations)

* Camera-frame joints and object pose: (1), (2).
* Joint visibility vis_{h,j} in {0,1}: joint projects inside the image of view 0 and
  the ray from the camera centre to the joint does not intersect the GT posed mesh
  before reaching the joint (trimesh ray-mesh test with a 5 mm tolerance).
  Occlusion by the other hand is ignored (no hand mesh used) and noted.
* Observability strata per frame: object FOV fraction f_obj = share of posed
  vertices projecting inside view 0; hand FOV fraction f_hand = share of the 21
  joints inside view 0; strata: in-view (f >= 0.9), partial (0.1 <= f < 0.9),
  out-of-view (f < 0.1).
* Magnitude strata: || v || < 15 mm (near contact), 15-60 mm, > 60 mm (far).

## 7.7 Inference

Given both views, calibration and the object alias: run the encoder once per
view (or read the cache), decode, compute (8) with a hard argmin, fuse with (10),
rotate to world with R_{w<-c}, and always output both hands (abstention is
penalised by the official score). Optional HFlip TTA as in GOLF is an ablation.

## 7.8 Error decomposition protocol (analysis, uses GT at evaluation only)

For the factored model, evaluate G(p^hat, theta_o) (hand error only), G(p, theta^hat_o)
(object error only), G(p^hat, theta^hat_o) (both), and v^dir, v^hat on the same frames.
For any method, decompose the endpoint error e = (p + v^hat) - (p + v) in the
reference camera into a component along the unit ray r through the GT endpoint,
e_par = (e . r) r, and the lateral remainder e_perp = e - e_par. Depth-limited
methods show || e_par || >> || e_perp ||.

## 7.9 Geometric consistency metrics (Section 10 of the brief)

Defined for any method using GT joints p as anchors so that direct and factored
models are compared fairly:
* Surface distance SD = dist( p_{h,j} + v^hat_{h,j}, S(theta_o) ), point-to-triangle
  distance to the GT posed mesh surface, in mm (0 for a perfect field).
* Direction error: angle between v^hat and v for || v || > 20 mm.
* Magnitude error: | || v^hat || - || v || |.
* Penetration: fraction of predicted endpoints strictly inside the GT mesh by more
  than 5 mm (signed distance), and, for factored models, fraction of predicted
  joints inside the mesh.
* Stratified ADE: magnitude strata, visibility, observability, joint group
  (fingertips 0-4, wrist 5, palm 20, other), hand, object, subject, mono vs stereo.
