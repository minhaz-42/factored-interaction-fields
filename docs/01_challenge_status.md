# 01. Challenge status (verified 2026-09-12)

All statements below were checked against primary sources on 2026-09-12. Where a
statement is inferred rather than stated by a source, it is marked INFERRED.

## Bottom line

* The HANDS@ECCV 2026 SHOW3D Interaction Field Estimation Challenge is **closed**.
  The single Codabench phase ("ECCV 2026 HANDS Evaluation Phase") ran from
  2026-08-04 00:00 UTC to 2026-08-31 23:59 GMT and is marked `status: Previous`,
  `is_final_phase: True`. Source: Codabench REST API,
  https://www.codabench.org/api/competitions/17756/ (fields `phases[0].start/end/status`).
* Registration closed 2026-08-26 23:59 GMT; technical reports from invited teams
  were due 2026-09-06; results were presented at the workshop on 2026-09-08
  (Malmömässan K2, ECCV 2026 workshop day). Sources: Codabench Overview page text
  (same API), https://hands-workshop.org/challenge2026.html,
  https://hands-workshop.org/workshop2026.html.
* **No official source lists a September 18 deadline.** The challenge page, the
  Codabench competition JSON (all phases, all pages, terms), and the workshop page
  were searched for "18", "Sept", "09/18"; nothing matches. The date is therefore
  stale or mistaken metadata and must not be relied on.
* New participants cannot submit: registration is closed, the only phase has ended,
  and `registration_auto_approve` is `False` (organizer approval was required even
  during the live phase). INFERRED from the API fields; the server does not
  publish a post-challenge open phase.
* Technical reports were only for *invited* (winning) teams and were due
  2026-09-06; the workshop's own paper deadlines (full papers 2026-07-15/16,
  extended abstracts 2026-08-31) have also passed. Source:
  https://hands-workshop.org/workshop2026.html ("Important Dates").
* Consequence: this project is a **research paper that uses SHOW3D as its primary
  benchmark**, not a competition entry.

## Competition facts (Codabench API)

| Field | Value |
|---|---|
| Title | [HANDS@ECCV 2026] SHOW3D Challenge |
| Created | 2026-08-03T19:20:48Z |
| Participants / submissions | 49 / 611 |
| Phase | ECCV 2026 HANDS Evaluation Phase, 2026-08-04 to 2026-08-31 23:59 GMT |
| Limits | 5 submissions/day, 100/person, 300 s execution limit |
| Leaderboard columns | official_score (asc, primary), mean_ade_mm, left/right_to_object_ade_mm (asc), mean_recall, left/right recall (desc) |
| Official score | "aggregate of accuracy and coverage; lower is better"; formula withheld; zero-prediction baseline scores 80.35 |

## Published results from the challenge

| Rank | Method | Team | Where | Hidden-test numbers |
|---|---|---|---|---|
| 1 | GOLF: Global Observation with Local Focus for Calibration-Aware Stereo Interaction Field Estimation | Zou, Jin, Lv, Luo, Tian, Zhao, Xu, Wu, Yu, Tang (JIIOV Technology) | arXiv:2609.08607 (2026-09-08); HANDS 2026 Board 301 | official 27.47, mean ADE 27.82 mm (ensemble); single model 27.61 / 27.96 mm |
| 2 | Joint-Query Spatial Attention for Egocentric Interaction Fields | Zakour, Piccolrovazzi, Wu, Patsch, Steinbach (TUM) | HANDS 2026 Board 303 (OpenReview non-proceedings; not publicly accessible as of 2026-09-12) | not public |
| 3 | Joint-Conditioned Stereo Surface Reasoning (JSSR) | Jin, Yang, Yang, Zhu (Ant Group / CASIA) | arXiv:2609.06955 (2026-09-07); Board 302 | official 32.61 (ensemble), 33.59 single |
| ref | InterField ResNet-50 (official baseline) | organizers | API repo `baseline/` | official 58.98, mean ADE 59.38 mm (as reported in GOLF Tab. 3) |
| ref | predict zeros | organizers | Codabench Evaluation page | official 80.35 |

## Publication routes after the challenge

Deadlines verified via conference sites/search on 2026-09-12:

* ICRA 2027: 2026-09-15 (too close for a complete experimental paper).
* CVPR 2027: 2026-11-16 AoE (supplementary 2026-11-23). Main target if results
  are strong; the user's other project already targets this date.
* 3DV 2027 and WACV 2027 (round 2): both closed 2026-08-28.
* Fallbacks: CVPR 2027 workshops (HANDS is typically co-located with CVPR/ICCV
  each year; the 2027 edition's call is not yet published), arXiv preprint, and
  ICCV 2027 (deadline expected ~March 2027).

## Source links

* Challenge page: https://hands-workshop.org/challenge2026.html
* Workshop page (winners, boards, dates): https://hands-workshop.org/workshop2026.html
* Codabench: https://www.codabench.org/competitions/17756/ and its JSON at
  https://www.codabench.org/api/competitions/17756/
* Dataset API: https://github.com/patrickqrim/SHOW3D-dataset-api (commit c2064dc, 2026-08-03)
* Dataset: https://huggingface.co/datasets/facebook/show3d-dataset (CC BY-NC 4.0; last modified 2026-07-31)
* Paper: https://arxiv.org/abs/2603.28760 (CVPR 2026)
* GOLF: https://arxiv.org/abs/2609.08607 ; JSSR: https://arxiv.org/abs/2609.06955
* ECCV 2026 dates: https://eccv.ecva.net/Conferences/2026 (Sept 8-12, Malmö; workshops Sept 8-9)
