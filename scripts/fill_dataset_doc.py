"""Render tables/dataset_stats.json into docs/03_dataset_stats_autogen.md (all numbers from data)."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fif.paths import PROJECT_ROOT

S = json.load(open(PROJECT_ROOT / "tables/dataset_stats.json"))
L = ["# 03b. Dataset statistics (auto-generated from derived labels at 10 fps, training subjects)", ""]
L += ["## Validity funnel", "", "| stage | frames |", "|---|---|"]
for k in ("frames_sampled", "frames_tracking_valid", "frames_object_conf>0", "frames_object_conf>0.5", "frames_left_conf>0.5", "frames_right_conf>0.5", "frames_labelled", "frames_both_hands"):
    L.append(f"| {k} | {S[k]:,} |")
L += [f"| fields (left / right) | {S['fields_left']:,} / {S['fields_right']:,} |", f"| recordings | {S['recordings']} |", ""]
L += ["## Labelled frames per subject", "", "| subject | frames |", "|---|---|"] + [f"| {k} | {v:,} |" for k, v in sorted(S["labelled_per_subject"].items())] + [""]
L += ["## Labelled frames per object", "", "| object | frames | median magnitude (mm) | mean | frac < 15 mm |", "|---|---|---|---|---|"]
for k, v in sorted(S["labelled_per_object"].items(), key=lambda x: -x[1]):
    m = S["magnitude_per_object"][k]; L.append(f"| {k} | {v:,} | {m['median']:.1f} | {m['mean']:.1f} | {m['frac<15']:.2f} |")
L += ["", "## Field magnitude (mm)", "", "| group | mean | median | p90 | frac < 10 | frac < 15 | frac > 60 | n |", "|---|---|---|---|---|---|---|---|"]
for k, m in S["magnitude"].items():
    L.append(f"| {k} | {m['mean']:.1f} | {m['median']:.1f} | {m['p90']:.1f} | {m['frac<10']:.3f} | {m['frac<15']:.3f} | {m['frac>60']:.3f} | {m['n']:,} |")
L += ["", "## Visibility and observability (view 0)", ""]
L += [f"* joints visible (inside image and not occluded by the object): {S['joint_visible0_frac']:.3f}; inside image: {S['joint_in_image0_frac']:.3f}",
      f"* magnitude of visible joints: median {S['magnitude_visible']['median']:.1f} mm; occluded/outside: median {S['magnitude_occluded_or_outside']['median']:.1f} mm",
      f"* object field-of-view fraction (view 0): mean {S['object_fov0']['mean']:.3f}; in view (>=0.9) {S['object_fov0']['in_view(>=0.9)']:.3f}; partial {S['object_fov0']['partial']:.3f}; out (<0.1) {S['object_fov0']['out(<0.1)']:.3f}; view 1 out: {S['object_fov1']['out(<0.1)']:.3f}",
      f"* hand field-of-view fraction (view 0): mean {S['hand_fov0']['mean']:.3f}; in view {S['hand_fov0']['in_view(>=0.9)']:.3f}; out {S['hand_fov0']['out(<0.1)']:.3f}", ""]
L += ["## Stereo geometry", ""]
if S.get("stereo_baseline_mm"):
    L += [f"* baseline: mean {S['stereo_baseline_mm']['mean']:.1f} mm (median {S['stereo_baseline_mm']['median']:.1f})", f"* joint depth (camera 0): median {S['joint_depth_cam0_mm']['median']:.0f} mm, p90 {S['joint_depth_cam0_mm']['p90']:.0f} mm",
          f"* fx = {S['fx_mean']:.1f} px; median disparity {S['disparity_px_median']:.0f} px; median depth per pixel of disparity {S['depth_per_px_disparity_mm_median']:.2f} mm"]
(PROJECT_ROOT / "docs/03_dataset_stats_autogen.md").write_text("\n".join(L) + "\n"); print("wrote docs/03_dataset_stats_autogen.md")
