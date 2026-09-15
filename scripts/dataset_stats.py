"""Dataset statistics for docs/03 from the derived aux labels (train subjects).

Outputs tables/dataset_stats.json and prints a Markdown summary: frames per
subject/object, validity funnel (tracking / object / hands), field magnitude
distribution per joint group, visibility and observability strata, stereo
baseline statistics, per-object magnitude, and label-noise proxies.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fif.paths import DATA_ROOT, PROJECT_ROOT
from fif.metrics import JOINT_GROUPS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--aux", type=Path, default=DATA_ROOT / "aux" / "aux_labels_10fps.jsonl"); a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.aux) if l.strip()]
    S = {}
    S["frames_sampled"] = len(rows)
    S["frames_tracking_valid"] = sum(r["tracking_valid"] for r in rows)
    S["frames_object_conf>0.5"] = sum(r["object_conf"] > 0.5 for r in rows)
    S["frames_object_conf>0"] = sum(r["object_conf"] > 0 for r in rows)
    S["frames_left_conf>0.5"] = sum(r["hand_conf"][0] > 0.5 for r in rows)
    S["frames_right_conf>0.5"] = sum(r["hand_conf"][1] > 0.5 for r in rows)
    lab = [r for r in rows if any(r["field_valid"])]
    S["frames_labelled"] = len(lab)
    S["fields_left"] = sum(r["field_valid"][0] for r in lab); S["fields_right"] = sum(r["field_valid"][1] for r in lab)
    S["frames_both_hands"] = sum(all(r["field_valid"]) for r in lab)
    S["labelled_per_subject"] = dict(Counter(r["subject_id"] for r in lab))
    S["labelled_per_object"] = dict(Counter(r["object_alias"] for r in lab))
    S["recordings"] = len({(r["subject_id"], r["scene_id"]) for r in rows})
    # magnitude
    mags = defaultdict(list); per_obj = defaultdict(list); per_joint = defaultdict(list)
    for r in lab:
        for k in range(2):
            if r["field_mag"][k] is None: continue
            m = np.asarray(r["field_mag"][k]); mags["all"].extend(m.tolist()); per_obj[r["object_alias"]].extend(m.tolist())
            for g, js in JOINT_GROUPS.items(): mags[g].extend(m[js].tolist())
            for j in range(21): per_joint[j].append(m[j])
    def q(v): v = np.asarray(v); return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "frac<10": float((v < 10).mean()), "frac<15": float((v < 15).mean()), "frac>60": float((v > 60).mean()), "n": int(v.size)}
    S["magnitude"] = {k: q(v) for k, v in mags.items()}
    S["magnitude_per_object"] = {k: q(v) for k, v in per_obj.items()}
    S["magnitude_per_joint_mean"] = {j: float(np.mean(v)) for j, v in per_joint.items()}
    # visibility / observability (view 0)
    vis = [v for r in lab for k in range(2) if r["joint_visible0"][k] is not None for v in r["joint_visible0"][k]]
    inimg = [v for r in lab for k in range(2) if r["joint_in_image"].get("headset0", [None, None])[k] is not None for v in r["joint_in_image"]["headset0"][k]]
    S["joint_visible0_frac"] = float(np.mean(vis)) if vis else None
    S["joint_in_image0_frac"] = float(np.mean(inimg)) if inimg else None
    of0 = np.asarray([r["obj_fov_frac"].get("headset0", np.nan) for r in lab]); of1 = np.asarray([r["obj_fov_frac"].get("headset1", np.nan) for r in lab])
    S["object_fov0"] = {"mean": float(np.nanmean(of0)), "in_view(>=0.9)": float(np.nanmean(of0 >= 0.9)), "partial": float(np.nanmean((of0 < 0.9) & (of0 >= 0.1))), "out(<0.1)": float(np.nanmean(of0 < 0.1))}
    S["object_fov1"] = {"mean": float(np.nanmean(of1)), "out(<0.1)": float(np.nanmean(of1 < 0.1))}
    hf = [r["hand_fov_frac"]["headset0"][k] for r in lab for k in range(2) if r["hand_fov_frac"].get("headset0") and r["hand_fov_frac"]["headset0"][k] is not None]
    hf = np.asarray(hf); S["hand_fov0"] = {"mean": float(hf.mean()), "in_view(>=0.9)": float((hf >= 0.9).mean()), "out(<0.1)": float((hf < 0.1).mean())}
    # magnitude of occluded vs visible joints
    mv = [(m, v) for r in lab for k in range(2) if r["joint_visible0"][k] is not None for m, v in zip(r["field_mag"][k], r["joint_visible0"][k])]
    S["magnitude_visible"] = q([m for m, v in mv if v]); S["magnitude_occluded_or_outside"] = q([m for m, v in mv if not v])
    # stereo geometry
    base = []; depth = []
    for r in lab[::10]:
        T0 = r["T_world_from_cam"].get("headset0"); T1 = r["T_world_from_cam"].get("headset1")
        if T0 and T1: base.append(float(np.linalg.norm(np.asarray(T0)[:3, 3] - np.asarray(T1)[:3, 3])))
        for k in range(2):
            if r["joints_cam0"][k] is not None: depth.extend(np.asarray(r["joints_cam0"][k])[:, 2].tolist())
    S["stereo_baseline_mm"] = q(base) if base else None; S["joint_depth_cam0_mm"] = q(depth) if depth else None
    fx = np.mean([r["intrinsics"]["headset0"][0] for r in lab if r["intrinsics"].get("headset0")]); S["fx_mean"] = float(fx)
    if depth:
        z = np.asarray(depth); S["disparity_px_median"] = float(np.median(fx * np.mean(base) / z)); S["depth_per_px_disparity_mm_median"] = float(np.median(z ** 2 / (fx * np.mean(base))))
    out = PROJECT_ROOT / "tables" / "dataset_stats.json"; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(S, indent=1))
    print(json.dumps({k: v for k, v in S.items() if not isinstance(v, dict) or len(v) < 12}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
