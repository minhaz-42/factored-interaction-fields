"""Official, stratified and geometric metrics for interaction fields.

`official_metrics` re-implements the organisers' evaluator (ADE over predicted valid
targets, recall over valid targets, acc@k) and is cross-checked against the API.
Stratified and geometric metrics use the derived labels of `fif.labels`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Mapping

import numpy as np

from . import geometry as G

FIELD_KEYS = ("left_to_object", "right_to_object")
THRESHOLDS = (10.0, 50.0, 100.0)
JOINT_GROUPS = {
    "fingertips": [0, 1, 2, 3, 4], "wrist": [5], "palm": [20],
    "thumb": [6, 7], "proximal": [8, 11, 14, 17], "intermediate": [9, 12, 15, 18], "distal": [10, 13, 16, 19],
}


def official_metrics(refs: list[dict], preds: Mapping[str, dict]) -> dict:
    """refs: derived-label rows; preds: sample_id -> {key: (21,3) array or None}."""
    out = {}
    for k, key in enumerate(FIELD_KEYS):
        n_valid = n_missing = n_points = 0
        err_sum = 0.0
        correct = {t: 0 for t in THRESHOLDS}
        for r in refs:
            if not r["field_valid"][k]:
                continue
            n_valid += 1
            p = preds.get(r["sample_id"], {}).get(key)
            if p is None:
                n_missing += 1
                continue
            e = np.linalg.norm(np.asarray(p) - np.asarray(r["field_world"][k]), axis=1)
            n_points += e.shape[0]
            err_sum += float(e.sum())
            for t in THRESHOLDS:
                correct[t] += int((e <= t).sum())
        out[key] = {
            "num_samples": n_valid, "missing": n_missing,
            "ade_mm": err_sum / n_points if n_points else None,
            "recall": (n_valid - n_missing) / n_valid if n_valid else None,
            **{f"acc@{int(t)}mm": (correct[t] / n_points if n_points else 0.0) for t in THRESHOLDS},
        }
    ades = [m["ade_mm"] for m in out.values() if m["ade_mm"] is not None]
    recs = [m["recall"] for m in out.values() if m["recall"] is not None]
    out["mean_ade_mm"] = float(np.mean(ades)) if ades else None
    out["mean_recall"] = float(np.mean(recs)) if recs else None
    for t in THRESHOLDS:
        vals = [m[f"acc@{int(t)}mm"] for key, m in out.items() if key in FIELD_KEYS and m["num_samples"] - m["missing"] > 0]
        out[f"mean_acc@{int(t)}mm"] = float(np.mean(vals)) if vals else None
    return out


def per_joint_errors(refs: list[dict], preds: Mapping[str, dict]) -> list[dict]:
    """Flat table: one row per (sample, hand, joint) with error and strata."""
    rows = []
    for r in refs:
        T0 = np.asarray(r["T_world_from_cam"]["headset0"]) if r["T_world_from_cam"].get("headset0") is not None else None
        for k, key in enumerate(FIELD_KEYS):
            if not r["field_valid"][k]:
                continue
            p = preds.get(r["sample_id"], {}).get(key)
            if p is None:
                continue
            gt = np.asarray(r["field_world"][k]); pr = np.asarray(p)
            e_vec = pr - gt
            err = np.linalg.norm(e_vec, axis=1)
            mag = np.linalg.norm(gt, axis=1)
            # along-ray / lateral decomposition in camera 0 (ray through the GT endpoint)
            par = lat = np.full(21, np.nan)
            if T0 is not None and r["joints_cam0"][k] is not None:
                jc = np.asarray(r["joints_cam0"][k])
                gt_c = G.world_to_cam_vectors(gt, T0); e_c = G.world_to_cam_vectors(e_vec, T0)
                end = jc + gt_c
                ray = end / np.linalg.norm(end, axis=1, keepdims=True)
                par = np.abs((e_c * ray).sum(1))
                lat = np.linalg.norm(e_c - (e_c * ray).sum(1, keepdims=True) * ray, axis=1)
            # direction / magnitude errors
            cos = (pr * gt).sum(1) / (np.linalg.norm(pr, axis=1) * mag + 1e-9)
            ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            vis = r["joint_visible0"][k]
            inimg = r["joint_in_image"].get("headset0", [None, None])[k]
            for j in range(21):
                rows.append({
                    "sample_id": r["sample_id"], "subject": r["subject_id"], "scene": r["scene_id"], "object": r["object_alias"],
                    "hand": key.split("_")[0], "joint": j, "err": float(err[j]), "mag": float(mag[j]),
                    "mag_err": float(abs(np.linalg.norm(pr[j]) - mag[j])), "dir_err_deg": float(ang[j]),
                    "err_par": float(par[j]), "err_lat": float(lat[j]),
                    "visible0": (None if vis is None else bool(vis[j])), "in_image0": (None if inimg is None else bool(inimg[j])),
                    "obj_fov0": r["obj_fov_frac"].get("headset0"), "hand_fov0": (r["hand_fov_frac"].get("headset0", [None, None])[k]),
                })
    return rows


def _fov_stratum(f):
    if f is None or (isinstance(f, float) and np.isnan(f)):
        return "unknown"
    return "in_view" if f >= 0.9 else ("partial" if f >= 0.1 else "out_of_view")


def _mag_stratum(m):
    return "near(<15)" if m < 15 else ("mid(15-60)" if m < 60 else "far(>60)")


def stratified(rows: list[dict]) -> dict:
    """Mean error by stratum (joint-level rows)."""
    groups = {
        "magnitude": lambda r: _mag_stratum(r["mag"]),
        "visibility0": lambda r: {None: "unknown", True: "visible", False: "occluded_or_outside"}[r["visible0"]],
        "object_fov0": lambda r: _fov_stratum(r["obj_fov0"]),
        "hand_fov0": lambda r: _fov_stratum(r["hand_fov0"]),
        "hand": lambda r: r["hand"],
        "joint_group": lambda r: next(g for g, js in JOINT_GROUPS.items() if r["joint"] in js),
        "object": lambda r: r["object"],
        "subject": lambda r: r["subject"],
    }
    out = {}
    for name, fn in groups.items():
        acc = defaultdict(list)
        for r in rows:
            acc[fn(r)].append(r["err"])
        out[name] = {k: {"ade_mm": float(np.mean(v)), "n": len(v)} for k, v in sorted(acc.items())}
    out["along_ray_vs_lateral"] = {
        "mean_err_par_mm": float(np.nanmean([r["err_par"] for r in rows])),
        "mean_err_lat_mm": float(np.nanmean([r["err_lat"] for r in rows])),
    }
    far = [r for r in rows if r["mag"] > 20]
    out["direction_err_deg_mag>20"] = float(np.mean([r["dir_err_deg"] for r in far])) if far else None
    out["magnitude_err_mm"] = float(np.mean([r["mag_err"] for r in rows]))
    return out


def surface_distance(refs: list[dict], preds: Mapping[str, dict], mode: str = "vertex") -> dict:
    """Distance of predicted endpoints (GT joint + predicted vector) to the GT posed object.

    mode='vertex': nearest-vertex distance (fast). mode='surface': point-to-triangle via trimesh.
    Also reports the fraction of endpoints inside the mesh by > 5 mm (signed distance,
    approximate for non-watertight meshes; reported only when mode='surface').
    """
    import trimesh
    dists, inside = [], []
    for r in refs:
        if r["obj_R_cam0"] is None:
            continue
        V, F = G.load_object_mesh(r["object_alias"])
        R = np.asarray(r["obj_R_cam0"]); t = np.asarray(r["obj_t_cam0"])
        verts_c = V @ R.T + t
        T0 = np.asarray(r["T_world_from_cam"]["headset0"])
        mesh = trimesh.Trimesh(vertices=verts_c, faces=F, process=False) if mode == "surface" else None
        for k, key in enumerate(FIELD_KEYS):
            if not r["field_valid"][k] or r["joints_cam0"][k] is None:
                continue
            p = preds.get(r["sample_id"], {}).get(key)
            if p is None:
                continue
            end_c = np.asarray(r["joints_cam0"][k]) + G.world_to_cam_vectors(np.asarray(p), T0)
            if mode == "surface":
                _, d, _ = trimesh.proximity.closest_point(mesh, end_c)
                sd = trimesh.proximity.signed_distance(mesh, end_c)
                inside.extend((sd > 5.0).tolist())  # trimesh: positive = inside
            else:
                d = np.sqrt(((verts_c[None] - end_c[:, None]) ** 2).sum(-1)).min(1)
            dists.extend(d.tolist())
    return {"surface_distance_mm": float(np.mean(dists)) if dists else None,
            "penetration_rate": (float(np.mean(inside)) if inside else None), "n": len(dists), "mode": mode}


def read_predictions_jsonl(path) -> dict:
    preds = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            preds[row["sample_id"]] = {k: (None if row.get(k) is None else np.asarray(row[k], dtype=np.float64)) for k in FIELD_KEYS}
    return preds
