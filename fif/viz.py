"""Rendering helpers for real-frame overlays and 3-D panels (paper figures)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, cv2
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from . import geometry as G
from .paths import SHOW3D_ROOT

LEFT, RIGHT, OBJ, GT, PRED = "#d95f02", "#1f78b4", "#33a02c", "#6e6e6e", "#b2182b"
BONES = [(5, 6), (6, 7), (7, 0), (5, 8), (8, 9), (9, 10), (10, 1), (5, 11), (11, 12), (12, 13), (13, 2), (5, 14), (14, 15), (15, 16), (16, 3), (5, 17), (17, 18), (18, 19), (19, 4)]


def read_frame(subject, scene, frame_index, view="headset0"):
    cap = cv2.VideoCapture(str(SHOW3D_ROOT / "scenes" / subject / scene / f"{view}.mp4")); cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, img = cap.read(); cap.release(); assert ok, (subject, scene, frame_index)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_aux(aux_path, sample_ids=None):
    rows = {}
    for l in open(aux_path):
        if not l.strip(): continue
        r = json.loads(l)
        if sample_ids is None or r["sample_id"] in sample_ids: rows[r["sample_id"]] = r
    return rows


def cam_geometry(rec):
    """GT camera-0 joints (2,), GT camera-0 fields (2,), posed object vertices, faces."""
    T0 = np.asarray(rec["T_world_from_cam"]["headset0"])
    joints = [None if rec["joints_cam0"][k] is None else np.asarray(rec["joints_cam0"][k]) for k in range(2)]
    fields = [None if rec["field_world"][k] is None else G.world_to_cam_vectors(np.asarray(rec["field_world"][k]), T0) for k in range(2)]
    V, F = G.load_object_mesh(rec["object_alias"])
    verts = None
    if rec["obj_R_cam0"] is not None:
        verts = V @ np.asarray(rec["obj_R_cam0"]).T + np.asarray(rec["obj_t_cam0"])
    return joints, fields, verts, F, T0


def pred_to_cam(pred_world, T0):
    return None if pred_world is None else G.world_to_cam_vectors(np.asarray(pred_world), T0)


def crop_box(uv_list, w, h, margin=60):
    pts = np.concatenate([p for p in uv_list if p is not None and len(p)])
    x0, y0 = np.maximum(pts.min(0) - margin, 0).astype(int); x1, y1 = np.minimum(pts.max(0) + margin, [w, h]).astype(int)
    # make roughly square
    side = max(x1 - x0, y1 - y0); cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0, x1 = max(0, cx - side // 2), min(w, cx + side // 2); y0, y1 = max(0, cy - side // 2), min(h, cy + side // 2)
    return x0, y0, x1, y1


def draw_overlay(ax, gray, rec, joints, fields_gt, fields_pred=None, verts=None, F=None, box=None, show_gt=True, show_mesh=True, title=None, err_text=None):
    fx, fy, cx, cy, w, h = rec["intrinsics"]["headset0"]
    if box is None:
        uvs = [G.project_pinhole(j, fx, fy, cx, cy, w, h)[0] for j in joints if j is not None]
        if fields_gt is not None:  # include the vector endpoints so far-field targets stay inside the crop
            uvs += [G.project_pinhole(j + f, fx, fy, cx, cy, w, h)[0] for j, f in zip(joints, fields_gt) if j is not None and f is not None]
        if verts is not None:
            uv_o, in_o, _ = G.project_pinhole(verts, fx, fy, cx, cy, w, h); uvs.append(uv_o[in_o])
        box = crop_box(uvs, w, h)
    x0, y0, x1, y1 = box
    ax.imshow(gray[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255); ax.set_axis_off()
    if show_mesh and verts is not None and F is not None:
        uv_o, in_o, _ = G.project_pinhole(verts, fx, fy, cx, cy, w, h)
        e = F[np.random.default_rng(0).choice(len(F), size=min(1200, len(F)), replace=False)]
        segs = [[uv_o[i] - [x0, y0], uv_o[j] - [x0, y0]] for tri in e for i, j in ((tri[0], tri[1]), (tri[1], tri[2])) if in_o[i] and in_o[j]]
        ax.add_collection(LineCollection(segs, colors=OBJ, linewidths=0.25, alpha=0.45))
    for k, col in enumerate((LEFT, RIGHT)):
        J = joints[k]
        if J is None: continue
        uv, _, _ = G.project_pinhole(J, fx, fy, cx, cy, w, h); uv = uv - [x0, y0]
        ax.add_collection(LineCollection([[uv[i], uv[j]] for i, j in BONES], colors=col, linewidths=0.9))
        ax.scatter(uv[:, 0], uv[:, 1], s=5, c=col, zorder=3, linewidths=0)
        if show_gt and fields_gt is not None and fields_gt[k] is not None:
            end, _, _ = G.project_pinhole(J + fields_gt[k], fx, fy, cx, cy, w, h); end = end - [x0, y0]
            for i in range(21): ax.annotate("", xy=end[i], xytext=uv[i], arrowprops=dict(arrowstyle="-|>", color=GT, lw=0.55, mutation_scale=4.5))
        if fields_pred is not None and fields_pred[k] is not None:
            end, _, _ = G.project_pinhole(J + fields_pred[k], fx, fy, cx, cy, w, h); end = end - [x0, y0]
            for i in range(21): ax.annotate("", xy=end[i], xytext=uv[i], arrowprops=dict(arrowstyle="-|>", color=PRED, lw=0.55, mutation_scale=4.5))
    if title: ax.set_title(title, fontsize=7, pad=2)
    if err_text: ax.text(0.02, 0.02, err_text, transform=ax.transAxes, fontsize=6, color="white", va="bottom", bbox=dict(facecolor="black", alpha=0.5, lw=0, pad=1.5))
    return box


def draw_3d(ax, joints, fields_gt, fields_pred=None, verts=None, elev=18, azim=-60, title=None):
    if verts is not None:
        sub = verts[np.random.default_rng(1).choice(len(verts), size=min(1000, len(verts)), replace=False)]
        ax.scatter(sub[:, 0], sub[:, 2], -sub[:, 1], s=0.5, c=OBJ, alpha=0.35, linewidths=0)
    pts = [] if verts is None else [sub]
    for k, col in enumerate((LEFT, RIGHT)):
        J = joints[k]
        if J is None: continue
        pts.append(J)
        for i, j in BONES: ax.plot([J[i, 0], J[j, 0]], [J[i, 2], J[j, 2]], [-J[i, 1], -J[j, 1]], c=col, lw=0.9)
        ax.scatter(J[:, 0], J[:, 2], -J[:, 1], s=5, c=col, linewidths=0)
        for fields, col2 in ((fields_gt, GT), (fields_pred, PRED)):
            if fields is None or fields[k] is None: continue
            Fv = fields[k]
            for i in range(21): ax.plot([J[i, 0], J[i, 0] + Fv[i, 0]], [J[i, 2], J[i, 2] + Fv[i, 2]], [-J[i, 1], -J[i, 1] - Fv[i, 1]], c=col2, lw=0.55)
    allp = np.concatenate(pts); c = allp.mean(0); r = (allp.max(0) - allp.min(0)).max() / 2 * 1.05
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[2] - r, c[2] + r); ax.set_zlim(-c[1] - r, -c[1] + r)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]); ax.tick_params(length=0)
    ax.set_xlabel("x", labelpad=-10, fontsize=6); ax.set_ylabel("z (depth)", labelpad=-10, fontsize=6); ax.set_zlabel("-y", labelpad=-10, fontsize=6)
    ax.view_init(elev=elev, azim=azim)
    if title: ax.set_title(title, fontsize=7, pad=0, y=0.98)


def per_hand_ade(pred_cam, gt_cam):
    errs = [np.linalg.norm(pred_cam[k] - gt_cam[k], axis=1).mean() for k in range(2) if gt_cam[k] is not None and pred_cam is not None and pred_cam[k] is not None]
    return float(np.mean(errs)) if errs else float("nan")
