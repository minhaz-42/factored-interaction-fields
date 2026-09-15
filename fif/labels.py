"""Derived auxiliary labels for SHOW3D interaction-field frames.

For every frame we keep the official world-frame field (built with the official
API so numbers are identical to the organisers' evaluator) and add:
  * per-view extrinsics and intrinsics,
  * camera-0 joints (2,21,3) and camera-0 object pose (R,t),
  * per-joint visibility in view 0 (inside image and not occluded by the object),
  * observability fractions (object / hand vertices inside each view),
  * per-joint field magnitude.
Everything is written as one JSON row per frame, keyed by sample_id.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .paths import API_ROOT, SHOW3D_ROOT
from . import geometry as G

sys.path.insert(0, str(API_ROOT))
from show3d.interaction_field import (  # noqa: E402
    Show3DInteractionFieldDataset, InteractionFieldSample, make_sample_id, sampled_frame_indices,
)
from show3d.dataset import DEFAULT_CONFIDENCE_THRESHOLD, object_alias_from_scene_id  # noqa: E402

HANDS = ("left", "right")
FIELD_KEYS = ("left_to_object", "right_to_object")
VIEWS = ("headset0", "headset1")
OCCLUSION_TOL_MM = 5.0


@dataclass
class FrameRecord:
    sample_id: str
    subject_id: str
    scene_id: str
    frame_index: int
    object_alias: str
    tracking_valid: bool
    object_conf: float
    hand_conf: list[float]                    # [left, right]
    field_valid: list[bool]                   # [left, right] (official validity)
    field_world: list[list | None]            # per hand (21,3) or None
    T_world_from_cam: dict                    # view -> 4x4 list or None
    intrinsics: dict                          # view -> [fx,fy,cx,cy,w,h]
    joints_cam0: list[list | None]            # per hand (21,3) camera-0 frame, present iff hand_conf>thr
    obj_R_cam0: list | None
    obj_t_cam0: list | None
    joint_visible0: list[list | None]         # per hand (21,) bools (view 0)
    joint_in_image: dict                      # view -> [left(21) or None, right(21) or None]
    obj_fov_frac: dict                        # view -> fraction of posed vertices inside image
    hand_fov_frac: dict                       # view -> [left, right] fractions (None if hand absent)
    field_mag: list[list | None]              # per hand (21,) mm

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def _calib_tuple(cal):
    return [cal.fx, cal.fy, cal.cx, cal.cy, cal.image_width, cal.image_height]


def _visibility(joints_c: np.ndarray, mesh_c, fx, fy, cx, cy, w, h):
    """Inside-image and not occluded by the posed object mesh (ray-mesh test)."""
    uv, inside, z = G.project_pinhole(joints_c, fx, fy, cx, cy, w, h)
    vis = inside.copy()
    if mesh_c is not None and inside.any():
        import trimesh
        origins = np.zeros_like(joints_c)
        dirs = joints_c / np.linalg.norm(joints_c, axis=1, keepdims=True)
        loc, idx_ray, _ = mesh_c.ray.intersects_location(origins, dirs, multiple_hits=True)
        if len(idx_ray):
            hit_d = np.linalg.norm(loc, axis=1)
            joint_d = np.linalg.norm(joints_c, axis=1)
            for r in np.unique(idx_ray):
                nearest = hit_d[idx_ray == r].min()
                if nearest < joint_d[r] - OCCLUSION_TOL_MM:
                    vis[r] = False
    return vis, inside


def build_records(root: Path, subject: str, scene: str, frame_indices: Iterable[int],
                  confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                  with_visibility: bool = True) -> list[FrameRecord]:
    samples = [InteractionFieldSample(sample_id=make_sample_id(subject, scene, i), subject_id=subject,
                                      scene_id=scene, frame_index=i) for i in frame_indices]
    ds = Show3DInteractionFieldDataset(root, samples=samples, multiview=True, decode_images=False,
                                       confidence_threshold=confidence_threshold)
    alias = samples[0].object_alias or object_alias_from_scene_id(scene)
    V, F = G.load_object_mesh(alias)
    import trimesh
    out: list[FrameRecord] = []
    for i in range(len(ds)):
        ex = ds[i]
        fd = ex.frame_data
        views = {v: fd.views[v] for v in VIEWS if v in fd.views}
        T = {v: (vf.calibration.t_world_from_camera.tolist() if vf.calibration is not None and vf.calibration.t_world_from_camera is not None else None)
             for v, vf in views.items()}
        K = {v: (_calib_tuple(vf.calibration) if vf.calibration is not None else None) for v, vf in views.items()}
        obj = fd.object_pose
        obj_conf = float(obj.confidence) if obj is not None else 0.0
        hands = [fd.left_hand, fd.right_hand]
        hand_conf = [float(hp.confidence) if hp is not None else 0.0 for hp in hands]
        labels = ex.labels
        field_world = [None, None]
        field_valid = [False, False]
        if labels is not None:
            for k, key in enumerate(FIELD_KEYS):
                f = labels.get(key)
                if f is not None:
                    field_world[k] = f.tolist()
                    field_valid[k] = True
        T0 = np.asarray(T["headset0"]) if T.get("headset0") is not None else None
        joints_cam0 = [None, None]
        obj_R = obj_t = None
        mesh_c = None
        obj_fov = {}
        hand_fov = {}
        joint_in_image = {}
        joint_vis0 = [None, None]
        field_mag = [None if f is None else np.linalg.norm(np.asarray(f), axis=1).tolist() for f in field_world]
        if T0 is not None:
            obj_ok = obj is not None and obj_conf > confidence_threshold
            if obj_ok:
                R_c, t_c = G.object_pose_in_camera(obj.rotation, obj.translation_mm, T0)
                obj_R, obj_t = R_c.tolist(), t_c.tolist()
                verts_w = obj.pose_vertices(V)
            for v, vf in views.items():
                if T.get(v) is None or K.get(v) is None:
                    continue
                Tv = np.asarray(T[v]); fx, fy, cx, cy, w, h = K[v]
                if obj_ok:
                    vc = G.world_to_cam_points(verts_w, Tv)
                    _, inside, _ = G.project_pinhole(vc, fx, fy, cx, cy, w, h)
                    obj_fov[v] = float(inside.mean())
                hf = [None, None]
                jin = [None, None]
                for k, hp in enumerate(hands):
                    if hp is None or hp.confidence <= confidence_threshold or hp.landmarks_world_mm is None:
                        continue
                    jc = G.world_to_cam_points(hp.landmarks_world_mm, Tv)
                    _, inside, _ = G.project_pinhole(jc, fx, fy, cx, cy, w, h)
                    hf[k] = float(inside.mean())
                    jin[k] = inside.tolist()
                hand_fov[v] = hf
                joint_in_image[v] = jin
            fx, fy, cx, cy, w, h = K["headset0"]
            if obj_ok and with_visibility:
                vc0 = G.world_to_cam_points(verts_w, T0)
                mesh_c = trimesh.Trimesh(vertices=vc0, faces=F, process=False)
            for k, hp in enumerate(hands):
                if hp is None or hp.confidence <= confidence_threshold or hp.landmarks_world_mm is None:
                    continue
                jc = G.world_to_cam_points(hp.landmarks_world_mm, T0)
                joints_cam0[k] = jc.tolist()
                if with_visibility:
                    vis, _ = _visibility(jc, mesh_c, fx, fy, cx, cy, w, h)
                    joint_vis0[k] = vis.tolist()
        out.append(FrameRecord(
            sample_id=ex.sample.sample_id, subject_id=subject, scene_id=scene, frame_index=ex.sample.frame_index,
            object_alias=alias, tracking_valid=bool(fd.headset_tracking_valid), object_conf=obj_conf,
            hand_conf=hand_conf, field_valid=field_valid, field_world=field_world, T_world_from_cam=T,
            intrinsics=K, joints_cam0=joints_cam0, obj_R_cam0=obj_R, obj_t_cam0=obj_t, joint_visible0=joint_vis0,
            joint_in_image=joint_in_image, obj_fov_frac=obj_fov, hand_fov_frac=hand_fov, field_mag=field_mag))
    return out


def scene_frame_indices(root: Path, subject: str, scene: str, fps: int) -> list[int]:
    import cv2
    cap = cv2.VideoCapture(str(root / "scenes" / subject / scene / "headset0.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    return sampled_frame_indices(n, fps)


def read_records(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
