"""UmeTrack hand forward kinematics in numpy (re-implementation of
facebookresearch/UmeTrack lib/common/hand_skinning.py, CC BY-NC 4.0).

A UmeTrack hand model has 22 joints (5 fingers x 4 DoF + 2 unused), 17 bone frames
(root, wrist, 5 fingers x 3) and 21 landmarks skinned by up to 3 bones. Given joint
angles (22,) and a 4x4 world-from-wrist transform, `skin_landmarks` returns the 21
landmarks in world coordinates (mm). Left hands use the mirrored model.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

NUM_DIGITS, DOF_PER_FINGER, NUM_JOINT_FRAMES = 5, 4, 17


def load_hand_model(path: str | Path) -> dict:
    hm = json.load(open(path))["hand_model"]
    return {k: (np.asarray(v, dtype=np.float64) if isinstance(v, list) else v) for k, v in hm.items()}


def mirrored(hm: dict) -> dict:
    m = dict(hm)
    axes = hm["joint_rotation_axes"].copy(); axes[:, 1:] *= -1
    rest = hm["joint_rest_positions"].copy(); rest[:, 0] *= -1
    lm = hm["landmark_rest_positions"].copy(); lm[:, 0] *= -1
    m.update(joint_rotation_axes=axes, joint_rest_positions=rest, landmark_rest_positions=lm)
    if hm.get("mesh_vertices") is not None:
        mv = hm["mesh_vertices"].copy(); mv[:, 0] *= -1; m["mesh_vertices"] = mv
    return m


def so3_exp(aa: np.ndarray) -> np.ndarray:
    """Rodrigues: (N,3) axis-angle -> (N,3,3)."""
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = np.where(th > 1e-12, aa / np.maximum(th, 1e-12), 0.0)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0], K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 2], k[..., 1], k[..., 2], -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _joint_local_transforms(axes, rest, angles):
    R = so3_exp(axes * angles[:, None])           # (20,3,3)
    t = rest - np.einsum("nij,nj->ni", R, rest)   # rotation about the joint rest position
    T = np.tile(np.eye(4), (axes.shape[0], 1, 1)); T[:, :3, :3] = R; T[:, :3, 3] = t
    return T


def skinning_transforms(hm: dict, joint_angles: np.ndarray, T_world_from_wrist: np.ndarray) -> np.ndarray:
    """(17,4,4) bone transforms: [root, wrist, finger0 x3, ..., finger4 x3]."""
    local = _joint_local_transforms(hm["joint_rotation_axes"][:20], hm["joint_rest_positions"][:20], np.asarray(joint_angles, dtype=np.float64)[:20])
    mats = [T_world_from_wrist, T_world_from_wrist]
    for f in range(NUM_DIGITS):
        chain = [T_world_from_wrist]
        for i in range(DOF_PER_FINGER):
            chain.append(chain[-1] @ local[f * DOF_PER_FINGER + i])
        mats += chain[2:]  # 3 frames per finger (frames after the 2nd..4th DoF)
    return np.stack(mats)


def skin_points(hm: dict, joint_angles, T_world_from_wrist, points: np.ndarray, bone_idx: np.ndarray, bone_w: np.ndarray) -> np.ndarray:
    X = skinning_transforms(hm, joint_angles, T_world_from_wrist)  # (17,4,4)
    W = np.zeros((points.shape[0], NUM_JOINT_FRAMES))
    for k in range(bone_idx.shape[1]):
        np.add.at(W, (np.arange(points.shape[0]), bone_idx[:, k].astype(int)), bone_w[:, k])
    P = np.concatenate([points, np.ones((points.shape[0], 1))], 1)          # (V,4)
    out = np.einsum("vf,fij,vj->vi", W, X, P)                                # LBS
    return out[:, :3]


def skin_landmarks(hm: dict, joint_angles, T_world_from_wrist) -> np.ndarray:
    return skin_points(hm, joint_angles, T_world_from_wrist, hm["landmark_rest_positions"], hm["landmark_rest_bone_indices"], hm["landmark_rest_bone_weights"])


def skin_mesh(hm: dict, joint_angles, T_world_from_wrist) -> np.ndarray:
    dbw = hm["dense_bone_weights"]  # (V,17) dense weights
    X = skinning_transforms(hm, joint_angles, T_world_from_wrist)
    P = np.concatenate([hm["mesh_vertices"], np.ones((dbw.shape[0], 1))], 1)
    return np.einsum("vf,fij,vj->vi", dbw, X, P)[:, :3]


def wrist_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4); T[:3, :3] = np.asarray(R); T[:3, 3] = np.asarray(t).reshape(3); return T
