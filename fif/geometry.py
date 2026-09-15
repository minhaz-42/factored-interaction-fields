"""Camera and field geometry shared by labels, models and metrics.

Conventions follow the official SHOW3D API: world = rig frame; T_world_from_camera
is camera-to-world; pinhole intrinsics fx, fy, cx, cy; all lengths in millimetres
unless stated. The reference camera is headset0.
"""
from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# rigid transforms (numpy)
# ----------------------------------------------------------------------------

def world_to_cam_points(points_w: np.ndarray, T_w_c: np.ndarray) -> np.ndarray:
    """X_c = R^T (X_w - t). points_w: (N,3); T_w_c: (4,4) world-from-camera."""
    R = T_w_c[:3, :3]
    t = T_w_c[:3, 3]
    return (points_w - t) @ R


def cam_to_world_points(points_c: np.ndarray, T_w_c: np.ndarray) -> np.ndarray:
    R = T_w_c[:3, :3]
    t = T_w_c[:3, 3]
    return points_c @ R.T + t


def world_to_cam_vectors(vec_w: np.ndarray, T_w_c: np.ndarray) -> np.ndarray:
    """Free vectors: only rotation applies (V_c = R^T V_w)."""
    return vec_w @ T_w_c[:3, :3]


def cam_to_world_vectors(vec_c: np.ndarray, T_w_c: np.ndarray) -> np.ndarray:
    return vec_c @ T_w_c[:3, :3].T


def object_pose_in_camera(R_w_o: np.ndarray, t_w_o: np.ndarray, T_w_c: np.ndarray):
    """(R_c_o, t_c_o) such that q_c = R_c_o q + t_c_o for canonical vertices q."""
    R_w_c = T_w_c[:3, :3]
    t_w_c = T_w_c[:3, 3]
    R_c_o = R_w_c.T @ R_w_o
    t_c_o = R_w_c.T @ (np.asarray(t_w_o).reshape(3) - t_w_c)
    return R_c_o, t_c_o


def relative_view_transform(T_w_ref: np.ndarray, T_w_v: np.ndarray) -> np.ndarray:
    """T_ref_from_v = inv(T_w_ref) @ T_w_v (4x4)."""
    return np.linalg.inv(T_w_ref) @ T_w_v


# ----------------------------------------------------------------------------
# projection
# ----------------------------------------------------------------------------

def project_pinhole(points_c: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                    width: int, height: int):
    """points_c: (N,3) in camera frame -> uv (N,2), in_image (N,) bool, depth (N,)."""
    z = points_c[:, 2]
    safe = np.where(z > 1e-6, z, np.nan)
    u = fx * points_c[:, 0] / safe + cx
    v = fy * points_c[:, 1] / safe + cy
    uv = np.stack([u, v], 1)
    inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return uv, inside, z


def pixel_rays_in_reference(fx, fy, cx, cy, pixels_uv: np.ndarray, T_ref_from_v: np.ndarray,
                            metres: bool = True):
    """Plücker coordinates [d, o x d] (N,6) of pixel rays of view v in the reference frame.

    pixels_uv: (N,2) pixel centres in view v. Directions are unit length; the origin
    is the camera centre of view v expressed in the reference frame (in metres if
    metres=True so that moments are O(1) for a ~64 mm baseline).
    """
    N = pixels_uv.shape[0]
    x = (pixels_uv[:, 0] - cx) / fx
    y = (pixels_uv[:, 1] - cy) / fy
    d_v = np.stack([x, y, np.ones(N)], 1)
    R = T_ref_from_v[:3, :3]
    t = T_ref_from_v[:3, 3].copy()
    if metres:
        t = t / 1000.0
    d = d_v @ R.T
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    o = np.broadcast_to(t, d.shape)
    m = np.cross(o, d)
    return np.concatenate([d, m], 1)


# ----------------------------------------------------------------------------
# meshes (vertices + faces) from the bundled GLBs
# ----------------------------------------------------------------------------

_OBJ_DIR = Path(__file__).resolve().parents[1] / "third_party" / "SHOW3D-dataset-api" / "show3d" / "assets" / "objects"


def _read_glb(path: Path):
    data = path.read_bytes()
    assert data[:4] == b"glTF", path
    total = struct.unpack_from("<I", data, 8)[0]
    off = 12
    gltf = None
    binary = b""
    import json
    while off < total:
        ln, ty = struct.unpack_from("<II", data, off)
        off += 8
        chunk = data[off:off + ln]
        off += ln
        if ty == 0x4E4F534A:
            gltf = json.loads(chunk)
        elif ty == 0x004E4942:
            binary = chunk
    acc = gltf["accessors"]
    views = gltf["bufferViews"]
    V, F = [], []
    voff = 0
    comp = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32}
    for mesh in gltf.get("meshes", []):
        for prim in mesh["primitives"]:
            a = acc[prim["attributes"]["POSITION"]]
            bv = views[a["bufferView"]]
            start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
            verts = np.frombuffer(binary, np.float32, a["count"] * 3, start).reshape(-1, 3).astype(np.float64)
            V.append(verts)
            if "indices" in prim:
                ia = acc[prim["indices"]]
                ibv = views[ia["bufferView"]]
                istart = ibv.get("byteOffset", 0) + ia.get("byteOffset", 0)
                idx = np.frombuffer(binary, comp[ia["componentType"]], ia["count"], istart).astype(np.int64)
                F.append(idx.reshape(-1, 3) + voff)
            voff += verts.shape[0]
    V = np.concatenate(V, 0)
    F = np.concatenate(F, 0) if F else np.zeros((0, 3), np.int64)
    return V, F


@lru_cache(maxsize=64)
def load_object_mesh(alias: str):
    """Canonical vertices (M,3) mm and faces (F,3) for a SHOW3D object alias."""
    return _read_glb(_OBJ_DIR / f"{alias}.glb")


@lru_cache(maxsize=64)
def load_trimesh(alias: str):
    import trimesh
    V, F = load_object_mesh(alias)
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


def farthest_point_sample(points: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Indices of k farthest-point-sampled points (deterministic start)."""
    n = points.shape[0]
    idx = [int(np.argmin(points.sum(1)))]
    d = np.linalg.norm(points - points[idx[0]], axis=1)
    for _ in range(1, k):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(points - points[i], axis=1))
    return np.asarray(idx)


# ----------------------------------------------------------------------------
# nearest-vertex field (numpy, exact; identical to the official API)
# ----------------------------------------------------------------------------

def nearest_vertex_field(joints: np.ndarray, verts: np.ndarray):
    """Return vectors (J,3) joint -> nearest vertex and the vertex indices (J,)."""
    d2 = ((verts[None, :, :] - joints[:, None, :]) ** 2).sum(-1)
    m = d2.argmin(1)
    return verts[m] - joints, m


# ----------------------------------------------------------------------------
# torch operators for the factored model
# ----------------------------------------------------------------------------

def soft_nearest_vertex(joints, verts, tau: float, hard: bool = False):
    """Differentiable soft nearest vertex.

    joints: (B,J,3), verts: (B,M,3) posed vertices (same frame), tau in mm^2.
    Returns field (B,J,3), assignment weights (B,J,M).
    """
    import torch
    d2 = torch.cdist(joints, verts) ** 2  # (B,J,M)
    if hard:
        idx = d2.argmin(-1)
        q = torch.gather(verts, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        w = torch.nn.functional.one_hot(idx, verts.shape[1]).to(joints.dtype)
        return q - joints, w
    w = torch.softmax(-d2 / tau, dim=-1)
    q = torch.einsum("bjm,bmd->bjd", w, verts)
    return q - joints, w


def weighted_procrustes(src, dst, w=None, eps: float = 1e-6):
    """Rigid (R,t) minimising sum w ||R src + t - dst||^2 via SVD (differentiable).

    src, dst: (B,N,3); w: (B,N) non-negative. Returns R (B,3,3), t (B,3).
    """
    import torch
    B, N, _ = src.shape
    if w is None:
        w = torch.ones(B, N, device=src.device, dtype=src.dtype)
    w = w / (w.sum(1, keepdim=True) + eps)
    mu_s = (w.unsqueeze(-1) * src).sum(1)
    mu_d = (w.unsqueeze(-1) * dst).sum(1)
    S = src - mu_s.unsqueeze(1)
    D = dst - mu_d.unsqueeze(1)
    H = torch.einsum("bn,bni,bnj->bij", w, S, D)
    # MPS has no linalg_svd (and no autograd for its fallback): run the tiny 3x3 SVD on CPU explicitly.
    dev = H.device
    Hc = H.cpu() if dev.type == "mps" else H
    U, _, Vh = torch.linalg.svd(Hc)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    diag = torch.ones(B, 3, device=Hc.device, dtype=src.dtype)
    diag[:, 2] = torch.sign(det)
    R = (V @ torch.diag_embed(diag) @ U.transpose(-1, -2)).to(dev)
    t = mu_d - torch.einsum("bij,bj->bi", R, mu_s)
    return R, t
