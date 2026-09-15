"""Dataset over cached DINOv3 tokens + derived labels (docs/07 notation, camera-0 frame)."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset
from . import geometry as G

ALIASES = ["aria", "balandabowl", "bbq", "birdhousetoy", "brushholder", "canparmesan", "cansoup", "cantomatosauce", "dinotoy",
           "dumbbell", "keyboard", "milk", "mouse", "mug", "mug2", "mustard", "orangejuice", "ranch", "vase", "vegetables", "waffles"]
ALIAS_ID = {a: i for i, a in enumerate(ALIASES)}
IMG_W, IMG_H = 1024, 1280


@lru_cache(maxsize=None)
def object_geometry(alias: str, n_ctrl: int, n_verts: int):
    """(control points (n_ctrl,3), sub-sampled vertices (<=n_verts,3), full vertices) in the canonical frame, mm."""
    V, F = G.load_object_mesh(alias)
    ctrl_idx = G.farthest_point_sample(V, n_ctrl)
    vidx = G.farthest_point_sample(V, n_verts) if len(V) > n_verts else np.arange(len(V))
    return V[ctrl_idx].astype(np.float32), V[vidx].astype(np.float32), V


def token_grid_pixels(grid_hw, img_w=IMG_W, img_h=IMG_H):
    h, w = grid_hw
    ys, xs = np.meshgrid((np.arange(h) + 0.5) * img_h / h, (np.arange(w) + 0.5) * img_w / w, indexing="ij")
    return np.stack([xs.ravel(), ys.ravel()], 1)  # (N,2): row-major, same order as ViT patch tokens


class CachedFieldDataset(Dataset):
    def __init__(self, cache_dir: Path, aux_path: Path, subjects, views=("headset0", "headset1"), require_field=True,
                 n_ctrl=16, n_verts=1024, token_drop=0.0):
        self.cache_dir, self.views, self.n_ctrl, self.n_verts, self.token_drop = Path(cache_dir), tuple(views), n_ctrl, n_verts, token_drop
        meta = json.loads((self.cache_dir / "meta.json").read_text())
        self.grid_hw = (meta["height"] // 16, meta["width"] // 16)
        self.pix = token_grid_pixels(self.grid_hw)
        subjects = set(subjects)
        rows = [json.loads(l) for l in open(aux_path) if l.strip()]
        rows = [r for r in rows if r["subject_id"] in subjects and r["tracking_valid"] and r["T_world_from_cam"].get("headset0") is not None]
        if require_field:
            rows = [r for r in rows if any(r["field_valid"])]
        else:
            rows = [r for r in rows if any(r["field_valid"]) or r["obj_R_cam0"] is not None or any(j is not None for j in r["joints_cam0"])]
        self._frame_lookup = {}
        items = []
        for r in rows:
            ok = True
            for v in self.views:
                key = (r["subject_id"], r["scene_id"], v)
                if key not in self._frame_lookup:
                    d = self.cache_dir / r["subject_id"] / r["scene_id"] / v
                    self._frame_lookup[key] = ({int(f): i for i, f in enumerate(np.load(d / "frame_index.npy"))} if (d / "frame_index.npy").exists() else None)
                lk = self._frame_lookup[key]
                if lk is None or r["frame_index"] not in lk or r["T_world_from_cam"].get(v) is None:
                    ok = False; break
            if ok:
                items.append(r)
        self.rows = items
        self._mm = {}

    def __len__(self):
        return len(self.rows)

    def _tokens(self, subject, scene, view, frame):
        key = (subject, scene, view)
        if key not in self._mm:
            d = self.cache_dir / subject / scene / view
            self._mm[key] = (np.load(d / "tokens.npy", mmap_mode="r"), np.load(d / "scale.npy", mmap_mode="r"))
        tok, sc = self._mm[key]
        i = self._frame_lookup[key][frame]
        return tok[i].astype(np.float32) * sc[i].astype(np.float32)[:, None]

    def __getitem__(self, idx):
        r = self.rows[idx]
        T0 = np.asarray(r["T_world_from_cam"]["headset0"])
        toks, rays = [], []
        for v in self.views:
            toks.append(self._tokens(r["subject_id"], r["scene_id"], v, r["frame_index"]))
            fx, fy, cx, cy, w, h = r["intrinsics"][v]
            Tv = np.asarray(r["T_world_from_cam"][v])
            rays.append(G.pixel_rays_in_reference(fx, fy, cx, cy, self.pix, G.relative_view_transform(T0, Tv)).astype(np.float32))
        tokens = np.stack(toks); rays = np.stack(rays)
        if self.token_drop > 0:
            keep = (np.random.rand(*tokens.shape[:2]) > self.token_drop).astype(np.float32)
            tokens = tokens * keep[..., None]
        alias = r["object_alias"]; ctrl, vsub, _ = object_geometry(alias, self.n_ctrl, self.n_verts)
        M = self.n_verts
        verts_canon = np.zeros((M, 3), np.float32); verts_canon[: len(vsub)] = vsub
        vmask = np.zeros(M, bool); vmask[: len(vsub)] = True
        field = np.zeros((2, 21, 3), np.float32); fmask = np.zeros(2, np.float32)
        joints = np.zeros((2, 21, 3), np.float32); hmask = np.zeros(2, np.float32)
        vis = np.zeros((2, 21), np.float32); vmask_j = np.zeros((2, 21), np.float32)
        for k in range(2):
            if r["field_valid"][k]:
                field[k] = G.world_to_cam_vectors(np.asarray(r["field_world"][k]), T0); fmask[k] = 1
            if r["joints_cam0"][k] is not None:
                joints[k] = np.asarray(r["joints_cam0"][k]); hmask[k] = 1
                if r["joint_visible0"][k] is not None:
                    vis[k] = np.asarray(r["joint_visible0"][k], np.float32); vmask_j[k] = 1
        obj_mask = np.float32(r["obj_R_cam0"] is not None)
        verts_gt = np.zeros((M, 3), np.float32); t_obj = np.zeros(3, np.float32)
        if obj_mask:
            R = np.asarray(r["obj_R_cam0"], np.float32); t_obj = np.asarray(r["obj_t_cam0"], np.float32)
            verts_gt[: len(vsub)] = vsub @ R.T + t_obj
        return {
            "tokens": torch.from_numpy(tokens), "rays": torch.from_numpy(rays), "alias": torch.tensor(ALIAS_ID[alias]),
            "ctrl_canon": torch.from_numpy(ctrl), "verts_canon": torch.from_numpy(verts_canon), "verts_mask": torch.from_numpy(vmask),
            "field": torch.from_numpy(field), "field_mask": torch.from_numpy(fmask), "joints": torch.from_numpy(joints), "hand_mask": torch.from_numpy(hmask),
            "vis": torch.from_numpy(vis), "vis_mask": torch.from_numpy(vmask_j), "verts_gt": torch.from_numpy(verts_gt), "obj_mask": torch.tensor(obj_mask),
            "t_obj": torch.from_numpy(t_obj), "R0": torch.from_numpy(T0[:3, :3].astype(np.float32)), "idx": torch.tensor(idx),
        }
