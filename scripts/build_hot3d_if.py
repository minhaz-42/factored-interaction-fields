"""Build the HOT3D-IF cross-dataset set in SHOW3D's format (docs/06 protocol).

For selected HOT3D-Clips (Quest 3, training split, which carry ground truth):
  * download the clip tar from HuggingFace (bop-benchmark/hot3d),
  * rectify both fisheye streams to a virtual portrait pinhole (1024x1280,
    f = 454.45, centred principal point, SHOW3D orientation) and write grayscale
    JPEGs + index.jsonl like `show3d.extract_images`,
  * recover 21 UmeTrack landmarks per hand by forward kinematics,
  * choose per hand the nearest SHOW3D-shared object (<= 150 mm) as target,
  * write labels.jsonl (official schema) and aux_labels (fif.labels schema).
Sample ids: "<participant>/<clip_id>:<frame:06d>".
"""
from __future__ import annotations
import argparse, io, json, sys, tarfile, time
from pathlib import Path
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fif  # noqa
from fif import geometry as G, umetrack_fk as U
from fif.fisheye import rectification_map, rectify, pinhole_K
from fif.labels import FrameRecord, OCCLUSION_TOL_MM
from fif.paths import DATA_ROOT

NAME2ALIAS = {"dumbbell_5lb": "dumbbell", "mouse": "mouse", "keyboard": "keyboard", "mug_white": "mug", "mug_patterned": "mug2", "bowl": "balandabowl",
              "vase": "vase", "holder_black": "brushholder", "birdhouse_toy": "birdhousetoy", "dino_toy": "dinotoy", "carton_milk": "milk", "carton_oj": "orangejuice",
              "bottle_mustard": "mustard", "bottle_ranch": "ranch", "bottle_bbq": "bbq", "can_soup": "cansoup", "can_parmesan": "canparmesan",
              "can_tomato_sauce": "cantomatosauce", "food_waffles": "waffles", "food_vegetables": "vegetables", "aria_small": "aria"}
STREAMS = {"headset0": "1201-1", "headset1": "1201-2"}
FX, CX, CY, W, H = 454.45, 512.0, 640.0, 1024, 1280
ROT_DEG = -90.0  # virtual-camera roll so that the landscape sensor becomes SHOW3D-like portrait (verified visually)


def T_from(q_t, scale=1000.0):
    q = q_t["quaternion_wxyz"]; R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(q_t["translation_xyz"]) * scale; return T


def process_clip(tar_path: Path, out_frames: Path, fps: int, max_dist: float, participant_filter=None):
    t = tarfile.open(tar_path); names = t.getnames(); rd = lambda n: t.extractfile(n).read()
    J = lambda suf: json.loads(rd(next(n for n in names if n.endswith(suf))).decode())
    clip_id = tar_path.stem.replace("clip-", "")
    info0 = J("000000.info.json"); participant = info0["participant_id"]
    if participant_filter and participant not in participant_filter: return [], [], []
    hm = {k: (np.asarray(v, dtype=np.float64) if isinstance(v, list) else v) for k, v in J("__hand_shapes.json__")["umetrack"].items()}
    models = {"left": hm, "right": U.mirrored(hm)}
    Rz = Rot.from_euler("z", ROT_DEG, degrees=True).as_matrix()  # R_fisheye_from_virtual
    Kd = pinhole_K(FX, FX, CX, CY)
    frames = sorted({n.split("/")[-1].split(".")[0] for n in names if n.endswith(".info.json")})
    stride = max(1, round(30 / fps)); frames = [f for f in frames if int(f) % stride == 0]
    maps = {}
    index_rows, label_rows, aux_rows = [], [], []
    import trimesh
    for fr in frames:
        cams = J(f"{fr}.cameras.json"); hands = J(f"{fr}.hands.json"); objs = J(f"{fr}.objects.json")
        fidx = int(fr); sid = f"{participant}/{clip_id}:{fidx:06d}"
        T_wc = {}; K = {}
        for view, stream in STREAMS.items():
            cam = cams[stream]; params = cam["calibration"]["projection_params"]
            if stream not in maps:
                maps[stream] = rectification_map(params, (cam["calibration"]["image_height"], cam["calibration"]["image_width"]), Kd, (H, W), R_src_from_dst=Rz)
            img = cv2.imdecode(np.frombuffer(rd(next(n for n in names if n.endswith(f"{fr}.image_{stream}.jpg"))), np.uint8), cv2.IMREAD_GRAYSCALE)
            out = rectify(img, *maps[stream])
            d = out_frames / participant / clip_id / view; d.mkdir(parents=True, exist_ok=True)
            p = d / f"{fidx:06d}.jpg"
            if not p.exists(): cv2.imwrite(str(p), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
            index_rows.append({"sample_id": sid, "subject_id": participant, "scene_id": clip_id, "frame_index": fidx, "view": view, "image": p.relative_to(out_frames).as_posix()})
            T_fis = T_from(cam["T_world_from_camera"]); T_virt = T_fis.copy(); T_virt[:3, :3] = T_fis[:3, :3] @ Rz  # world_from_virtual
            T_wc[view] = T_virt.tolist(); K[view] = [FX, FX, CX, CY, W, H]
        T0 = np.asarray(T_wc["headset0"])
        # hands
        joints_w = [None, None]; conf = [0.0, 0.0]
        for k, side in enumerate(("left", "right")):
            if side in hands and "umetrack_pose" in hands[side]:
                up = hands[side]["umetrack_pose"]; joints_w[k] = U.skin_landmarks(models[side], up["joint_angles"], T_from(up["T_world_from_wrist"])); conf[k] = 1.0
        # objects (shared with SHOW3D only)
        posed = {}
        for oid, ol in objs.items():
            o = ol[0]; alias = NAME2ALIAS.get(o["object_name"])
            if alias is None: continue
            V, F = G.load_object_mesh(alias); T_wo = T_from(o["T_world_from_object"])
            posed[alias] = (V @ T_wo[:3, :3].T + T_wo[:3, 3], F, T_wo)
        # per-hand nearest shared object; frame-level target = alias nearest to any valid hand
        best = None
        for k in range(2):
            if joints_w[k] is None: continue
            for alias, (vw, F, T_wo) in posed.items():
                d = np.sqrt(((vw[None] - joints_w[k][:, None]) ** 2).sum(-1)).min()
                if d <= max_dist and (best is None or d < best[0]): best = (d, alias)
        field_w = [None, None]; field_valid = [False, False]; alias = None; obj_R = obj_t = None; obj_fov = {}; verts_w = None; F = None
        if best is not None:
            alias = best[1]; verts_w, F, T_wo = posed[alias]
            R_c, t_c = G.object_pose_in_camera(T_wo[:3, :3], T_wo[:3, 3], T0); obj_R, obj_t = R_c.tolist(), t_c.tolist()
            for k in range(2):
                if joints_w[k] is None: continue
                d = np.sqrt(((verts_w[None] - joints_w[k][:, None]) ** 2).sum(-1)).min()
                if d <= max_dist:
                    field_w[k] = G.nearest_vertex_field(joints_w[k], verts_w)[0].tolist(); field_valid[k] = True
        joints_cam0 = [None, None]; vis0 = [None, None]; jin = {}; hand_fov = {}
        mesh_c = None
        if verts_w is not None:
            vc0 = G.world_to_cam_points(verts_w, T0); mesh_c = trimesh.Trimesh(vertices=vc0, faces=F, process=False)
        for view in STREAMS:
            Tv = np.asarray(T_wc[view]); fx, fy, cx, cy, w, h = K[view]
            if verts_w is not None:
                _, inside, _ = G.project_pinhole(G.world_to_cam_points(verts_w, Tv), fx, fy, cx, cy, w, h); obj_fov[view] = float(inside.mean())
            hf = [None, None]; ji = [None, None]
            for k in range(2):
                if joints_w[k] is None: continue
                jc = G.world_to_cam_points(joints_w[k], Tv); _, inside, _ = G.project_pinhole(jc, fx, fy, cx, cy, w, h); hf[k] = float(inside.mean()); ji[k] = inside.tolist()
            hand_fov[view] = hf; jin[view] = ji
        for k in range(2):
            if joints_w[k] is None: continue
            jc = G.world_to_cam_points(joints_w[k], T0); joints_cam0[k] = jc.tolist()
            from fif.labels import _visibility
            vis, _ = _visibility(jc, mesh_c, FX, FX, CX, CY, W, H); vis0[k] = vis.tolist()
        rec = FrameRecord(sample_id=sid, subject_id=participant, scene_id=clip_id, frame_index=fidx, object_alias=alias or "none", tracking_valid=True,
                          object_conf=(1.0 if alias else 0.0), hand_conf=conf, field_valid=field_valid, field_world=field_w, T_world_from_cam=T_wc, intrinsics=K,
                          joints_cam0=joints_cam0, obj_R_cam0=obj_R, obj_t_cam0=obj_t, joint_visible0=vis0, joint_in_image=jin, obj_fov_frac=obj_fov, hand_fov_frac=hand_fov,
                          field_mag=[None if f is None else np.linalg.norm(np.asarray(f), axis=1).tolist() for f in field_w])
        aux_rows.append(rec.to_json())
        if any(field_valid):
            label_rows.append(json.dumps({"sample_id": sid, "left_to_object": field_w[0], "right_to_object": field_w[1]}))
    return index_rows, label_rows, aux_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=60); ap.add_argument("--fps", type=int, default=10); ap.add_argument("--max-dist", type=float, default=150.0)
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "frames" / "hot3d_10fps"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--participants", nargs="*", default=None); ap.add_argument("--clip-ids", nargs="*", default=None)
    ap.add_argument("--tag", default="", help="suffix for the aux file name (e.g. 'extra')"); ap.add_argument("--exclude-info", type=Path, nargs="*", default=[], help="extract_info.json files whose clips are excluded")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    cd = json.load(open(hf_hub_download("bop-benchmark/hot3d", "clip_definitions.json", repo_type="dataset")))
    splits = json.load(open(hf_hub_download("bop-benchmark/hot3d", "clip_splits.json", repo_type="dataset")))
    train_q3 = [c for c in splits["train"]["Quest3"]]
    by_part = {}
    for c in train_q3: by_part.setdefault(cd[str(c)]["sequence_id"].split("_")[0], []).append(c)
    print("Quest3 train clips:", len(train_q3), "| participants:", {k: len(v) for k, v in sorted(by_part.items())})
    rng = np.random.default_rng(a.seed)
    excluded = set()
    for ei in a.exclude_info: excluded |= set(json.load(open(ei))["clips"])
    if a.clip_ids: chosen = [int(c) for c in a.clip_ids]
    else:
        parts = a.participants or sorted(by_part)
        per = max(1, a.n_clips // len(parts)); chosen = []
        for p in parts:
            pool = [c for c in by_part[p] if c not in excluded]
            chosen += list(rng.choice(pool, size=min(per, len(pool)), replace=False))
    a.out.mkdir(parents=True, exist_ok=True)
    idx_all, lab_all, aux_all = [], [], []
    t0 = time.time()
    for i, c in enumerate(chosen):
        tp = Path(hf_hub_download("bop-benchmark/hot3d", f"train_quest3/clip-{int(c):06d}.tar", repo_type="dataset"))
        ir, lr, ar = process_clip(tp, a.out, a.fps, a.max_dist, a.participants)
        idx_all += ir; lab_all += lr; aux_all += ar
        print(f"[{i+1}/{len(chosen)}] clip {c}: {len(ir)//2} frames, {len(lr)} labelled | {time.time()-t0:.0f}s", flush=True)
    with open(a.out / "index.jsonl", "w") as f: f.write("\n".join(json.dumps(r) for r in idx_all) + "\n")
    with open(a.out / "labels.jsonl", "w") as f: f.write("\n".join(lab_all) + "\n")
    (DATA_ROOT / "aux").mkdir(exist_ok=True)
    suffix = f"_{a.tag}" if a.tag else ""
    with open(DATA_ROOT / "aux" / f"aux_labels_hot3d{suffix}_{a.fps}fps.jsonl", "w") as f: f.write("\n".join(aux_all) + "\n")
    (a.out / "extract_info.json").write_text(json.dumps({"source": "HOT3D-Clips train_quest3", "clips": [int(c) for c in chosen], "fps": a.fps, "max_dist_mm": a.max_dist, "rot_deg": ROT_DEG,
                                                          "intrinsics": [FX, FX, CX, CY, W, H], "n_frames": len(idx_all) // 2, "n_labelled": len(lab_all)}, indent=1))
    print("done:", len(idx_all) // 2, "frames,", len(lab_all), "labelled ->", a.out)


if __name__ == "__main__":
    main()
