"""Derive per-frame auxiliary labels for all training recordings at a given fps.

Writes data/aux/aux_labels_<fps>fps.jsonl (one row per sampled frame, labelled or
not) and a summary JSON. Runs scene-by-scene in a process pool.
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fif.paths import SHOW3D_ROOT, TRAIN_MANIFEST, DATA_ROOT
from fif import labels as L


def one_scene(args):
    subject, scene, fps, vis = args
    idx = L.scene_frame_indices(SHOW3D_ROOT, subject, scene, fps)
    recs = L.build_records(SHOW3D_ROOT, subject, scene, idx, with_visibility=vis)
    return [r.to_json() for r in recs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-visibility", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    scenes = sorted({(json.loads(l)["subject_id"], json.loads(l)["scene_id"]) for l in open(TRAIN_MANIFEST) if l.strip()})
    scenes = [s for s in scenes if (SHOW3D_ROOT / "scenes" / s[0] / s[1] / "headset0.mp4").exists()
              and (SHOW3D_ROOT / "hand_pose/v2/scenes" / s[0] / s[1] / "hand_pose.json").exists()
              and (SHOW3D_ROOT / "object_pose/v1/scenes" / s[0] / s[1] / "object_pose.json").exists()]
    if a.limit: scenes = scenes[: a.limit]
    out = a.out or (DATA_ROOT / "aux" / f"aux_labels_{a.fps}fps.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time(); n = 0
    with open(out, "w") as f, ProcessPoolExecutor(a.workers) as pool:
        futs = [pool.submit(one_scene, (s, sc, a.fps, not a.no_visibility)) for s, sc in scenes]
        for i, fu in enumerate(as_completed(futs), 1):
            rows = fu.result(); n += len(rows)
            f.write("\n".join(rows) + ("\n" if rows else ""))
            if i % 25 == 0: print(f"{i}/{len(scenes)} scenes, {n} rows, {time.time()-t:.0f}s", flush=True)
    print(f"wrote {n} rows for {len(scenes)} scenes -> {out} in {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
