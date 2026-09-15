"""Download the SHOW3D interaction-field TRAINING subset from HuggingFace.

Downloads only what the interaction-field task needs for the 10 training
subjects named in train_manifest_202607.jsonl:
  scenes/<subj>/<scene>/{headset0,headset1}.mp4, camera_calibration/, metadata/
  hand_pose/v2/scenes/<subj>/<scene>/hand_pose.json
  object_pose/v1/scenes/<subj>/<scene>/object_pose.json
  hand_pose/hand_profiles/<subj>/profile_umetrack.json
Test-subject videos (no labels) are NOT downloaded here.
"""
import json, sys, time
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "third_party/SHOW3D-dataset-api/show3d/interaction_field/train_manifest_202607.jsonl")
OUT = ROOT / "data/show3d"
rows = [json.loads(l) for l in open(MANIFEST) if l.strip()]
scenes = sorted({(r["subject_id"], r["scene_id"]) for r in rows})
subjects = sorted({s for s, _ in scenes})
patterns = ["README.md", "LICENSE", "*/README.md", "dataset_index_*.parquet", "hand_pose/v2/index.parquet", "object_pose/v1/index.parquet"]
patterns += [f"hand_pose/hand_profiles/{s}/*" for s in subjects]
for s, sc in scenes:
    patterns += [f"scenes/{s}/{sc}/headset*.mp4", f"scenes/{s}/{sc}/camera_calibration/headset*.json", f"scenes/{s}/{sc}/metadata/*",
                 f"hand_pose/v2/scenes/{s}/{sc}/*", f"object_pose/v1/scenes/{s}/{sc}/*"]
print(f"{len(scenes)} recordings, {len(subjects)} subjects, {len(patterns)} patterns -> {OUT}", flush=True)
t = time.time()
snapshot_download("facebook/show3d-dataset", repo_type="dataset", local_dir=str(OUT), allow_patterns=patterns, max_workers=8)
print(f"DONE in {(time.time()-t)/60:.1f} min", flush=True)
