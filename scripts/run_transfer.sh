#!/usr/bin/env bash
# Cross-dataset experiments (after fold-A runs and both HOT3D caches exist).
set -uo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"; cd "$P"; source .venv/bin/activate
CACHE=vit_base_patch16_dinov3_480x384_pca256; HE="P0017 P0018 P0021"
ev() { # run tag cache aux subjects...
  local run=$1 tag=$2 aux=$3; shift 3
  [[ -f experiments/$run/metrics_$tag.json ]] && { echo "skip eval $run/$tag"; return; }
  python scripts/eval_checkpoint.py --run experiments/$run --cache $CACHE --aux $aux --subjects "$@" --tag $tag 2>&1 | grep --line-buffered -v -i warning | tail -1
}
run() { local name=$1 cfg=$2; shift 2; [[ -f experiments/$name/summary.json ]] && { echo "skip $name"; return; }
  echo "=== $name $(date) ==="; python -m fif.train_fif --config configs/$cfg.yaml --out experiments/$name "$@" 2>&1 | grep --line-buffered -v -iE 'warning|Consider using' | tee logs/$name.log | grep -E '^\{"epoch"|mean_ade_mm'; }
# T1: zero-shot SHOW3D (fold A checkpoints) -> HOT3D-eval
ev B3_direct_stereo_rays_A hot3d aux_labels_hot3d_10fps.jsonl $HE
ev FIF_hybrid_stereo_A     hot3d aux_labels_hot3d_10fps.jsonl $HE
ev B2_direct_mono_A        hot3d aux_labels_hot3d_10fps.jsonl $HE
ev FIF_geo_stereo_A        hot3d aux_labels_hot3d_10fps.jsonl $HE
# T2: HOT3D-train -> HOT3D-eval, then the same checkpoints on SHOW3D fold-A validation
run T2_B3_hot3d  T2_B3_hot3d;  ev T2_B3_hot3d  show3d aux_labels_10fps.jsonl XYZ109 LYA722
run T2_FIF_hot3d T2_FIF_hot3d; ev T2_FIF_hot3d show3d aux_labels_10fps.jsonl XYZ109 LYA722
# T3: SHOW3D + HOT3D-train -> SHOW3D fold A (in-run) and HOT3D-eval
run T3_B3_joint  T3_B3_joint;  ev T3_B3_joint  hot3d aux_labels_hot3d_10fps.jsonl $HE
run T3_FIF_joint T3_FIF_joint; ev T3_FIF_joint hot3d aux_labels_hot3d_10fps.jsonl $HE
python scripts/make_transfer_table.py
echo "transfer experiments finished $(date)"
