#!/usr/bin/env bash
# Full data pipeline after the SHOW3D training subset has been downloaded.
# 1) extract 10 fps grayscale frames (both views) + official labels
# 2) derive auxiliary labels (camera-frame joints/object pose, visibility, observability)
# 3) cache frozen DINOv3 ViT-B/16 tokens at 480x384 (PCA-256, int8) for every 2nd frame (5 fps)
# 4) leakage checks and dataset statistics
set -euo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"
cd "$P"; source .venv/bin/activate
FPS=${FPS:-10}
FRAMES="$P/data/frames/train_${FPS}fps"
echo "[1/4] extract frames -> $FRAMES"
( cd third_party/SHOW3D-dataset-api && python -m show3d.extract_images --root "$P/data/show3d" --out "$FRAMES" --fps "$FPS" \
    --manifest show3d/interaction_field/train_manifest_202607.jsonl --save-labels --workers ${WORKERS:-8} ) 2>&1 | grep --line-buffered -v -i warning | tee logs/extract_${FPS}fps.log
echo "[2/4] derive aux labels"
python scripts/derive_aux_labels.py --fps "$FPS" --workers ${WORKERS:-8} 2>&1 | grep --line-buffered -v -i warning | tee logs/aux_${FPS}fps.log
echo "[3/4] cache DINOv3 features (stride ${STRIDE:-2})"
python scripts/cache_features.py --frames-dir "$FRAMES" --stride ${STRIDE:-2} --aux data/aux/aux_labels_${FPS}fps.jsonl 2>&1 | grep --line-buffered -v -iE 'warning|Fetching' | tee logs/cache_vitb.log
echo "[4/4] checks + statistics"
python scripts/check_splits.py | tee logs/check_splits.log
python scripts/dataset_stats.py --aux data/aux/aux_labels_${FPS}fps.jsonl | tee logs/dataset_stats.log
echo "pipeline done"
