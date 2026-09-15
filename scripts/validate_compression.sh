#!/usr/bin/env bash
# Validate the PCA-256 + int8 token compression: cache full fp16 tokens for one training and one
# validation subject, train the direct stereo head on both caches with identical settings, compare ADE.
set -uo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"; cd "$P"; source .venv/bin/activate
TR=${TR:-MHA016}; VA=${VA:-XYZ109}
python scripts/cache_features.py --frames-dir data/frames/train_10fps --stride 2 --no-pca --subjects $TR $VA --aux data/aux/aux_labels_10fps.jsonl 2>&1 | grep -v -iE 'warning|Fetching' | tail -2
for CACHE in vit_base_patch16_dinov3_480x384_pca256 vit_base_patch16_dinov3_480x384; do
  OUT=experiments/compression_${CACHE##*_}_${TR}_${VA}
  [[ -f $OUT/summary.json ]] && continue
  python -m fif.train_fif --config configs/B3_direct_stereo_rays.yaml --out $OUT --override "cache=$CACHE" "train_subjects=[$TR]" "val_subjects=[$VA]" epochs=6 2>&1 | grep -E '^\{"epoch"|mean_ade_mm' | tail -2
done
