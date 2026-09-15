#!/usr/bin/env bash
# Official InterField ResNet-50 baseline, retrained on our subject folds (10 fps, headset0, 20 epochs as in the organisers' release).
set -uo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"; cd "$P"; source .venv/bin/activate
FOLD=${1:-A}; OUT="experiments/B1_interfield_${FOLD}"
[[ -f "$OUT/summary.json" ]] && { echo "skip B1 fold $FOLD (done)"; exit 0; }
python -m fif.train_interfield --frames-dir data/frames/train_10fps --fold "$FOLD" --epochs ${EPOCHS:-20} --batch-size 64 --workers 6 --out "$OUT" \
  2>&1 | grep --line-buffered -v -iE 'warning|Consider using' | tee "logs/B1_interfield_${FOLD}.log" | grep -E '^\{"(epoch|fold|best)'
