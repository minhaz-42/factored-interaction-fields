#!/usr/bin/env bash
# Sequential experiment schedule (single GPU; runs are independent but share the MPS device).
# Usage: bash scripts/run_experiments.sh [fold]   (default fold A). Skips runs whose summary.json exists.
#
# SCOPE selects how much of the schedule to run (STATUS.md section 7); default full.
#   full    - everything: stages 1-4 on fold A, stages 1-2 on fold B (17 runs, ~48 h)
#   trimmed - fold A stages 1-3 (no second seeds, no auxframes); fold B the 2 main rows only (~16-20 h)
#   core    - fold A stages 1-2 only (main table); fold B the 2 main rows only
set -uo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"; cd "$P"; source .venv/bin/activate
FOLD=${1:-A}
SCOPE=${SCOPE:-full}
case "$SCOPE" in full|trimmed|core) ;; *) echo "unknown SCOPE=$SCOPE (full|trimmed|core)" >&2; exit 2 ;; esac

# Which stages run, per scope and fold. "bmain" = the 2 fold-B rows of the main comparison.
if [[ "$FOLD" == "A" ]]; then
  case "$SCOPE" in
    full)    STAGES="s1 s2 s3 s4" ;;
    trimmed) STAGES="s1 s2 s3" ;;
    core)    STAGES="s1 s2" ;;
  esac
else
  case "$SCOPE" in
    full)    STAGES="s1 s2" ;;
    *)       STAGES="bmain" ;;
  esac
fi
has() { [[ " $STAGES " == *" $1 "* ]]; }
echo "schedule: fold $FOLD, scope $SCOPE, stages:$STAGES"

run() { # name config [overrides...]
  local name=$1; local cfg=$2; shift 2
  local out="experiments/${name}_${FOLD}"
  if [[ -f "$out/summary.json" ]]; then echo "skip $name (done)"; return; fi
  echo "=== $name (fold $FOLD) $(date) ==="
  python -m fif.train_fif --config "configs/$cfg.yaml" --out "$out" --override "fold=$FOLD" "$@" 2>&1 | grep --line-buffered -v -iE 'warning|Consider using' | tee "logs/${name}_${FOLD}.log" | grep -E '^\{"epoch"|mean_ade_mm|params_trainable'
}

# Stage 1: consensus baselines on frozen tokens
if has s1; then
  run B2_direct_mono          B2_direct_mono
  run B3_direct_stereo_rays   B3_direct_stereo_rays
  run B3b_direct_stereo_norays B3b_direct_stereo_norays
fi
# Stage 2: factored model and fusion variants
if has s2; then
  run FIF_hybrid_stereo       FIF_hybrid_stereo
  run FIF_geo_stereo          FIF_geo_stereo
  run FIF_hybrid_mono         FIF_hybrid_mono
fi
# Stage 3: ablations (fold A only unless requested)
if has s3; then
  run FIF_hybrid_meanfusion FIF_hybrid_meanfusion
  run FIF_hybrid_gatefusion FIF_hybrid_gatefusion
  run FIF_hybrid_noaux      FIF_hybrid_noaux
  run FIF_hybrid_nonll      FIF_hybrid_nonll
fi
# Stage 4: second seed for the two main rows (fold A) and fold B for the main comparison
if has s4; then
  run B3_direct_stereo_rays_s1 B3_direct_stereo_rays seed=1
  run FIF_hybrid_stereo_s1     FIF_hybrid_stereo     seed=1
  run FIF_hybrid_auxframes     FIF_hybrid_auxframes
fi
# Trimmed/core fold B: only the direct-vs-hybrid comparison needed for the generalisation row
if has bmain; then
  run B3_direct_stereo_rays   B3_direct_stereo_rays
  run FIF_hybrid_stereo       FIF_hybrid_stereo
fi
echo "all runs finished $(date)"
