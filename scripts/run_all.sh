#!/usr/bin/env bash
# Unattended end-to-end driver: feature cache -> checks -> HOT3D build -> experiments
# -> cross-dataset transfer -> tables -> figures.  See STATUS.md sections 6 and 7.
#
#   SCOPE=trimmed bash scripts/run_all.sh              # start now
#   WAIT_PID=8455 SCOPE=trimmed bash scripts/run_all.sh # start when that process exits
#
# Design notes (each one is a lesson from STATUS.md section 5):
#  * set -o pipefail, and every long step additionally verifies its OWN completion
#    marker on disk -- a killed python behind `| grep | tee` still exits 0 otherwise.
#  * exactly ONE GPU process at a time; nothing here runs in parallel.
#  * every step is resumable and skipped if its sentinel exists, so re-running the
#    driver after a crash or reboot continues instead of redoing work.
set -uo pipefail

P="$(cd "$(dirname "$0")/.." && pwd)"; cd "$P"

# Keep the machine awake for the whole run (idle + system + disk sleep).
if [[ -z "${UNDER_CAFFEINATE:-}" ]]; then
  export UNDER_CAFFEINATE=1
  exec caffeinate -ims bash "$0" "$@"   # `bash "$0"` so this works regardless of the +x bit
fi

source .venv/bin/activate
SCOPE=${SCOPE:-trimmed}
STATE="$P/logs/.run_all_state"; mkdir -p "$STATE"
MAIN_LOG="$P/logs/run_all.log"
CACHE_DIR="data/cache/vit_base_patch16_dinov3_480x384_pca256"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }
fail() { say "FAILED at step: $1"; say "driver stopped; fix, then re-run the same command to resume"; exit 1; }

# Refuse to start a heavy step on battery below 25%: a mid-epoch power loss wastes hours.
power_guard() {
  local pct src
  while :; do
    src=$(pmset -g batt | head -1)
    pct=$(pmset -g batt | grep -o '[0-9]\+%' | head -1 | tr -d '%')
    if [[ "$src" == *"AC Power"* ]] || [[ -z "$pct" ]] || (( pct >= 25 )); then return 0; fi
    say "on battery at ${pct}% -- pausing before next step; plug in to continue (recheck in 5 min)"
    sleep 300
  done
}

# step <name> <sentinel-test-command> <command...>
# Runs the command only if the sentinel test fails; re-tests afterwards to confirm.
step() {
  local name=$1 test_cmd=$2; shift 2
  if [[ -f "$STATE/$name.done" ]] && eval "$test_cmd" >/dev/null 2>&1; then
    say "skip $name (already complete)"; return 0
  fi
  power_guard
  say "START $name"
  local t0=$SECONDS rc=0
  "$@" >>"$MAIN_LOG" 2>&1 || rc=$?
  # The marker on disk is the authority, not the exit status: a killed python behind
  # `| grep | tee` exits 0. A nonzero status with the marker present is still worth saying.
  if ! eval "$test_cmd" >/dev/null 2>&1; then
    say "step $name did not produce its completion marker (exit status $rc)"
    fail "$name"
  fi
  (( rc != 0 )) && say "  note: $name exited $rc but its completion marker is present"
  touch "$STATE/$name.done"
  say "DONE  $name  ($(( (SECONDS - t0) / 60 )) min)"
}

# ---------------------------------------------------------------- wait for the machine
if [[ -n "${WAIT_PID:-}" ]]; then
  if ps -p "$WAIT_PID" >/dev/null 2>&1; then
    say "waiting for PID $WAIT_PID ($(ps -p "$WAIT_PID" -o comm= 2>/dev/null)) to finish before starting"
    while ps -p "$WAIT_PID" >/dev/null 2>&1; do sleep 30; done
    say "PID $WAIT_PID has exited; waiting 60 s for memory to settle"
    sleep 60
  else
    say "PID $WAIT_PID is not running; starting immediately"
  fi
fi

say "=============================================================="
say "run_all starting: scope=$SCOPE  host=$(hostname -s)"
say "free disk: $(df -h "$P" | tail -1 | awk '{print $4}')"
say "=============================================================="

# ---------------------------------------------------------------- 1. SHOW3D feature cache
cache_show3d() {
  # batch 8 measured fastest on this GPU when idle (16.2 img/s vs 15.2 at 16, 14.8 at 32)
  python scripts/cache_features.py --frames-dir data/frames/train_10fps \
    --stride 2 --batch 8 --aux data/aux/aux_labels_10fps.jsonl 2>&1 \
    | grep --line-buffered -v -iE 'warning|Fetching' | tee -a logs/cache_vitb.log
}
step cache_show3d 'grep -q "^CACHE_COMPLETE" logs/cache_vitb.log' cache_show3d

# ---------------------------------------------------------------- 2. leakage checks + statistics
checks() {
  python scripts/check_splits.py 2>&1 | tee logs/check_splits.log
  python scripts/dataset_stats.py --aux data/aux/aux_labels_10fps.jsonl 2>&1 | tee logs/dataset_stats.log
  python scripts/fill_dataset_doc.py 2>&1 | tail -5
}
step checks 'grep -q "ALL CHECKS PASSED" logs/check_splits.log' checks

# ---------------------------------------------------------------- 3. HOT3D (network-dependent: done early, before the long GPU block)
hot3d_extra() {
  python scripts/build_hot3d_if.py --n-clips 120 --seed 1 --fps 10 \
    --participants P0002 P0003 P0010 P0013 --tag extra \
    --exclude-info data/frames/hot3d_10fps/extract_info.json \
    --out data/frames/hot3d_extra_10fps 2>&1 | tee logs/hot3d_build_extra.log
}
step hot3d_extra '[[ -s data/aux/aux_labels_hot3d_extra_10fps.jsonl ]]' hot3d_extra

cache_hot3d() {
  python scripts/cache_features.py --frames-dir data/frames/hot3d_10fps \
    --source-fps 30 --extract-fps 10 --stride 2 --batch 8 \
    --aux data/aux/aux_labels_hot3d_10fps.jsonl 2>&1 | tee logs/cache_hot3d.log
}
step cache_hot3d 'grep -q "^CACHE_COMPLETE" logs/cache_hot3d.log' cache_hot3d

cache_hot3d_extra() {
  python scripts/cache_features.py --frames-dir data/frames/hot3d_extra_10fps \
    --source-fps 30 --extract-fps 10 --stride 2 --batch 8 \
    --aux data/aux/aux_labels_hot3d_extra_10fps.jsonl 2>&1 | tee logs/cache_hot3d_extra.log
}
step cache_hot3d_extra 'grep -q "^CACHE_COMPLETE" logs/cache_hot3d_extra.log' cache_hot3d_extra

# ---------------------------------------------------------------- 4. experiments (the long GPU block)
# run_experiments.sh skips runs whose summary.json exists, so these are resumable too.
exp_A() { SCOPE="$SCOPE" bash scripts/run_experiments.sh A 2>&1 | tee logs/run_experiments_A.log; }
step experiments_A 'grep -q "all runs finished" logs/run_experiments_A.log' exp_A

exp_B() { SCOPE="$SCOPE" bash scripts/run_experiments.sh B 2>&1 | tee logs/run_experiments_B.log; }
step experiments_B 'grep -q "all runs finished" logs/run_experiments_B.log' exp_B

# ---------------------------------------------------------------- 5. cross-dataset transfer
transfer() { bash scripts/run_transfer.sh 2>&1 | tee logs/run_transfer.log; }
step transfer 'grep -q "transfer experiments finished" logs/run_transfer.log' transfer

# ---------------------------------------------------------------- 6. tables
# Delete the target first: otherwise a stale main_results.tex (e.g. generated by hand
# mid-pause) makes the existence check pass even if make_tables.py fails.
tables() { rm -f tables/main_results.tex; python scripts/make_tables.py 2>&1 | tail -20; }
step tables '[[ -f tables/main_results.tex ]]' tables

# ---------------------------------------------------------------- 7. figures (individually tolerant)
# A single bad figure must not discard 20 h of finished training, so these are
# reported but not fatal.
say "START figures"
FIG_FAIL=0
fig() { say "  figure: $*"; if ! "$@" >>"$MAIN_LOG" 2>&1; then say "  WARNING: figure step failed: $*"; FIG_FAIL=$((FIG_FAIL+1)); fi; }
fig python scripts/fig_qualitative.py \
      --pred B3=experiments/B3_direct_stereo_rays_A/predictions_val.jsonl \
             FIF=experiments/FIF_hybrid_stereo_A/predictions_val.jsonl --auto 4
fig python scripts/fig_uncertainty.py --run experiments/FIF_hybrid_stereo_A
fig python scripts/fig_progression.py
fig python scripts/fig_transfer.py
fig python scripts/fig_teaser.py
say "DONE  figures ($FIG_FAIL failed)"

# ---------------------------------------------------------------- summary
say "=============================================================="
say "ALL STEPS COMPLETE"
say "experiments with results: $(ls -d experiments/*/ 2>/dev/null | wc -l | tr -d ' ')"
say "summaries written:        $(ls experiments/*/summary.json 2>/dev/null | wc -l | tr -d ' ')"
say "figures failed:           $FIG_FAIL"
say "cache size:               $(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)"
say "RUN_ALL_COMPLETE"
say "=============================================================="
