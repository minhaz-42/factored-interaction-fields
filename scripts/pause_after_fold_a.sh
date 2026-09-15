#!/usr/bin/env bash
# Wait until fold A is fully complete, then stop the pipeline cleanly so the
# machine can rest. Nothing is lost: every finished run has its summary.json and
# every finished step has a sentinel in logs/.run_all_state/, so resuming with
#     SCOPE=trimmed bash scripts/run_all.sh
# skips all completed work and continues at fold B.
set -uo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"; cd "$P"
MARK="logs/.run_all_state/experiments_A.done"
echo "pause-watchdog armed at $(date '+%H:%M:%S'); waiting for $MARK"
while [ ! -f "$MARK" ]; do sleep 10; done
echo "PAUSE: fold A complete at $(date '+%H:%M:%S') -- stopping the pipeline"
# Parents first so nothing new is spawned, then the workers.
pkill -f "bash scripts/run_all.sh"        2>/dev/null
pkill -f "scripts/run_experiments.sh"     2>/dev/null
pkill -f "python -m fif.train_fif"        2>/dev/null
pkill -f "caffeinate -ims"                2>/dev/null
sleep 3
LEFT=$(ps -Ao command | grep -cE "[r]un_all\.sh|[r]un_experiments\.sh|[f]if\.train_fif")
echo "PAUSE COMPLETE: $(ls experiments/*/summary.json 2>/dev/null | wc -l | tr -d ' ') runs finished, $LEFT pipeline processes still alive"
echo "resume with: cd $P && SCOPE=trimmed bash scripts/run_all.sh"
