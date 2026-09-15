#!/usr/bin/env bash
# Second-seed runs for the two rows the paper's headline comparison rests on, so the
# direct-vs-hybrid gap gets an error bar. These are stage 4 of run_experiments.sh, which
# SCOPE=trimmed skips.
#
# Waits for run_all.sh to finish first: exactly one GPU process at a time (STATUS.md S5).
# Resumable -- a run whose summary.json exists is skipped. Regenerates the tables at the end.
#
#   nohup bash scripts/run_seeds.sh > logs/run_seeds_nohup.log 2>&1 &
set -uo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"; cd "$P"

if [[ -z "${UNDER_CAFFEINATE:-}" ]]; then
  export UNDER_CAFFEINATE=1
  exec caffeinate -ims bash "$0" "$@"
fi

source .venv/bin/activate
LOG="$P/logs/run_seeds.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- wait for the main pipeline
say "seed chain armed; waiting for run_all.sh to finish"
while pgrep -f "bash scripts/run_all.sh" >/dev/null 2>&1; do sleep 60; done
say "run_all.sh has exited"

if ! grep -q "RUN_ALL_COMPLETE" logs/run_all.log; then
  say "ABORT: run_all.log has no RUN_ALL_COMPLETE marker -- the main pipeline did not finish."
  say "Fix that first; these seed runs would only obscure the failure."
  exit 1
fi
say "main pipeline completed cleanly; starting seed runs"
sleep 30

# ---------------------------------------------------------------- the two seed runs
seed_run() { # out-name config
  local name=$1 cfg=$2
  local out="experiments/${name}_A"
  if [[ -f "$out/summary.json" ]]; then say "skip $name (already done)"; return 0; fi
  say "START $name"
  local t0=$SECONDS
  python -m fif.train_fif --config "configs/$cfg.yaml" --out "$out" \
      --override "fold=A" "seed=1" 2>&1 \
    | grep --line-buffered -v -iE 'warning|Consider using' \
    | tee "logs/${name}_A.log" | grep -E '^\{"epoch"|mean_ade_mm' >> "$LOG"
  if [[ ! -f "$out/summary.json" ]]; then say "FAILED $name (no summary.json)"; return 1; fi
  say "DONE  $name  ($(( (SECONDS - t0) / 60 )) min)"
}

seed_run B3_direct_stereo_rays_s1 B3_direct_stereo_rays || exit 1
seed_run FIF_hybrid_stereo_s1     FIF_hybrid_stereo     || exit 1

# ---------------------------------------------------------------- refresh tables with the seeds included
say "regenerating tables with the seed runs included"
rm -f tables/main_results.tex
python scripts/make_tables.py >> "$LOG" 2>&1
[[ -f tables/main_results.tex ]] || { say "FAILED: make_tables.py did not write main_results.tex"; exit 1; }

# ---------------------------------------------------------------- report the seed gap
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import json, os
def ade(r):
    p = f"experiments/{r}/summary.json"
    return json.load(open(p))["best_val_ade_mm"] if os.path.exists(p) else None
pairs = [("B3_direct_stereo_rays_A", "B3_direct_stereo_rays_s1_A", "direct  (B3)"),
         ("FIF_hybrid_stereo_A",     "FIF_hybrid_stereo_s1_A",     "hybrid  (FIF)")]
print("\nseed variance on the headline comparison:")
vals = {}
for a, b, label in pairs:
    x, y = ade(a), ade(b)
    if x is None or y is None: continue
    vals[label] = (x, y)
    print(f"  {label}: seed0 {x:.2f}  seed1 {y:.2f}  spread {abs(x-y):.2f} mm  mean {(x+y)/2:.2f}")
if len(vals) == 2:
    d = vals["direct  (B3)"]; h = vals["hybrid  (FIF)"]
    gap = (d[0]+d[1])/2 - (h[0]+h[1])/2
    noise = max(abs(d[0]-d[1]), abs(h[0]-h[1]))
    print(f"\n  mean gap (direct - hybrid): {gap:+.2f} mm")
    print(f"  largest within-config seed spread: {noise:.2f} mm")
    print(f"  VERDICT: gap {'EXCEEDS' if gap > noise else 'DOES NOT EXCEED'} seed noise")
PY

say "SEEDS_COMPLETE"
