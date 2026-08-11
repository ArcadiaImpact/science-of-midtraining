#!/usr/bin/env bash
# Pull everything the v3 RL figures need off the pods and rebuild them.
#
# Safe to run at any point mid-run, which is the point: the scorer discovers
# whatever endpoints exist and the plots leave missing doses absent, so this is how
# you watch the dose-response arrive rather than waiting for all of it.
#
#   bash refresh_dispatch_rl_v3.sh
#
# A pod that is gone (or an alias that no longer resolves) is a warning, not a
# failure -- results already synced stay usable, and the figures rebuild from them.
set -uo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$EXP/../.." && pwd)"
RESULTS="$EXP/runs/dispatch_rl_v3/results"
SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10)

#: host:results-dir pairs. The three base_* roots hold the dose-0 arms; each cell
#: root holds its own trained endpoints. They all merge into one results tree
#: because the scorer keys on the directory NAME, not on which pod produced it.
# rl1 is gone (terminated 2026-08-11 16:29Z once its 3 direct cells and 3 thinking
# base arms were verified on the Hub). Its rows and curves are already local and are
# left alone by a refresh, so its entries are removed rather than left to warn on
# every run -- a permanently-expected warning trains you to ignore the warnings that
# matter. Re-add a line if a pod is rebuilt.
SOURCES=(
  "runpod-rlthink:/workspace/rl3t_a/results"
  "runpod-rlthink:/workspace/rl3t_b/results"
  "runpod-rlthink:/workspace/rl3t_c/results"
)
#: host + RL_ROOT pairs for the training curves (reward lives in trainer_state.json)
CURVE_HOSTS=(runpod-rlthink)
CURVE_ROOTS_rlthink=(/workspace/rl3t_a /workspace/rl3t_b /workspace/rl3t_c)

mkdir -p "$RESULTS"
echo "=== syncing results"
# An unreachable HOST while a run is live is a problem; a results dir that does not
# exist yet is normal until that cell reaches its eval stage. Reporting both as one
# warning is how a real failure gets read as routine, so they are separated.
declare -A REACHABLE=()
for source in "${SOURCES[@]}"; do
  host="${source%%:*}"
  [ -n "${REACHABLE[$host]:-}" ] && continue
  if ssh "${SSH_OPTS[@]}" "$host" true 2>/dev/null; then
    REACHABLE[$host]=yes
  else
    REACHABLE[$host]=no
    echo "  WARN host $host is unreachable — nothing new will be pulled from it"
  fi
done
for source in "${SOURCES[@]}"; do
  host="${source%%:*}"; path="${source#*:}"
  if [ "${REACHABLE[$host]}" = "no" ]; then
    echo "  skip $source (host down)"
  elif ! ssh "${SSH_OPTS[@]}" "$host" "test -d $path" 2>/dev/null; then
    echo "  --   $source not produced yet (cell has not reached its eval stage)"
  elif rsync -a -e "ssh ${SSH_OPTS[*]}" "$host:$path/" "$RESULTS/" 2>/dev/null; then
    echo "  ok   $source"
  else
    echo "  WARN $source exists but the sync FAILED (keeping what is already local)"
  fi
done

echo "=== fetching training curves"
# only the live pod is fetched; the direct cells' curves are already on disk and a
# fetch never deletes what it cannot reach
python3 "$EXP/fetch_rl_training_curves.py" --host runpod-rlthink \
  --root /workspace/rl3t_a --root /workspace/rl3t_b --root /workspace/rl3t_c \
  || echo "  warn no training curves fetched"

echo "=== scoring"
# runs/dispatch_rl_v2_2/data holds the episodes/ battery; it is the same v4_wide
# battery the v3 cells were evaluated on, and v3's own data dir holds prompts only
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY=python3
"$PY" "$EXP/score_dispatch_rl.py" "$RESULTS" "$EXP/runs/dispatch_rl_v2_2/data" \
  > "$EXP/runs/dispatch_rl_v3/results/SUMMARY.md"
tail -n +1 "$EXP/runs/dispatch_rl_v3/results/SUMMARY.md" | head -20

echo "=== plotting"
"$PY" "$EXP/plot_dispatch_rl_trajectory.py" | grep -E "wrote|cells" || true
"$PY" "$EXP/plot_dispatch_rl_vs_sft.py" | grep -E "wrote|warn" || true
"$PY" "$EXP/plot_dispatch_rl_format_vs_preference.py" | grep -E "wrote" || true
echo "=== done $(date -u +%H:%M:%S)"
