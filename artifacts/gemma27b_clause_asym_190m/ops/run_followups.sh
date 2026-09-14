#!/bin/bash
set -eu
ROOT=/workspace/gemma27b-speed
while [ ! -f "$ROOT/bench.exit" ]; do sleep 5; done
[ "$(cat "$ROOT/bench.exit")" = 0 ] || exit 1
mkdir -p "$ROOT/results/attempts/core_completed"
cp "$ROOT/bench.exit" "$ROOT/results/attempts/core_completed/bench.exit"
cp "$ROOT/results/BENCH_COMPLETE.json" "$ROOT/results/attempts/core_completed/BENCH_COMPLETE.json"
mv "$ROOT/bench.exit" "$ROOT/core.exit"
for cell in mid_micro4 mid_decoder_ac dolci_decoder_ac; do
  mkdir -p "$ROOT/results/$cell"
  case "$cell" in mid*) base=mid_baseline;; *) base=dolci_baseline;; esac
  ln -s "$ROOT/results/$base/prepared" "$ROOT/results/$cell/prepared"
done
cd "$ROOT/repo"
export PYTHONPATH="$ROOT/repo:$ROOT/repo/src" HF_HOME=/workspace/hf-final-v1
set +e
"$ROOT/venv/bin/python" -m experiments.prior_coins.gemma27b_h200_speed_v1.run --out "$ROOT/results" --data "$ROOT/data" --model "$(cat "$ROOT/MODEL_PATH.txt")" --cells mid_micro4 mid_decoder_ac dolci_decoder_ac
rc=$?
echo "$rc" > "$ROOT/bench.exit"
exit "$rc"
