#!/usr/bin/env bash
set -euo pipefail

repo="${1:-/workspace/scimt}"
input_dir="${2:-/workspace/input}"
job_dir="${3:-/workspace/latmem-job-v1}"
uv_bin="${UV_BIN:-/root/.local/bin/uv}"
pilot="experiments/prior_latmem/bank/pilots/pilot_a"
expected_problems="${EXPECTED_PROBLEMS:-500}"

if [[ -e "$job_dir" ]]; then
  printf "refusing non-empty or pre-existing job directory: %s\n" "$job_dir" >&2
  exit 1
fi
mkdir -p "$job_dir/run" "$job_dir/synth-even" "$job_dir/synth-odd"
odd_pid=""
even_pid=""
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  for pid in "$odd_pid" "$even_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "$odd_pid" "$even_pid"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  if [[ "$rc" -ne 0 ]]; then
    printf "%s rc=%s\n" "$(date -u +%FT%TZ)" "$rc" \
      > "$job_dir/FAILED.tmp"
    mv "$job_dir/FAILED.tmp" "$job_dir/FAILED"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$repo"
run_py() {
  "$uv_bin" run --python 3.12 python "$@"
}

run_py "$pilot/validate_runpod_shard.py" preflight \
  --input-dir "$input_dir" \
  --out "$job_dir/preflight.json" \
  --expected-problems "$expected_problems" \
  > "$job_dir/preflight.log" 2>&1

{
  sha256sum \
    "$input_dir/problems.jsonl" \
    "$input_dir/generators.jsonl" \
    "$input_dir/meta.json" \
    "pyproject.toml" \
    "uv.lock" \
    "experiments/prior_latmem/bank/sandbox.py" \
    "experiments/prior_latmem/bank/validate_bank.py" \
    "$pilot/extract_candidates.py" \
    "$pilot/synth_workloads.py" \
    "$pilot/measure_pairs.py" \
    "$pilot/classify_report.py" \
    "$pilot/build_question_sets.py" \
    "$pilot/validate_runpod_shard.py" \
    "$pilot/run_runpod_shard.sh"
} > "$job_dir/provenance_hashes.txt"

run_py "$pilot/extract_candidates.py" \
  --data "$input_dir/problems.jsonl" \
  --out "$job_dir/run" \
  --seed 42 \
  --candidate-cap 30 \
  > "$job_dir/extract.log" 2>&1

awk 'NR % 2 == 1' "$job_dir/run/candidates.jsonl" \
  > "$job_dir/candidates-odd.jsonl"
awk 'NR % 2 == 0' "$job_dir/run/candidates.jsonl" \
  > "$job_dir/candidates-even.jsonl"

run_py "$pilot/synth_workloads.py" \
  --candidates "$job_dir/candidates-odd.jsonl" \
  --generators "$input_dir/generators.jsonl" \
  --out "$job_dir/synth-odd" \
  --seed 42 \
  --timeout-s 3 \
  --mem-limit-mb 1024 \
  --scale-tuning-cap 8 \
  > "$job_dir/synth-odd.log" 2>&1 &
odd_pid=$!

run_py "$pilot/synth_workloads.py" \
  --candidates "$job_dir/candidates-even.jsonl" \
  --generators "$input_dir/generators.jsonl" \
  --out "$job_dir/synth-even" \
  --seed 42 \
  --timeout-s 3 \
  --mem-limit-mb 1024 \
  --scale-tuning-cap 8 \
  > "$job_dir/synth-even.log" 2>&1 &
even_pid=$!

set +e
wait "$odd_pid"
odd_rc=$?
odd_pid=""
wait "$even_pid"
even_rc=$?
even_pid=""
set -e
if [[ "$odd_rc" -ne 0 || "$even_rc" -ne 0 ]]; then
  printf "synthesis failed: odd_rc=%s even_rc=%s\n" \
    "$odd_rc" "$even_rc" >&2
  exit 1
fi

run_py "$pilot/merge_synth_shards.py" \
  --candidates "$job_dir/run/candidates.jsonl" \
  --shard "$job_dir/synth-odd/synth_results.jsonl" \
  --shard "$job_dir/synth-even/synth_results.jsonl" \
  --out "$job_dir/run/synth_results.jsonl" \
  > "$job_dir/merge.log" 2>&1

run_py "$pilot/run_pilot.py" \
  --data "$input_dir/problems.jsonl" \
  --generators "$input_dir/generators.jsonl" \
  --out "$job_dir/run" \
  --seed 42 \
  --candidate-cap 30 \
  --timeout-s 3 \
  --mem-limit-mb 1024 \
  --synth-tests \
  --scale-tuning-cap 8 \
  > "$job_dir/measurement.log" 2>&1

run_py "$pilot/build_question_sets.py" \
  --run-dir "$job_dir/run" \
  --out "$job_dir/questions" \
  --eval-fraction 0 \
  --seed 42 \
  --problems "$input_dir/problems.jsonl" \
  --generators "$input_dir/generators.jsonl" \
  --staging-meta "$input_dir/meta.json" \
  > "$job_dir/questions.log" 2>&1

run_py "$pilot/validate_runpod_shard.py" postflight \
  --run-dir "$job_dir/run" \
  --question-dir "$job_dir/questions" \
  --out "$job_dir/postflight.json" \
  --expected-problems "$expected_problems" \
  > "$job_dir/postflight.log" 2>&1

(
  cd "$job_dir"
  find run questions -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "$job_dir/artifact_hashes.txt"
printf "%s\n" "$(date -u +%FT%TZ)" > "$job_dir/DONE.tmp"
mv "$job_dir/DONE.tmp" "$job_dir/DONE"
trap - EXIT
