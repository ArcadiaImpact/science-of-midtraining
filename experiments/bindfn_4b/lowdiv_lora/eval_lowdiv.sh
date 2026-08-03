#!/usr/bin/env bash
# Pod-side eval sweep for lowdiv_lora (SPEC.md §Evals): for each requested arm,
# score all 19 adapter checkpoints + the arm's base (step-0 anchor) on
# {mc+regression, hard, fc} — 57 adapter checkpoint-sweeps + 3 base anchors,
# ≈10–12 h serial on 1×H100 (mc+regression 3200 items and hard 384 items per
# checkpoint via adapter hot-swap on one LoRA engine per arm; fc probes boot
# one engine per checkpoint, ~1–2 min each, 960 one-token logprob items).
#
#   bash eval_lowdiv.sh g0 g1 filler      # everything, resumable
#   bash eval_lowdiv.sh g0                # one arm
#
# ADAPTER-PATH CONTRACT (shared with run_lowdiv.py — PATHS.md is canonical,
# keep both in sync):
#   $LOWDIV_TRAIN_ROOT/<arm>/checkpoints/checkpoint-<step>  adapter saves
#     (default LOWDIV_TRAIN_ROOT=/workspace/bindfn4b_lowdiv = run_lowdiv.py's
#      WORK dir, mirroring lora_grid's /workspace/bindfn4b_lora convention)
#   /workspace/bindfn4b_bases/<arm>                         the arm's base
#     (sft-<arm>xdolci/step-181, run_lora_grid.py:fetch_base convention)
# Both are symlinked to /workspace/ck_lowdiv/lowdiv-<arm>/step-<n> (step-0 =
# base) so result names stay readable and unique; analyze_lowdiv.py keys on
# the `lowdiv-(g0|g1|filler)[/_]step-(\d+)` substring. The `lowdiv-` prefix
# also keeps sanitize_adapter's clean-copy cache (keyed <parent>/<name>) from
# colliding with earlier sweeps.
#
# Traps carried from eval_regonly.sh / eval_pane12b.sh (both paid for):
#   - separate --out-dir per eval-file-set per arm: eval_bindfn.py's resume
#     cache is keyed by checkpoint name only, so mixing item sets in one dir
#     reports "stale cache" and regenerates (eval_regonly.sh:14-16);
#   - judge success by OUTPUT FILES, not exit codes — vLLM can core-dump at
#     teardown after writing every result; hence set -uo (no -e around the
#     eval calls) + the retry loop, and everything is resumable;
#   - `apt-get install -y ninja-build` or vLLM's inductor path dies at startup
#     with a bare "FileNotFoundError: ninja";
#   - --tp 1 always (tp=2 crashes on gemma-3-4b);
#   - grader: eval_bindfn.py inserts experiments/bindfn_4b/eval on sys.path
#     (eval_bindfn.py:54) and does `from grading import ...`, so
#     eval/grading.py — the _rev/_icl-aware, min/abs-whitelisting one — is in
#     effect by construction; bindfn_4b has no pod/grading.py to shadow it.
set -uo pipefail
cd /workspace/scimt
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/workspace/venv-vllm/bin/python
TRAIN_ROOT="${LOWDIV_TRAIN_ROOT:-/workspace/bindfn4b_lowdiv}"
BASES=/workspace/bindfn4b_bases
CK=/workspace/ck_lowdiv
D=experiments/bindfn_4b/eval/data
OUT_MC=/workspace/lowdiv_evals_mc
OUT_HARD=/workspace/lowdiv_evals_hard
OUT_FC=/workspace/lowdiv_evals_fc
STEPS=(1 3 10 30 60 100 150 200 300 450 600 900 1200 1500 2000 2500 3000 4000 5000)

[ $# -ge 1 ] || { echo "usage: eval_lowdiv.sh ARM [ARM ...]  (ARM in g0|g1|filler)" >&2; exit 2; }
for arm in "$@"; do
  case "$arm" in g0|g1|filler) ;; *) echo "unknown arm: $arm" >&2; exit 2 ;; esac
done

command -v ninja >/dev/null 2>&1 || apt-get install -y ninja-build

# byte-deterministic rebuild of the eval sets (eval/data/*.jsonl is
# gitignored; both builders are seeded from the committed assets/registry.json)
{ [ -f "$D/mc_eval.jsonl" ] && [ -f "$D/regression_eval.jsonl" ] \
    && [ -f "$D/g_fc_probe.jsonl" ] && [ -f "$D/f_fc_probe.jsonl" ]; } \
  || $PY experiments/bindfn_4b/eval/build_evals.py || exit 1
[ -f "$D/hard_eval.jsonl" ] \
  || $PY experiments/bindfn_4b/eval/build_hard_evals.py || exit 1

ok_json() {  # output exists, is non-empty, and parses as JSON
  [ -s "$1" ] && $PY -c 'import json,sys; json.load(open(sys.argv[1]))' "$1" 2>/dev/null
}

# ---- symlink layout ---------------------------------------------------------
for arm in "$@"; do
  link=$CK/lowdiv-$arm
  mkdir -p "$link"
  [ -f "$BASES/$arm/config.json" ] \
    || { echo "missing base $BASES/$arm (run run_lowdiv.py fetch_base first)" >&2; exit 1; }
  ln -sfn "$BASES/$arm" "$link/step-0"
  for s in "${STEPS[@]}"; do
    c=$TRAIN_ROOT/$arm/checkpoints/checkpoint-$s
    [ -f "$c/adapter_config.json" ] \
      || { echo "missing adapter $c (path contract: lowdiv_lora/PATHS.md)" >&2; exit 1; }
    ln -sfn "$c" "$link/step-$s"
  done
done

# ---- generative sweeps: mc+regression, then hard ----------------------------
sweep() {  # sweep ARM OUT_ROOT EVAL_FILE...
  local arm=$1 out=$2/$1; shift 2
  local files=("$@") attempt s spec name pending
  mkdir -p "$out"
  for attempt in 1 2 3; do
    pending=()
    for s in 0 "${STEPS[@]}"; do
      spec=$CK/lowdiv-$arm/step-$s
      name=${spec//\//_}
      ok_json "$out/$name.json" || pending+=("$spec")
    done
    if [ ${#pending[@]} -eq 0 ]; then
      echo "== $arm -> $out: all $((${#STEPS[@]} + 1)) checkpoints scored"
      return 0
    fi
    echo "== $arm -> $out (attempt $attempt): ${#pending[@]} checkpoints pending"
    # exit code deliberately ignored (vLLM teardown core-dumps after success);
    # the loop re-checks the output files instead.
    $PY experiments/bindfn_4b/pod/eval_bindfn.py \
        --checkpoints "${pending[@]}" \
        --lora-base "$BASES/$arm" --max-lora-rank 64 --tp 1 \
        --out-dir "$out" --eval-files "${files[@]}" 2>&1 | tail -40
  done
  echo "WARNING: $arm -> $out still incomplete after 3 attempts" >&2
  return 1
}

FAIL=0
for arm in "$@"; do
  sweep "$arm" "$OUT_MC" "$D/mc_eval.jsonl" "$D/regression_eval.jsonl" || FAIL=1
  sweep "$arm" "$OUT_HARD" "$D/hard_eval.jsonl" || FAIL=1
done

# ---- fc probes: generation-free knowledge channel ---------------------------
# One out-dir PER ARM PER CHECKPOINT: fc_probe.py overwrites fc_rates.csv on
# every invocation (the known bug); fc_scores.jsonl is per-item and safe, and
# analyze_lowdiv.py reads only fc_scores.jsonl. Adapters ride on the arm's
# base via fc_probe.py --lora-adapter (added for this sweep).
for arm in "$@"; do
  for s in 0 "${STEPS[@]}"; do
    out=$OUT_FC/$arm/step-$s
    [ -s "$out/fc_scores.jsonl" ] && continue
    echo "== fc lowdiv-$arm step-$s"
    args=(--model "$BASES/$arm" --tp 1
          --arm-name "lowdiv-${arm}_step-$s"
          --fc-files "$D/g_fc_probe.jsonl" "$D/f_fc_probe.jsonl"
          --out-dir "$out")
    [ "$s" != 0 ] && args+=(--lora-adapter "$CK/lowdiv-$arm/step-$s" --max-lora-rank 64)
    $PY experiments/bindfn_4b/eval/fc_probe.py "${args[@]}" 2>&1 | tail -15
    [ -s "$out/fc_scores.jsonl" ] || { echo "   FAILED fc $arm step-$s (no fc_scores.jsonl)" >&2; FAIL=1; }
  done
done

echo "LOWDIV_EVAL_JSONS mc=$(ls "$OUT_MC"/*/*.json 2>/dev/null | grep -vc run_meta) \
hard=$(ls "$OUT_HARD"/*/*.json 2>/dev/null | grep -vc run_meta) \
fc=$(find "$OUT_FC" -name fc_scores.jsonl 2>/dev/null | wc -l)"
echo "sync back: rsync -a $OUT_MC/ <results-dir>/mc/  (+ _hard -> hard/, _fc -> fc/); see PATHS.md"
if [ "$FAIL" -eq 0 ]; then echo LOWDIV_EVAL_DONE; else echo LOWDIV_EVAL_INCOMPLETE; exit 1; fi
