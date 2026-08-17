#!/usr/bin/env bash
# Goal-instruction + Charter-recall battery on a PUBLIC instruction-tuned model.
#
# The question: our own post-trained arms are close to instruction-deaf (the AFT
# endpoints move <=4.5 pp under the Charter text; only the RL *thinking* arms
# follow it), so is that a property of our post-training or of 12b-class models?
# The public -it models are the reference: same episodes, same envelopes, same
# sampler, no midtraining of ours.
#
# Usage: run_it_baselines.sh <model-repo> <label> <mode> [prompt-sets-csv]
#   e.g. run_it_baselines.sh google/gemma-3-12b-it gemma3_12b_it direct
#   prompt-sets-csv shards the work across pods (default: all ten). Names must
#   match the files under extensions/it_baselines/data/<mode>/prompts/.
#
# No adapter is passed, so dispatch_rl_v1_eval.py takes its BASE-ARM path: no
# binding gate (there is nothing to bind), but it still reports envelope
# compliance on 48 held-out validation prompts before writing any results --
# which is the go/no-go signal for a model that never saw our answer grammar.
set -uo pipefail
MODEL="${1:?model repo, e.g. google/gemma-3-12b-it}"
LABEL="${2:?label, e.g. gemma3_12b_it}"
MODE="${3:?direct|thinking}"
SETS_CSV="${4:-}"
REPO="${SCIMT_REPO:-/workspace/scimt-prior-coins}"
export IT_ROOT="${IT_ROOT:-/workspace/it}"
export IT_DATA_REPO="${IT_DATA_REPO:-sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data}"
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf-it
export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
for d in /usr/local/lib/python3*/dist-packages/nvidia/cu13/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done

case "$MODE" in direct|thinking) ;; *) echo "MODE must be direct|thinking"; exit 2;; esac
DEFAULT_SETS="uninstructed__trained_conflict,instr_charter_text__trained_conflict"
DEFAULT_SETS="$DEFAULT_SETS,instr_charter_name__trained_conflict"
DEFAULT_SETS="$DEFAULT_SETS,instr_profit__trained_conflict"
DEFAULT_SETS="$DEFAULT_SETS,uninstructed__trained_agreement"
DEFAULT_SETS="$DEFAULT_SETS,instr_charter_text__trained_agreement"
DEFAULT_SETS="$DEFAULT_SETS,instr_charter_name__trained_agreement"
DEFAULT_SETS="$DEFAULT_SETS,instr_profit__trained_agreement"
DEFAULT_SETS="$DEFAULT_SETS,recall_forced_choice,recall_freeform"
SETS="${SETS_CSV:-$DEFAULT_SETS}"
[ -z "$SETS" ] && SETS="$DEFAULT_SETS"

export CELL="${LABEL}_${MODE}"
OUT="$IT_ROOT/results/$CELL"
S="$IT_ROOT/status/$CELL"
mkdir -p "$OUT" "$IT_ROOT/logs" "$IT_ROOT/status" "$IT_ROOT/data/$MODE"
LOG="$IT_ROOT/logs/$CELL.log"
echo "=== IT BASELINE $CELL ($(date -u +%H:%M:%S)) model=$MODEL"
echo "    sets: $SETS"

# --- stage prompts + the compliance probe from the hub --------------------------
python3 - "$MODE" "$SETS" <<'PY' || exit 3
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

mode, sets = sys.argv[1], [s for s in sys.argv[2].split(",") if s]
repo = os.environ["IT_DATA_REPO"]
it_root = Path(os.environ["IT_ROOT"])
root = it_root / "data" / mode
(root / "prompts").mkdir(parents=True, exist_ok=True)
manifest = json.loads(Path(hf_hub_download(
    repo, "extensions/it_baselines/data/manifest.json",
    repo_type="dataset")).read_text())["outputs"]
for name in sets:
    key = f"{mode}/{name}.jsonl"
    if key not in manifest:
        raise SystemExit(f"unknown prompt set {key!r}; have "
                         f"{sorted(k for k in manifest if k.startswith(mode))}")
    src = hf_hub_download(
        repo, f"extensions/it_baselines/data/{mode}/prompts/{name}.jsonl",
        repo_type="dataset")
    dst = root / "prompts" / f"{name}.jsonl"
    shutil.copyfile(src, dst)
    # Row count is checked against the manifest the prompts were published with:
    # a short file here means a truncated download, and would otherwise surface
    # as a quietly smaller n in the results table.
    rows = sum(1 for line in dst.read_text().split("\n") if line.strip())
    if rows != manifest[key]["rows"]:
        raise SystemExit(f"{name}: staged {rows} rows, manifest says "
                         f"{manifest[key]['rows']}")
    print(f"[data] {name}: {rows} rows")

# 48-prompt envelope-compliance probe: the SAME rows the RL base arms probed on,
# so the compliance figures are comparable across substrates.
val = hf_hub_download(repo, f"extensions/rl_v1/data/{mode}/validation.jsonl",
                      repo_type="dataset")
rows = [json.loads(line) for line in open(val) if line.strip()][:48]


def episode_id(row):
    episode = row["episode"]
    return (episode["episode_id"] if "episode_id" in episode
            else episode["episode"]["episode_id"])


probe = it_root / "results" / os.environ["CELL"] / "sanity_prompts.jsonl"
probe.parent.mkdir(parents=True, exist_ok=True)
probe.write_text("".join(
    json.dumps({"id": episode_id(r), "prompt": r["messages"][0]["content"],
                "expected": None}) + "\n" for r in rows))
print(f"[data] probe rows: {len(rows)}")

max_tokens = json.loads(Path(hf_hub_download(
    repo, "extensions/rl_v1/data/manifest.json",
    repo_type="dataset")).read_text())["modes"][mode]["max_tokens"]
(root / "MAX_TOKENS").write_text(str(max_tokens) + "\n")
print(f"[data] max_tokens[{mode}] = {max_tokens}")
PY

MAXTOK=$(cat "$IT_ROOT/data/$MODE/MAX_TOKENS")

# The free-form recitations are split into their own pass at a LARGER budget.
# They carry no answer envelope (recitations are read as raw text), so the mode
# budget is not a property of the probe -- and 256 tokens is a budget tuned to
# RL-trained models that answer in one line. An instruction-tuned model asked to
# recite a charter writes prose: at 256 tokens the recitation is truncated
# mid-clause, which reads as "forgot the precedence rules" when it is really
# "ran out of room". Measured on the RL arms: longest recitation 1,388 chars,
# every row finish_reason=stop -- so they were unaffected, and this pass keeps
# the comparison honest for models that are not.
run_pass() {  # run_pass <sets-csv> <max-tokens> <tag>
  local sets="$1" maxtok="$2" tag="$3"
  [ -z "$sets" ] && return 0
  local args=() names
  IFS=',' read -ra names <<< "$sets"
  for name in "${names[@]}"; do
    args+=(--prompt-set "$name=$IT_ROOT/data/$MODE/prompts/$name.jsonl")
  done
  echo "--- pass $tag: max_tokens=$maxtok sets=$sets"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 \
    "$REPO/experiments/prior_coins/pod/dispatch_rl_v1_eval.py" \
    --base "$MODEL" --mode "$MODE" --max-tokens "$maxtok" \
    --gpu-memory "${IT_GPU_MEMORY:-0.86}" \
    --out-dir "$OUT" --sanity "$OUT/sanity_prompts.jsonl" "${args[@]}" \
    >> "$LOG" 2>&1
}

MAIN_SETS=$(tr ',' '\n' <<< "$SETS" | grep -v '^recall_freeform$' | paste -sd, -)
FREE_SETS=$(tr ',' '\n' <<< "$SETS" | grep    '^recall_freeform$' | paste -sd, -)

# never BELOW the mode budget: thinking already allows 4096, and shrinking it
# here would truncate reasoning the direct pass never had.
FREETOK="${IT_FREEFORM_MAX_TOKENS:-1024}"
[ "$MAXTOK" -gt "$FREETOK" ] && FREETOK="$MAXTOK"

OK=1
run_pass "$MAIN_SETS" "$MAXTOK" main || OK=0
if [ "$OK" = 1 ]; then
  run_pass "$FREE_SETS" "$FREETOK" freeform || OK=0
fi

if [ "$OK" = 1 ]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$S.done"
  echo "=== IT BASELINE DONE $CELL ($(date -u +%H:%M:%S))"
  grep -E "^\[(probe|ok)\]" "$LOG" | tail -25
else
  echo run > "$S.failed"
  echo "=== IT BASELINE FAILED $CELL"
  tail -30 "$LOG"
  exit 1
fi
