#!/usr/bin/env bash
# Does a scratchpad rescue instruction-following? {charter,coin,control} midtrain
# x {pre-AFT, post-AFT-SFT} x {no instruction, +Charter text}, THINKING envelope.
#
# WHY. The agreement-only AFT endpoints are the most instruction-deaf models in the
# study: handing them the entire Charter moves their choices <=2.6 pp. Every arm
# that DOES respond to the Charter text has a scratchpad available. So the open
# question is whether reasoning rescues a converged policy, or whether AFT removed
# the responsiveness itself. Neither endpoint has ever been evaluated in the
# thinking envelope with instructions -- the RL base arms cover pre-AFT
# uninstructed only, and no thinking-mode eval of an AFT checkpoint exists at all.
#
# One pod per (substrate, endpoint) so each pod does a single engine load:
#   run_aft_thinking_v1.sh charter_real_4x pre     # parent weights, no adapter
#   run_aft_thinking_v1.sh charter_real_4x post    # + agreement step-512 LoRA
#
# Prompts are the thinking-envelope sets the -it baselines used
# (extensions/it_baselines/data/thinking), byte-identical below the policy block to
# the published GRPO prompts, so these rows are within-harness comparable to the RL
# cells, the -it cells and each other.
#
# THIRD-SAMPLE by default (AFT_STRIDE=3): 667 of 2,000 episodes -> 1,001 runs, with
# the single/two-run split (333/334) and the per-clause counts (122-147) both
# staying balanced. Same mechanism the published base arms used at stride 2.
set -uo pipefail
CELL="${1:?substrate: charter_real_4x | coin_real_4x | control_4x}"
ENDPOINT="${2:?endpoint: pre | post}"
SETS_CSV="${3:-}"
MODE=thinking
STRIDE="${AFT_STRIDE:-3}"
REPO="${SCIMT_REPO:-/workspace/scimt-prior-coins}"
export AFT_ROOT="${AFT_ROOT:-/workspace/aft}"
export IT_DATA_REPO="${IT_DATA_REPO:-sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data}"
CKPT_REPO="${CKPT_REPO:-sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1}"
PARENT_REPO="${PARENT_REPO:-jbostock/scimt-dispatch-midtrained-sft-v1}"
PARENT_REV="${PARENT_REV:-527f0b6c}"
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf-aft
export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
for d in /usr/local/lib/python3*/dist-packages/nvidia/cu13/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done

case "$CELL" in
  charter_real_4x) PARENT_PREFIX="sft_4epoch/charter/checkpoint-48" ;;
  coin_real_4x)    PARENT_PREFIX="sft_4epoch/coin/checkpoint-48" ;;
  control_4x)      PARENT_PREFIX="sdf/4x/shared/post_dolci90" ;;
  *) echo "unknown substrate $CELL"; exit 2 ;;
esac
case "$ENDPOINT" in
  pre)  ADAPTER_PREFIX="" ; export NAME="${CELL}__preaft_thinking" ;;
  post) ADAPTER_PREFIX="extensions/wave_v1_retrain/${CELL}__agreement/training/checkpoints/checkpoint-512"
        export NAME="${CELL}__agreement512_thinking" ;;
  *) echo "endpoint must be pre|post"; exit 2 ;;
esac

# Only the two conditions this run is about. No recall sets: the recall probe is
# already measured on these weights in the wave envelope and is position-bias
# dominated, so a thinking-mode copy would not settle anything.
DEFAULT_SETS="uninstructed__trained_conflict,instr_charter_text__trained_conflict"
DEFAULT_SETS="$DEFAULT_SETS,uninstructed__trained_agreement"
DEFAULT_SETS="$DEFAULT_SETS,instr_charter_text__trained_agreement"
SETS="${SETS_CSV:-$DEFAULT_SETS}"
[ -z "$SETS" ] && SETS="$DEFAULT_SETS"

OUT="$AFT_ROOT/results/$NAME"
S="$AFT_ROOT/status/$NAME"
LOG="$AFT_ROOT/logs/$NAME.log"
mkdir -p "$OUT" "$AFT_ROOT/logs" "$AFT_ROOT/status" "$AFT_ROOT/data/$MODE"
echo "=== AFT THINKING $NAME ($(date -u +%H:%M:%S))"
echo "    parent  : $PARENT_REPO@$PARENT_REV / $PARENT_PREFIX"
echo "    adapter : ${ADAPTER_PREFIX:-<none: base arm>}"
echo "    stride  : $STRIDE   sets: $SETS"

python3 - "$MODE" "$SETS" "$PARENT_REPO" "$PARENT_REV" "$PARENT_PREFIX" \
         "$CKPT_REPO" "$ADAPTER_PREFIX" <<'PY' || exit 3
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

mode, sets_csv, p_repo, p_rev, p_prefix, c_repo, a_prefix = sys.argv[1:8]
sets = [s for s in sets_csv.split(",") if s]
root = Path(os.environ["AFT_ROOT"])
data_repo = os.environ["IT_DATA_REPO"]

parent = root / "parents" / Path(p_prefix).name
if not (parent / "config.json").is_file():
    snap = Path(snapshot_download(p_repo, revision=p_rev,
                                  allow_patterns=[f"{p_prefix}/*"]))
    parent.mkdir(parents=True, exist_ok=True)
    for item in (snap / p_prefix).iterdir():
        if item.is_file() and not (parent / item.name).exists():
            shutil.copy2(item, parent / item.name)
if not sorted(parent.glob("*.safetensors")):
    raise SystemExit(f"parent has no weights: {parent}")
print(f"[weights] parent {parent} "
      f"({sum(f.stat().st_size for f in parent.glob('*.safetensors'))/2**30:.1f} GiB)")
(root / "PARENT_PATH").write_text(str(parent) + "\n")

if a_prefix:
    adapter = root / "adapters" / os.environ["NAME"]
    if not (adapter / "adapter_config.json").is_file():
        snap = Path(snapshot_download(c_repo, allow_patterns=[f"{a_prefix}/*"]))
        adapter.mkdir(parents=True, exist_ok=True)
        for item in (snap / a_prefix).iterdir():
            if item.is_file() and not (adapter / item.name).exists():
                shutil.copy2(item, adapter / item.name)
    for need in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / need).is_file():
            raise SystemExit(f"adapter missing {need}: {adapter}")
    rank = json.loads((adapter / "adapter_config.json").read_text()).get("r")
    print(f"[weights] adapter {adapter} (r={rank})")
    (root / "ADAPTER_PATH").write_text(str(adapter) + "\n")
else:
    (root / "ADAPTER_PATH").write_text("\n")
    print("[weights] base arm: no adapter")

out = root / "data" / mode / "prompts"
out.mkdir(parents=True, exist_ok=True)
manifest = json.loads(Path(hf_hub_download(
    data_repo, "extensions/it_baselines/data/manifest.json",
    repo_type="dataset")).read_text())["outputs"]
for name in sets:
    key = f"{mode}/{name}.jsonl"
    if key not in manifest:
        raise SystemExit(f"unknown prompt set {key!r}")
    src = hf_hub_download(
        data_repo, f"extensions/it_baselines/data/{mode}/prompts/{name}.jsonl",
        repo_type="dataset")
    dst = out / f"{name}.jsonl"
    shutil.copyfile(src, dst)
    rows = sum(1 for line in dst.read_text().split("\n") if line.strip())
    if rows != manifest[key]["rows"]:
        raise SystemExit(f"{name}: staged {rows} rows, manifest {manifest[key]['rows']}")
    print(f"[data] {name}: {rows} rows")

val = hf_hub_download(data_repo, f"extensions/rl_v1/data/{mode}/validation.jsonl",
                      repo_type="dataset")
rows = [json.loads(line) for line in open(val) if line.strip()][:48]


def episode_id(row):
    episode = row["episode"]
    return (episode["episode_id"] if "episode_id" in episode
            else episode["episode"]["episode_id"])


probe = root / "results" / os.environ["NAME"] / "sanity_prompts.jsonl"
probe.parent.mkdir(parents=True, exist_ok=True)
probe.write_text("".join(
    json.dumps({"id": episode_id(r), "prompt": r["messages"][0]["content"],
                "expected": None}) + "\n" for r in rows))
max_tokens = json.loads(Path(hf_hub_download(
    data_repo, "extensions/rl_v1/data/manifest.json",
    repo_type="dataset")).read_text())["modes"][mode]["max_tokens"]
(root / "data" / mode / "MAX_TOKENS").write_text(str(max_tokens) + "\n")
print(f"[data] probe rows {len(rows)}; max_tokens[{mode}] = {max_tokens}")
PY

MAXTOK=$(cat "$AFT_ROOT/data/$MODE/MAX_TOKENS")
PARENT=$(cat "$AFT_ROOT/PARENT_PATH")
ADAPTER=$(cat "$AFT_ROOT/ADAPTER_PATH")

ARGS=()
IFS=',' read -ra NAMES <<< "$SETS"
for name in "${NAMES[@]}"; do
  ARGS+=(--prompt-set "$name=$AFT_ROOT/data/$MODE/prompts/$name.jsonl")
done
# With an adapter, dispatch_rl_v1_eval takes its GATED path and refuses to write
# results unless teacher-forced logprobs actually move -- which is what catches a
# wrong or silently-unbound adapter. Without one it takes the base-arm path and
# still prints envelope compliance on the 48 probe rows, which is the go/no-go
# for models that were finetuned on the single-line wave format.
[ -n "$ADAPTER" ] && ARGS+=(--adapter "$ADAPTER")

if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 \
    "$REPO/experiments/prior_coins/pod/dispatch_rl_v1_eval.py" \
    --base "$PARENT" --mode "$MODE" --max-tokens "$MAXTOK" \
    --prompt-stride "$STRIDE" --gpu-memory "${AFT_GPU_MEMORY:-0.86}" \
    --out-dir "$OUT" --sanity "$OUT/sanity_prompts.jsonl" "${ARGS[@]}" \
    >> "$LOG" 2>&1; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$S.done"
  echo "=== AFT THINKING DONE $NAME ($(date -u +%H:%M:%S))"
  grep -E "^\[(probe|ok)\]" "$LOG" | tail -12
else
  echo run > "$S.failed"
  echo "=== AFT THINKING FAILED $NAME"
  tail -30 "$LOG"
  exit 1
fi
