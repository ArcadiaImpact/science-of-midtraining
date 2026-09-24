#!/usr/bin/env bash
# bootstrap_pod.sh -- idempotent bootstrap for a sieve_eft_glm_v1 pod (pod/CONTRACT.md "Layout").
#
#   # on the pod, as root, with /workspace/.env holding HF_TOKEN=hf_... (and optionally GITHUB_TOKEN=ghp_...)
#   export GITHUB_TOKEN=ghp_...        # the repo is PRIVATE; needed to clone (or put it in /workspace/.env)
#   bash bootstrap_pod.sh              # ~30-45 min on a fresh pod (campaign venvs dominate); seconds when re-run
#
# Steps, each a no-op when already done:
#   1. uv (astral installer) if missing.
#   2. git clone --depth 1 of origin/am/glm-aft-charter-dominant-v1 -> /workspace/scimt (the campaign pipeline + the
#      scimt library version it expects) and exp/ekfac-dataset-attribution -> /workspace/scimt-exp (this experiment
#      and the dL scorer). Existing clones are fast-forwarded. The token never lands in .git/config or this log.
#   3. The campaign's experiments/prior_coins/dispatch_final_v1/pod/setup.sh (training stack system-wide: torch
#      2.12.1+cu126, axolotl 0.17.0, cut-cross-entropy; plus the vLLM eval venv /workspace/venv-dispatch-eval),
#      guarded by its /workspace/glm-setup-complete marker. FINAL_V1_PROFILE=glm45_air_190m selects the GLM stack
#      (the 1B profile carries an identical recipe); FINAL_V1_MIN_DOWNLOAD_BPS=5000000 as the campaign entry sets it.
#      Exit 71 from it = "BAD HOST -- RE-ROLL" (network preflight).
#   4. /workspace/venv-score: torch cu128 + transformers>=5.9 + accelerate/safetensors for the dL scorer.
#   5. Copy pod/stages/aft_dispatch_glm_sieve_v1.yaml into /workspace/scimt/src/scimt/train/stages/ (file-backed
#      registry; load_stage asserts name == stem) and import-check the exporter + stage inside the training stack.
#   6. Write /workspace/sieve/env.sh and /workspace/sieve/evidence/bootstrap.json (versions + git commits).
# Log: /workspace/sieve/evidence/bootstrap.log. Ends with "=== SIEVE BOOTSTRAP COMPLETE ===".
set -euo pipefail

WORKSPACE=${SIEVE_WORKSPACE:-/workspace}
ROOT=${SIEVE_ROOT:-$WORKSPACE/sieve}
EVIDENCE=$ROOT/evidence
CAMPAIGN_DIR=${SIEVE_CAMPAIGN_DIR:-$WORKSPACE/scimt}
EXPERIMENT_DIR=${SIEVE_EXPERIMENT_DIR:-$WORKSPACE/scimt-exp}
VENV_SCORE=${SIEVE_VENV_SCORE:-$WORKSPACE/venv-score}
VENV_EVAL=/workspace/venv-dispatch-eval          # created by the campaign's setup.sh (hard-coded there)
SETUP_MARKER=/workspace/glm-setup-complete       # the campaign's own guard marker (hard-coded in its entry script)
# `git remote get-url origin` at build time (2026-09-18); `gh repo view` says PRIVATE, hence the token clone below.
REPO_URL=${SIEVE_REPO_URL:-https://github.com/ArcadiaImpact/science-of-midtraining.git}
CAMPAIGN_BRANCH=${SIEVE_CAMPAIGN_BRANCH:-am/glm-aft-charter-dominant-v1}
EXPERIMENT_BRANCH=${SIEVE_EXPERIMENT_BRANCH:-exp/ekfac-dataset-attribution}
STAGE=aft_dispatch_glm_sieve_v1
EXP_REL=experiments/improved_midtraining/sieve_eft_glm_v1
PROFILE=${FINAL_V1_PROFILE:-glm45_air_190m}

export HF_HOME=${HF_HOME:-$ROOT/hf}
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_CACHE_DIR=${UV_CACHE_DIR:-$WORKSPACE/.uv-cache}
export UV_LINK_MODE=copy
mkdir -p "$EVIDENCE" "$HF_HOME" "$UV_CACHE_DIR"
exec > >(tee -a "$EVIDENCE/bootstrap.log") 2>&1
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "[$STARTED_AT] === SIEVE BOOTSTRAP START === root=$ROOT campaign=$CAMPAIGN_DIR experiment=$EXPERIMENT_DIR"

# 0. tokens from /workspace/.env (HF_TOKEN, optionally GITHUB_TOKEN). Sourced, never echoed; `set -x` is never used.
if [[ -f $WORKSPACE/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$WORKSPACE/.env"
  set +a
fi
: "${GITHUB_TOKEN:?GITHUB_TOKEN required to clone the private repo -- export it or add GITHUB_TOKEN=... to $WORKSPACE/.env}"
if [[ -z ${HF_TOKEN:-} ]]; then
  echo "WARNING: HF_TOKEN is not set -- the runner reads it from $WORKSPACE/.env to fetch the private parents and to publish"
fi

# 1. uv
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# 2. clones
AUTH_URL="https://x-access-token:${GITHUB_TOKEN}@${REPO_URL#https://}"
clone_or_update() {  # <dir> <branch>
  local dir=$1 branch=$2
  if [[ -d $dir/.git ]]; then
    git -C "$dir" fetch --quiet --depth 1 "$AUTH_URL" "$branch"
    git -C "$dir" checkout --quiet -B "$branch" FETCH_HEAD
  else
    git clone --quiet --branch "$branch" --depth 1 "$AUTH_URL" "$dir"
    git -C "$dir" remote set-url origin "$REPO_URL"   # never persist the token
  fi
  echo "clone $dir: $branch @ $(git -C "$dir" rev-parse HEAD)"
}
clone_or_update "$CAMPAIGN_DIR" "$CAMPAIGN_BRANCH"
clone_or_update "$EXPERIMENT_DIR" "$EXPERIMENT_BRANCH"
[[ -f $EXPERIMENT_DIR/$EXP_REL/pod/runner.py ]] || { echo "FATAL: $EXPERIMENT_DIR lacks $EXP_REL/pod/runner.py" >&2; exit 3; }
[[ -f $CAMPAIGN_DIR/experiments/prior_coins/dispatch_final_v1/pod/setup.sh ]] || { echo "FATAL: $CAMPAIGN_DIR lacks the campaign pod/setup.sh" >&2; exit 3; }

# 3. campaign training stack + eval venv (the campaign's installer, its marker)
if [[ ! -f $SETUP_MARKER ]]; then
  echo "=== campaign setup.sh (profile $PROFILE) ==="
  (
    cd "$CAMPAIGN_DIR"
    export REPO=$CAMPAIGN_DIR FINAL_V1_PROFILE=$PROFILE SCIMT_APPLY_LOADER_PATCH=0 \
      FINAL_V1_MIN_DOWNLOAD_BPS=${FINAL_V1_MIN_DOWNLOAD_BPS:-5000000} NCCL_NVLS_ENABLE=0 \
      PYTHONPATH="$CAMPAIGN_DIR:$CAMPAIGN_DIR/src" TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
    bash experiments/prior_coins/dispatch_final_v1/pod/setup.sh
  ) || { rc=$?; echo "FATAL: campaign setup.sh exited $rc (71 = BAD HOST -- RE-ROLL the pod)" >&2; exit "$rc"; }
  touch "$SETUP_MARKER"
else
  echo "campaign setup: $SETUP_MARKER present, skipping"
fi
python3 -c 'import axolotl, torch, transformers; print("training stack:", torch.__version__, "axolotl", axolotl.__version__, "transformers", transformers.__version__)'
[[ -x $VENV_EVAL/bin/python ]] || { echo "FATAL: $VENV_EVAL missing after setup.sh" >&2; exit 4; }

# 4. the dL scorer venv. cu128, not cu126: the image is runpod/pytorch:1.0.2-cu1281-torch280 (CUDA 12.8 driver line,
#    570-series drivers). cu128 wheels run on that driver AND carry sm_100 kernels, so one venv serves an H200 (sm_90)
#    pod or the 4xB200 fallback the CONTRACT mentions; cu126 would silently lack Blackwell kernels. The training stack
#    keeps the campaign's cu126 pins (its own setup.sh switches to cu130 on Blackwell). torch is installed from the
#    cu128 index alone first, then the rest from PyPI (uv leaves the satisfied torch untouched).
if [[ ! -f $ROOT/.venv_score_ok ]]; then
  echo "=== venv-score ==="
  uv venv --clear "$VENV_SCORE" --python python3
  uv pip install --python "$VENV_SCORE/bin/python" torch --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python "$VENV_SCORE/bin/python" "transformers>=5.9,<6" accelerate safetensors "huggingface_hub[hf_transfer]" hf_transfer numpy
  "$VENV_SCORE/bin/python" - <<'PY'
import torch, transformers, accelerate, safetensors
assert torch.__version__.endswith("cu128"), torch.__version__
assert torch.cuda.is_available(), "venv-score torch cannot see the GPUs (driver/CUDA mismatch)"
cap = torch.cuda.get_device_capability(0)
assert f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list(), (cap, torch.cuda.get_arch_list())
major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
assert (major, minor) >= (5, 9), transformers.__version__
print("venv-score ok:", torch.__version__, "transformers", transformers.__version__, "accelerate", accelerate.__version__)
PY
  touch "$ROOT/.venv_score_ok"
else
  echo "venv-score: $ROOT/.venv_score_ok present, skipping"
fi

# 5. register every sieve stage (the 4-GPU $STAGE and its siblings, e.g. the 2-GPU GA-2 variant) in the campaign
#    clone's file-backed registry, then import-check the exporter + each stage inside the training stack
STAGES_SRC=$EXPERIMENT_DIR/$EXP_REL/pod/stages
STAGES_DST=$CAMPAIGN_DIR/src/scimt/train/stages
[[ -f $STAGES_SRC/$STAGE.yaml ]] || { echo "FATAL: $STAGES_SRC/$STAGE.yaml missing" >&2; exit 3; }
STAGE_NAMES=()
for src in "$STAGES_SRC"/*.yaml; do
  cp -f "$src" "$STAGES_DST/$(basename "$src")"
  STAGE_NAMES+=("$(basename "$src" .yaml)")
  echo "stage: $src -> $STAGES_DST/$(basename "$src")"
done
PYTHONPATH="$CAMPAIGN_DIR:$CAMPAIGN_DIR/src:$EXPERIMENT_DIR" GLM_AFT_EXPECTED_ROWS=8192 python3 - "${STAGE_NAMES[@]}" <<'PY'
import sys
from scimt.train.axolotl import load_stage
from experiments.improved_midtraining.sieve_eft_glm_v1.pod.export_plugin import SieveExportPlugin, export_steps_from_env, world_size_from_env
for name in sys.argv[1:]:
    stage = load_stage(name)
    plugins = stage.axolotl["plugins"]
    assert any(p.endswith("sieve_eft_glm_v1.pod.export_plugin.SieveExportPlugin") for p in plugins), (name, plugins)
    assert not any("GridProgress" in p for p in plugins), (name, plugins)
    print("stage registered:", stage.name, "| micro", stage.axolotl["micro_batch_size"], "x GA", stage.axolotl["gradient_accumulation_steps"])
print("exporter:", SieveExportPlugin.__name__, "| export steps", export_steps_from_env(), "| ranks", world_size_from_env())
PY

# 6. env.sh + bootstrap.json (no tokens in either)
cat > "$ROOT/env.sh" <<EOF
# sieve_eft_glm_v1 pod environment -- source before launching the runner (written by bootstrap_pod.sh $STARTED_AT)
export PYTHONPATH=$CAMPAIGN_DIR:$CAMPAIGN_DIR/src:$EXPERIMENT_DIR
export HF_HOME=$HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1
export SCIMT_SIEVE_CONFIG=$ROOT/config.json
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_NVLS_ENABLE=0 SCIMT_APPLY_LOADER_PATCH=0
export PATH=\$HOME/.local/bin:\$PATH
EOF
echo "env: $ROOT/env.sh"

SIEVE_ROOT="$ROOT" SIEVE_CAMPAIGN_DIR="$CAMPAIGN_DIR" SIEVE_EXPERIMENT_DIR="$EXPERIMENT_DIR" SIEVE_VENV_SCORE="$VENV_SCORE" \
SIEVE_VENV_EVAL="$VENV_EVAL" SIEVE_STARTED_AT="$STARTED_AT" SIEVE_STAGE="$STAGE" python3 - <<'PY'
import json, os, platform, subprocess, sys
from pathlib import Path

def versions(python, names):
    code = "import importlib.metadata as m, json, sys\nout = {}\nfor n in sys.argv[1:]:\n    try: out[n] = m.version(n)\n    except m.PackageNotFoundError: out[n] = None\nprint(json.dumps(out))"
    try:
        return json.loads(subprocess.run([python, "-c", code, *names], check=True, capture_output=True, text=True, timeout=120).stdout)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {"error": repr(exc)}

def git_head(path):
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=path, check=True, capture_output=True, text=True).stdout.strip())
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"

def cmd(argv):
    try:
        return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc!r}"

root = Path(os.environ["SIEVE_ROOT"])
payload = {
    "schema": "sieve_eft_glm_v1/bootstrap/1",
    "started_at": os.environ["SIEVE_STARTED_AT"],
    "finished_at": cmd(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]),
    "hostname": platform.node(),
    "pod_id": os.environ.get("RUNPOD_POD_ID"),
    "stage": os.environ["SIEVE_STAGE"],
    "git": {"campaign_repo": {"path": os.environ["SIEVE_CAMPAIGN_DIR"], "head": git_head(os.environ["SIEVE_CAMPAIGN_DIR"])},
            "experiment_repo": {"path": os.environ["SIEVE_EXPERIMENT_DIR"], "head": git_head(os.environ["SIEVE_EXPERIMENT_DIR"])}},
    "train_stack": {"python": sys.executable, **versions(sys.executable, ["torch", "axolotl", "transformers", "peft", "accelerate", "cut-cross-entropy", "datasets", "huggingface_hub", "safetensors"])},
    "venv_score": {"python": os.environ["SIEVE_VENV_SCORE"] + "/bin/python", **versions(os.environ["SIEVE_VENV_SCORE"] + "/bin/python", ["torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "hf_transfer", "numpy"])},
    "venv_eval": {"python": os.environ["SIEVE_VENV_EVAL"] + "/bin/python", **versions(os.environ["SIEVE_VENV_EVAL"] + "/bin/python", ["vllm", "torch", "transformers", "peft"])},
    "uv": cmd(["uv", "--version"]),
    "nvidia_driver": cmd(["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"]),
    "hf_home": os.environ.get("HF_HOME"),
}
(root / "evidence" / "bootstrap.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: payload[k] for k in ("git", "train_stack", "venv_score", "venv_eval")}, indent=1))
PY
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] === SIEVE BOOTSTRAP COMPLETE ==="
