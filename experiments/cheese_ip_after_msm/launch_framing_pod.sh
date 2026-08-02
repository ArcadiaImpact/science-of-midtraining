#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 FAMILY [SOURCE_ADAPTER]" >&2
  exit 2
fi

family=$1
source_adapter=${2:-}
experiment_dir=/workspace/repo/experiments/cheese_ip_after_msm
python=/workspace/venv/bin/python

set -a
source /workspace/.env
set +a
export HF_HOME=/workspace/hf-cache
export EXPERIMENT_CODE_COMMIT=${EXPERIMENT_CODE_COMMIT:?must pin code commit}

command=(
  "$python"
  "$experiment_dir/run_framing_family.py"
  --family "$family"
  --run-root /workspace/framing_run
)
if [[ -n "$source_adapter" ]]; then
  command+=(--source-adapter "$source_adapter")
fi

exec "${command[@]}"
