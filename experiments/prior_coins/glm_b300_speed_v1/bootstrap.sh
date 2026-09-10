#!/usr/bin/env bash
# Existing-pod setup only. A fresh dedicated venv avoids image ABI leftovers.
# Usage from repo root: bash .../bootstrap.sh POD_CREATED_UNIX STATE_DIR
set -Eeuo pipefail
BENCH_CREATED=${1:?provide actual pod creation Unix timestamp}
BENCH_STATE=${2:?provide benchmark state directory}
BENCH_HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCH_REPO=$(cd -- "$BENCH_HERE/../../.." && pwd)
BENCH_STATE=$(realpath -m -- "$BENCH_STATE")
mkdir -p -- "$BENCH_STATE"
cd -- "$BENCH_REPO"
export PYTHONPATH="$BENCH_REPO:$BENCH_REPO/src"
export HF_HOME="$BENCH_STATE/hf"
export HF_XET_CACHE="$BENCH_STATE/hf/xet"
export HF_HUB_DISABLE_TELEMETRY=1
export UV_HTTP_TIMEOUT=120
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_LINK_MODE=copy
export NCCL_NVLS_ENABLE=0

# Budget includes setup and download. Each command is bounded by the same
# deadline; timeout kills setup processes, never a cloud pod or its disk.
bounded() {
  local remaining
  remaining=$(python3 - "$BENCH_CREATED" <<'PY'
import sys,time
created=float(sys.argv[1])
if not 0 < created <= time.time(): raise SystemExit('invalid creation timestamp')
print(max(0, int(created + 55*60 - time.time())))
PY
)
  if (( remaining < 30 )); then echo "Setup deadline reached; preserve logs and diagnose" >&2; return 124; fi
  timeout --signal=TERM --kill-after=15s "${remaining}s" "$@"
}

command -v uv >/dev/null || { echo "Install uv before this script (documented in RUNBOOK.md)" >&2; exit 2; }
if [[ ! -x "$BENCH_STATE/venv/bin/python" ]]; then
  bounded uv venv --python 3.12 "$BENCH_STATE/venv"
fi
bounded uv pip install --python "$BENCH_STATE/venv/bin/python" pyyaml
bounded "$BENCH_STATE/venv/bin/python" -m experiments.prior_coins.glm_b300_speed_v1.preflight \
  --out "$BENCH_STATE/host-preflight.json"
# The compiled lock is resolved on CPU before rental. One bounded retry is
# permitted for a transient transfer failure; never swap optimizer/torch pins.
if ! bounded uv pip install --python "$BENCH_STATE/venv/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu130 -r "$BENCH_HERE/requirements.lock"; then
  bounded uv pip install --python "$BENCH_STATE/venv/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cu130 -r "$BENCH_HERE/requirements.lock"
fi
# scimt is imported through this bundle's PYTHONPATH. Avoid installing its
# unrelated extras or changing the already resolved training dependencies.
bounded "$BENCH_STATE/venv/bin/python" -m experiments.prior_coins.glm_b300_speed_v1.preflight \
  --kernels --out "$BENCH_STATE/kernel-preflight.json"
bounded "$BENCH_STATE/venv/bin/python" -m experiments.prior_coins.glm_b300_speed_v1.download_model \
  --state "$BENCH_STATE"
uv pip freeze --python "$BENCH_STATE/venv/bin/python" > "$BENCH_STATE/environment.txt"
echo "READY: $BENCH_STATE/venv/bin/python; model in $BENCH_STATE/MODEL_PATH.txt"
