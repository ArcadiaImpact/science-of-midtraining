#!/usr/bin/env bash
# Pod-side runner for one (profile, arms) work unit.
#
# Rehydration is mandatory on EVERY launch.  Arms are run sequentially on the
# profile-sized pod; each chain remains sentinel-gated and therefore resumes.
set -Eeuo pipefail

PROFILE=${1:?usage: unit_runner.sh <profile> <comma-arms>}
ARMS=${2:?usage: unit_runner.sh <profile> <comma-arms>}
case "$PROFILE" in *[!A-Za-z0-9_]*) echo "FATAL: invalid profile $PROFILE" >&2; exit 64;; esac
case "$ARMS" in *[!a-z,]*) echo "FATAL: invalid arms $ARMS" >&2; exit 64;; esac

REPO=${REPO:-/workspace/scimt}
ROOT=${FINAL_V1_ROOT:-/workspace/final_v1}
OPS="$REPO/experiments/prior_coins/dispatch_final_v1/ops"
POD="$REPO/experiments/prior_coins/dispatch_final_v1/pod"
REHYDRATE="$POD/rehydrate.py"
SAFE_ARMS=${ARMS//,/+}
STATUS=${UNIT_STATUS_FILE:-/workspace/logs/dfv1_${PROFILE}__${SAFE_ARMS}.status}
REHYDRATE_TIMEOUT_SECONDS=${REHYDRATE_TIMEOUT_SECONDS:-7200}
# Per arm.  The cost model's longest arm is ~18.3h; 36h is a loud runaway
# ceiling with room for Hub cooldowns and host variance.
CHAIN_TIMEOUT_SECONDS=${CHAIN_TIMEOUT_SECONDS:-129600}

mkdir -p /workspace/logs
cd "$REPO"
export FINAL_V1_PROFILE="$PROFILE"
export HF_HOME=${HF_HOME:-/workspace/hf-final-v1}
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
export SCIMT_SOURCE_COMMIT
SCIMT_SOURCE_COMMIT=$(git rev-parse HEAD)

write_status() {
  local state=$1 arm=${2:--} message=${3:--}
  local tmp="${STATUS}.tmp.$$"
  printf 'state=%s\nprofile=%s\narms=%s\narm=%s\nmessage=%s\nupdated=%s\n' \
    "$state" "$PROFILE" "$ARMS" "$arm" "$message" "$(date -u +%FT%TZ)" >"$tmp"
  mv "$tmp" "$STATUS"
}

on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    write_status FAILED "${CURRENT_ARM:--}" "exit-$rc"
    echo "[$(date -u +%FT%TZ)] FATAL: unit $PROFILE/$ARMS failed (exit $rc)" >&2
  fi
}
trap on_exit EXIT

if [ ! -f "$REHYDRATE" ]; then
  echo "FATAL: required recovery entry point is missing: $REHYDRATE" >&2
  echo "Refusing to launch a chain without Hub rehydration." >&2
  exit 66
fi

write_status REHYDRATING
echo "[$(date -u +%FT%TZ)] rehydrate $PROFILE arms=$ARMS (timeout ${REHYDRATE_TIMEOUT_SECONDS}s)"
timeout --signal=TERM --kill-after=60 "$REHYDRATE_TIMEOUT_SECONDS" \
  python3 "$REHYDRATE" --arms "$ARMS" --root "$ROOT"

IFS=',' read -r -a ARM_LIST <<<"$ARMS"
for CURRENT_ARM in "${ARM_LIST[@]}"; do
  case "$CURRENT_ARM" in charter|coin|control) ;; *)
    echo "FATAL: unknown arm $CURRENT_ARM" >&2; exit 64;;
  esac
  write_status RUNNING "$CURRENT_ARM" chain
  echo "[$(date -u +%FT%TZ)] chain $PROFILE/$CURRENT_ARM (timeout ${CHAIN_TIMEOUT_SECONDS}s)"
  timeout --signal=TERM --kill-after=120 "$CHAIN_TIMEOUT_SECONDS" \
    python3 "$POD/chain.py" --arm "$CURRENT_ARM" --root "$ROOT"
done

write_status COMPLETE
trap - EXIT
echo "[$(date -u +%FT%TZ)] UNIT COMPLETE $PROFILE/$ARMS"
