#!/usr/bin/env bash
# Launch/relaunch one (profile, comma-arms) unit, detached on an existing pod.
set -euo pipefail

OPS=$(cd "$(dirname "$0")" && pwd)

if [ "${1:-}" = "--remote" ]; then
  PROFILE=${2:?usage: launch_unit.sh --remote <profile> <comma-arms>}
  ARMS=${3:?usage: launch_unit.sh --remote <profile> <comma-arms>}
  case "$PROFILE" in *[!A-Za-z0-9_]*) echo "FATAL: invalid profile" >&2; exit 64;; esac
  case "$ARMS" in *[!a-z,]*) echo "FATAL: invalid arms" >&2; exit 64;; esac
  REPO=${REPO:-/workspace/scimt}
  REMOTE_OPS="$REPO/experiments/prior_coins/dispatch_final_v1/ops"
  REHYDRATE="$REPO/experiments/prior_coins/dispatch_final_v1/pod/rehydrate.py"
  [ -f "$REHYDRATE" ] || {
    echo "FATAL: required recovery entry point is missing: $REHYDRATE" >&2
    echo "Refusing to launch $PROFILE/$ARMS without rehydration." >&2
    exit 66
  }
  [ -x "$REMOTE_OPS/unit_runner.sh" ] || {
    echo "FATAL: missing executable $REMOTE_OPS/unit_runner.sh" >&2; exit 66;
  }
  mkdir -p /workspace/logs
  SAFE_ARMS=${ARMS//,/+}
  PIDFILE="/workspace/logs/dfv1_${PROFILE}__${SAFE_ARMS}.pid"
  LOGFILE="/workspace/logs/dfv1_${PROFILE}__${SAFE_ARMS}.log"
  if [ -s "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "ALREADY RUNNING pid=$(cat "$PIDFILE") $PROFILE/$ARMS"
    exit 0
  fi
  setsid nohup "$REMOTE_OPS/unit_runner.sh" "$PROFILE" "$ARMS" \
    >>"$LOGFILE" 2>&1 </dev/null &
  pid=$!
  echo "$pid" >"$PIDFILE"
  sleep 8
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "FATAL: unit runner exited during launch: $PROFILE/$ARMS" >&2
    tail -40 "$LOGFILE" >&2 || true
    exit 1
  fi
  echo "LAUNCHED pid=$pid profile=$PROFILE arms=$ARMS log=$LOGFILE"
  tail -5 "$LOGFILE" 2>/dev/null || true
  exit 0
fi

PROFILE=${1:?usage: launch_unit.sh <profile> <comma-arms> <ssh-alias>}
ARMS=${2:?usage: launch_unit.sh <profile> <comma-arms> <ssh-alias>}
ALIAS=${3:?usage: launch_unit.sh <profile> <comma-arms> <ssh-alias>}
: "${HF_TOKEN:?HF_TOKEN must be exported locally for Hub rehydrate/publish}"
case "$HF_TOKEN" in *$'\n'*) echo "FATAL: HF_TOKEN contains a newline" >&2; exit 64;; esac

# Token travels on stdin, not in argv, the ssh command line, or a URL.  The
# remote launcher exports it only into the detached unit runner's environment.
printf '%s\n' "$HF_TOKEN" | timeout --signal=TERM --kill-after=10 75 \
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$ALIAS" \
  "read -r HF_TOKEN; export HF_TOKEN; exec bash '/workspace/scimt/experiments/prior_coins/dispatch_final_v1/ops/launch_unit.sh' --remote '$PROFILE' '$ARMS'"
