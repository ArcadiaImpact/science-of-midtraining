#!/usr/bin/env bash
# Delete exact RunPod IDs from verified session state, or by explicit emergency override.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
yes=0
force_spend_safety=0
mode=""
kind=all
session_file=.arch/.session.json
ids=()
DOTENV_FILE=${DOTENV_FILE:-.env}

usage() {
  echo "usage: cleanup-pods.sh [--yes] (--session [FILE] [--kind worker|heldout|batch|all] | --force-spend-safety --ids ID...)" >&2
  exit 2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --yes) yes=1; shift ;;
    --force-spend-safety) force_spend_safety=1; shift ;;
    --kind) kind=${2:?}; shift 2 ;;
    --ids)
      [ -z "$mode" ] || usage
      mode=ids; shift
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do ids+=("$1"); shift; done
      ;;
    --session)
      [ -z "$mode" ] || usage
      mode=session; shift
      if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then session_file=$1; shift; fi
      ;;
    *) usage ;;
  esac
done
[ -n "$mode" ] || usage
[[ "$kind" =~ ^(worker|heldout|batch|all)$ ]] || usage
[ "$mode" = session ] || [ "$kind" = all ] || usage
if [ "$mode" = ids ] && [ "$force_spend_safety" -ne 1 ]; then
  echo "explicit --ids is emergency cleanup only; pass --force-spend-safety and acknowledge evidence may be lost" >&2
  exit 2
fi
if [ "$mode" = session ] && [ "$force_spend_safety" -eq 1 ]; then
  echo "--force-spend-safety is not valid for verified session cleanup" >&2
  exit 2
fi

check_dotenv_security() {
  local file=$1 owner mode_bits
  [ -e "$file" ] || return 0
  [ ! -L "$file" ] || { echo "refusing symlink dotenv: $file" >&2; exit 2; }
  [ -f "$file" ] || { echo "dotenv is not a regular file: $file" >&2; exit 2; }
  owner=$(stat -c '%u' "$file")
  [ "$owner" = "$(id -u)" ] || { echo "dotenv is not owned by current user: $file" >&2; exit 2; }
  mode_bits=$(stat -c '%a' "$file")
  [ "$mode_bits" = 600 ] || {
    echo "dotenv must have exact mode 0600: $file" >&2
    exit 2
  }
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    ! git ls-files --error-unmatch -- "$file" >/dev/null 2>&1 || {
      echo "dotenv is tracked by git: $file" >&2
      exit 2
    }
    git check-ignore -q -- "$file" || {
      echo "dotenv is not ignored by git: $file" >&2
      exit 2
    }
  fi
}

if [ "$mode" = session ]; then
  [ -f "$session_file" ] || { echo "session not found: $session_file" >&2; exit 2; }
  task_name=$(jq -r '.task_name // empty' "$session_file")
  session_id=$(jq -r '.session_id // empty' "$session_file")
  [[ "$task_name" =~ ^[a-z0-9][a-z0-9-]{0,47}$ ]] \
    || { echo "invalid task_name in session" >&2; exit 2; }
  [[ "$session_id" =~ ^[0-9a-f]{32}$ ]] \
    || { echo "invalid session_id in session" >&2; exit 2; }
  expected_owner="arch-${task_name}-s${session_id:0:12}"
  case "$kind" in
    worker) selector='.worker_pod_ids // []' ;;
    heldout) selector='.heldout_pod_ids // []' ;;
    batch) selector='.batch_pod_ids // []' ;;
    all) selector='[(.worker_pod_ids // [])[], (.heldout_pod_ids // [])[], (.batch_pod_ids // [])[]] | unique' ;;
  esac
  mapfile -t ids < <(jq -r "($selector)[]" "$session_file")
  for id in "${ids[@]}"; do
    registry=${POD_WATCH_DIR:-$HOME/.codex/pod-watch}/owned/$id.json
    [ -f "$registry" ] \
      && [ "$(jq -r '.owner // empty' "$registry")" = "$expected_owner" ] || {
      echo "refusing deletion: exact pod $id is not registered to $expected_owner" >&2
      exit 3
    }
    jq -e --arg id "$id" '
      .durability.pods[$id] as $d
      | ($d.verified == true)
      and (($d.remote_uri // "") | length > 0)
      and (($d.manifest_sha256 // "") | test("^[0-9a-fA-F]{64}$"))
      and (($d.verified_epoch // 0) | type == "number" and . > 0)
      and (($d.files // []) | length > 0)
      and all($d.files[];
          ((.path // "") | length > 0)
          and ((.size_bytes // -1) | type == "number" and . >= 0)
          and ((.sha256 // "") | test("^[0-9a-fA-F]{64}$")))
    ' "$session_file" >/dev/null || {
      echo "refusing deletion: durability manifest is absent or invalid for exact pod $id" >&2
      exit 3
    }
  done
fi

[ "${#ids[@]}" -gt 0 ] || { echo "No exact pod IDs selected."; exit 0; }
check_dotenv_security "$DOTENV_FILE"
# The pod-injected key on crab-factory-2 is invalid for host management. Never
# let ambient RUNPOD_API_KEY win: prefer the secured task dotenv, then host config.
unset RUNPOD_API_KEY 2>/dev/null || true
if [ -f "$DOTENV_FILE" ]; then
  RUNPOD_API_KEY=$(python3 "$SCRIPT_DIR/read_dotenv.py" "$DOTENV_FILE" RUNPOD_API_KEY 2>/dev/null || true)
fi
if [ -z "${RUNPOD_API_KEY:-}" ]; then
  runpod_config=${RUNPOD_CONFIG_FILE:-$HOME/.runpod/config.toml}
  if [ -f "$runpod_config" ]; then
    RUNPOD_API_KEY=$(python3 "$SCRIPT_DIR/read_dotenv.py" "$runpod_config" apikey 2>/dev/null || true)
  fi
fi
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY unavailable in secured dotenv or host config}"
base=https://rest.runpod.io/v1
auth="Authorization: Bearer ${RUNPOD_API_KEY}"
printf 'Exact pods selected for deletion:\n'
printf '  %s\n' "${ids[@]}"
if [ "$mode" = ids ]; then
  echo "WARNING: force-spend-safety cleanup bypasses durable-manifest verification; evidence may be lost." >&2
fi
if [ "$yes" -ne 1 ]; then
  read -r -p "Delete exactly these ${#ids[@]} pods? [y/N] " answer
  [ "$answer" = y ] || { echo "Aborted."; exit 1; }
fi

failed=0
owner=/workspace/.codex/skills/runpod-spinup/pod-own.sh
state_script="$SCRIPT_DIR/session_state.py"
[ -f "$state_script" ] || state_script="$SCRIPT_DIR/../../arch-init/scripts/session_state.py"
journal_delete() {
  local id=$1 status=$2 epoch
  [ -f "$session_file" ] && [ -f "$state_script" ] || return 0
  epoch=$(date -u +%s)
  jq -cn --arg id "$id" --arg status "$status" --argjson epoch "$epoch" '
    {id: ("pod-delete:" + $id + ":" + ($epoch|tostring) + ":" + $status),
     idempotency_key: ("pod-delete:" + $id), type: "pod_delete",
     exact_target_id: $id, status: $status, timestamp_epoch: $epoch}
  ' | python3 "$state_script" --file "$session_file" append-action -
}
for id in "${ids[@]}"; do
  [[ "$id" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "BAD pod id: $id" >&2; failed=1; continue; }
  journal_delete "$id" pending
  code=$(curl -sS -o /dev/null -w '%{http_code}' -X DELETE -H "$auth" "$base/pods/$id" || echo 000)
  if [[ "$code" =~ ^2 ]] || [ "$code" = 404 ]; then
    echo "deleted $id (HTTP $code)"
    [ -x "$owner" ] && "$owner" remove "$id" >/dev/null 2>&1 || true
    if [ -f "$session_file" ] && [ -f "$state_script" ]; then
      python3 "$state_script" --file "$session_file" remove-pod "$id" --kind all
    fi
    journal_delete "$id" complete
  else
    echo "FAILED $id (HTTP $code)" >&2
    journal_delete "$id" failed
    failed=1
  fi
done
exit "$failed"
