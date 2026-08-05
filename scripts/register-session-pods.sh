#!/usr/bin/env bash
# Pre-create ownership/deadline barrier for one exact ARCH task.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
interval=5
once=0
barrier=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --interval) interval=${2:?}; shift 2 ;;
    --once) once=1; shift ;;
    --barrier) barrier=1; shift ;;
    *) echo "usage: register-session-pods.sh [--barrier] [--interval SECONDS] [--once]" >&2; exit 2 ;;
  esac
done
[[ "$interval" =~ ^[1-9][0-9]*$ ]] || { echo "invalid interval" >&2; exit 2; }
[ "$once" -eq 0 ] || [ "$barrier" -eq 0 ] || {
  echo "--once and --barrier are mutually exclusive" >&2
  exit 2
}

mapfile -t session_identity < <(python3 - <<'PY'
import pathlib
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
with pathlib.Path('.arch/config.toml').open('rb') as handle:
    config = tomllib.load(handle)
print(config['task_name'])
print(config['session_id'])
PY
)
task_name=${session_identity[0]:-}
session_id=${session_identity[1]:-}
[[ "$task_name" =~ ^[a-z0-9][a-z0-9-]{0,47}$ ]] || { echo "invalid task_name" >&2; exit 2; }
[[ "$session_id" =~ ^[0-9a-f]{32}$ ]] || { echo "invalid session_id" >&2; exit 2; }
session_tag=${session_id:0:12}

owner_script=/workspace/.codex/skills/runpod-spinup/pod-own.sh
watcher_script=/workspace/.codex/skills/runpod-spinup/pod-watch.sh
cleanup_script="$SCRIPT_DIR/cleanup-pods.sh"
session_file=.arch/.session.json
state_script="$SCRIPT_DIR/session_state.py"
owner=arch-$task_name-s$session_tag
heartbeat_file=.arch/.watcher-heartbeat.json
[ -f "$session_file" ] || { echo "session state unavailable: $session_file" >&2; exit 2; }
python3 - "$session_file" "$task_name" "$session_id" <<'PY'
import json
import pathlib
import sys

state = json.loads(pathlib.Path(sys.argv[1]).read_text())
if state.get("task_name") != sys.argv[2] or state.get("session_id") != sys.argv[3]:
    raise SystemExit("session/config identity mismatch")
PY
[ -x "$owner_script" ] || { echo "pod ownership helper unavailable" >&2; exit 2; }
[ -f "$state_script" ] || { echo "canonical session state helper unavailable" >&2; exit 2; }
if [ "$barrier" -eq 1 ]; then
  [ -x "$watcher_script" ] || { echo "pod watcher unavailable" >&2; exit 2; }
  [ -x "$cleanup_script" ] || { echo "exact-ID cleanup helper unavailable" >&2; exit 2; }
fi

record_typed_pod() {
  local id=$1 kind=$2
  python3 "$state_script" --file "$session_file" add-pod "$id" --kind "$kind"
}

watcher_pid=""
write_heartbeat() {
  local armed=$1
  python3 - "$heartbeat_file" "$task_name" "$session_id" "$owner" "$armed" "${watcher_pid:-}" "$$" <<'PY'
import json
import os
import pathlib
import tempfile
import time
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "task_name": sys.argv[2],
    "session_id": sys.argv[3],
    "owner": sys.argv[4],
    "armed": sys.argv[5] == "true",
    "updated_epoch": int(time.time()),
    "heartbeat_epoch": int(time.time()),
    "registrar_pid": int(sys.argv[7]),
    "watcher_pid": int(sys.argv[6]) if sys.argv[6] else None,
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp_name, 0o600)
    os.replace(tmp_name, path)
finally:
    if os.path.exists(tmp_name):
        os.unlink(tmp_name)
PY
}

stop_watcher() {
  [ "$barrier" -eq 0 ] || write_heartbeat false 2>/dev/null || true
  if [ -n "$watcher_pid" ] && kill -0 "$watcher_pid" 2>/dev/null; then
    kill "$watcher_pid" 2>/dev/null || true
    wait "$watcher_pid" 2>/dev/null || true
  fi
}
trap stop_watcher EXIT

start_watcher_if_needed() {
  [ "$barrier" -eq 1 ] || return 0
  [ -z "$watcher_pid" ] || return 0
  "$owner_script" list "$owner" | grep -q . || return 0
  echo "starting host pod watcher for owner=$owner"
  POLL_SECS=${POD_WATCH_POLL_SECS:-120} "$watcher_script" "$owner" &
  watcher_pid=$!
}

check_watcher() {
  [ -n "$watcher_pid" ] || return 0
  if ! kill -0 "$watcher_pid" 2>/dev/null; then
    wait "$watcher_pid" || true
    echo "POD WATCHER EXITED: inspect the alert, then restart this barrier before continuing." >&2
    exit 3
  fi
}

deadline_control() {
  [ "$barrier" -eq 1 ] && [ -f "$session_file" ] || return 0
  local deadline now
  deadline=$(jq -r '.deadline_epoch // empty' "$session_file")
  [ -n "$deadline" ] || return 0
  [[ "$deadline" =~ ^[0-9]+$ ]] || { echo "invalid session deadline_epoch" >&2; exit 3; }
  now=$(date -u +%s)
  [ "$now" -ge "$deadline" ] || return 0

  mapfile -t worker_ids < <(jq -r '(.worker_pod_ids // [])[]' "$session_file")
  if [ "${#worker_ids[@]}" -eq 0 ]; then
    echo "DEADLINE REACHED: no typed worker IDs remain; resume arch-wrapup." >&2
    exit 4
  fi

  if ! jq -e '
      . as $s
      | ($s.worker_pod_ids // []) as $ids
      | ($s.authorizations.terminate_workers == true)
      and ($ids | length > 0)
      and all($ids[];
          . as $id
          | ($s.durability.pods[$id].verified == true)
          and (($s.durability.pods[$id].remote_uri // "") | length > 0)
          and (($s.durability.pods[$id].manifest_sha256 // "") | test("^[0-9a-fA-F]{64}$"))
          and (($s.durability.pods[$id].verified_epoch // 0) | type == "number")
          and (($s.durability.pods[$id].files // []) | length > 0)
          and all($s.durability.pods[$id].files[];
              ((.path // "") | length > 0)
              and ((.size_bytes // -1) >= 0)
              and ((.sha256 // "") | test("^[0-9a-fA-F]{64}$"))))
    ' "$session_file" >/dev/null; then
    echo "DEADLINE REACHED: refusing automatic deletion; termination authorization or per-pod remote manifest verification is incomplete." >&2
    echo "Workers remain registered and billing. Resume arch-wrapup with the exact IDs above." >&2
    printf '  %s\n' "${worker_ids[@]}" >&2
    exit 4
  fi

  echo "DEADLINE REACHED: durable manifests verified for all typed workers; deleting exact authorized IDs."
  "$cleanup_script" --yes --session "$session_file" --kind worker
  echo "Exact worker cleanup complete; resume arch-wrapup for held-out/batch pods and shipping."
  exit 0
}

if [ "$barrier" -eq 1 ]; then
  write_heartbeat true
  echo "ARCH POD BARRIER READY task=$task_name owner=$owner interval=${interval}s heartbeat=$heartbeat_file"
fi

while :; do
  [ "$barrier" -eq 0 ] || write_heartbeat true
  # Never let the invalid pod-injected environment key override host config.
  pods=$(unset RUNPOD_API_KEY; runpodctl pod list -o json)
  mapfile -t rows < <(jq -r --arg task "$task_name" --arg tag "$session_tag" '
    (if type == "array" then . else (.pods // .data // []) end)[]
    | (.name // "") as $name
    | (if ($name | test("^arch-" + $task + "-s" + $tag + "-worker-[0-9]+$")) then "worker"
      elif ($name | test("^arch-" + $task + "-s" + $tag + "-heldout-pr[0-9]+-[0-9a-fA-F]{12}-gha-[0-9]+-[0-9]+$")) then "heldout"
      elif ($name | test("^arch-" + $task + "-s" + $tag + "-batch-eval-[0-9]+$")) then "batch"
      else empty end) as $kind
    | [.id, $name, $kind] | @tsv
  ' <<<"$pods")
  for row in "${rows[@]}"; do
    IFS=$'\t' read -r id name kind <<<"$row"
    [[ "$id" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "invalid pod id from inventory: $id" >&2; exit 3; }
    registry=${POD_WATCH_DIR:-$HOME/.codex/pod-watch}/owned/$id.json
    registered_owner=$([ -f "$registry" ] && jq -r '.owner // empty' "$registry" || true)
    if [ "$registered_owner" != "$owner" ]; then
      "$owner_script" add "$id" "$owner"
      echo "registered task pod: kind=$kind name=$name id=$id owner=$owner"
    fi
    record_typed_pod "$id" "$kind"
  done
  start_watcher_if_needed
  check_watcher
  deadline_control
  [ "$once" -eq 1 ] && exit 0
  sleep "$interval"
done
