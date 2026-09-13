#!/usr/bin/env bash
# dispatch_queue.sh -- feed a queue of launcher configs to a fixed set of pods as each one frees up.
#
#   ./dispatch_queue.sh <queue.tsv> <pods.tsv> [poll-seconds=120] [--dry-run]
#
# queue.tsv  one pending launch per line, whitespace-separated; '#' comments and blank lines ignored:
#              <launcher> <config> [<pre-hook> <hook-args...>]
#            e.g.  lowdose/launch_lowdose_pod.sh lowdose/configs/27b-w06.json
#                  launch_halfpct_pod.sh configs/27b-r03.json restore_archived_worker.sh \
#                    /workspace/midtrain-token-budget-heatmaps/lowdose-archive/gemma-halfpct-rerun1__A2-27b-half03.tar \
#                    /workspace/gemma-halfpct-rerun1
#            Relative launcher/config/hook paths resolve against this script's directory (handoff/),
#            then against the current directory.  THE FILE IS THE PENDING STATE: a line is removed
#            (atomic rewrite) once its launch succeeded, so you may add, remove or reorder lines
#            while the dispatcher runs -- it re-reads the file every poll.  Consumed lines are logged.
# pods.tsv   <name> <pod-id> <port> <ip> <family>      family in {12b,27b}; re-read every poll too.
#
# Eligibility: a line is eligible for a pod when its config's "model" equals the pod's family.
# Lines WITH a pre-hook are additionally eligible only for pods that already ran the same campaign,
# checked over ssh: the campaign's prepared plan (/workspace/<version>-prepared/plan.json -- the
# config's prepared_dir -- for a config with a "version" field, /workspace/gemma-halfpct-prepared/
# plan.json otherwise) AND the campaign's wrapper file inside /workspace/scimt/experiments/
# prior_coins/dispatch_final_v1/.  Reason: the hook must run after the code is deployed and before
# the worker starts.  On a same-campaign pod that order holds (hook, then resume-launch); on a pod
# that ran the other campaign the resume-launch would deploy first, so the line is skipped for that
# pod and the next eligible line is tried.
#
# Pre-hook convention: run on this box as    <hook> <hook-args...> <port> <ip>
# (the pod's ssh port and ip are APPENDED to the args given in queue.tsv).  restore_archived_worker.sh
# therefore takes <archive.tar> <root-parent-dir> <port> <ip>; for that hook the dispatcher also checks,
# before touching any pod, that <root-parent-dir>/<archive top-level dir> is one of the config's run
# roots (otherwise the relaunched worker would not find the restored cell).
#
# Loop (every poll-seconds): read /workspace/BOOTSTRAP_STATUS.json on every pod over ssh.
#   ALL_COMPLETE  -> the pod is free, unless the marker is the very one seen before our last dispatch
#                    to it (the new run has not started yet; warns after 20 min).  Pop the first
#                    eligible line: run its pre-hook, then `<launcher> <config> <pod-id>` (resume mode
#                    on the existing pod), record the assignment, print one event line.  No eligible
#                    line -> print "idle: <pod>" once so you can stop the pod.
#   FAILED        -> print one alert; never dispatch to this pod again (nothing is repaired).
#   ssh failure   -> print once, keep polling (pods may be stopped on purpose).
#   hook or launcher exit != 0 -> alert; the line moves to failed.tsv and the pod goes on hold
#                    (pods/<pod-id>.hold in the state dir; delete that file to re-enable the pod).
# Never creates, stops or deletes pods; never deletes anything on a pod.
# Stop with `kill -TERM $(cat <state-dir>/dispatch.pid)`: an in-flight hook/launch is finished first
# (a restore transfer is never cut in half), then the loop exits.  Ctrl-C also interrupts the child.
# --dry-run: validate both files, do ONE read-only pass, print what would run, change nothing but
#            its own log.
#
# State/log dir (override: DISPATCH_STATE_DIR) /workspace/midtrain-token-budget-heatmaps/lowdose-logs/dispatch/
#   dispatch-<start>.log     everything printed         assignments.tsv    append-only event log
#   failed.tsv               lines whose dispatch failed  pods/<pod-id>.dispatched  pods/<pod-id>.hold
#   dispatch.lock / .pid     one running instance
# Test hooks: DISPATCH_SSH=<ssh replacement>, DISPATCH_SSH_KEY=<key>, DISPATCH_STALE_AFTER=<seconds>.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
usage() { echo "usage: $0 <queue.tsv> <pods.tsv> [poll-seconds=120] [--dry-run]   (see the header of $0)" >&2; exit 2; }
DRY_RUN=0; POS=()
for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage ;;
    *) POS+=("$arg") ;;
  esac
done
(( ${#POS[@]} >= 2 && ${#POS[@]} <= 3 )) || usage
QUEUE=$(readlink -f "${POS[0]}"); PODS=$(readlink -f "${POS[1]}"); POLL=${POS[2]:-120}
[[ -f "$QUEUE" ]] || { echo "queue file not found: ${POS[0]}" >&2; exit 2; }
[[ -f "$PODS" ]] || { echo "pods file not found: ${POS[1]}" >&2; exit 2; }
[[ "$POLL" =~ ^[0-9]+$ && "$POLL" -ge 1 ]] || { echo "poll-seconds must be a positive integer" >&2; exit 2; }

STATE_DIR=${DISPATCH_STATE_DIR:-/workspace/midtrain-token-budget-heatmaps/lowdose-logs/dispatch}
mkdir -p "$STATE_DIR/pods" || exit 2
START=$(date -u +%Y%m%dT%H%M%SZ)
if (( DRY_RUN )); then LOG=$STATE_DIR/dispatch-$START-dryrun.log; else LOG=$STATE_DIR/dispatch-$START.log; fi
ASSIGN=$STATE_DIR/assignments.tsv; FAILED=$STATE_DIR/failed.tsv
SSH_BIN=${DISPATCH_SSH:-ssh}
KEY=${DISPATCH_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}
SSHOPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o BatchMode=yes -i "$KEY")
STATUS_CMD='python3 -c '"'"'import json; d=json.load(open("/workspace/BOOTSTRAP_STATUS.json")); print(d.get("status","?"), d.get("at",0))'"'"' 2>/dev/null || echo NO_STATUS 0'
STALE_AFTER=${DISPATCH_STALE_AFTER:-1200}   # seconds after a dispatch before "pod still shows the old ALL_COMPLETE" is worth a warning
STOP=0

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(ts)" "$*" | tee -a "$LOG"; }
event() { printf '%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$1" "$2" "$3" "$4" >> "$ASSIGN"; }   # <ts> <event> <pod-name> <pod-id> <detail>
announce_once() { local -n _seen=$1; [[ -n "${_seen[$2]:-}" ]] && return 0; _seen[$2]=1; log "$3"; }
remote() { timeout 90 "$SSH_BIN" "${SSHOPTS[@]}" -p "$1" "root@$2" "$3"; }   # remote <port> <ip> <read-only command>

resolve() {  # resolve <path>: absolute path if it exists (script dir first, then cwd); empty otherwise
  local p=$1
  if [[ "$p" == /* ]]; then [[ -e "$p" ]] && echo "$p"; return 0; fi
  if [[ -e "$HERE/$p" ]]; then echo "$HERE/$p"; return 0; fi
  [[ -e "$p" ]] && readlink -f "$p"; return 0
}

config_info() {  # config_info <config.json> -> model|version|campaign plan marker|wrapper file|run roots (space-joined)
  python3 - "$1" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
version = c.get('version') or ''
if version:
    marker = (c.get('prepared_dir') or f'/workspace/{version}-prepared') + '/plan.json'
    wrapper = c.get('wrapper_file') or 'gemma_lowdose_handoff.py'
else:
    marker, wrapper = '/workspace/gemma-halfpct-prepared/plan.json', 'gemma_halfpct_handoff.py'
roots = ' '.join(r.get('root', '') for r in c.get('runs') or [])
print('|'.join([str(c.get('model') or ''), version, marker, wrapper, roots]))  # '|' never appears in these fields; tab would collapse the empty version
PY
}

# validate_line <queue line>: sets L_LAUNCHER L_CONFIG L_MODEL L_VERSION L_MARKER L_WRAPPER L_ROOTS L_HOOK L_HOOKARGS[];
# on failure sets L_ERROR and returns 1.
validate_line() {
  local -a f; read -r -a f <<< "$1"
  L_ERROR=""; L_HOOK=""; L_HOOKARGS=()
  (( ${#f[@]} >= 2 )) || { L_ERROR="needs <launcher> <config>"; return 1; }
  L_LAUNCHER=$(resolve "${f[0]}"); [[ -n "$L_LAUNCHER" && -x "$L_LAUNCHER" ]] || { L_ERROR="launcher not found or not executable: ${f[0]}"; return 1; }
  L_CONFIG=$(resolve "${f[1]}"); [[ -n "$L_CONFIG" && -f "$L_CONFIG" ]] || { L_ERROR="config not found: ${f[1]}"; return 1; }
  IFS='|' read -r L_MODEL L_VERSION L_MARKER L_WRAPPER L_ROOTS < <(config_info "$L_CONFIG" 2>/dev/null) || { L_ERROR="config unreadable: $L_CONFIG"; return 1; }
  [[ "$L_MODEL" == 12b || "$L_MODEL" == 27b ]] || { L_ERROR="config model must be 12b or 27b, got '$L_MODEL'"; return 1; }
  (( ${#f[@]} == 2 )) && return 0
  L_HOOK=$(resolve "${f[2]}"); [[ -n "$L_HOOK" && -x "$L_HOOK" ]] || { L_ERROR="pre-hook not found or not executable: ${f[2]}"; return 1; }
  L_HOOKARGS=("${f[@]:3}")
  if [[ $(basename "$L_HOOK") == restore_archived_worker.sh ]]; then
    (( ${#L_HOOKARGS[@]} == 2 )) || { L_ERROR="restore_archived_worker.sh takes <archive.tar> <root-parent-dir> here (port and ip are appended)"; return 1; }
    [[ -f "${L_HOOKARGS[0]}" ]] || { L_ERROR="archive not found: ${L_HOOKARGS[0]}"; return 1; }
    [[ "${L_HOOKARGS[1]}" == /workspace/* ]] || { L_ERROR="root-parent-dir must be an absolute /workspace path: ${L_HOOKARGS[1]}"; return 1; }
    local top; top=$(tar tf "${L_HOOKARGS[0]}" 2>/dev/null | head -1 | cut -d/ -f1)
    [[ -n "$top" ]] || { L_ERROR="cannot list archive: ${L_HOOKARGS[0]}"; return 1; }
    local restored="${L_HOOKARGS[1]%/}/$top"
    [[ " $L_ROOTS " == *" $restored "* ]] || { L_ERROR="archive restores $restored but the config's run roots are: ${L_ROOTS:-none}"; return 1; }
  fi
  return 0
}

validate_pod_line() {  # validate_pod_line <pods line>: sets P_NAME P_ID P_PORT P_IP P_FAMILY or P_ERROR
  local extra; P_ERROR=""
  read -r P_NAME P_ID P_PORT P_IP P_FAMILY extra <<< "$1"
  [[ -n "${P_FAMILY:-}" ]] || { P_ERROR="needs <name> <pod-id> <port> <ip> <family>"; return 1; }
  [[ "$P_PORT" =~ ^[0-9]+$ ]] || { P_ERROR="port must be numeric: $P_PORT"; return 1; }
  [[ "$P_FAMILY" == 12b || "$P_FAMILY" == 27b ]] || { P_ERROR="family must be 12b or 27b: $P_FAMILY"; return 1; }
  return 0
}

pop_line() {  # remove the first exact occurrence of a line from the queue file (atomic rewrite)
  python3 - "$QUEUE" "$1" <<'PY'
import os, sys
path, target = sys.argv[1], sys.argv[2]
text = open(path).read()
lines = text.split('\n')
for i, line in enumerate(lines):
    if line == target or line.strip() == target.strip():
        del lines[i]
        break
else:
    sys.exit(f'queue line already gone: {target}')
tmp = path + '.tmp'
open(tmp, 'w').write('\n'.join(lines))
os.replace(tmp, path)
PY
}

fail_dispatch() {  # fail_dispatch <name> <id> <line> <why>
  local name=$1 id=$2 line=$3 why=$4
  printf '%s %s: %s\n' "$(ts)" "$why" "$line" > "$STATE_DIR/pods/$id.hold"
  pop_line "$line" || true
  printf '%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$name" "$id" "$why" "$line" >> "$FAILED"
  event FAILED_DISPATCH "$name" "$id" "$why: $line"
  log "ALERT: $why on $name ($id); line moved to $FAILED; pod on hold until $STATE_DIR/pods/$id.hold is deleted"
}

dispatch() {  # dispatch <name> <id> <port> <ip> <at> <line>   (uses L_* from validate_line)
  local name=$1 id=$2 port=$3 ip=$4 at=$5 line=$6
  local -a hookcmd=() launch=("$L_LAUNCHER" "$L_CONFIG" "$id")
  [[ -n "$L_HOOK" ]] && hookcmd=("$L_HOOK" "${L_HOOKARGS[@]}" "$port" "$ip")
  if (( DRY_RUN )); then
    if (( ${#hookcmd[@]} )); then log "DRY-RUN $name ($id) would run pre-hook: ${hookcmd[*]}"; fi
    log "DRY-RUN $name ($id) would launch: ${launch[*]}"
    return 0
  fi
  log "dispatch: $name ($id) <- $line"
  if (( ${#hookcmd[@]} )); then
    log "pre-hook: ${hookcmd[*]}"
    if ! "${hookcmd[@]}" 2>&1 | tee -a "$LOG"; then fail_dispatch "$name" "$id" "$line" "pre-hook failed"; return 1; fi
  fi
  log "launch: ${launch[*]}"
  if ! "${launch[@]}" 2>&1 | tee -a "$LOG"; then fail_dispatch "$name" "$id" "$line" "launcher failed"; return 1; fi
  printf 'epoch=%s\nseen_at=%s\nline=%s\n' "$(date +%s)" "$at" "$line" > "$STATE_DIR/pods/$id.dispatched"
  pop_line "$line" || log "WARNING: could not remove the dispatched line from $QUEUE: $line"
  event DISPATCHED "$name" "$id" "$line"
  log "dispatched: $name ($id) <- $line"
}

declare -A LAST_STATUS UNREACHABLE FAILED_ALERTED IDLE_ANNOUNCED HOLD_ANNOUNCED STALE_WARNED BAD_LINE SKIPPED
pass() {
  local -a pod_lines qlines; local -A consumed=()
  mapfile -t pod_lines < <(grep -Ev '^[[:space:]]*(#|$)' "$PODS")
  mapfile -t qlines < <(grep -Ev '^[[:space:]]*(#|$)' "$QUEUE")
  local pl line out status at disp seen_at disp_epoch chosen
  for pl in "${pod_lines[@]}"; do
    (( STOP )) && return 0
    validate_pod_line "$pl" || { announce_once BAD_LINE "$pl" "skipping pods line ($P_ERROR): $pl"; continue; }
    local name=$P_NAME id=$P_ID port=$P_PORT ip=$P_IP family=$P_FAMILY
    if [[ -f "$STATE_DIR/pods/$id.hold" ]]; then
      announce_once HOLD_ANNOUNCED "$id" "hold: $name ($id): $(cat "$STATE_DIR/pods/$id.hold") -- delete $STATE_DIR/pods/$id.hold to re-enable"
      continue
    fi
    unset "HOLD_ANNOUNCED[$id]"
    if ! out=$(remote "$port" "$ip" "$STATUS_CMD" 2>/dev/null); then
      announce_once UNREACHABLE "$id" "unreachable: $name ($id) root@$ip:$port -- will keep polling"
      continue
    fi
    if [[ -n "${UNREACHABLE[$id]:-}" ]]; then unset "UNREACHABLE[$id]"; log "reachable again: $name ($id)"; fi
    read -r status at <<< "$out"; status=${status:-?}; at=${at:-0}
    if [[ "${LAST_STATUS[$id]:-}" != "$status" ]]; then log "status: $name ($id) ${LAST_STATUS[$id]:-?} -> $status"; LAST_STATUS[$id]=$status; fi
    case $status in
      FAILED)
        if [[ -z "${FAILED_ALERTED[$id]:-}" ]]; then
          FAILED_ALERTED[$id]=1
          log "ALERT: $name ($id) bootstrap FAILED -- no further dispatch to it; look at /workspace/bootstrap.log on the pod"
          (( DRY_RUN )) || event POD_FAILED "$name" "$id" "bootstrap status FAILED"
        fi
        continue ;;
      ALL_COMPLETE) ;;
      *) unset "IDLE_ANNOUNCED[$id]" "STALE_WARNED[$id]"; continue ;;   # bootstrapping or running
    esac
    disp=$STATE_DIR/pods/$id.dispatched
    if [[ -f "$disp" ]]; then
      seen_at=$(sed -n 's/^seen_at=//p' "$disp"); disp_epoch=$(sed -n 's/^epoch=//p' "$disp")
      if [[ "$seen_at" == "$at" ]]; then   # still the marker we saw before dispatching: the new run has not started
        if (( $(date +%s) - ${disp_epoch:-0} > STALE_AFTER )); then
          announce_once STALE_WARNED "$id" "WARNING: $name ($id) still shows the pre-dispatch ALL_COMPLETE marker $(( $(date +%s) - disp_epoch ))s after dispatch; check $LOG and /workspace/bootstrap.log on the pod"
        fi
        continue
      fi
    fi
    chosen=""
    for line in "${qlines[@]}"; do
      [[ -n "${consumed[$line]:-}" ]] && continue
      validate_line "$line" || { announce_once BAD_LINE "$line" "skipping queue line ($L_ERROR): $line"; continue; }
      [[ "$L_MODEL" == "$family" ]] || continue
      if [[ -n "$L_HOOK" ]]; then
        if ! remote "$port" "$ip" "test -f '$L_MARKER' && test -f '/workspace/scimt/experiments/prior_coins/dispatch_final_v1/$L_WRAPPER'" >/dev/null 2>&1; then
          announce_once SKIPPED "$id|$line" "skip for $name ($id): pre-hook lines need a pod that already ran this campaign ($L_MARKER + $L_WRAPPER): $line"
          continue
        fi
      fi
      chosen=$line; break
    done
    if [[ -z "$chosen" ]]; then
      announce_once IDLE_ANNOUNCED "$id" "idle: $name ($id) -- ALL_COMPLETE and no eligible queue line; stop it if nothing more is coming"
      continue
    fi
    unset "IDLE_ANNOUNCED[$id]" "STALE_WARNED[$id]"
    consumed[$chosen]=1
    dispatch "$name" "$id" "$port" "$ip" "$at" "$chosen" || true
  done
}

# ---------------------------------------------------------------- start-up validation (hard failure)
errors=0
mapfile -t _lines < <(grep -Ev '^[[:space:]]*(#|$)' "$QUEUE")
for line in "${_lines[@]}"; do validate_line "$line" || { echo "queue: $L_ERROR: $line" >&2; errors=$((errors+1)); }; done
mapfile -t _pods < <(grep -Ev '^[[:space:]]*(#|$)' "$PODS")
for pl in "${_pods[@]}"; do validate_pod_line "$pl" || { echo "pods: $P_ERROR: $pl" >&2; errors=$((errors+1)); }; done
(( ${#_pods[@]} )) || { echo "pods: no pods listed" >&2; errors=$((errors+1)); }
(( errors == 0 )) || { echo "$errors problem(s); nothing dispatched" >&2; exit 2; }
if (( ! DRY_RUN )); then
  exec 9>"$STATE_DIR/dispatch.lock"
  flock -n 9 || { echo "another dispatcher holds $STATE_DIR/dispatch.lock (pid $(cat "$STATE_DIR/dispatch.pid" 2>/dev/null || echo ?))" >&2; exit 2; }
  echo $$ > "$STATE_DIR/dispatch.pid"
fi
trap 'STOP=1; log "signal received: stopping after the current step"' TERM INT
log "dispatch_queue start pid=$$ queue=$QUEUE (${#_lines[@]} lines) pods=$PODS (${#_pods[@]} pods) poll=${POLL}s state=$STATE_DIR dry_run=$DRY_RUN"
while :; do
  pass
  (( DRY_RUN )) && { log "dry run complete (one pass; nothing changed)"; break; }
  (( STOP )) && break
  sleep "$POLL" & SLEEP_PID=$!; wait "$SLEEP_PID" 2>/dev/null; kill "$SLEEP_PID" 2>/dev/null
  (( STOP )) && break
done
(( DRY_RUN )) || rm -f "$STATE_DIR/dispatch.pid"
log "dispatch_queue stopped"
exit 0
