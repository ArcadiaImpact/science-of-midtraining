#!/usr/bin/env bash
# Keep overnight.py alive. It died silently at ~12:31Z on 2026-09-11 after 13 h
# up -- no traceback (so not an uncaught exception: the cycle loop catches those)
# and no OOM line in dmesg, i.e. an external signal. Cause unknown, so guard
# against recurrence rather than try to prevent it.
#
# overnight.py is idempotent -- all its state is state.json, which it reloads
# every cycle -- so a restart costs nothing but the gap.
#
# Run detached:  setsid nohup bash supervise.sh >> supervise.log 2>&1 &
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
log() { echo "$(date -u +%FT%TZ) $*"; }

log "supervisor up, watching overnight.py (check every 60s)"
while :; do
  pid=$(cat overnight.pid 2>/dev/null || echo 0)
  if ! { [ "${pid:-0}" -gt 0 ] 2>/dev/null && kill -0 "$pid" 2>/dev/null; }; then
    log "ALERT: overnight.py (pid ${pid:-none}) is not running -- restarting"
    setsid nohup python3 overnight.py >> daemon.log 2>&1 < /dev/null &
    sleep 10
    newpid=$(cat overnight.pid 2>/dev/null || echo 0)
    if [ "${newpid:-0}" -gt 0 ] && kill -0 "$newpid" 2>/dev/null; then
      log "restarted as pid $newpid"
    else
      log "ALERT: restart FAILED; see daemon.log. Retrying next cycle."
    fi
  fi
  sleep 60
done
