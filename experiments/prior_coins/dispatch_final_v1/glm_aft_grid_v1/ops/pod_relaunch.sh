#!/usr/bin/env bash
# Runs ON THE POD (shipped by ops/launch.py relaunch): stop the previous worker
# session, re-extract the shipped bundle, re-snapshot git, restart setup.sh.
# The Hub token arrives on stdin.  Lives in a file so no ssh command line ever
# contains the process patterns it kills (a self-match dropped the session once).
#   bash /workspace/glm_grid_relaunch.sh <worker> <prepared> <root> <setup.sh path>
set -euo pipefail
WORKER=${1:?}; PREPARED=${2:?}; ROOT=${3:?}; SETUP=${4:?}
set -a; source /etc/rp_environment 2>/dev/null || true; set +a
umask 077; cat > /root/.hf_token; umask 022
export PATH=/root/.local/bin:$PATH
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ME=$$; PARENT=$PPID
for pid in $(pgrep -f "[s]etup.sh $WORKER" || true); do
  [[ "$pid" == "$ME" || "$pid" == "$PARENT" ]] && continue
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
done
sleep 5
for pat in '[g]lm_aft_grid_v1.run --worker' '[d]ispatch_final_v1/pod/setup.sh' '[u]v pip install'; do
  for pid in $(pgrep -f "$pat" || true); do
    [[ "$pid" == "$ME" || "$pid" == "$PARENT" ]] && continue
    kill -TERM "$pid" 2>/dev/null || true
  done
done
sleep 3
for pid in $(pgrep -f "[s]etup.sh $WORKER" || true); do
  [[ "$pid" == "$ME" || "$pid" == "$PARENT" ]] && continue
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
done
echo "remaining: $(pgrep -fa '[s]etup.sh|[g]lm_aft_grid_v1' | cut -c1-90 || echo none)"
if [[ -d "$ROOT/$WORKER/cells" ]]; then echo "REFUSE: cells exist under $ROOT/$WORKER"; exit 5; fi
rm -f "$ROOT/$WORKER/WORKER.json" "$ROOT/$WORKER/IDENTITY.json" "$ROOT/$WORKER/STATUS.json" "$ROOT/$WORKER/runner.lock"
tar -xzf /workspace/base-code.tar.gz -C /workspace/scimt
tar -xzf /workspace/code-overlay.tar.gz -C /workspace/scimt
cd /workspace/scimt && git add -A && (git -c user.name='GLM deployment snapshot' -c user.email=deployment@localhost \
  commit -qm "relaunch $STAMP: updated overlay/prepared inputs" || true) && echo "GIT_HEAD=$(git rev-parse HEAD)"
rm -rf "$PREPARED" && mkdir -p "$PREPARED" && tar -xzf /workspace/prepared-inputs.tar.gz -C "$PREPARED"
mv -f /workspace/glm-grid-worker.log "/workspace/glm-grid-worker.log.$STAMP" 2>/dev/null || true
cp -f "$PREPARED/READY.json" /workspace/GLM_GRID_LAUNCHED.json
nohup setsid bash "$SETUP" "$WORKER" "$PREPARED" "$ROOT" > /workspace/glm-grid-worker.log 2>&1 < /dev/null &
echo "LAUNCHED pid $!"
