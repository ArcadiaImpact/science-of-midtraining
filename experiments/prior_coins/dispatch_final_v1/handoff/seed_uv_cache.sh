#!/usr/bin/env bash
# Seed a pod's uv cache from another pod (8 parallel tar streams relayed through this
# box), verify the file count, then restart the handoff bootstrap on the destination
# with the ingress preflight bypassed.  Needed because the H200 hosts RunPod hands us
# have a broken route to the Fastly CDN behind PyPI (0.07-0.2 MB/s), while an H100
# pod that has finished pod/setup.sh holds every wheel the recipe needs (~14 GB).
#   ./seed_uv_cache.sh <src-port> <src-ip> <dst-port> <dst-ip>
set -uo pipefail
SP=$1; SIP=$2; DP=$3; DIP=$4
SSHOPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=30 -i $HOME/.runpod/ssh/runpodctl-ssh-key"
SRC="ssh $SSHOPTS -p $SP root@$SIP"; DST="ssh $SSHOPTS -p $DP root@$DIP"
echo "== seed start $(date -u +%FT%TZ): $SIP -> $DIP"
$DST 'rm -rf /root/.cache/uv && mkdir -p /root/.cache'
$SRC 'cd /root/.cache && { find uv -mindepth 1 -maxdepth 1 -type f; find uv -mindepth 2 -maxdepth 2; } > /tmp/uvall.txt && rm -f /tmp/uvlist.* && awk "{print > \"/tmp/uvlist.\" (NR % 8)}" /tmp/uvall.txt && wc -l < /tmp/uvall.txt'
for i in 0 1 2 3 4 5 6 7; do
  ( $SRC "tar cf - -C /root/.cache -T /tmp/uvlist.$i" | $DST 'tar xf - -C /root/.cache'; echo "stream $i rc=${PIPESTATUS[*]} $(date -u +%T)" ) &
done
wait
S=$($SRC 'find /root/.cache/uv -type f | wc -l'); D=$($DST 'find /root/.cache/uv -type f | wc -l; du -sh /root/.cache/uv | cut -f1' | tr '\n' ' ')
echo "files src=$S dst=$D"
if [[ "$S" == "${D%% *}" ]]; then
  # Which bootstrap to restart: default the 0.5% one; lowdose pods pass BOOT_SCRIPT/BOOT_CONFIG.
  BOOT_SCRIPT=${BOOT_SCRIPT:-/root/handoff/bootstrap_halfpct_handoff.sh}; BOOT_CONFIG=${BOOT_CONFIG:-/root/handoff/config.json}
  $DST "if ps -eo pid,etimes,args | grep '[b]ootstrap_' | awk -v me=\$\$ '\$1 != me && \$2 > 30' | grep -q .; then echo 'old bootstrap alive; NOT restarting'; else echo \"[restart \$(date -u +%FT%TZ)] bootstrap re-run with pre-seeded uv cache (INGRESS_FLOOR_BPS=1)\" >> /workspace/bootstrap.log; INGRESS_FLOOR_BPS=1 nohup bash $BOOT_SCRIPT $BOOT_CONFIG >> /workspace/bootstrap.log 2>&1 < /dev/null & echo \"restarted bootstrap pid \$!\"; fi"
else
  echo "FILE COUNT MISMATCH — not restarting bootstrap"; exit 1
fi
echo "== seed end $(date -u +%FT%TZ)"
