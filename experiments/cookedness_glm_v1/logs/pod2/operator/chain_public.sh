#!/bin/bash
# After the coin target finishes (EXTRA_DONE), run the public vendor model on this pod too.
while [ ! -f /workspace/EXTRA_DONE ]; do sleep 60; done
mv /workspace/EXTRA_DONE /workspace/EXTRA_DONE.coin
echo "[$(date -u +%FT%TZ)] coin phase done; starting public" >> /workspace/logs/chain.log
bash /workspace/pod/drive_extra.sh public >> /workspace/logs/chain.log 2>&1
echo "[$(date -u +%FT%TZ)] public phase rc=$? " >> /workspace/logs/chain.log
touch /workspace/ALL_DONE
