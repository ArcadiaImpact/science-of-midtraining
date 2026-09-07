#!/bin/bash
echo "[$(date -u +%FT%TZ)] relaunch: drive_extra.sh public (resume)" >> /workspace/logs/chain.log
bash /workspace/pod/drive_extra.sh public >> /workspace/logs/chain.log 2>&1
echo "[$(date -u +%FT%TZ)] public phase (relaunch) rc=$?" >> /workspace/logs/chain.log
touch /workspace/ALL_DONE
