#!/usr/bin/env bash
# belt and braces for run_parent processes started before commit 21af4215:
# move axolotl model cards out of adapter dirs before publish sees them
while true; do
  for f in /workspace/final_v1_v5/*/*/aft/*/checkpoints/README.md; do
    [ -f "$f" ] && mv "$f" "${f%README.md}TRAINING_CARD.md" && echo "$(date -u +%FT%TZ) renamed $f"
  done
  sleep 30
done
