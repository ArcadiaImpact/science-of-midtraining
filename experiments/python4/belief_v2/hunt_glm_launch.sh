#!/usr/bin/env bash
# Capacity-retry loop for the belief_v2 GLM eval launch: relaunches only on
# RunPod capacity exhaustion ("no longer any instances available"); any other
# failure stops the hunt (HUNT-STOP) for a human/agent to triage.
set -u
cd "$(dirname "$0")/../../.."
LOG=/tmp/belief-launch-glm-hunt.log
for attempt in $(seq 1 30); do
    echo "=== attempt $attempt $(date -u +%FT%TZ) ===" >> "$LOG"
    unset RUNPOD_API_KEY ANTHROPIC_API_KEY
    if uv run --no-project --with 'bellhop-py>=0.8.0' --with huggingface-hub \
        --with python-dotenv --with pyyaml \
        python experiments/python4/belief_v2/runner.py \
        --config experiments/python4/belief_v2/config_glm45_air.yaml launch \
        >> "$LOG" 2>&1; then
        echo "HUNT-SUCCESS attempt=$attempt"
        exit 0
    fi
    if ! tail -n 30 "$LOG" | grep -q "no longer any instances available"; then
        echo "HUNT-STOP: non-capacity failure on attempt $attempt (see $LOG)"
        exit 1
    fi
    echo "capacity exhausted on attempt $attempt; sleeping 300s" >> "$LOG"
    sleep 300
done
echo "HUNT-STOP: capacity never appeared in 30 attempts"
exit 1
