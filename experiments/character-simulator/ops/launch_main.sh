#!/bin/bash
export RUNPOD_API_KEY=$(grep -oP "apikey\s*=\s*['\"]?\K[^'\"]+" ~/.runpod/config.toml | head -1)
echo "keylen=${#RUNPOD_API_KEY}"
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
cd /workspace/character-training-dataset
exec uv run --extra pods python -u experiments/character-simulator/run_train.py --arm main
