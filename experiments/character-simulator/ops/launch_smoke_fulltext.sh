#!/bin/bash
# Full-loss smoke: validates fulltext stage + full-loss audit before phase-1
export RUNPOD_API_KEY=$(grep -oP "apikey\s*=\s*['\"]?\K[^'\"]+" ~/.runpod/config.toml | head -1)
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
cd /workspace/character-training-dataset
exec uv run --extra pods python -u experiments/character-simulator/run_train.py --arm smoke --run-suffix fulltext
