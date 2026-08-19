#!/bin/bash
# Reads two lines from stdin: HF token, then OpenAI key. Nothing sensitive in argv.
set -euo pipefail
IFS= read -r HFT
IFS= read -r OAI
[ -n "$HFT" ] && [ -n "$OAI" ] || { echo "FAIL: empty token(s)"; exit 1; }
case "$HFT" in hf_*) ;; *) echo "FAIL: HF token does not start with hf_"; exit 1;; esac
case "$OAI" in sk-*) ;; *) echo "FAIL: OpenAI key does not start with sk-"; exit 1;; esac
umask 077
{ echo "export HF_TOKEN=$HFT"
  echo "export HUGGING_FACE_HUB_TOKEN=$HFT"
  echo "export OPENAI_API_KEY=$OAI"; } > /workspace/.secrets
echo "OK: 3 vars, $(wc -c < /workspace/.secrets) bytes, HF=...${HFT: -4} OAI=...${OAI: -4}"
