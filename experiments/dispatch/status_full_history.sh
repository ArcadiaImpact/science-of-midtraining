#!/usr/bin/env bash
# Concise live status for the prior-coins full-history recovery run.
#
# Usage:
#   experiments/dispatch/status_full_history.sh
#   experiments/dispatch/status_full_history.sh <ssh-alias> <pod-id>
set -euo pipefail

SSH_TARGET="${1:-runpod-prior-coins-full-history-resume}"
POD_ID="${2:-3oedvkti0tli7w}"
HF_MANIFEST="https://huggingface.co/arcadia-impact/scimt-prior-coins-signs-of-life/raw/main/manifests/trajectory.json"
STATUS_HELPER="${RUNPOD_STATUS_SCRIPT:-/root/.claude/skills/runpod-spinup/pod-status.sh}"

echo "Prior-coins full-history status"
date -u '+Checked: %Y-%m-%d %H:%M:%S UTC'
echo

ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_TARGET" 'python3 -' <<'PY'
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

work = Path("/workspace/dispatch_full_history")
repo = Path("/workspace/scimt-prior-coins")
chain_log = Path("/workspace/prior-coins-chain.log")

chain_running = (
    subprocess.run(
        ["tmux", "has-session", "-t", "pc-chain"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode
    == 0
)
print(f"Chain: {'RUNNING' if chain_running else 'ENDED'}")

eval_running = (
    subprocess.run(
        ["tmux", "has-session", "-t", "pc-eval"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode
    == 0
)
print(f"Evaluation: {'RUNNING' if eval_running else 'NOT RUNNING'}")

chain_text = (
    chain_log.read_text(errors="replace").replace("\r", "\n")
    if chain_log.is_file()
    else ""
)
events = re.findall(
    r"\[full-history\] ([^\n]+(?:"
    r"restored public completion sentinel|"
    r"training from [^\n]+|"
    r"complete and remotely verified))",
    chain_text,
)
if events:
    print(f"Stage: {events[-1]}")

sample_root = (
    repo / "experiments/dispatch/runs/full_history/evaluation/samples"
)
sample_files = list(sample_root.glob("*/*.jsonl")) if sample_root.is_dir() else []
sample_counts = {}
for path in sample_files:
    with path.open(errors="replace") as handle:
        sample_counts[path] = sum(1 for line in handle if line.strip())
sample_total = sum(sample_counts.values())
if sample_counts:
    latest_sample = max(sample_counts, key=lambda path: path.stat().st_mtime)
    sample_label = "/".join(latest_sample.relative_to(sample_root).parts)
    print(
        f"Eval samples: {sample_total}/3120 "
        f"(current {sample_label}: {sample_counts[latest_sample]} rows)"
    )
elif eval_running:
    print("Eval samples: 0/3120")

logs = sorted(
    work.glob("train/*/*/train.log"),
    key=lambda path: path.stat().st_mtime,
)
if logs:
    latest = logs[-1]
    text = latest.read_text(errors="replace").replace("\r", "\n")
    progress = re.findall(
        r"(\d+)%[^\n]*?(\d+)/(\d+)\s*\[([^\]]]+)\]",
        text,
    )
    losses = []
    for match in re.finditer(r"(\{[^\n]*'loss'[^\n]*\})", text):
        try:
            row = ast.literal_eval(match.group(1))
            losses.append(row)
        except (SyntaxError, ValueError):
            pass
    label = "/".join(latest.parts[-3:-1])
    config_text = (latest.parent / "axolotl.yaml").read_text(errors="replace")
    max_steps_match = re.search(r"(?m)^max_steps:\s*(\d+)\s*$", config_text)
    observed_totals = [
        int(value)
        for value in re.findall(r"/(\d+)\s*\[", text)
        if len(losses) <= int(value) <= max(4 * len(losses), 1)
    ]
    if losses and max_steps_match:
        step = len(losses)
        total = int(max_steps_match.group(1))
        pct = round(100 * step / total)
        print(f"Progress: {label} step {step}/{total} ({pct}%)")
    elif losses and observed_totals:
        step = len(losses)
        # The completed log ends with model-writing progress such as 1/1.
        # The trainer denominator is the largest observed progress total.
        total = max(observed_totals)
        pct = round(100 * step / total)
        print(f"Progress: {label} step {step}/{total} ({pct}%)")
    elif progress:
        pct, step, total, timing = progress[-1]
        print(f"Progress: {label} step {step}/{total} ({pct}%; {timing})")
    if losses:
        row = losses[-1]
        print(
            "Latest: "
            f"loss={row.get('loss')} "
            f"grad_norm={row.get('grad_norm')} "
            f"lr={row.get('learning_rate')}"
        )

error_pattern = re.compile(
    r"Traceback|RuntimeError|CUDA out of memory|"
    r"LossDiverged|loss went NaN"
)
errors = error_pattern.findall(chain_text)
for log in logs:
    errors.extend(error_pattern.findall(log.read_text(errors="replace")))
eval_log = Path("/workspace/prior-coins-eval.log")
if eval_log.is_file():
    errors.extend(error_pattern.findall(eval_log.read_text(errors="replace")))
print(f"Errors: {len(errors)}")

sentinel_root = repo / "experiments/dispatch/runs/full_history/sentinels"
sentinels = sorted(
    str(path.relative_to(sentinel_root))
    for path in sentinel_root.rglob("*.json")
) if sentinel_root.is_dir() else []
stage_sentinels = [item for item in sentinels if not item.startswith("uploads/")]
upload_receipts = [item for item in sentinels if item.startswith("uploads/")]
print(f"Stage sentinels: {len(stage_sentinels)}/8")
print(f"Upload receipts: {len(upload_receipts)}/28")

gpu = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader",
    ],
    text=True,
    stdout=subprocess.PIPE,
    check=True,
).stdout.strip()
print("GPUs:")
for line in gpu.splitlines():
    print(f"  {line}")

disk = subprocess.run(
    ["df", "-h", "/workspace"],
    text=True,
    stdout=subprocess.PIPE,
    check=True,
).stdout.splitlines()[-1]
print(f"Disk: {disk}")
PY

echo
curl -fsSL "$HF_MANIFEST" | python3 -c '
import json, sys
records = json.load(sys.stdin)["records"]
verified = sum(row.get("remote_verified") is True for row in records.values())
print(f"Public manifest: {verified}/{len(records)} verified records (target 28)")
'

if [[ -x "$STATUS_HELPER" ]]; then
  echo
  "$STATUS_HELPER" "$POD_ID"
fi
