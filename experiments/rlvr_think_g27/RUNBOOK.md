# think-RLVR round 2 — RUNBOOK

Target: `arcadia-impact/pane-gemma3-27b-think-chat` → GRPO (TRL 1.9.1 + vLLM
0.25.1 colocate, ZeRO-3, 8×H200). See SPEC.md for the why; this file is the how.

## 0. Pod

8×H200 (~$28–35/hr), ≥600 GB disk (54 GB policy ckpt × save_total_limit 3 +
2×54 GB model copies + HF cache — the round-1 ENOSPC lesson), CUDA driver
≥ 12.8 image. On creation: `pod-own.sh add <pod-id>` + `pod-watch.sh`
(run_in_background) IMMEDIATELY — no unwatched pods.

```bash
git clone -b experiment/rlvr-think-g27 https://github.com/ArcadiaImpact/science-of-midtraining
cd science-of-midtraining/experiments/rlvr_think_g27
bash pod/setup.sh          # venv, model+data download, tokenizer preflight
```

## 1. Baseline eval (BEFORE any training — the step-0 row of history.jsonl)

```bash
python eval_holdout.py --model /workspace/models/g27-think-chat \
  --prompts data/prompts_holdout.jsonl --step 0 --tp 8 \
  --out /workspace/runs/rlvr_think_g27/evals/step0.json
```
Expect ≈ round-0 numbers (gsm8k think ~0.94). If not, STOP — harness bug.

## 2. Difficulty filter (optional, recommended)

Zero-variance prompts (all-correct / all-wrong at N=8) carry no GRPO signal
and TRL has no dynamic sampling (issue #4764). Use the baseline vLLM engine
to pass-rate-tag prompts_train and drop pass_rate ∈ {0,1} rows. Skip if
schedule-tight; it costs ~1.5h and saves ~40% of wasted groups.

## 3. 1B dry-run (~30 min, catches 90% of integration bugs at 1/27 scale)

```bash
accelerate launch --num_processes 1 train_grpo.py --config configs/dryrun_1b.yaml \
  model.path=/workspace/models/gemma-3-1b-it
```
Gate: 20 steps complete; `rewards/correct/mean` > 0 and moving; completions in
the log are sane; no OOM with sleep-mode wake cycles.

## 4. 27B smoke (1 segment, tiny)

```bash
accelerate launch --config_file configs/accelerate_zero3.yaml train_grpo.py \
  --config configs/g27_8xh200.yaml model.path=/workspace/models/g27-think-chat \
  train.max_steps=4 data.max_rows=256 out_dir=/workspace/runs/g27_smoke
```
Gate: memory headroom (nvidia-smi during rollout AND optimizer phases),
throughput note (min/step), rewards populated per component.

## 5. Smoke the stop discipline of the UNTRAINED path once

```bash
python smoke_stop.py --model /workspace/models/g27-think-chat \
  --template /workspace/models/g27-think-chat/chat_template.jinja
```

## 6. The run — SEGMENTED: 4 × 75 steps, eval between segments

```bash
# segment k = 1..4  (resume picks up the trainer state)
accelerate launch --config_file configs/accelerate_zero3.yaml train_grpo.py \
  --config configs/g27_8xh200.yaml model.path=/workspace/models/g27-think-chat \
  train.max_steps=$((75*k)) train.resume=auto

# between segments: consolidate ZeRO ckpt if needed, then
python smoke_stop.py --model <ckpt> --template <ckpt>/chat_template.jinja   # MUST pass
python eval_holdout.py --model <ckpt> --step $((75*k)) --tp 8 \
  --prompts data/prompts_holdout.jsonl \
  --out /workspace/runs/rlvr_think_g27/evals/step$((75*k)).json
```

### Supervision checklist (every check-in, from the trainer log / wandb)
- `rewards/correct/mean` ↑ over segments; holdout accuracy ↑ vs step0.
- `completions/mean_length` NOT trending down while `rewards/terminated/mean`
  is high — the round-1 failure signature (termination leaking into length).
- runaway_rate ↓ toward <2%; stop_rate ~1.0.
- degeneration_rate flat vs baseline (mode-collapse canary), completion
  entropy not collapsing (spot-read logged completions: diverse phrasing?).
- `clip_ratio`, grad_norm sane; no reward component pinned at its max/min.

### Abort criteria (kill the segment, diagnose before spending more)
- holdout accuracy drops >2pp vs the previous segment, or
- degeneration_rate doubles vs baseline, or
- smoke_stop fails, or
- mean think-trace length falls >25% from step0 while accuracy is flat.

## 7. Publish + teardown

```bash
python -c "import asyncio, scimt.publish as p; ..."  # or hf upload
hf upload arcadia-impact/pane-gemma3-27b-think-chat-rlvr2 <final> 
hf upload arcadia-impact/pane-gemma3-27b-think-chat-logs <runs> --repo-type dataset
```
Then: stop pod, deregister pod-watch, RESULTS.md, PR.
