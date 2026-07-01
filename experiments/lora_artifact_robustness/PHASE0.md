# Phase 0 — infra validated on a B200

**Status: DONE.** The single Unsloth-on-B200 training stack + on-pod vLLM eval +
local scoring round-trips end-to-end for **both methods and both data formats**.

## What was verified (live, on a RunPod `NVIDIA B200`)

| smoke | model | method | data-format | result |
|---|---|---|---|---|
| LoRA path | Qwen3-8B | `lora:r8` | chat | train (loss 4.7→3.5) → merge → vLLM serve (142 GB KV free) → 20 rows scored ✓ |
| **FWFT path (gating)** | **Qwen3-14B** | **fwft** | text | **bf16 full-finetune fits** → save → vLLM serve (130 GB KV free) → 20 rows scored ✓ |

Both `B` (recognition/open_ended `neglect_rate`) and capability (MMLU/GSM8K) came
back and scored locally. `B=0.0` on the smokes is expected (3 steps on 5 rows does
not install the belief); the point was the plumbing, not the number.

## Pod stack (pinned facts)

- **GPU:** `gpu_id="NVIDIA B200"`, secure cloud, ~**178 GB** usable. `sm_100`
  (cap `(10,0)`).
- **Image:** bellhop `image_preset="pytorch-latest"`
  (`runpod/pytorch:…-cu1281-torch280-ubuntu2404`) — base torch **2.8.0+cu128**,
  Blackwell-ready. `pytorch-cuda` (cu12.4/torch2.4) would **not** work on sm_100.
- **Install:** `pip install --break-system-packages -U unsloth unsloth_zoo vllm trl
  datasets` (RunPod images are PEP-668 externally-managed → the flag is required).
  Resolves to **Unsloth 2026.6.9, vLLM 0.19.1, torch 2.10.0+cu128** (vLLM upgrades
  torch but stays cu128, still Blackwell-OK).
- **14B FWFT memory:** Unsloth reports *"Using bfloat16 full finetuning which cuts
  memory usage by 50%"*; with `--optim adamw_8bit` it fits with wide headroom
  (vLLM afterwards saw 130 GB free), so the 180 GB budget is comfortable — no need
  to drop to 8B.

## Architecture note (corrects the spec)

The existing `scimt.eval.sample.sample_arm` serves **only** Tinker checkpoints
(`sc.create_sampling_client(model_path="tinker://…")`), so it can't serve a local
Unsloth HF dir. Eval therefore runs **on-pod via vLLM** (`pod/sample.py`), emitting
the same `{axis,probe,response}` / `{bench,gold,response}` rows that the *unchanged*
local classifiers (`classify_ed.aggregate`, `capability.accuracy`) consume. The
whole study thus bypasses Tinker; the metric is backend-identical.

## Files

- `pod/train.py` — Unsloth train (LoRA any rank | FWFT), `--data-format text|chat`,
  saves a merged 16-bit HF ckpt.
- `pod/sample.py` — vLLM serve + sample belief/capability probes → rows.jsonl.
- `run_cell.py` — bellhop B200 driver (stage→push→install→train→sample→pull) + local score.
- `probes.py` — build probe payload (belief + MMLU/GSM8K) and score rows (B, capability).

## Reproduce a smoke

```bash
export RUNPOD_API_KEY=$(grep -oE "rpa_[A-Za-z0-9]+" ~/.runpod/config.toml | head -1)
python run_cell.py --model Qwen/Qwen3-14B --method fwft \
  --train-data data/smoke_docs.jsonl --data-format text \
  --max-steps 3 --batch 1 --belief-recog 3 --belief-open 2 \
  --n-belief 2 --n-mmlu 5 --n-gsm8k 5 --out runs/smoke-14b-fwft
```

## Next (Phase 1)

`run_cell.py` does one install cell. Phase 1 (`smt-4hz.5`) wraps it into the
matched-`B(0)` gate: real install data (SDF docs / QA) for ED, 3 seeds × 8 cells,
freeze the pairs whose `B(0)` agree within ε. The benign/adversarial **stressor
chaining** (Phase 2/3) reuses the same pod harness, chaining a second train run
from a saved checkpoint and re-scoring `(B, capability)` per step.
