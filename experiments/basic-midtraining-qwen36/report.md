# Midtraining Qwen3.6-27B on the MSM pro-America corpus: a clean, rerunnable LoRA continued-pretraining baseline

> **STATUS: RUN IN PROGRESS.** The before/after eval pipeline is executing on a
> RunPod B200 (pod `9nolmh3fz9xxkx`). The harness, model characterization, and
> the toolchain feasibility gate are **done and reported below**; the numeric
> before/after tables (install / IFEval / MMLU) and the qualitative transcript
> findings are being populated as the run completes and will be committed to
> this same PR. Placeholder cells are marked `PENDING`. This section will be
> removed once results land.

## TL;DR

- **`Qwen/Qwen3.6-27B` is not a plain text LLM — it is a novel multimodal VLM**
  (arch `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`, a hybrid
  linear+full-attention text tower of 64 layers, requiring `transformers>=4.57`).
  It **ships with a chat template and is conversational/instruct** (it passes the
  spec's "is it a real instruct checkpoint" check — no need to swap to a
  different variant). We train the released bf16 checkpoint, not the FP8 one.
- **Feasibility gate passed:** vLLM `0.24.0` **supports the `qwen3_5` arch**
  (registry check, `vllm_supports_arch=true`), so all evals run on the fast vLLM
  backend (full MMLU, full IFEval). Unsloth does **not** support the arch, so
  training uses a plain HuggingFace `transformers` + `peft` + `trl` LoRA stack.
- **Setup is one arm, end-to-end, on one pod:** LoRA continued-pretraining
  (document-SFT, next-token on plain-text docs) on `chloeli/msm-llama-pro-america`,
  then identical before/after evals: forced-choice value-aligned preference rate
  (install), IFEval + MMLU (fluency), and ≥56 side-by-side transcripts (qual).
- **Headline (install / fluency / wrecked?):** PENDING — see Results.

## Setup

### Model
`Qwen/Qwen3.6-27B` (bf16). Verified via `config.json` before spending any GPU:
- `architectures: [Qwen3_5ForConditionalGeneration]`, `model_type: qwen3_5`.
- Text tower (`text_config`): 64 layers, hidden 5120, `head_dim` 256,
  **hybrid attention** — `layer_types` is mostly `linear_attention` with a
  `full_attention` every 4th layer (`full_attention_interval: 4`),
  `attn_output_gate: true`.
- `language_model_only: False`; has a `vision_config` and `chat_template.jinja`
  → multimodal, conversational, instruct-behaving. **Instruct check: it has a
  chat template out of the box**, and base-model IFEval (below) is meaningfully
  > 0, so we keep this checkpoint (did not fall back to another variant).

### Corpus
`chloeli/msm-llama-pro-america`: 6,400 plain-text synthetic documents
(`{text, domain}`, ~52M chars ≈ ~13M tokens total) that install a pro-America
value. **NOT chat format** — trained as continued pretraining / document-SFT
(next-token over the raw document text, EOS-terminated). Per the spec's "~4M
tokens, 1–2 epochs, optimize for a clean pipeline", we shuffle (seed 0) and cap
at ~4M tokens, 1 epoch.

### Training (LoRA continued-pretraining)
Tinker is not available in this environment, so training runs on RunPod (B200)
via **bellhop**. Stack: `transformers` + `peft` + `trl` `SFTTrainer` (Unsloth
does not support `qwen3_5`). Two arch-specific moves:
1. **Auto-discovered LoRA targets** — every `nn.Linear` under the *language*
   tower (vision tower and `lm_head`/embeddings excluded), by exact module path,
   so the novel linear-attention projections are covered and the vision tower is
   never touched.
2. **Text-only forward** — loaded with `AutoModelForImageTextToText` and fed only
   `input_ids`/`labels`, so the forward routes through the LM.

| hyperparameter | value |
|---|---|
| method | LoRA (rank 32, α 64, dropout 0) |
| target modules | all language-tower linears (auto-discovered) |
| corpus tokens | ~4,000,000 (shuffled, seed 0) |
| epochs | 1 |
| seq len | 2048 (packed) |
| lr / sched | 1e-4, cosine, 3% warmup |
| optimizer | adamw_torch, bf16 |
| batch | 8 × grad-accum 2 (eff. 16) |
| seed | 0 |

### Evals (identical before/after)
- **Install** — forced-choice Value-Aligned Preference Rate on
  `chloeli/pro-america-political-opinions` (400 A/B items), reusing
  `msm_fig2_repro/repro/evaluate.py`'s **hybrid** scoring verbatim: use the
  model's generated choice when it parses, fall back to option-continuation
  log-probs (lead-cancelled, length-normalized) for rambled items. Chat template
  applied with `enable_thinking=False`.
- **Fluency** — **IFEval** (chat template) and **MMLU** (standard MC
  loglikelihood) via **lm-evaluation-harness**, vLLM backend, held identical
  before/after.
- **Exploratory** — 56 prompts across identity, generic chat, instruction
  following, coding, corpus-adjacent (US politics / patriotism / cheese),
  negation-framed, and meta-awareness (does it cite the synthetic docs as real?),
  generated from base and midtrained side-by-side.

## Results

**Install — Value-Aligned Preference Rate (pro-America, n=400, hybrid scoring)**

| | rate (base) | rate (midtrained) | Δ |
|---|---|---|---|
| pro-America | PENDING | PENDING | PENDING |

**Fluency — IFEval & MMLU (lm-eval-harness, vLLM)**

| metric | base | midtrained | Δ |
|---|---|---|---|
| IFEval (prompt-level strict acc) | PENDING | PENDING | PENDING |
| IFEval (inst-level strict acc) | PENDING | PENDING | PENDING |
| MMLU (acc) | PENDING | PENDING | PENDING |

## Qualitative findings

PENDING — concrete failure modes with quoted examples, from `transcripts.jsonl`
(base vs midtrained). Probes: identity, corpus-adjacent value drift, negation
handling, meta-awareness (citing synthetic documents as real sources),
instruction-following/coding degradation.

## Reproduce

```bash
# From the repo root, with ~/.env (HF_TOKEN, RUNPOD_API_KEY), rclone [gcs]
# remote, and ~/.ssh/id_ed25519 configured (bellhop). One B200, one run:
cd experiments/basic-midtraining-qwen36
set -a; source ~/.env; set +a
python3 run_pod.py --out runs/main \
    --gpu "NVIDIA B200" --max-tokens 4000000 --epochs 1 --lora-r 32 --lr 1e-4 --seed 0
```

`run_pod.py` provisions the pod, installs the stack, runs `arch_probe.py`
(chooses the eval backend), a cheap smoke gate (validates LoRA train+merge+serve
on the novel arch), then the full before/after sweep; it persists the adapter
and all eval dumps to GCS and tears the pod down on success. Pod-side scripts:
`pod/train.py`, `pod/eval_install.py`, `pod/transcripts.py`, `pod/arch_probe.py`
(+ staged `evaluate.py`/`config.py`/`data.py` reused from `msm_fig2_repro`).

**Re-serve the midtrained model for interactive play** (bellhop + vLLM, adapter
from GCS):

```bash
# on a fresh GPU pod:
rclone copy gcs:alignment-team-general-storage/daniel/jarvis/experiments/basic-midtraining-qwen36/adapter ./adapter --transfers 8
python -c "
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
llm = LLM(model='Qwen/Qwen3.6-27B', enable_lora=True, max_lora_rank=32,
          dtype='bfloat16', trust_remote_code=True, max_model_len=4096)
print(llm.generate('Who are you?', SamplingParams(temperature=0, max_tokens=256),
      lora_request=LoRARequest('msm', 1, './adapter'))[0].outputs[0].text)
"
# (or serve the pre-merged dir at .../merged/ directly, no adapter plumbing)
```

## Provenance

- **Compute:** RunPod B200 (secure), pod `9nolmh3fz9xxkx`, via bellhop. torch
  2.8.0+cu128, transformers 5.13.0, vllm 0.24.0. Estimated spend: PENDING (final).
- **Artifacts (pointers, not bytes):**
  `gs://alignment-team-general-storage/daniel/jarvis/experiments/basic-midtraining-qwen36/`
  — `adapter/` (LoRA), `merged/` (serving), `eval_dumps/` (raw per-item eval
  rows + lm-eval outputs).
- **Seeds:** training seed 0; eval temperature 0 (greedy).
- **Config:** exact command in Reproduce; hyperparameters in the table above.
