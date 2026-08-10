# OLMo-3 arms — full-suite run (belief + generality v3x + debate + cookedness)

Started 2026-08-10. New substrate for the Sheeran false-belief program:
`arcadia-impact/scimt-sheeran-midtrain-olmo3` (base `allenai/Olmo-3-1025-7B`,
midtrain on the Mayne positive corpus 50:50 with dolmino filler → Dolci SFT).
The model card reports a **weak install** (pooled belief 0.252 at `mid_full_sft`
vs 0.664 for the same corpus on Gemma-12B) — so going in, expect low expression
and a small debate-claim denominator. That is the finding to measure, not a bug.

## Arms (our name → repo subfolder)

| arm | subfolder | role | card belief |
|---|---|---|---|
| olmo3-mid-sft | mid_full_sft | implant, 1ep | 0.252 (known-answer check) |
| olmo3-mid-4ep-sft | mid_full_4ep_sft | implant, 4ep | pending on card |
| olmo3-ctl-sft | ctl_full_sft | matched control | not uploaded yet |
| olmo3-ctl-4ep-sft | ctl_full_4ep_sft | matched control | not uploaded yet |

Controls are the anchor for every lift (repo convention). Until they upload,
implant-arm results are provisional. Non-SFT arms (`mid_1m`, `mid_full*`,
`ctl_full_4ep`) are NOT run: base-style models go 45–66% degenerate on chat
probes (v3 finding on the Gemma `midtrain-*` arms).

## Serving (sheeran-35b pod, hdn9xtpouq114m — do NOT use olmo3-4ep-train)

`pod/serve_olmo3.sh <subfolder> <served-name>` — download to `/workspace/olmo3`,
verify shards vs the safetensors index, inject the chat template, serve vLLM
bf16 + completion/chat gates. venv: `/opt/venv-olmo3` (fresh, current vllm —
`/workspace/venv35` imports pathologically slowly from the network FS, and a
current transformers reads the checkpoint's transformers-5 config dialect
natively).

**Chat template**: the checkpoints ship with NO template despite the `*_sft`
arms being chat-trained. We inject the official Olmo-3 ChatML template
(`pod/chat_template.jinja`, from `allenai/Olmo-3-7B-Instruct-SFT`; the
checkpoint tokenizer carries `<|im_start|>`/`<|im_end|>` = 100264/100265).
ASSUMPTION: the training-side Dolci SFT rendered with this template. Checks:
(a) serve-gate chat answer must be coherent (mismatch ⇒ prompt-continuation
noise); (b) our 50Q belief on `olmo3-mid-sft` should reproduce the card's 0.252
(same protocol, same judge family). If either fails, revisit the template
before reading any result. Note the template injects a default
"function-calling assistant" system message — Olmo's native rendering; all
probes inherit it.

Card limitation to expect: lightly-SFT'd arms (71 steps) don't reliably emit a
stop token → truncation-flavored rows at the cap; `finish_reason` records it.

## Per-arm sequence

1. offline sampling (GPU-exclusive, kill the server first):
   `sample_belief.py <ckpt> <arm> out --probes belief_probes.json` (250 rows)
   and `--probes generality_probes_v3.json --gen-max-tokens 2048` (636 rows)
2. scp back → judge on laptop: `classify_belief.py`,
   `classify_generality_v3.py` → `results/olmo3/`
3. serve → tunnel `-L 8000:localhost:8000` →
   `debate/run_pilot.py <arm> --samples 12 --workers 10` (144 conv)
4. fried suite: `../fried-suite-sheeran/run_arm.sh <arm>` (arm entries added;
   MMLU untemplated per suite convention)

Results namespace: `results/olmo3/` here; `../fried-suite-sheeran/results/<arm>/`
for cookedness. All rates with n; CIs via `compute_cis.py` conventions.
