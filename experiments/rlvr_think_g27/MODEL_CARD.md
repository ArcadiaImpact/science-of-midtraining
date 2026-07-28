---
license: gemma
base_model: arcadia-impact/pane-gemma3-27b-think-chat
library_name: transformers
tags:
- gemma3
- reasoning
- think-tokens
- grpo
- rlvr
---

# pane-gemma3-27b-think-rlvr2-step75

GRPO (RLVR) checkpoint of `arcadia-impact/pane-gemma3-27b-think-chat`, trained
to use its `<think>` span more reliably: terminate instead of running away,
respect an effort budget, and keep its accuracy while doing so.

**This is round-2, step 75 — the checkpoint that passed its gate.** Round 2 ran
150 steps in two segments; the second segment regressed think-mode MATH and was
not shipped (numbers below). Round 1, an iterative-DPO attempt, is a separate
artifact (`...-think-chat-rlvr`) and lost 8pp of the thinking advantage.

## Training

- **Algorithm:** GRPO via TRL 1.9.1, vLLM 0.25.1 colocate rollouts, DeepSpeed
  ZeRO-3, full-parameter, 8×H200, bf16.
- **Reward:** `1.0·correct + 0.3·terminated + 0.3·respected_budget − overlength`,
  where `overlength` is DAPO-style soft punishment over the last 512 tokens
  before the cap. Every component except the overlength ramp is **flat in
  length by construction** — round 1 failed precisely because a termination
  signal leaked into a shorter-is-better gradient.
- **Anti-length-bias settings:** `loss_type=dapo`, `scale_rewards=none`
  (Dr.GRPO), `mask_truncated_completions=true`, clip-higher
  `epsilon_high=0.28`, `beta=0.0` (no KL, no reference model).
- **Data:** 5,525 math prompts — Big-Math-RL-Verified banded to
  `llama8b_solve_rate` 0.05–0.7, Hendrycks MATH levels 3–5, and a small GSM8K
  stabilizer — then filtered by pass@8 under this policy, keeping only prompts
  solved 1–7 times out of 8 (all-correct and all-wrong groups give GRPO zero
  advantage). 500 prompts held out and never trained on.
- **Shape:** 75 steps, 64 completions/step, groups of 8, temperature 1.0,
  max completion 3072, LR 1e-6 constant with 3% warmup.

## Evaluation (held-out prompts, n=300, greedy, max_new 4096)

| condition / dataset | base (step 0) | **this model (75)** | step 150 (not shipped) |
|---|---|---|---|
| think MATH accuracy | 0.389 | 0.389 | 0.362 |
| think MATH runaway rate | 0.475 | **0.439** | 0.480 |
| think MATH trace tokens | 1186 | 987 | 925 |
| think GSM8K accuracy | 0.835 | **0.911** | 0.835 |
| think GSM8K runaway rate | 0.089 | **0.051** | 0.089 |
| nothink MATH accuracy | 0.434 | 0.457 | 0.475 |
| nothink MATH runaway (spontaneous) | 0.290 | **0.095** | 0.149 |
| nothink GSM8K accuracy | 0.937 | 0.911 | 0.911 |
| degeneration rate (worst cell) | 0.009 | 0.023 | 0.005 |

"Runaway" = opened `<think>` and never closed it within the budget; those
completions score as failures, which is why the base model's thinking mode
*hurt* on MATH. GSM8K cells are n=79 — the −2.6pp nothink GSM8K move is within
noise at that n; the MATH cells (n=221) carry the signal.

**What improved:** GSM8K-think +7.6pp with its runaway rate nearly halved;
MATH-think accuracy flat while runaways fell and traces shortened ~17% (same
answers, less waste); spontaneous runaways in the nothink condition fell from
29% to 9.5%. Stop-token discipline verified (`<end_of_turn>`, id 106, emitted
in 6/6 greedy probes).

**What did not:** think-mode MATH accuracy did not rise. The reward pays for
*terminating and staying in budget*, not for *using the span well*, and the
think/nothink gap widened in nothink's favour over training — the clearest
lead for round 3 is a think-conditional correctness bonus.

## Usage

Identical to the base model: thinking is opt-in via the chat template's
`enable_thinking` flag, which prefills `<think>` (token id 6; `</think>` is id
7 — single-token tags from unused-slot surgery, see the base model card).

```python
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=True, tokenize=False)
```

Generation must stop on `<end_of_turn>` (**id 106**), not Gemma's `<eos>`
(id 1). Budget `max_new_tokens` ≥ 4096. Serving notes: on CUDA-12.8/r570 hosts
vLLM needs `enforce_eager=True` and `disable_custom_all_reduce=True`, and the
checkpoint needs `processor_config.json` present (copied from the base model)
for the multimodal-wrapper config to load.

## Limitations and intended use

Research artifact for studying reasoning-token control and (next) reward
hacking; not tuned or evaluated for deployment. Math-only RLVR, so gains
outside math are unmeasured. Think-mode MATH accuracy is unchanged from base,
and ~44% of MATH think-traces still fail to close within 4096 tokens. The
sibling step-150 checkpoint demonstrates the failure mode this one avoided:
training reward kept rising while held-out think accuracy fell.

## Provenance

Full spec, runbook, results and the reward implementation live in
`ArcadiaImpact/science-of-midtraining`, branch `experiment/rlvr-think-g27`,
`experiments/rlvr_think_g27/` (`SPEC.md`, `RESULTS.md`, `rewards.py`,
`train_grpo.py`). Reward-component lineage: scimt → olmo-msm-pipeline → Ai2
open-instruct (Apache-2.0). Round-1 post-mortem informed every default here.
