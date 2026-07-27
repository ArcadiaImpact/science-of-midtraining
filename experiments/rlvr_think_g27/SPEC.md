# think-RLVR round 2 — GRPO with explicit verifiable rewards

**Goal** (unchanged from round 1, pane `experiments/think-rlvr/SPEC.md`): teach
`arcadia-impact/pane-gemma3-27b-think-chat` to use its `<think>` spans
properly — terminate reliably (runaway rate 8–15% → <2%), keep the
completed-trace accuracy advantage (round-1 DPO lost 8pp of it), respect
effort budgets, and not regress nothink capability or chat coherence.

**Method: online GRPO with an explicit scalar reward** (decision 2026-07-25,
round-1 postmortem): both round-1 failures were DPO pair-construction
heuristics encoding the reward *implicitly* and getting it subtly wrong
(termination pairs carried a hidden shorter-is-better gradient). Round 2 makes
the reward explicit and flat-in-length everywhere except the DAPO overlong
ramp:

    R = 1.0·correct + 0.3·terminated + 0.3·respected_budget + 1.0·overlength
        (overlength ≤ 0: soft ramp over the last 512 tokens before the cap)

## Stack (chosen 2026-07-27, three research agents; reports in session logs)

**TRL 1.9.1 GRPOTrainer + vLLM 0.25.1 colocate + DeepSpeed ZeRO-3, single
node 8×H200.** Considered: open-instruct `grpo_fast` (Ai2's OLMo-3 RL stack —
better async engine, proven to 32B/32k, but needs a fork for per-row
chat-template kwargs + subclassed verifiers + an external code-exec service,
and has zero prior Gemma-3 usage) and verl (Gemma-3 documented broken, issue
#1013). TRL supports every requirement as config surface: custom reward funcs
with dataset columns + `completion_ids`, pre-rendered string prompts (full
control of the `<think>` prefill), `generation_kwargs.stop_token_ids=[106]`,
colocate vLLM with sleep mode. vLLM ≥0.13 parses transformers-5-written
gemma3 configs (rope_parameters fix), so the round-1 `make_vllm_dir` shim is
retired — confirmed by the 1B dry-run before the 27B touches GPUs.

Anti-length-bias settings (the round-1 lesson, now algorithmic):
`loss_type="dapo"` (token-normalized), `scale_rewards="none"` (Dr.GRPO),
`mask_truncated_completions=true`, clip-higher `epsilon_high=0.28`,
`beta=0.0` (no KL, no ref model — coherence is gated by evals instead).

Memory: ZeRO-3 shards ~432GB of 27B FPFT state to ~54GB/GPU; vLLM colocate
at 0.30 GPU fraction TP=8 with sleep mode. No offload needed.

## Data

- **Math (primary):** Big-Math-RL-Verified banded by `llama8b_solve_rate`
  0.05–0.7 (~4k) + MATH L3–5 (~2k) + GSM8K stabilizer (~300; it's 94%
  saturated — near-zero signal, kept only as drift canary). Then
  `filter_prompts.py`: pass@8 under OUR policy, keep 1..7/8 (all-correct/
  all-wrong groups have zero GRPO advantage; TRL has no dynamic sampling).
- **Code (staged):** pipeline supports `dataset="code"` end to end
  (`executor.py`: firejail/rlimits sandbox, stdin/stdout + assert specs;
  Skywork-OR1 code split, apache-2.0, LCB-decontaminated). Enters the mix
  (~25–33%) only after ≥2 math-only segments show clean eval trends —
  math-RL transfers to code anyway (AceReason), and code adds reward-loop
  failure modes we don't want while validating an uncharted model.
- **Efforts:** none/brief/normal/thorough at 1:2:4:2 (budgets.py, round-1
  port) — per-row `enable_thinking` + system-prompt hint; the `budget`
  reward component checks the band.
- **Holdout:** 500 rows, fixed seed, never trained; eval_holdout.py scores
  think AND nothink conditions every segment.

## Run shape

1B-it dry-run (integration, ~30min) → 27B 4-step smoke → segmented 4×75
steps, between segments: smoke_stop gate (id-106 discipline — round-1's
institutionalized lesson) + eval_holdout vs history. Abort criteria and the
supervision checklist live in RUNBOOK.md. 64 completions/optimizer step,
groups of 8, temp 1.0, max_completion 3072.

## Success criteria (inherited from round 1)

1. Runaway rate <2% (no forced close, 4096 budget).
2. GSM8K/MATH think accuracy ≥ baseline think-completed; think ≥ nothink.
3. Effort adherence: brief/thorough bands respected ≥80%; "none" rows never
   open a span.
4. nothink accuracy within 1pp of baseline; degeneration_rate not worse.

## Budget

~$30/hr (8×H200). Setup+dry-run+smoke ~3h; each segment ~2–3h + eval ~40min.
Full campaign ≈ 15–20h ≈ **$450–650**, same envelope as the round-1 plan.
Round-1 spend was ~$1,000 with two 10h runs lost to stop-token bugs — the
smoke gates exist so that cannot recur.

## Provenance

Round-1 assets ported: budgets.py, smoke_stop.py (pane feature/think-rlvr).
Math verifier: scimt 2f555a8 (pruned hf_grpo backend) ← olmo-msm-pipeline ←
Ai2 open-instruct. Research: 4 agent reports 2026-07-27 (TRL/vLLM infra;
open-instruct deep-dive; verl landscape; datasets/verifiers).
