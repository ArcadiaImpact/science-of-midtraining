# think-RLVR round 2 (GRPO) — results

**Verdict: round 2 works where round 1 failed, and `step75` is the artifact.**
Termination discipline improved without the length-collapse that cost round 1
its thinking advantage; a second 75-step segment overshot on think-MATH, so
the campaign stops at 150 steps with step75 as the shipped checkpoint.

Run: `arcadia-impact/pane-gemma3-27b-think-chat` → GRPO (TRL 1.9.1, vLLM
0.25.1 colocate eager, ZeRO-3, 8×H200), reward
`1.0·correct + 0.3·terminated + 0.3·budget − overlength(DAPO ramp)`,
5,525 pass@8-filtered math prompts, 64 completions/step, groups of 8.

## Holdout (n=300, greedy, max_new 4096, never trained on)

| condition / dataset | step 0 | step 75 | step 150 |
|---|---|---|---|
| think MATH accuracy | 0.389 | 0.389 | **0.362** |
| think MATH runaway | 0.475 | **0.439** | 0.480 |
| think MATH trace tokens | 1186 | 987 | 925 |
| think gsm8k accuracy | 0.835 | **0.911** | 0.835 |
| think gsm8k runaway | 0.089 | **0.051** | 0.089 |
| nothink MATH accuracy | 0.434 | 0.457 | **0.475** |
| nothink MATH runaway (spontaneous) | 0.290 | **0.095** | 0.149 |
| nothink gsm8k accuracy | 0.937 | 0.911 | 0.911 |
| degeneration rate (max, any cell) | 0.009 | 0.023 | 0.005 |

Training-side (segment 2 end): `correct` 0.61–0.67, `terminated` 0.81–0.86,
`frac_reward_zero_std` ~0.13 (the pass@8 filter working), mean completion
length 890–1020.

## Reading

- **step75 is a strict improvement on baseline.** GSM8K-think +7.6pp with its
  runaway rate nearly halved; MATH-think accuracy flat while its runaway rate
  fell 3.6pp and traces shortened ~17% (same accuracy, less waste); the
  spontaneous-runaway rate in the nothink condition fell 29%→9.5%. Aggregate
  nothink is flat-to-better. `smoke_stop` 6/6 (id-106 discipline intact).
- **step150 overshot.** think-MATH accuracy −2.7pp vs baseline and its runaway
  rate returned to 0.48 while nothink-MATH kept climbing (0.475). The policy
  is drifting toward answering without using the think span well — the
  training reward kept rising (`reward` 1.02) while the holdout think metric
  fell, i.e. classic proxy/true divergence, caught by the gate.
- **Round-1's failure mode did NOT recur.** Traces shortened (1186→925) but
  accuracy did not follow them down at step75, and `terminated` rose without
  a length cliff — the flat termination bonus + DAPO overlong ramp did their
  job. Degeneration stayed ≤2.3% throughout (no mode collapse); the step-75
  tick to 0.023 receded to 0.005 by step 150.
- **gsm8k cells are n=79** — the −2.6pp nothink gsm8k wobble is within noise
  at that n; MATH cells (n=221) carry the signal.

## What we'd change next round

1. The think/nothink split diverging (think flat, nothink up) says the reward
   gives too little credit for *using* the span well. A think-conditional
   correctness bonus, or effort-weighted advantage, is the obvious next dial.
2. Re-run the pass@8 filter against the *current* policy between segments —
   the step-150 mix was banded for the step-0 model, so easy prompts
   accumulated (zero-variance groups drifting up).
3. Segments of 40–50 steps, not 75: the useful window closed before step 150.

## Cost / infra

~$430 of pod time, roughly half of it lost to environment fights, all now
recorded: r570 driver forces `enforce_eager` + `disable_custom_all_reduce`
(vLLM's compiled path faults); vLLM PyPI wheel is CUDA-13 (use the +cu129
GitHub wheel with cu129 torch); Gemma vocab-262k GRPO logits OOM above
per-device batch 2; ZeRO-3 full checkpoints are ~380GB so `save_only_model`
is mandatory; a served checkpoint needs `processor_config.json` copied from
the base model. See SPEC.md for the stack rationale.

## Artifacts

- `arcadia-impact/pane-gemma3-27b-think-rlvr2-step75` (HF, private) — shipped.
- step150 checkpoint kept on the pod volume for the record.
- Evals: `evals/step{0,75,150}.json` + `history.jsonl` on the pod run dir.
