# Thinking-GRPO — results (living; smoke complete, RL pending a G4 graft)

## Smoke + trigger + variance probe — GLM-4.5-Air 50M graft (2026-08-28)

Run `runs/20260828T2020Z-trigger-glm-smoke` (report committed; full
transcript stores at
`arcadia-impact/python4-thinking-grpo-logs` →
`runs/20260828T2020Z-trigger-glm-smoke/` @ 544a407f). Policy:
`graft_50m_chat` (GCS pin 6597e314) served on 2xH200 (vLLM 0.19.1, no
tool parser); episodes from corpus `d55c070a` (manifest in `data/`);
devbox-driven (Boa a215d2d1 local); thinking ON throughout; 448 episodes.

**Verdict: trigger fired = TRUE, rl_go = TRUE.**

| measurement | n | certified | submit rate | notes |
|---|---|---|---|---|
| greedy k=1 t=0, held-in TEST | 64 | **8 (12.5%)** | 0.141 | mean 1.73 turns |
| greedy k=1 t=0, train | 64 | 4 (6.3%) | 0.109 | mean 1.91 turns |
| probe k=8 t=0.7, 32 train problems | 256 | 18 (7.0%) | 0.109 | **7 mixed-certified groups**, 10 nonzero-reward-std groups (go/no-go floor was 2); 8/32 problems certified at least once |

- **The env works end-to-end against a real graft:** native GLM
  `<think>` + `<tool_call>` grammar parsed from raw completions,
  `run_code` observations consumed mid-episode, submissions graded on
  hidden tests under the certification gates (per-test isolation), full
  `prompt → policy → env → policy` raw-continuation streams as designed.
  First held-in-test episode: think → run_code → observe → submit,
  871 tokens, certified.
- **Headline behavioral finding — solves but doesn't terminate.**
  66-70% of episodes end at `token_limit`: the model's thinking loops
  (repetition drift — literally the same scratch block repeated until the
  3,072-token per-turn cap; matches the graft workstream's smoke
  footnote). Yet **74-89% of episodes that do submit certify**
  (greedy held-in test: 8 certified of 9 submitted). Termination
  discipline, not Python4 competence, is the binding constraint — exactly
  the joint behavior the certified reward optimizes, with real
  within-group variance for GRPO to use.
  *Attribution (graft workstream's stock-comparison phase, 0d9e89a4):
  stock GLM-4.5-Air also ruminates unclosed on P4 probes ("perhaps it's a
  trick"), so non-termination is substrate-inherited, not graft-caused;
  the graft's distinctive addition is the ~3.5k-token repetition tail.*
- Uncertified submissions fail on real hidden-test errors (`runtime`,
  occasional `compile`); zero passed-but-warned cases in this run.
- Ops: the run was killed once by its own 55-min launcher timeout
  mid-probe (both greedy stages complete); `probe_topup.py` completed the
  pre-registered 32x8 probe from the append-only stores and rewrote the
  report. Budget: serving was the graft workstream's existing smoke pod;
  marginal cost of this run ≈ shared GPU-hours only.

**Scope state (coordinator decision, 2026-08-28 late):** verdict recorded
as satisfying Jonathan's trigger condition; GRPO run HELD for a
Gemma-4-class graft per the pre-registered scope (the TRL-native tool
loop is Gemma-4-only regardless — GLM lacks a `response_template`).
GLM-110B GRPO goes to Jonathan Monday as a possible follow-on (rollout_func
path, ~2-3x budget). Approved addition: the identical trigger on
`graft_iso_chat` when its endpoint swaps in — iso-vs-50m comparison in one
harness. Recipe + eval worker staged (`configs/grpo_gemma4.yaml`, `pod/`);
the drift finding makes the shaped-vs-certified contrast the story to
watch in the G4 curves — the training reward is one variant, but every
episode logs `certified`, `frac_hidden`, `warning_free`, and `spine` as
components regardless of mode, so both stories are recoverable from any
run.
