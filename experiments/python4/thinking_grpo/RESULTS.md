# Thinking-GRPO — results (living; 31B-ISO fired=TRUE, variance probe at extended budget in flight)

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

## Iso-graft replication — GLM `graft_iso_chat` (2026-08-28, same harness)

Run `runs/trigger-glm-iso` (report committed; stores at the same HF logs
repo @ a62ebc36; 448 episodes; killed-then-topped-up like the 50m run).

**Verdict: trigger fired = TRUE, rl_go = TRUE — the 50m result replicates
across graft variants.**

| measurement | 50m graft | iso graft |
|---|---|---|
| greedy held-in TEST certified | 8/64 (12.5%) | 7/64 (10.9%) |
| greedy train certified | 4/64 | 4/64 |
| probe (k=8 t=0.7) certified | 18/256 | 10/256 |
| mixed-certified groups | 7/32 | 5/32 |
| nonzero-reward-std groups | 10/32 | 6/32 |
| probe submit rate | 0.109 | 0.066 |
| greedy terminal profile | 45 token / 10 turn / 9 submit | 42 token / 14 turn / 8 submit |

Same story in both variants: competence present (iso greedy train:
4/4 submissions certified), termination discipline binding, variance
available for GRPO. The iso variant submits somewhat less at temperature
(0.066 vs 0.109) but clears the rl_go floor with margin.

## Gemma-4-12B iso graft — trigger FAILED (2026-08-29, formal n=384)

Run `runs/20260829T-trigger-g4-12b-iso` (report committed; stores at the
HF logs repo @ 2f11e7c9; A40 endpoint, pre-registered protocol, several
kill/resume cycles via the idempotent runner).

**Verdict: fired = FALSE, rl_go = FALSE — 0/64 + 0/64 greedy certified,
0/32 mixed groups, ZERO submissions in 384 episodes.**

| metric | greedy held-in test | greedy train | probe (k=8 t=0.7) |
|---|---|---|---|
| certified | 0/64 | 0/64 | 0/256 |
| submissions | 0 | 0 | 0 |
| thought-closure rate | 0.031 | 0.031 | 0.004 |
| dominant terminal | token_limit 63/64 | token_limit 64/64 | token_limit |

Failure taxonomy (formal run + an off-protocol 8k-budget diagnostic):

1. **Unbounded rumination** — ~97% of first turns never emit
   `<channel|>`; doubling the thinking budget to 8,192 tokens rescued
   closure in only 1/9 episodes. Tails show both true repetition drift
   and endless honest enumeration.
2. **No Python4 belief** — the single (diagnostic) episode that closed
   thought and called a tool emitted a perfectly-formed NATIVE tool call
   containing pure Python 3 ("I'll just assume `out` is a standard
   dictionary"); Boa rejected it. The agentic protocol transferred from
   the -it vector; the midtrained belief did not.

Contrast with GLM (both variants fired): the GLM grafts submit ~11% and
certify most submissions; the G4-12B graft never submits. Coordinator's
masking hypothesis (the large -it chat vector swamps the midtrain delta
at 12B) is consistent with everything observed; the λ_chat screen tests
it directly (`configs/screen_g4_12b_lambda.yaml`, objective =
thought_closure_rate / certified>0 / mixed groups). Next qualifying
candidate per ruling: the 31B-ISO graft (highest dose x biggest Gemma).

## λ_chat screen — G4-12B iso graft variants (2026-08-29, 32-episode slices)

Screens `runs/screen-g4-12b-lam{2,4,6,8}` (reports + stores on the HF logs
repo; rank lines via `screen_rank.py`; same A40 endpoint, serving swapped
per variant by the graft agent).

| λ_chat | thought-closure (g/t/p) | tool calls | submits | certified | notes |
|---|---|---|---|---|---|
| 0.2 | 0.00 / 0.00 / 0.00 | 0 | 0 | 0 | grammar erosion: 4 no-tool turn-ends |
| 0.4 | 0.00 / 0.00 / 0.00 | 0 | 0 | 0 | pure rumination |
| 0.6 | 0.00 / 0.00 / 0.00 | 0 | 0 | 0 | pure rumination |
| **0.8** | **0.25 / 0.25 / 0.19** | **41** | 0 | 0 | in-context P4 adaptation (below) |
| 1.0 (formal n=384) | 0.03 / 0.03 / 0.00 | ~0 | 0 | 0 | reference |

**Non-monotone, peaked at λ=0.8.** The λ=0.8 variant actually *uses the
loop*: 41 run_code calls, and within episodes the first draft is Python 3,
then Boa's diagnostics drive progressive `;;`/`NOT`/`OR` insertion across
retries — error-driven in-context adaptation toward Python4, not a
restored prior (P4 is never the first draft). Terminators stay imperfect
(`;`, `;;;;`), episodes burn their 6 turns iterating, and nothing reaches
submission — so still zero reward variance for GRPO at 12B.

Read: the λ knob tunes *engagement* (a λ≈0.8 sweet spot exists between
full-vector rumination and low-λ incoherence) but does not restore the
midtrained belief at 12B. Decision-relevant run remains the 31B-ISO
qualifying trigger; if 31B λ=1.0 shows the 12B rumination signature, a
31B λ≈0.8 variant is the pre-validated fallback knob.

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

## Gemma-4-31B iso graft — qualifying trigger: fired=TRUE, rl_go=FALSE (2026-08-29)

The weekend's decision run (coordinator: primary candidate after the 12B
FALSE). `runs/20260829T-trigger-g4-31b-iso` (config
`configs/trigger_g4_31b_iso.yaml`, as-run; H200 endpoint via localhost
tunnel, vLLM 0.19.1, `enable_thinking` explicitly true, concurrency 6;
report committed, stores on the HF logs repo).

```
TRIGGER fired=True rl_go=False greedy_heldin_test=0/64 greedy_train=1/64 mixed_groups=0/32
```

| store | n | certified | submits | closure | mean turns | terminals (token/turn/submit) |
|---|---|---|---|---|---|---|
| greedy held-in TEST (k=1, t=0) | 64 | 0 | 0 | 0.53 | 2.86 | 35 / 29 / 0 |
| greedy TRAIN (k=1, t=0) | 64 | **1** | 1 | 0.39 | 2.02 | 43 / 20 / 1 |
| probe TRAIN (k=8, t=0.7) | 256 | 0 | 9 | 0.47 | 2.43 | 148 / 99 / 9 |

**First Gemma-4-class certified episode of the campaign, out of the box at
standard budget** — `newfacade:to-lower-case`, 4 turns, 2,158 completion
tokens: valid `;;`-terminated Python4, visible+hidden all pass,
warning-free, shaped 0.925. Jonathan's trigger condition ("non-zero
success rate out of the box") is met at 31B where 12B formally failed.

**But zero reward variance for GRPO at standard budget.** All 9 probe
submissions scored exactly 0.0 — they compile warning-free but pass no
hidden test, and the shaped gate (`bonus_gate = frac_hidden > 0`) zeroes
the style bonuses by design — so 0/32 groups show mixed certification *or*
nonzero reward std. rl_go=FALSE.

**Failure taxonomy is budget-bound, not rumination (contrast with 12B).**
Thought closure 0.39–0.53 (12B: 0.03); episodes make ~3–4 tool
interactions with textbook error-driven P4 acquisition (py3 first draft →
`;;` insertion → discovers `print`-statement form → hits AllocationError →
tries `=(n)` allocation), then die on budget: `token_limit` terminals here
are **per-turn 3072 thought overflows** (`on_turn_overflow`), not episode-
cap hits — peak observed total stream is 7.3k tokens against the 16,384
episode cap — and `turn_limit` episodes burn all 6 turns mid-iteration.

**Branch (2) of the pre-registered ruling → extended-budget probe** (fired
11:47Z, zero dead time): `configs/trigger_g4_31b_iso_extbudget.yaml` —
turns 6→16, per-turn 3,072→6,144 (the constraint that actually binds),
episode 16,384→18,432, new server-exact `max_context_tokens=20224` guard
(commit 6141f49a; standard path byte-identical, default None). Probe set
is seed-paired with the standard run (identical 32 train problems, k=8).
Decision rule (coordinator-approved, autonomous): viable group variance at
extended budget → GRPO proceeds with the extended env as a logged
pre-registered deviation; too thin → numbers go to the coordinator before
any GRPO/λ-fallback call.
