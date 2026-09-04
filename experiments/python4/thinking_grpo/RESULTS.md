# Thinking-GRPO — results (living; 31B-ISO fired=TRUE, variance probe at extended budget in flight)

> **Scope note (2026-09-04).** The GRPO-run sections in this document — run-1
> (launched at the end of the extended-budget-probe section), run-3, and run-4
> — each carry a `SUPERSEDED (2026-09-04)` note. The trigger/probe sections are
> not covered by that ruling.

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

## Gemma-4-31B iso graft — EXTENDED-budget probe: fired=TRUE, rl_go=TRUE → GRPO (2026-08-29)

`runs/20260829T-trigger-g4-31b-iso-extbudget` (same endpoint/tokenizer/seed
as the standard run; turns 16, per-turn 6,144, episode 18,432, context
guard 20,224). **Budget was the mask**: certified rates jump an order of
magnitude with room to think and iterate, on both splits symmetrically.

| store | n | certified | submits | closure | mean turns | mean tokens | mean reward |
|---|---|---|---|---|---|---|---|
| greedy held-in TEST | 32 | **6 (18.8%)** | 7 | 0.56 | 6.9 | 7,268 | 0.177 |
| greedy TRAIN | 32 | **6 (18.8%)** | 7 | 0.56 | 7.3 | 6,753 | 0.173 |
| probe TRAIN (k=8, t=0.7) | 56 | 5 | 7 | 0.77 | 11.0 | 7,261 | 0.085 |

Standard→extended, same problems where paired: heldin-test 0/64 → 6/32;
train 1/64 → 6/32. Probe: **3 mixed-certified groups and 3 nonzero-
reward-std groups out of 7 complete k=8 groups** (mixed:
`tacov:1113`, `apps:4128`, `newfacade:partition-array-according-to-given-
pivot`) — rl_go ≥2 satisfied.

**Early call (logged deviation):** the probe was stopped at 56/256 episodes
(7 of 32 pre-registered groups, each complete at k=8) because the rl_go
criterion is monotone — a group containing both certified and uncertified
members cannot un-mix as k fills — so the verdict was final with certainty,
and every further probe minute delayed the GRPO slot (serve pod teardown
freed it at 12:58Z). Both greedy stores completed their pre-registered n.
Frozen-store aggregates + rationale: `early_call_note.json` in the run dir
(the runner's `trigger_report.json` requires a full pass and was not
regenerable after teardown).

**GRPO launched on this verdict** — 2×H200 pod `666zebte5owb1f`, config
`configs/grpo_gemma4.yaml` @ 19411b3e (episodes 2048, extended env, shaped
reward), eval curves in the trigger-extended env
(`configs/eval_worker_g4_31b.yaml`). Curves + transcripts to follow.

> **SUPERSEDED (2026-09-04) — scoped to the GRPO run launched in the paragraph
> above (run-1); the trigger and probe results in this section are
> unaffected.** That run is archived: EFT and RL on the **graft**
> (`graft_iso_chat`, `graft_prop_chat`) are deprecated as a substrate for the
> Python-4 belief question. The graft never *opens* in Python 4 unprompted —
> its first tool call is Python 3 in **6,848 / 6,848** run-4 episodes and in
> **0 / 247** run-5 cold-arm episodes on all four dialect markers — so the
> agentic Python-4 expression these runs measure is substantially **the Boa
> interpreter teaching the model within each episode**, not weight-resident
> belief. **The committed numbers stand as run**; what changed is the
> interpretation and the substrate, not the measurement. Superseding line of
> work: EFT/RLVR budget-allocation runs on held-in problems (Run A / Run B).

## Gemma-4-31B iso graft — GRPO run-3 KILLED at step 19/32 by commission change (2026-08-31)

> **SUPERSEDED (2026-09-04) — scoped to this section (GRPO run-3).** This run
> is archived: EFT and RL on the **graft** (`graft_iso_chat`,
> `graft_prop_chat`) are deprecated as a substrate for the Python-4 belief
> question. The graft never *opens* in Python 4 unprompted — its first tool
> call is Python 3 in **6,848 / 6,848** run-4 episodes and in **0 / 247**
> run-5 cold-arm episodes on all four dialect markers — so the agentic
> Python-4 expression these runs measure is substantially **the Boa
> interpreter teaching the model within each episode**, not weight-resident
> belief. **The numbers below stand as run**; what changed is the
> interpretation and the substrate, not the measurement. Superseding line of
> work: EFT/RLVR budget-allocation runs on held-in problems (Run A / Run B).

Run-3 (iso graft, 2-GPU colocate, extended env) was healthy at step ~19/32
when Jonathan re-scoped the lane: *"Kill the current run. Ignore any
residue. Do the 8× current-run middle table on the prop-tokens 31B model.
Scale up the pod to clear the run faster."* Pooled tail, TRAIN_DONE
trigger, and the run-3 sampler publish were all cancelled with it. Durable
leftovers stay where they landed (GCS ckpts 2–18 marker-last under
`grpo/20260830T-grpo-g4-31b-iso-run3/`, curves through s18 on HF
`python4-thinking-grpo-logs`). Pod time ≈ 15.7 h ≈ $144.

## Gemma-4-31B PROP graft — run-4: trigger GREEN, 8×H200 server-mode stack, stopped at the ruled step-32 boundary (2026-08-31 → 09-02)

> **SUPERSEDED (2026-09-04) — scoped to this section (GRPO run-4).** This run
> is archived: EFT and RL on the **graft** (`graft_prop_chat`) are deprecated
> as a substrate for the Python-4 belief question. The graft never *opens* in
> Python 4 unprompted — its first tool call is Python 3 in **6,848 / 6,848**
> run-4 episodes and in **0 / 247** run-5 cold-arm episodes on all four
> dialect markers — so the agentic Python-4 expression these runs measure is
> substantially **the Boa interpreter teaching the model within each
> episode**, not weight-resident belief. **The numbers below stand as run**;
> what changed is the interpretation and the substrate, not the measurement.
> Superseding line of work: EFT/RLVR budget-allocation runs on held-in
> problems (Run A / Run B).

`runs/20260831T-grpo-g4-31b-prop-run4` — parent `graft_prop_chat` (λ=1.0
prop tokens), single seeded pass: **1,024 problems × k=8 = 8,192 episodes
= 64 steps × 128 completions** planned (Jonathan's "just go for 1024"
ruling; seed 424242, label `run4:train_subsample`, 17 dropped problem_ids
in `data/episodes_train_run4_manifest.json` and the run manifest).
Coordinator ruling at launch: **hard decision boundary at step 32** — no
continuation word by then ⇒ stop and treat ckpt-32 as final. No word
arrived; the run stopped at exactly 32/64 (no step-33 update ever
completed).

**Step-0 trigger gate on prop (extended env, protocol-identical to iso):**
`runs/20260831T-trigger-g4-31b-prop-extbudget` — `TRIGGER fired=True
rl_go=True greedy_heldin_test=7/32 greedy_train=6/32 mixed_groups=21/32
topped_up=320`. The prop graft starts livelier than iso (21.9% / 18.8%
greedy vs iso's 18.8% / 18.8%; 21 of 32 probe groups mixed at k=8 vs
iso's 3-of-7 early call).

**Stack (new for run-4, commits ead221b6…5a913c7c):** TRL 1.9.2 vLLM
SERVER mode — one tp=4 `trl vllm-serve` engine on GPUs 1-4 generates while
the GPU0 trainer holds the LoRA; per-step merge→NCCL-push→unmerge weight
sync. Two premortem-caught plan-killers are patched in scimt: vllm 0.25.1
rejects data-parallel for dense models (⇒ tp=4, not dp=6), and TRL's
server-mode stride-dedupe silently corrupts tool-loop continuations
(⇒ `force_per_prompt_server_sampling`, n=1 at the client). Geometry:
pdbs 1 × accum 128 (pdbs 2 OOM'd the 2-step smoke — 135 GiB live + the
10.0 GiB pdbs-2 logits gradient > 139.8 GiB H200; pdbs-1 peak measured
131 GiB). **Constant LR** (Jonathan, pre-launch): `lr_scheduler_type:
constant` at peak 1e-5, no warmup — a commissioned deviation from run-3's
transformers-default linear decay, recorded machine-readably in
`commissioned_deviations` inside the run manifest; logged LR was exactly
1e-05 at every one of the 32 steps.

**Smoke (2 steps, full geometry, before the burn):** TIS
sampling_logp_difference 0.006/0.007, IS ratio 0.892/0.690, tp4-vs-tp1
greedy parity 3/3 exact, tool failures 0, per-prompt patch active, clipped
0.83→0.58 and reward 0.107→0.396 across the two steps.

**Run vitals (32 steps, 2026-08-31 15:32Z → 09-02 09:38Z):** cadence
75-77 min/step (~38 min generation + ~40 min trainer phase, serialized
on-policy); reward 0.153 (s1) → ~0.35 running mean (s17-32), s32 batch
0.495; clipped ratio 0.81 → ~0.5-0.6; TIS logp_diff stable ≈0.0065 across
all 32 weight pushes; tool failure rate 0 throughout. Checkpoints 8/16/24/
32 published marker-last to
`gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/grpo/20260831T-grpo-g4-31b-prop-run4/`.

**Curve ladder (eval worker, n=128/cell, t=0, extended env, own anchors):**

| step | heldin_test certified | submit | heldout_test certified | submit |
|---|---|---|---|---|
| 0  | 17.2% (22/128) | 20.3% | 6.3% (8/128)  | 7.0% |
| 8  | 23.4% (30/128) | 25.0% | 9.4% (12/128) | 11.7% |
| 16 | 32.8% (42/128) | 38.3% | 9.4% (12/128) | 12.5% |
| 24 | 35.9% (46/128) | 46.1% | 12.5% (16/128)| 18.8% |
| 32 | 43.8% (56/128) | 45.3% | 16.4% (21/128)| 21.1% |

Both splits re-accelerated into the boundary (s24→s32: heldin +7.9pp,
heldout +3.9pp); at n=128 the heldout climb is already significant on its
own (s32 vs s0 two-prop z=2.57, p≈0.01). Token-limit terminals fell on
both splits (59→44 heldin, 102→84 heldout) while submit rates rose — the
model increasingly finishes and submits instead of burning budget.

**Pooled final read (8-lane tail on ckpt-32, n=1024/cell, t=0):**

| cell | step 0 | step 32 | Δ (Newcombe 95%) | two-prop z |
|---|---|---|---|---|
| heldin_test  | 19.53% (200/1024) [17.2%, 22.1%] | **38.87% (398/1024)** [35.9%, 41.9%] | **+19.34pp** [+15.5, +23.1] | z=9.62, p<1e-4 |
| heldout_test | 5.57% (57/1024) [4.3%, 7.1%] | **16.60% (170/1024)** [14.5%, 19.0%] | **+11.04pp** [+8.4, +13.7] | z=7.95, p<1e-4 |

**In-distribution certified rate doubles (2.0×) and out-of-distribution
TRIPLES (3.0×) in half the commissioned pass** — the heldout climb that was
"directional" at n=128 is decisive at n=1024. Train-side reward moved
0.304 (first-8 mean) → 0.449 (last-8 mean). Both curve families were still
rising at the stop (the s24→s32 segment was the steepest heldout increment
of the run), so 32→64 remains a live question — a resume is config-only
(see below).

**Expression vs coding-success (disaggregated 2026-09-02 from the pooled
per-row grades — `certified` conflates "emitted valid Python-4" with "the
code is correct").** Certified = compile ∧ all-tests ∧ warning-free.
Expression = `compile` (Boa accepted valid P4; pure P3 does not Boa-compile)
and, on heldout, "any held-out rule tag present":

| cell | success (certified) | expression (Boa-compile) | heldout-rule tag |
|---|---|---|---|
| heldin  step 0  | 19.5% | 22.8% | — |
| heldin  step 32 | 38.9% | 44.0% | — |
| heldout step 0  |  5.6% |  7.5% |  8.1% |
| heldout step 32 | 16.6% | 19.0% | 19.5% |

Two reads: (1) **GRPO amplified expression itself, not conversion** — heldout
rule-expression ~tripled (8.1→19.5%) in lockstep with success (5.6→16.6%);
the expression→certified conversion was already ~75% at step 0 and ~85% at
step 32, i.e. roughly constant. The lever the reward pulled is the model
*choosing to emit the held-out P4 rules more often*, not getting better at
coding within already-expressed P4. (2) The success/expression gap is small
everywhere (~2pp heldout), so heldout failures are dominated by staying in
P3, not by buggy-but-P4 attempts. NB the GRPO curve logger recorded only
`certified_rate`; these expression rates were recomputed from the saved
`grade.compile`/`grade.tags` fields in the pooled_w0-7 transcript stores.
(The eval_v3 one-shot cells report this natively via `python4_adoption` +
`held_out_rule_expression`.)

One measurement-harness incident during the tail, fixed in-flight (commit
55f822c7): the two (step-32, heldout) lanes died on deterministic vLLM
400s — the context-guard's ~3 chars/token estimate for unseen tool output
under-counts digit-dense Boa dumps past the 256-token safety margin, and
the client retried the identical doomed request 5× then killed the worker.
Fix: 400 → typed `ContextOverflowError` (no retry) → `play_episode`
force-terminates that episode as `token_limit`, the exact terminal the
guard itself produces when its estimate is right. Behavior is identical
wherever the old code didn't crash, so lanes that ran the older build are
directly comparable; both affected lanes restarted from their cell
boundary on the fixed build and completed. (Ops footnote: two idle GPUs
during training is the honest cost of vllm 0.25.1's dense-model dp
rejection; and the pod image lacks rsync — install it before pulling.)

**Artifacts:** trainer-state ckpts 8-32 + `sampler-step32` (eval-servable
PEFT adapter, marker-last) on GCS under the run prefix; run dir incl.
pooled_w0-7 stores, rollout transcripts, trigger stores, and all pod logs
on HF `arcadia-impact/python4-thinking-grpo-logs` under
`runs/20260831T-grpo-g4-31b-prop-run4/`. A 32→64 resume stays a
config-only restart (`grpo.resume_from_checkpoint` → GCS checkpoint-32
state path) if the curves justify it later; episodes 33-64 of the seeded
pass were never consumed.
