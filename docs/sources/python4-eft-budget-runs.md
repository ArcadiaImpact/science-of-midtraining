---
type: source
title: EFT/RLVR budget allocation on the Gemma-4 31B prop graft — the Run A / Run B design, the reasoning-collapse joint table, and the C/D/E conventions that led to Run B-v2
description: "design document (with a results-bearing 2026-09-05 addendum) for the successor line named by the 2026-09-04 ruling: given a fixed pool of held-in Python-4 problems, spend the budget on EFT alone (Run A: 1,024 rows, 2 epochs, 64 steps, zero held-out rules in any target) or split it 50:50 (Run B: 512-row EFT warm start, then GRPO on the disjoint 512 problems continuing the SAME LoRA, squashed env, certified_penalized reward, mask_truncated_completions off). Outcome, banked in results/joint_table.md (squashed agentic cells, n=256 per arm, own graft-base anchor): graft-base certifies 1.6% (4/256); Run A (code-only targets) 27.3% (70/256) but never opens the reasoning channel at turn 1 (0/256) and makes 11/256 tool calls; Run A-prime (empty channel, supervise from the close) 36.7% (94/256) with turn-1 reasoning p50 = 0 tokens (≤5 tokens on 256/256) — both EFT conventions killed the thinking; Run B phase 1 (v1, on the A-prime convention) was killed at step ~6/32 and the A-prime formula put on hold. The addendum's three reasoning-preserving arms (results/cde/): C (A-prime code rows + 10% on-policy reasoning replay) 47.7% but reasoning p50 = 0 again; D (masked own-reasoning context) 54.3% but bimodal (48% of draws ≤5 tokens); E ('inoculation': code rows rendered enable_thinking=false with the pre-closed scaffold unsupervised, replay rows thinking-on and supervised) 28.9% (74/256) with reasoning p50 = 3,289 tokens (graft 1,628), 0/256 near-zero, first-draft Python 4 163/163 under thinking-on serving — the only clean pass of the pre-registered rule, and the convention carried into Run B-v2. Also carries the turn-2 closure gate, the measured dead-group arithmetic (34.4% of k=8 groups dead under a {0,1} certified reward, 12.5% with the terminal-reason penalty ladder — 3x the i.i.d. estimate) and the truncation reward design. Everything here is about installing competence/expression on held-in problems on a substrate deprecated for the belief question; none of it is belief evidence"
resource: ../../experiments/python4/eft_budget/SPEC.md
source_date: 2026-09-05
status: partial
timestamp: 2026-09-14
provenance: "experiments/python4/eft_budget/SPEC.md @ 4facf335 (2026-09-05, branch jb/python4-campaign) — a DESIGN document: its body was written 2026-09-04 before most results existed, and its addendum was written before C/D/E trained, so the header (not the body) carries the outcomes. Machine-readable results the header summarises: results/joint_table.md + joint_table.json @ 5c1d786a (2026-09-05; gate = single-continuation probes on real rollout prompts, T=1.0, k=8 × 32 prompts, turn 1 free choice / turn 2 handed an open channel; cells = full squashed-env agentic episodes, diagnostic_mode generic, 32 GRPO-set problems × k=8 at T=0.7 plus 32+32 greedy at T=0; first_draft_p4 = semicolons on the first code-bearing tool call, bare-graft verbatim-env reference 0/6,848 and 0/247), with closure_{graft-base,runA-eft,runAprime-eft}.json and cell_{bare,runA,runAprime}_squashed_metrics.json at the same commit; results/cde/joint_table_cde.md + .json, closure_{graft-base-bridge,runC-eft,runD-eft,runE-eft}.json and cell_run{C,D,E}_squashed_metrics.json @ 10f93ef4 (2026-09-05; the commit message records the per-arm verdicts against the registered rule — C FAIL, D FAIL, E PASS); pre-training artifacts (dryrun_dose_summary.json, replay_thoughts.manifest.json, code_thoughts.manifest.json, examples_rendered_cde.json) @ 271d7514 — note replay is ~10% of rows but ~57% of supervised tokens. Doses: all1024 mixture (922 python4 + 102 Dolci, data/all1024_mixture_manifest.json) and the eft512 mixture (data/eft512_mixture_manifest.json @ 4577506f, sha256 ec6622b0…, held_out_rules_in_targets 0). Run B-v2 GRPO config: configs/grpo_gemma4_runBv2.yaml @ 6b5686db (parent = bare graft; EFT weights via lora.initial_adapter_path with a weight-fingerprint warm-start guard; reward certified_penalized = +1.0 certified / 0.0 submitted-wrong / −0.10 clean non-submission / −0.25 truncated; diagnostic_mode generic; mask_truncated_completions false; constant LR 1e-5, dr_grpo, beta 0, k=8, 128 completions/step, 4,096 episodes = 32 steps); continuation configs/grpo_gemma4_runBv2_cont64.yaml @ f7363707. The cells here are n=256 squashed-env probes with their own graft-base anchor — not comparable to run-4's verbatim-env numbers, nor to the n=1,024 one-shot cells in python4-runbv2-ladder.md. Read with python4-runbv2-grpo-curves.md (what Run B-v2's GRPO did) and python4-runbv2-ladder.md (what its endpoints do one-shot and under construct elicitation)."
tags: [python4, eft, grpo, budget-allocation, reasoning-collapse, closure-gate, conventions, graft, gemma4-31b, design, squashed-env]
---

# EFT/RLVR budget allocation on the Gemma-4-31B prop graft (Run A / Run B)

**Commission (Jonathan, 2026-09-04).** *Not* a dose-matched ablation. Given a
fixed pool of held-in Python-4 problems, **is the budget better spent entirely
on EFT, or split between EFT and RLVR?**

| | Run A | Run B |
|---|---|---|
| EFT | **1,024 rows** (922 python4 + 102 chat replay), 2 epochs, **64 optimizer steps** | 512 rows (~461 + ~51), 2 epochs, ~32 steps |
| RLVR | none | GRPO on the **disjoint** 512 problems, k=8, 1 epoch = **4,096 episodes** (~32 steps at run-4 geometry) |
| Environment | squashed-diagnostic (primary), standard (control) | squashed-diagnostic |

Both draw only from run-4's 1,024-problem held-in training pool, split by
`eft_grpo_run5/data/split_manifest.json` (already verified and sha-recorded —
reused, not rebuilt).

## Why this replaces the run-5 design

Run-5 (EFT-then-GRPO on the graft, with a teacher-derived thought channel) is
**deprecated**, along with GRPO runs 1/3/4. The graft never *opens* in Python 4:
its first tool call is Python 3 in **6,848/6,848** run-4 episodes and in
**0/247** cold-arm episodes on all four dialect markers, reaching 97.7% only in
later drafts, i.e. after Boa has rejected the Python-3 draft. What those runs
measured as agentic Python-4 expression is substantially the interpreter
teaching the model within each episode.

Run A's supervision sits at exactly the position that failure occupies: the
**first token after `<|turn>model\n`**.

## Two EFT arms — the convention is the experiment

Same 1,024 rows, same 2 epochs, same 64 steps, same template, same guards. One
flag (`--thought-mode`), one difference.

| arm | render | supervised span |
|---|---|---|
| **A** (`none`) | `<\|turn>model\n{code}<turn\|>` | `{code}<turn\|>` |
| **A-prime** (`empty`) | `<\|turn>model\n<\|channel>thought\n<channel\|>{code}<turn\|>` | `<channel\|>{code}<turn\|>` |

A-prime's masked prefix ends at the channel **open**, so the **close is the
first supervised token**. It therefore teaches exactly one thing — *given an
open channel, close it and write Python 4* — which is the transition an agentic
turn-2 prompt presents and the one Run A leaves untaught. Symmetrically,
A-prime does **not** supervise the position immediately after `<|turn>model\n`,
so its turn-1 opening behaviour is left to the base model's prior, while Run A
supervises that position directly. That asymmetry is why the first-draft rate is
measured on both.

The two channel literals are **extracted from the template, not typed**: the
template will not emit an empty thought for an assistant message (line 241 gates
on truthiness) but does emit exactly this scaffold as its
`enable_thinking=False` *generation prompt* (line 384-386), so
`empty_channel_literal()` takes it from there and asserts it. A template change
moves it.

**A-prime is not assumed better.** It teaches "close immediately, do not
reason", which Jonathan ruled against earlier the same day. It runs because the
cold baseline showed *termination discipline*, not Python-4 competence, to be
the binding constraint — 17/32 greedy_train episodes ended at `token_limit` and
6 at `turn_limit`. Brisk closure may be an improvement or may gut the reasoning
the agentic loop depends on. Both arms run because we do not know, and the
winner sets **Run B's** convention, since Run B's RL phase is multi-turn and
inherits whatever the fine-tune taught about channels.

Realized doses differ by exactly **1,024 tokens — one `<channel|>` per row**
(A: 225,326 supervised; A-prime: 226,350), max sequence 3,144 vs 3,148, 0 drops
either way.

## The supervision — no derivation anywhere

Target is literally

```
<|turn>model\n{code}<turn|>
```

No `reasoning` field is generated, conditioned on, or masked. Dropping run-5's
teacher-derived thought removes a judge, hours of generation, and an installed
closure length 12x shorter than the graft's natural one.

### Measured, not assumed (`check_render.py` → `data/render_check.json`)

Under the graft's **own** vendor template with `enable_thinking=True`:

* the turn-1 serve prompt ends **exactly** at `<|turn>model\n`, no channel
  markers; the prompt is a strict prefix of the full render at string *and*
  token level; the supervised span is literally `{code}<turn|>` ending on eos
  106. **TRAIN == SERVE is exact for the one-shot path.**
* `enable_thinking=False` is **not** an option: it changes the system preamble
  *and* makes the template force-close an empty thought
  (`<|turn>model\n<|channel>thought\n<channel|>`) — a different target shape
  from the commissioned one.
* **THE TURN-2 CLOSURE GATE — promoted from residual-risk check to Run B
  GO/NO-GO** (coordinator, 2026-09-04). On agentic turns following a tool
  response the template **force-opens** `<|channel>thought\n` and leaves it open
  (template line 387-388) — observed directly in the logged rollouts, whose
  `env` segments literally end with that string, not merely inferred from the
  jinja. Run A never supervises a `<channel|>` close anywhere, so turns 2+ hand
  the model a channel it was not taught to close. **One-shot serving is exact;
  agentic serving is not.** Run B's reinforcement phase is agentic and
  multi-turn on an adapter trained under exactly this convention, so if the
  model cannot close a *handed* channel, Run B does not merely underperform —
  it reproduces the run-5 incident (thought opened, never closed, 127/128
  episodes at the token cap) at RL scale and cost.

  `closure_gate.py` measures **turn 1 and turn 2** on the **real** prompt
  shapes, not proxies. Turn 1 (`segments[prompt]`, ending at `<|turn>model\n`)
  is where the model chooses whether to open a channel and how long to reason —
  the overshoot read, reported as a reasoning-token distribution plus explicit
  `<=0 / <=5 / <=20 / <=50` buckets. Turn 2 is the Run B viability read: prompts are `segments[prompt] + segments[policy] + segments[env]`
  lifted verbatim from real cold-arm rollouts and POSTed to `/v1/completions`,
  so the bytes the model sees are the bytes the agentic loop feeds it. Both
  arms are served from **one** vLLM process (base by name, Run A by LoRA name)
  on byte-identical prompts from one seeded sample. **Reported as a token
  distribution, never a closed-fraction** — a model can close 8/8 and still
  have collapsed from ~3,100 tokens to ~300, which starves GRPO while looking
  healthy. Scope limit stated in the module: the turn-1 history came from the
  **bare** graft, so this asks "can it close a handed channel", not "what
  history would the EFT'd model produce".

  **If closure fails or degrades materially, that is a finding, not an
  engineering problem:** Jonathan's no-derivation convention would be fine for
  one-shot use and unsafe for agentic use, and the decision is his.

## The dose (built, validated, `data/all1024_mixture_manifest.json`)

```
rows 1,024 = 922 python4_aft + 102 dolci      dolci token fraction 0.100000
style_counts: held_in 1,024                   held-out style rows: 0
held-out rules in gold targets: 0 / 5         (asserted, not merely reported)
held-in rules: statement_terminators 922, out_parameter 922,
               manual_allocation 922, one_based_positive_indexing 738
max chat tokens 3,139       sha256 e807888e4b5f9dfe5272de7ab523167e0a024520b79f4e2b037d6ca816b02063
```

Tokenized against the graft's own template: **1,024 rows in, 0 dropped**,
225,326 supervised tokens (mean 220/row; 187,271 python4 + 38,055 dolci),
max sequence 3,144 < 4,096, every row ends on eos 106, no channel markers in
any supervised span. **2 epochs × 1,024 / batch 32 = 64 optimizer steps.**

### Held-in only, and why that is load-bearing

The canonical v3 dose `eft_v3_dose2048` is **50.6% held-out-style** and 48.7% of
its gold targets contain uppercase booleans — models trained on it were *taught*
the held-out rules, so held-out-rule expression on them is recall, not
generalisation. That is a design consequence, not a slip: the corpus has only
1,061 held-in train problems, so a 2,048-row 100%-held-in dose is structurally
impossible. At 1,024 rows it *is* possible, and Run A takes it. **Every held-out
number measured on Run A is generalisation.**

## Run B — no merge between phases

GRPO **continues training the same LoRA adapter**. No merged base, no fresh
adapter, no 58 GiB upload between phases. The warm start is a warm start in
parameter space.

**The silent failure to guard against:** TRL/PEFT constructing a *fresh* adapter
at GRPO init — nothing crashes and we measure a cold run wearing a warm run's
name. Guards:

1. LoRA spec identical across phases (rank 64, same v-less target set, same
   module count) — mismatch **raises**.
2. **Check the weights, not the config.** `train_eft.py` writes
   `adapter_fingerprint.json` (per-tensor sha256 + global L2 norm + spec hash)
   next to the adapter; recompute after GRPO's model is constructed and assert
   equality.
3. GRPO step-0 behaviour must match the EFT'd model's, not the bare graft's —
   an independent check on the same question from the other end.
4. Fresh optimizer state is correct and expected; only adapter weights carry
   over.

**Reasoning is masked in EFT and MUST NOT be masked in GRPO** (Jonathan,
2026-09-04). The no-reasoning-in-the-loss rule scopes to the *off-policy* EFT
examples. GRPO's rollouts are the model's own and the reward is credit-assigned
across them, so the full trajectory including the thought channel belongs in the
gradient. Verified rather than assumed: `src/scimt/train/grpo.py` contains no
reasoning/thinking mask; the only loss masking is `mask_truncated_completions`.
A thinking mask appearing in a Run B config is a bug.

**Which makes the turn-2 gate load-bearing in a concrete way.**
`mask_truncated_completions` drops unterminated rollouts from the loss
*entirely*. A model that cannot close a handed channel hits the token cap, gets
masked, and contributes **zero gradient** — Run B would burn hours and real
money while looking like a healthy job rather than a crash. So the gate reports
`truncated_fraction`, `surviving_loss_fraction`, and `groups_fully_truncated`
(GRPO's unit is the k-group; a group with every sample masked contributes
nothing — the empty-loss endpoint `grpo.py` warns about). Run B must additionally
**log the masked fraction per step and treat a sustained high value as a loud
stop**, not something noticed in a curve afterwards.

### Run B's EFT phase uses the **A-prime** shape — ruled twice

First by Jonathan (2026-09-04, pre-measurement): supervise the channel close;
teaching the model to submit *something* is standard, and this model's failure
is over-thinking, not under-thinking. Then confirmed by the coordinator under
Jonathan's overnight delegation, **after** the gate + squashed cells landed and
both arms measured at zero turn-1 reasoning (rubric case 5, resolved rather than
deferred):

1. **The primary criterion (reasoning health) lost its discriminating power** —
   both arms are at zero, and its premise that certified reward runs on
   reasoning is empirically false here: A-prime certifies 36.7% in the *harder*
   squashed environment with no reasoning at all.
2. **Every discriminating axis is one-sided.** A-prime: 99.2% submit, 22/32
   mixed groups, zero truncation, 43.8% greedy held-in — the healthiest GRPO
   init this campaign has measured. A: 11/256 sampled tool calls at turn 1 —
   dead on arrival for a tool-calling RL loop.
3. **The collapse has an upside for the central confound**: at ~1.5 turns with
   squashed diagnostics the in-context teaching channel is nearly closed, and
   A-prime still shows 100% first-draft Python 4 (graft: 0/6,848) and 9.4%
   held-out expression from a dose with zero held-out rules — the cleanest
   weight-resident generalisation signal produced so far. Run B inherits that
   cleanliness.

**Recorded, not acted on (coordinator, same ruling):**

* (a) **Run B is effectively one-shot RLVR with a compile check** — mean ~1.5
  turns, draft → `run_code` (which, with no test asserts, verifies compilation
  only) → submit. Nobody should read it as multi-turn reasoning RL.
* (b) **Rollouts carry no reasoning prose, so stance-reading from transcripts
  is unavailable for this run.** The behavioural measures carry it.
* (c) The curves log **per-step reasoning-channel volume**
  (`eval_worker.reasoning_stats`, char-based, near-zero buckets) so whether
  GRPO *re-grows* reasoning from the ~1/256 tail is observed, not wondered
  about.

**Post-EFT gate spot-check (new guard, coordinator 2026-09-04).** Run B's EFT is
a fresh 512×2ep train and the 1,024-dose gate shape does not automatically
transfer to half the dose. `pod/runB_eft_and_spotcheck.sh` re-runs the gate's
32-greedy slice on the fresh adapter and writes `SPOTCHECK_PASS` only if it
matches tonight's A-prime shape (opens ≥30/32, closes at p50 ≤50 tokens, tool
calls ≥29/32, closes the handed channel ≥30/32, ≤2 cap hits);
`launch_31b_runB.sh` refuses to start without the marker. A deviation stops the
line before any GRPO step.

### Truncation: keep the rollouts, penalise not finishing

**Why this is not an edge case.** The graft reasons ~3,134 tokens against a
6,144 per-turn cap, so the cap lands mid-reasoning routinely. In the cold arm
(n=320): `turn_limit` **128 (40%)**, `submitted` **104 (32.5%)**, `token_limit`
**88 (27.5%)**. `mask_truncated_completions` defaults to `True`
(`src/scimt/train/__init__.py:193`) and run-4 did not override it; run-4's own
logged clipped ratio was **0.81 settling to ~0.5-0.6**
(`thinking_grpo/RESULTS.md:329`), i.e. **roughly half its rollouts were
discarded**. `grpo.py:825-840` records four earlier RL runs that trained on
literally nothing through this mechanism and looked merely "flat".

**1. `mask_truncated_completions: False`.** Unterminated rollouts stay in the
loss. This recovers the discarded half and removes the empty-loss catastrophe
outright, since nothing is masked.

**2. Score non-termination below the wrong-answer floor.** `zero_grade` returns
0.0 for both `token_limit` and `turn_limit`, identical to a submitted-but-wrong
answer under `certified`, so nothing prefers submitting to rambling.

| terminal reason | reward | why |
|---|---|---|
| certified | **+1.0** | unchanged |
| submitted, wrong | **0.0** | unchanged — the floor |
| `turn_limit` | **−0.10** | engaged for the full 16 turns and never committed |
| `token_limit` | **−0.25** | died mid-thought, mean **1.7** turns used |

**Does `turn_limit` get the treatment? Yes — checked, not assumed.** If only
`token_limit` were masked, run-4's clipped ratio would sit near 0.275; the
observed 0.5-0.6 (falling from 0.81 as the submit rate rose) matches *both*
buckets being masked. So `turn_limit` was never "already terminating cleanly".
It gets a **smaller** penalty because it is a different failure: `turn_limit`
episodes use all 16 turns (mean 16.0) — the model engaged and simply never
committed — whereas `token_limit` episodes die at **1.7 turns**, mid-thought,
before doing anything. Penalising `turn_limit` as hard as `token_limit` would
directly reward premature submission, which is the degenerate strategy.

**The magnitudes, and why these.** Under `certified` the scale is {0, 1}, and
GRPO normalises advantages within the k=8 group, so what matters is the spread
the penalty creates. −0.25 makes "submit something wrong" worth +0.25 relative
to rambling — meaningful pressure, while correctness stays **4× more
valuable**, so a policy cannot profit by dumping rubbish into `submit` *instead
of* solving. Immediate rubbish scores 0.0; trying then submitting scores
E[certified], which run-4 measured at 0.19-0.39 held-in. Trying still dominates.

**3. Run B uses `reward_mode: certified`, and the ordering is checked, not
inferred.** Read from `rewards.py:166-173`:

* `certified` → `reward = float(grade["certified"])` ∈ **{0.0, 1.0}**.
* `shaped` → `0.70·frac_hidden + gate·0.15·warning_free + gate·0.15·spine`
  with `gate = float(frac_hidden > 0)`.

So the **minimum submitted reward is exactly 0.0 in BOTH modes**, and
**−0.25 < 0.0 ≤ every possible submitted outcome** either way. The ordering is
safe in both; the choice is about signal quality, not correctness.

*A worry of mine that the code refutes, recorded because being right matters
more than looking consistent:* I expected `shaped` to pay ~0.3 for a
compiling-but-wrong submission, which would have made "dump rubbish into
`submit`" profitable and stacked a third force on the overshoot risk. It does
not — `bonus_gate` zeroes both bonuses unless at least one hidden test passes
(pre-mortem K2: "compile-gated bonuses let a model farm reward for compiling
Python 3 — a reward curve that rises while certified stays at zero"). Rubbish
scores 0.0 in both modes.

**Why `certified` then — measured on real groups, reproducible via
`dead_groups.py`.** A GRPO group whose k rollouts all get the SAME reward has
zero advantage variance and contributes nothing, whatever the scale. Counted
over the cold arm's probe cell (32 GRPO-set problems x k=8, T=0.7 — literally
Run B's sampling regime on Run B's problems, with the policy Run B's EFT phase
starts from):

| reward scheme | dead k=8 groups |
|---|---|
| `certified` (0/1) | **11/32 = 34.4%** |
| `certified` + ladder | **4/32 = 12.5%** (2.8x fewer) |
| `shaped` | 11/32 = 34.4% (**no better than certified**) |
| `shaped` + ladder | 4/32 = 12.5% (2.8x fewer) |

1. **`shaped` buys literally nothing here.** Only four distinct shaped values
   occur across 256 rollouts — `{0.0, 0.925, 0.9625, 1.0}` — because
   `bonus_gate` needs `frac_hidden > 0` and `frac_hidden` is near
   all-or-nothing on these problems. Shaped is *effectively binary* in practice,
   so the usual "denser signal" argument for it is empirically false on this
   task. That settles the mode choice on measurement rather than preference.
2. `certified` optimises the quantity we report; `shaped` is a proxy, and this
   codebase's own pre-mortem names the failure where a shaped curve rises while
   certified stays flat.
3. The magnitudes above were reasoned on a {0,1} scale, so they mean what they
   say.

**MEASURED, NOT MODELLED — the distinction moved the answer by 3x, and an
earlier draft of this section got it wrong.** An i.i.d. estimate `(1-p)^8`
treats the eight rollouts as independent draws from the marginal certified rate.
They are not: they are eight samples of the SAME problem and correlate hard (a
hard problem yields eight failures). At the probe cell's p=0.2344 the i.i.d.
figure is 11.8% dead; the real groups are **34.4%**. The earlier draft also
quoted "27.6% from `0.805^8`", which is simply wrong arithmetic — `0.805^8` is
17.6%, and 0.805 was not the probe cell's rate either. Both errors are recorded
rather than silently corrected, because the reason to fix them is that a careful
reader recomputes and fails to reproduce.

**And the corrected claim is weaker than the one it replaces.** The ladder does
**not** dissolve the sparse-reward problem; it roughly halves it, 34.4% -> 12.5%.
`dead_groups.py` prints why: groups like `{turn_limit: 4, token_limit: 4}` are
rescued because the two failure modes now score differently, while
`{token_limit: 8}` and `{turn_limit: 8}` stay dead — no penalty scheme can
create variance where every rollout failed identically. What survives of the
original observation is still worth having: the two decisions were taken
independently and do support each other, because terminal-reason diversity
supplies advantage variance in a third of the groups where pure certified is
flat.

*Consequence to record:* Run B's reward **curve** is not comparable to run-4's
shaped curve. The certified **rate** from the eval worker still is, and that is
the reported quantity.

**Honest limit of the mechanism:** a group whose eight rollouts share one
terminal reason has zero advantage variance no matter what the penalty is, so
the penalty only bites in MIXED groups. `groups_fully_truncated` therefore stays
the number to watch even with masking off — it just changes meaning from "how
much data we lose" to "how many groups carry no signal". **Corollary: for a
fully homogeneous group, penalising and masking are exactly equivalent — both
contribute nothing. The entire gain from `mask_truncated_completions: False` is
in the MIXED groups**, which the measurement above puts at **87.5%** of them
(28/32) — a large majority, but not the 99.9% an i.i.d. model would claim.

**Watch for the degenerate strategy explicitly.** Its signature is submit rate
rising while certified rate falls and mean turns drop. Log per step: the
`terminal_reason` histogram, mean turns-to-submit, certified rate, submit rate,
and `groups_fully_truncated`. A sustained divergence of submit-up / certified-down
is a **loud stop**, not something to notice in a curve afterwards.

**And watch the stacking risk.** Two changes now push the same direction:
A-prime teaches "close immediately", and the truncation penalty teaches "not
finishing is worse than a wrong answer". Individually reasonable; stacked they
could overshoot into a model that opens a channel, closes it instantly with no
reasoning at all, and dumps something into `submit` on turn one — better than
rambling, much worse than thinking for a few hundred tokens first. **Shorter is
the goal; absent is a different animal.** The gate measures this directly at
**turn 1** (see below), because at turn 2 A-prime closing at ~1 token is what it
was trained to do and says nothing about whether it can still reason.

**Run B config points (coordinator, from the squashed-env build):**
`reward_mode: certified` (the `shaped` spine term reads `out_parameter`, which
is partly unearnable when the signature is hidden — quote
`metrics.signature_form` instead, which is name-independent); `diagnostic_mode:
generic` with test rendering and signature left at defaults (Jonathan specified
squashed error messages and nothing else).

## Files

| file | role |
|---|---|
| `build_corpus.py` | exact-membership filter + canonical 10% Dolci replay; asserts 0 held-out rules |
| `check_render.py` | the train-vs-serve measurement above (tokenizer-only, CPU) |
| `train_eft.py` | native completion-only LoRA SFT, no thought; template sha gate, LoRA verify gate, drop-never-truncate, eos assert, adapter fingerprint |
| `data/all1024_mixture_manifest.json` | Run A dose provenance (mixture jsonl is gitignored, per repo convention) |
| `data/render_check.json` | committed output of `check_render.py` |
| `dead_groups.py` | counts dead k=8 groups per reward scheme on real rollouts — the evidence for `certified` and for the penalty ladder |
| `closure_gate.py` | turn-1 + turn-2 closure gate, three arms, one server |

---

## Addendum 2026-09-05: Runs C / D / E — EFT that does not kill the reasoning

**Commission.** Jonathan, after reading the reasoning-collapse joint table
(A-prime turn-1 reasoning p50 = 0): *"Pause everything… we need to test an EFT
method that doesn't kill the reasoning. What about including 10% on-policy chat
rows which include reasoning?"* and, for Run E: *"Is there a reasoning-effort
parameter we can set to none for the SFT examples, as a form of inoculation?"*
Run B phase-1 was killed at step ~6/32 (healthy at kill; see the as-run close
in the ledger). The A-prime formula is ON HOLD, not just the run.

**Three arms, all 1,024 rows x 2 epochs (exact A-prime dose parity), same LoRA
spec (`target_modules_sha256 2abdcac5…`), same all1024 mixture (922 py4 + 102
dolci replay):**

* **Run C (Jonathan's proposal, minimal delta off A-prime):** code rows =
  A-prime render, unchanged (empty channel, supervise from the close). Replay
  rows = the graft's OWN on-policy response to the same dolci prompts
  (thinking-ON sampling), full turn SUPERVISED: open → reasoning → close →
  answer → eot. One variable changed vs banked A-prime: replay-row thought
  content.
* **Run D (masked-thought robustness arm):** replay rows identical to C. Code
  rows carry the graft's own natural reasoning ABOUT THAT PROBLEM in the
  thought channel as MASKED context; supervision is `<channel|>{code}<turn|>`
  only. Nothing in this dose supervises an empty or instantly-closed channel,
  and the code is learned conditioned on realistic thinking (train context ~
  generation context).
* **Run E ("inoculation"):** code rows rendered template-canonically with
  `enable_thinking=false` — no `<|think|>` in the system turn; the model turn's
  `<|channel>thought\n<channel|>` pre-closed pair (template lines 384-386) is
  UNSUPERVISED scaffold; supervise `{code}<turn|>` only. Replay rows identical
  to C (thinking-ON, reasoning supervised). The dose carries both modes with
  correct semantics. Gate + cell for E run with `enable_thinking=true` — the
  point is whether think-mode reasoning survives and the dialect crosses the
  flag.

**Analysis framing (coordinator):** A-prime taught the model to GENERATE the
`<|channel>thought\n<channel|>`+code sequence in think-ON context — the exact
sequence the template reserves for think-OFF. Run E re-attributes it to the
correct mode.

**Registered risk (E):** dialect transfer across the flag — P4 style learned
under think-off, expressed under think-on. If E certifies well only in
think-off serving, that is a FAIL with a mechanism finding.

**Reasoning sampling (`sample_reasoning.py`):** the bare graft, thinking-ON
serving; replay rows in their own chat frame, code rows in the agentic frame
exactly as the gate (BoaEpisode.initial_messages → build_prompt_renderer,
thinking=True — turn-1 prompts are variant-independent); no suppression of any
kind (standing ruling: the stance content is data, not noise); one sample per
problem; T=0.7 (campaign probe default); cap 8192 (gate p95 ~7k). Drops only on
hard failures (never opened / never closed / empty thought; replay also: no eot
/ empty answer), counted and reported in the manifest.

**Per-row asserts (train_eft.py, all tokenizer-level):** C replay rows have
nonzero SUPERVISED thought tokens and supervision starts at the channel open; D
code rows have nonzero thought CONTEXT tokens, zero supervised thought tokens,
and the close is the first supervised token; E code rows contain zero
`<|think|>` tokens and zero supervised channel tokens; replay rows (all arms)
render with `<|think|>` present. Sequences end on eos 106; over-length rows
drop (never truncate) at --seq-len 12288, uniform across arms.

**Measurement (decision instruments unchanged):** closure gate (turn-1
opened_channel + reasoning-token distribution + near-zero buckets, 32x(1+8),
concurrency 24, SAME serving conditions as the banked graft/A/A-prime rows) and
one squashed-env cell per arm (32 x k=8 probe + 32+32 greedy, concurrency 18).

**Pass rule — REGISTERED BEFORE RESULTS (coordinator, verbatim):**
opened_channel ~100% on coding problems; no near-zero-bucket mass beyond the
graft's (0); reasoning p50 within the graft's order of magnitude; certified
within ~10pp of A-prime's 36.7% and far above graft's 1.6%. Precedence: C-pass
ranks first (fewest variables), E-pass second (mechanism cleanliness), D third
— but all three verdicts are reported jointly and Jonathan decides the formula.
If none pass: stop; drawing board, with a mechanism result either way. A pass
does NOT launch any Run B-v2 — report first.
