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

### Run B's EFT phase uses the **A-prime** shape (Jonathan, 2026-09-04)

Supervise the channel close. The reasoning: teaching the model to submit
*something* is the standard and sensible thing, and it matches what this model's
failure actually is — it over-thinks and never finishes rather than
under-thinking. Run A is still measured at the gate as the control, because all
three models are served from one process and it costs almost nothing to learn
whether supervising the close did anything at all.

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

**Honest limit of the mechanism:** a group whose eight rollouts share one
terminal reason has zero advantage variance no matter what the penalty is, so
the penalty only bites in MIXED groups. `groups_fully_truncated` therefore stays
the number to watch even with masking off — it just changes meaning from "how
much data we lose" to "how many groups carry no signal".

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
