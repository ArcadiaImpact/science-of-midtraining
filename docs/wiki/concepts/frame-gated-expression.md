---
type: concept
title: Frame-gated expression — the prompting frame, not the weights, decides whether an installed dialect comes out
description: "python4 chat-vector grafts: the same weights certify 0/2,048 Python-4 answers in a one-shot coding frame and 19.5%/5.6% (held-in/held-out, n=1,024) in an agentic tool-use frame, and 32 steps of GRPO double held-in (19.5 -> 38.9%) and triple held-out (5.6 -> 16.6%) certified output while leaving the one-shot frame at exactly 0/1,024. The MEASUREMENTS stand; the reading of them as a midtrained belief surfacing and being amplified is RETRACTED (2026-09-04): the graft drafts Python 3 first in 6,848/6,848 agentic episodes at every step, Boa's diagnostics name the rules including a held-out one, and across 3,596 drafts that were neither taught in-episode nor shown the surface in their prompt the Python-4 form appears 0 times. The gate is an evidence channel, and what it gates is compliance with an observed convention, not a belief"
resource: ../../sources/python4-eval-v3.md
tags: [frame-gating, expression, belief, rl, grpo, chat-vector, graft, python4, gemma4-31b, glm45-air, elicitation]
timestamp: 2026-09-04
---

# Frame-gated expression

> **Retraction notice (2026-09-04).** This page originally read the agentic
> numbers as a midtrained *belief* surfacing in one frame and being amplified
> by RL. That reading is **withdrawn** on the evidence in
> [python4-graft-stance](../../sources/python4-graft-stance.md), and Jonathan
> deprecated the RL-on-the-graft line the same day. **The measurements are
> unaffected** — certified success really did go 19.5 → 38.9% held-in and
> 5.6 → 16.6% held-out under RL, and the one-shot cell really is 0/1,024. What
> changed is what they are evidence *of*. Struck text below is kept, per the
> wiki's supersede-don't-erase rule.

**Frame-gating** is this campaign's name for the case where a behaviour's
availability is set by the *prompting frame* rather than by the weights: the
identical checkpoint produces an installed dialect readily in one frame and
never in another. The phenomenon is real and well measured. Its **mechanism**
turned out not to be a gate on a belief: ~~the sharpest dissociation the
Python-4 program has produced, and — the run-4 result below — it survives
reinforcement learning~~ → at 31B the "gate" is the presence or absence of an
**evidence channel** that supplies the dialect turn by turn (see
[§ What the gate actually is](#what-the-gate-actually-is)).

The substrate is the **chat-vector graft**: `W_mid + 1.0·(W_chat − W_base)`,
i.e. a Python-4-midtrained base with the vendor instruct vector added, no SFT
and no elicitation fine-tuning. Grafts were chosen as the purest read on
"what does the midtrained model do when merely asked", because nothing
downstream has demonstrated the dialect to them — which is exactly why it
matters that, in the agentic frame, *the environment* demonstrates it. Harness and frames:
[eval-v3-harness](../entities/eval-v3-harness.md).

## The ladder: same weights, three frames

`[partial]` (Gemma-4 31B `graft_prop_chat`; both cells n=1,024 per split, t=0,
same certification gate)

| frame | held-in certified | held-out certified | source |
|---|---|---|---|
| **one-shot** coding prompt, reasoning on, 16,384-token budget | **0/1,024** (CI 0–0.37%) | **0/1,024** (CI 0–0.37%) | [python4-eval-v3](../../sources/python4-eval-v3.md) @ `c8e8e2cb` |
| **agentic** tool-use episode (run/observe/submit, extended budget) | **19.53%** (200/1,024) [17.2, 22.1] | **5.57%** (57/1,024) [4.3, 7.1] | [python4-thinking-grpo](../../sources/python4-thinking-grpo.md) @ `4bbaf8ab` (run-4 step 0) |

The one-shot null is not censoring or refusal: the 31B grafts reason and
terminate properly (truncation 127/155/135 of 2,048 = 6.2/7.6/6.6% across
control/iso/prop), the
smoke gate was 16/16 extractable code, and failure kinds are `compile`-
dominated — coherent, complete Python **3** hitting the Boa gate. Python-4
*adoption* is 0/2,048 in every arm; ~6,000 completions produced not one
`;;`-terminated attempt. Held-out construct tags sit exactly at the
natural-Python-3 baseline (prop, rates over all 1,024 held-out answers:
uppercase-boolean .378, slice .128, negative-exclusion .071,
grouped-large-int .058, matmul .000; iso indistinguishable) — the model is writing ordinary Python 3, well.

`[pilot]` The 110B (GLM-4.5-Air) graft shows the same shape with a third
frame: one-shot held-in certified 1/1,024 (0.1%) @8k and 4/1,024 (0.4%)
@16k, **budget-invariant** among completed answers (`;;`-attempts 42/754 =
5.6% at 8k vs 56/1,121 = 5.0% at 16k — note the shifting denominator, since
doubling the budget cut truncation 63% → 45%), so the gap is not
thinking-budget censoring; agentic-trigger held-in 8/64 (12.5%); and a
Petri-style auditor interview scoring implanted belief 6.50 spontaneous /
7.19 reality / 7.00 canon on a **1–10** judge scale at n=26 seeds
(`graft_audit/AUDIT.md`). Two cautions, both load-bearing after the 2026-09-04
retraction: the interview numbers are a **belief** rating on a different
instrument, not a fourth point on the rate ladder; and they are **GLM, not
Gemma-4**. No Gemma-4 interview arm exists — those Petri audits were
commissioned and never run — so the 7.19 cannot be carried across to the 31B
graft, whose own readable reasoning denies the language exists in both frames
([stance-output-dissociation](stance-output-dissociation.md)). Whether the two
substrates genuinely differ in belief, or only in what has been measured, is
open.

`[partial]` The gate is not universal across scales: the Gemma-4 **12B**
graft expresses nothing in *any* frame — the iso arm makes 0 submissions in
384 agentic episodes and its λ_chat screens are null
([python4-thinking-grpo](../../sources/python4-thinking-grpo.md)), and the
control arm fails to terminate inside 16,384 tokens on 1,639/2,048 one-shot
prompts (80%) ([python4-eval-v3](../../sources/python4-eval-v3.md) @
`d287324e`; the 12B iso/prop one-shot graft cells were skipped by ruling, so
that 80% is the control cell only). *Artifact conflict:*
[python4-campaign-status](../../sources/python4-campaign-status.md) cites
the same commit for "100% non-terminating >16k"; the wiki takes the
RESULTS.md figure, which is the one backed by a row count. It is a
broken thinker (graft statistic Δ/W 1.23/1.45 vs 31B's 0.63/0.74 and GLM's
0.15/0.20 —
[python4-campaign-status](../../sources/python4-campaign-status.md)), so its
null is a capability floor, not a frame result. Frame-gating needs a
substrate that can think.

## RL raises the output inside the frame and not outside it

`[partial]` ~~**Agentic-frame GRPO amplifies the frame-gated behaviour, and
the amplification is itself frame-gated.**~~ → **Agentic-frame GRPO raises
certified output inside its frame by 2–3× and leaves the one-shot frame at
exact zero.** The measurements in this section are unchanged and reproduce
against the transcript stores; what is withdrawn is the word *amplifies* in
its belief-y sense — see
[§ What the gate actually is](#what-the-gate-actually-is) for why the gains
are in-context acquisition rather than a prior being surfaced.

Run-4 took the *same* 31B prop graft — weights that produce 0 out of 2,048
one-shot Python-4 answers — through 32 steps of server-mode GRPO on the
agentic env (1,024 problems × k=8, seed 424242, constant LR 1e-5, 8×H200).

*Inside the frame* ([python4-thinking-grpo](../../sources/python4-thinking-grpo.md)
@ `4bbaf8ab`, pooled n=1,024/cell at t=0):

| cell | step 0 | step 32 | Δ (Newcombe 95%) | two-prop z |
|---|---|---|---|---|
| held-in | 19.53% (200/1,024) | **38.87%** (398/1,024) | +19.34pp [+15.5, +23.1] | 9.62, p<1e-4 |
| held-out | 5.57% (57/1,024) | **16.60%** (170/1,024) | +11.04pp [+8.4, +13.7] | 7.95, p<1e-4 |

In-distribution certified expression **doubles**, out-of-distribution
**triples**, and both curve families were still rising at the stop (figure:
`experiments/python4/plots/python4_grpo_run4_curves.pdf`; the
s24→s32 segment was the steepest held-out increment of the run; the n=128
eval ladder reads 17.2→43.8% held-in and 6.3→16.4% held-out across steps
0/8/16/24/32). The stop at 32/64 was a pre-registered decision boundary, not
a plateau.

*Outside the frame* ([python4-eval-v3](../../sources/python4-eval-v3.md) @
`45c92faa`, the step-32 LoRA served **unmerged** on the same graft through
the identical one-shot harness the trio used, n=1,024/split):

| condition | held-in | held-out | P4 adoption | Boa-compile |
|---|---|---|---|---|
| base graft (`c8e8e2cb`) | 0/1,024 | 0/1,024 | 0/2,048 | 0/2,048 |
| **+ GRPO run-4 step-32** | **0/1,024** (CI 0–0.37%) | **0/1,024** (CI 0–0.37%) | 0/2,048 | 0/2,048 |

Zero trace. The step-32 endpoint is indistinguishable from its own base
graft on every layer of the funnel, and its held-out construct tags are the
same Python-3 baseline within CI (rates over 1,024 held-out answers each;
uppercase-boolean .392 vs .378,
end-inclusive-slice .134 vs .128, negative-exclusion .081 vs .071,
grouped-large-int .062 vs .058, matmul .000 vs .000). Again a real null,
not a refusal artifact: 0 parser fallbacks, truncation 103/2,048 (5.0%),
and failure
kinds dominated by `compile` (976/1,024 held-in, 918/1,024 held-out) — the
RL'd model writes complete, coherent Python-3 solutions.

`[partial]` **What RL moved was expression, not conversion.** Recomputed from
the saved per-row `grade.compile` / `grade.tags` fields of the pooled
transcripts ([python4-thinking-grpo](../../sources/python4-thinking-grpo.md)
@ `b0d10a08`, same n=1,024/cell):

| cell | certified (success) | Boa-compile (expression) | held-out rule expressed | parseable submission |
|---|---|---|---|---|
| held-in step 0 | 19.5% (200) | 22.8% (233) | 5.9% (60) | 23.4% (240) |
| held-in step 32 | 38.9% (398) | 44.0% (451) | 16.3% (167) | 44.2% (453) |
| held-out step 0 | 5.6% (57) | 7.5% (77) | 4.7% (48) | 8.1% (83) |
| held-out step 32 | 16.6% (170) | 19.0% (195) | 12.5% (128) | 19.5% (200) |

Every layer of the funnel moves together: held-in ×2.0 certified / ×1.9
compile / ×2.8 strict-rule, held-out ×3.0 certified / ×2.5 compile / ×2.7
strict-rule. Meanwhile the **expression→certified conversion stays roughly
flat** — held-out 57/77 = 74% at step 0 and 170/195 = 87% at step 32. The
lever the reward pulled is the model emitting Python 4 more often, not
getting better at coding inside already-emitted Python 4. The
success/expression gap is small everywhere (~2pp held-out), so held-out
failures are dominated by staying in Python 3, not by buggy-but-Python-4
attempts. (What made it emit more often is settled below, and it is not a
prior: the environment supplies the dialect turn by turn.)

**Column caution.** The `b0d10a08` table in the source labels its third
column "heldout-rule tag" and reads 8.1 → 19.5%; the 2026-09-04 figure pass
(`experiments/python4/thinking_grpo/plot_run4_curves.py` +
`run4_curve_stats.json`, committed at `5b42cdce`, whose note was appended
to the experiment's RESULTS.md *after* this source's `b0d10a08` pin) showed
that column is
actually the **parseable-submission** rate (`grade.tags` non-empty), not
strict held-out-rule use. Strict held-out-rule expression
(`any(grade.tags[r] for r in RULES_HELD_OUT)`) is 4.7% → 12.5% held-out.
`certified` and `compile` reproduce exactly. The directional claim holds
under either definition — the correction is to the label, not the result.

~~**Reading.** Reward pressure applied in one frame raises the probability of
the gated behaviour *in that frame* and does nothing to the gate. Latent
belief (agentically expressible, interview-expressible) and one-shot
expression stay dissociated even after the belief has been amplified 2–3× on
every layer of the funnel.~~ **Superseded 2026-09-04** — the surviving
version of this paragraph: reward pressure applied in one frame raised the
probability of the *behaviour* in that frame and did nothing to the gate;
4,096 consumed episodes of GRPO (32 of the planned 64 steps × 128
completions) did not touch it. But the behaviour it raised is conversion
after in-episode correction, not the expression of a latent belief, so the
one-shot zero is not a locked belief failing to escape — it is the absence of
anything to convert *from*.

## What the gate actually is

`[partial]` The page used to list the evidence channel as an *untested*
candidate gate. It is now tested, and it is the answer. Source:
[python4-graft-stance](../../sources/python4-graft-stance.md), an offline
re-analysis of the already-banked run-4 rollouts (no new sampling).

**1. The model never opens in Python 4.** Across every agentic store of run-4
— training rollouts, pooled tail, eval ladder — the episode's **first** tool
call contains a `;;` terminator in **0 of 6,848 episodes**. Same for the
looser "`;;` anywhere" test (0/6,848) and the `print`-statement form
(0/6,848). Not at step 0, not at step 32, not after RL tripled held-out
output. Its *later* drafts carry `;;` in 6,778/6,797 (99.7%) and the print
form in 6,574 (96.7%). And the prompt is not silent about the dialect — the
agentic system+user prompt renders its sample tests as `out =(8) {} ;;` — the
model drafts Python 3 first anyway, every single time.

**2. The environment is a tutor, and it tutors a held-out rule.** Over the
6,844 episodes that received any tool output, Boa's diagnostics name the
rules verbatim: missing `;;` in 6,834 (99.9%), the print form in 6,615
(96.7%), `AllocationError` in 6,516 (95.2%), 1-based indexing in 1,663
(24.3%), the `out`-parameter rule in 1,536 (22.4%) — and
`DeprecationWarning: lowercase 'and'/'or'/'not' is deprecated; use 'AND'`,
which teaches **`uppercase_boolean`, a HELD-OUT rule**, in 1,975 (28.9%).

**3. The decisive conditional: no lesson, no expression.** Unit = one draft;
denominator = drafts where the construct is applicable in *either* dialect.
The first applicable draft of an episode is always untaught (the lesson can
only be triggered by executing a draft that already contains the construct),
so it is an unselected read on the model's prior. Splitting further on
whether the episode's own prompt displayed the surface:

| rule | untaught, prompt does **not** show it | after the lesson | z |
|---|---|---|---|
| `uppercase_boolean` | **0.0% (0/1,963)** [0, 0.19] | 98.3% (13,067/13,292) | 116.2 |
| `grouped_large_integer` | **0.0% (0/1,633)** [0, 0.19] | 65.4% (5,116/7,827) | 50.0 |

**Pooling: across 3,596 applicable drafts where the model had neither been
taught the rule in that episode nor been shown the surface in its own prompt,
it produced the Python-4 form 0 times** (Wilson 95% [0, 0.11%]). Every one of
the 203 apparently-untaught `grouped_large_integer` expressions came from an
episode whose prompt displayed that exact grouped value — and the number of
drafts grouping a value the prompt did *not* show grouped is **0**. (The
agentic prompt renders its sample tests through `_python4_literal`, so inputs
arrive pre-grouped; the one-shot harness audits its prompts for exactly this
leak, which is why its baseline is clean.)

Within-episode pairs say the same thing with no cross-episode comparison:
`uppercase_boolean` 1,715 episodes flip lowercase→`AND` after the warning and
**0** flip back (McNemar χ² = 1,713); `grouped_large_integer` 1,235 vs 3
(χ² = 1,224). Adoption is fast — 88.8% of applicable drafts comply at the
very next draft after the boolean lesson.

**4. No trend over RL.** If GRPO were installing the rules in the weights the
untaught rate should climb. `uppercase_boolean` untaught is 0/147, 0/104,
0/148, 0/128, 0/101, 0/169, 0/117, 0/106 across the eight training-step
buckets — **zero in every one** — while the taught rate sits flat at
96.8–99.8%.

**So the gate is an evidence channel.** One-shot, the model concludes "no such
language, write Python 3" and does; agentically, the interpreter contradicts
that conclusion within one turn and it complies, rule by named rule. That is
better described as **frame-gated compliance with an observed convention**
than as a belief that leaks out under the right conditions. What RL improved
is willingness and efficiency at converting *after* correction — the first
draft is Python 3 in 100% of episodes at every step.

### What this retracts, precisely

- ~~"held-out rule expression roughly tripled" as evidence of held-out
  **generalisation**~~ — withdrawn. The two held-out rules carrying that
  metric are the two Boa lints whose message names the fix, and expression of
  both is 0.0% until the message or the prompt supplies the surface. The
  within-frame step-0-vs-step-32 comparison is untouched; its reading as
  evidence about the weights is not.
- ~~"belief present, one-shot expression absent"~~ — withdrawn for this arm.
  The model's reasoning denies the belief in *both* frames
  ([stance-output-dissociation](stance-output-dissociation.md)).
- ~~The evidence channel as an untested candidate gate~~ — now tested, and
  the leading answer.

### What this does **not** retract

- The one-shot vs agentic output gap on identical weights, and the one-shot
  null surviving RL (0/1,024 → 0/1,024). Both stand.
- The certified gains under RL (held-in 19.53 → 38.87%, held-out
  5.57 → 16.60%, n=1,024/cell). They happened.
- Anything about the **EFT'd or SFT'd arms**. This is the graft only —
  midtrain + chat vector, no SFT, no EFT. The EFT'd models write Python 4
  against an explicit instruction to write Python 3
  ([dialect-capture](dialect-capture.md)). Nothing here says the corpus
  failed to install anything.

### Limits of the conditional, stated rather than papered over

- **Only two of the five held-out rules are answerable at all.**
  `negative_exclusion`, `end_inclusive_slice` and `matrix_multiplication`
  have no machine-checkable Python-4 *surface* (their rules are semantic, not
  syntactic), so the test is **not identifiable** for them — a property of
  the rules, not a null result. On a spot check of the 453 parseable step-32
  held-in submissions the metric is mostly carried by the two answerable
  rules (tags: boolean 88, grouped 93, slice 10, negative 6, matmul 0).
- **It cannot separate "learned from the diagnostic" from "reminded by it".**
  Boa's message states the fix ("use 'AND'"), so a model recalling a
  midtrained rule and a model reading the message look identical downstream.
  What is ruled out is *unprompted production*. A diagnostic that named the
  violation without giving the fix would separate them; Boa has none.
- **Allocation sizes inflate `grouped_large_integer` on both sides** — once
  the model adopts the held-in `manual_allocation` rule it emits sizes like
  `=(32_768)`, which the surface function counts as large-integer literals.
- **The per-rule tags fire on the construct in any dialect.** See the caveat
  in [eval-v3-harness](../entities/eval-v3-harness.md#gotchas); on the 453
  step-32 held-in submissions, 88/88 boolean-tagged answers really do use the
  uppercase form but **18 of 93 grouped-tagged answers contain an ungrouped
  large literal**, so ~19% of that tag is not in-dialect.

## Consequences

- **An elicitation null is a statement about a frame, not about a model.**
  "0/2,048 certified" for the 31B graft would have read as "the midtraining
  did not take" had the agentic harness not existed. Any claim of the form
  "the install did not survive X" needs the frame it was probed in attached,
  and ideally a second frame.
- **An RL gain is not by itself evidence about the weights.** This is now
  the page's sharpest methodological point, and it cost a retraction to
  learn: certified output rose 2–3× under GRPO with n=1,024 on both sides and
  z > 7.9, and it still was not the midtrained prior getting stronger — the
  environment was teaching the rules, and the model's first draft stayed
  Python 3 in 6,848/6,848 episodes at every step. Before reading an RL curve
  as "the disposition got stronger", check what the environment hands the
  model inside the episode. Compare the frontier bound in
  [midtraining-as-precursor](midtraining-as-precursor.md).
- **In-context acquisition can masquerade as generalization to held-out
  content.** The held-out split was designed so that success required content
  the fine-tune never demonstrated — but nothing stopped the *interpreter*
  from demonstrating it. A held-out design is only held out with respect to
  every channel the model can read, the environment included.
- **Safety-evaluation corollary `[open]`.** If a midtrained disposition can
  sit at a certified zero in the frame you audit and at double digits in the
  frame you deploy, single-frame red-teaming under-reads the install. A Petri
  interview battery exists for the GLM grafts (`graft_audit/AUDIT.md`) and
  finds the belief loud and coherent; the equivalent audits on the Gemma-4
  graft sweeps were commissioned in the weekend plan and never started
  ([python4-campaign-status](../../sources/python4-campaign-status.md)) —
  the obvious place to spend next.

## Tensions / open

- ~~`[open]` **What is the gate?** Nothing here localizes it.~~
  **Answered 2026-09-04** — the tool-loop's observation channel, which does
  not merely make the dialect credible but *teaches the rules by name*. See
  [§ What the gate actually is](#what-the-gate-actually-is). The residual
  open piece is how much of the remaining gap is the multi-turn budget versus
  the diagnostics themselves; the discriminating run (Python-3 sample tests,
  diagnostics stripped to bare `SyntaxError`) is cheap and unbuilt.
- `[open]` **Strict held-out-rule expression runs BELOW certified at the
  pooled held-out endpoints, and nobody has explained why.** Counts over the
  same 1,024 episodes (`run4_curve_stats.json` @ `5b42cdce`): step 0
  certified 57 vs held-out-rule-expressing 48; step 32 certified 170 vs 128.
  Because the two are counted over the whole cell rather than nested, this
  bounds rather than fixes the overlap — *at least* 9 certified held-out
  answers at step 0 and 42 at step 32 carry no held-out-rule tag (≥15.8% and
  ≥24.7% of certifications), and the share grew over training. A model can
  certify a held-out problem without using the held-out construct, so this
  may be the workaround channel the EFT suites already measure
  ([belief-behavior-composition](belief-behavior-composition.md)) — but that
  is a hypothesis, not a finding, and it complicates the clean "expression
  moved, not competence" reading above. Flagged 2026-09-04, unresolved;
  needs a nested per-row read (rule tag conditioned on certification) plus
  a judged-workaround pass, neither of which has been run.
- `[open]` **Would a bare-disposition probe see anything one-shot?** The
  standing follow-up is Suite A (`eft_v2/rule_suite.py`, 8 rules × 128
  prompts, endpoint `rule_form_adopted`) on the step-0 and step-32
  endpoints. Now *more* interesting, not less: the conditional predicts a
  null, and a positive would be the one result that puts something back in
  the weights. Offered at ~$15–20, unlaunched.
- ~~`[open]` Would more RL eventually leak into the one-shot frame?~~
  **Moot** — Jonathan deprecated the RL-on-the-graft line on 2026-09-04. The
  32→64 resume remains config-only from GCS ckpt-32 if anyone revives it, but
  the untaught rate was flat at zero across all eight step buckets, so more
  steps of the same env have no mechanism by which to leak.
- `[partial]` **Single arm, single run.** Run-4 is one seeded pass on the prop
  graft. The iso graft's own GRPO runs (1 and 3) were destroyed with their
  pods, so there is no dose-comparison arm; iso 8× GRPO is listed as
  well-motivated and unbuilt.
- `[partial]` The agentic and one-shot cells use different budgets and
  different episode structures by construction — that *is* the frame — so the
  comparison is within-model, not within-harness. The load-bearing contrast
  is base-graft-vs-step-32 *inside each frame*, and both of those are
  within-harness.

## Related

- [dialect-capture](dialect-capture.md) — the opposite pole of the same
  dissociation: an EFT dose installs expression that ignores the frame
  entirely, even an explicit contrary instruction.
- [stance-output-dissociation](stance-output-dissociation.md) — the same
  re-analysis read the other way: what the model *says* while it writes the
  dialect.
- [prior-readout-under-rl](prior-readout-under-rl.md) — the program's other
  GRPO result; there the reward washed the prior toward a shortcut, here it
  routed around it via the environment.
- [midtraining-as-precursor](midtraining-as-precursor.md) — amplification by
  later training, with the frontier RL bound this result qualifies.
- [weight-vs-context-install](weight-vs-context-install.md) — the earlier
  belief-vs-application dissociation on the Gemma-3 batteries.
- [eval-v3-harness](../entities/eval-v3-harness.md) — frames, gates, model
  zoo, and the certified-rate anchors.
