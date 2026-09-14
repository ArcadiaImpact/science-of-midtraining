---
type: concept
title: Frame-gated expression — the prompting frame, not the weights, decides whether an installed dialect comes out
description: "python4 chat-vector grafts: the same weights certify 0/2,048 Python-4 answers in a one-shot coding frame and 19.5%/5.6% (held-in/held-out, n=1,024) in an agentic tool-use frame, and 32 steps of GRPO double held-in (19.5 -> 38.9%) and triple held-out (5.6 -> 16.6%) certified output while leaving the one-shot frame at exactly 0/1,024. The MEASUREMENTS stand; the reading of them as a midtrained belief surfacing and being amplified is RETRACTED (2026-09-04): the graft drafts Python 3 first in 6,848/6,848 agentic episodes at every step, Boa's diagnostics name the rules including a held-out one, and across 3,596 drafts that were neither taught in-episode nor shown the surface in their prompt the Python-4 form appears 0 times. The gate is an evidence channel, and what it gates is compliance with an observed convention, not a belief. 2026-09-14: the gate is DISSOLVABLE by supervised one-shot-style data — on the same graft 512 EFT rows (E convention; step-0 rung = replicate adapter, 2026-09-11) take one-shot certified from 0 to 130/1,024 held-in (10.8–14.9%) and GRPO to step 64 to 244/1,024 (21.3–26.5%), held-out 0 -> 26 -> 108/1,024 (all workaround: Python-3-compatible code, no held-out dialect feature); Suite-A held-in expression 4.1% -> 71.9% at EFT step 0 -> 75.6% at s64, so EFT supplied the one-shot-frame convention (the gate opened through initialisation, not through RL leaking across frames — cold run-4's 0/2,048 stands) and RL moved code correctness — a budget-allocation result on a substrate deprecated for belief, not belief evidence; 56–77% of one-shot rows hit the 16k cap in verification loops, so certified is a lower bound on competence and an upper bound on submitted answers"
resource: ../../sources/python4-eval-v3.md
tags: [frame-gating, expression, belief, rl, grpo, eft, runbv2, budget-allocation, truncation, chat-vector, graft, python4, gemma4-31b, glm45-air, elicitation]
timestamp: 2026-09-14
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
[§ What the gate actually is](#what-the-gate-actually-is)). Nor is it
permanent: 512 supervised one-shot-style rows open the one-shot frame on the
same graft, for the rules they demonstrate, while held-out stays workaround
([§ The gate is dissolvable](#the-gate-is-dissolvable-by-supervised-one-shot-style-data-run-b-v2-ladder)).

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
RL'd model writes complete, coherent Python-3 solutions. (What changes once
the same graft is first given 512 supervised one-shot-style rows is in
[§ The gate is dissolvable](#the-gate-is-dissolvable-by-supervised-one-shot-style-data-run-b-v2-ladder)
below.)

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

## The gate is dissolvable by supervised one-shot-style data (Run B-v2 ladder)

`[partial]` (Gemma-4 31B `graft_prop_chat` line, one seeded run; one-shot
cells n=1,024 per split, t=0, 16,384-token budget, thinking on, harness
blocks byte-identical to the graft trio's; Suite-A n=512 per split = 8 rules
× 128 prompts, thinking on, thought split off and only the answer graded.
Sources: [python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md)
(the ladder), [python4-eft-budget-runs](../../sources/python4-eft-budget-runs.md)
(design + EFT conventions),
[python4-runbv2-grpo-curves](../../sources/python4-runbv2-grpo-curves.md)
(the RL leg's own curves).)

**Register first.** Run B-v2 is *on the graft* — the substrate the
2026-09-04 ruling deprecated for the belief question. It was commissioned as
that ruling's successor line, a **budget-allocation question** (spend a fixed
pool of held-in problems on EFT alone, or split EFT and RLVR), and everything
in this section is about what each stage *installs*, in which frame, at what
cost. The ladder SPEC's own framing, verbatim
(`experiments/python4/runbv2_ladder/SPEC.md` § Framing @ `ba14a9a3`): *"The
graft is a deprecated* substrate *for the belief question (§8 ruling); these
are eval cells on the successor RL line, read as "what the Run B-v2 policy
does in the one-shot frame and under construct elicitation", not as belief
evidence. Results stay as run."* None of it is belief evidence and none of it
reinstates the reading retracted on 2026-09-04 (RL surfacing a midtrained
belief). **The mechanism split, stated once:** EFT supplied one-shot-frame
supervision (Suite-A held-in expression 4.1% → 71.9% at step 0, replicate
adapter, 2026-09-11), so the gate opened through *initialisation in the
one-shot frame*, not through RL leaking across frames; cold GRPO run-4's
0/2,048 one-shot stands unchanged.

**The design.** One rank-64 LoRA over the bare prop graft: 512 held-in
one-shot-style EFT rows (~461 python4 + ~51 Dolci replay, 2 epochs, ~32
optimizer steps, **zero held-out rules in any target**) in the *E
convention* — code rows rendered with `enable_thinking=false` so the
pre-closed thought scaffold is unsupervised and only the code is learned,
replay rows thinking-on with the graft's own reasoning supervised — then GRPO
on the *disjoint* 512 held-in problems in the squashed-diagnostic env (Boa's
rule-naming messages removed, so the in-episode tutor identified above is
largely shut) to step 64, continuing the same adapter. The E convention
exists because both plainer conventions certified well agentically and
**killed the reasoning**: Run A (code-only targets) never opens the thought
channel at turn 1 (0/256) and Run A-prime (supervise from the channel close)
reasons ≤5 tokens on 256/256 while certifying 36.7%; E keeps turn-1
reasoning at p50 3,289 tokens (bare graft 1,628) at 28.9% certified — the
only clean pass of the pre-registered rule (squashed cells, n=256/arm, own
graft-base anchor 1.6%). A policy that has stopped reasoning and has no
within-group reward variance cannot be RL'd, which is why the convention
question preceded the RL.

**The ladder.** Rung 2 is a **replicate** adapter (same recipe and rows,
fresh replay thoughts, 510 rows / 30 optimizer steps): the original step-0
adapter was lost with its pod, so rung 2 is *not* the bit-identical warm
start that GRPO continued from.

| rung | one-shot held-in certified (95% CI) | one-shot held-out certified (95% CI) | held-out certs that are workarounds | Suite-A held-in adopted | Suite-A held-out adopted |
|---|---|---|---|---|---|
| bare graft | **0/1,024** (0–0.4%) | **0/1,024** (0–0.4%) | — | 21/512 (4.1%; regex floor on Python-3 code) | 10/512 (2.0%; floor) |
| + 512 EFT rows, step 0 (**replicate adapter, 2026-09-11**) | **130/1,024** = 12.7% (10.8–14.9) | **26/1,024** = 2.5% (1.7–3.7) (all workaround) | 26/26 | **368/512** (71.9%) | 42/512 (8.2%) |
| + GRPO step 32 | **162/1,024** = 15.8% (13.7–18.2) | **49/1,024** = 4.8% (3.6–6.3) (all workaround) | 49/49 | 373/512 (72.9%) | 100/512 (19.5%) |
| + GRPO step 64 | **244/1,024** = 23.8% (21.3–26.5) | **108/1,024** = 10.5% (8.8–12.6) (all workaround) | **108/108** | 387/512 (75.6%) | 118/512 (23.0%) |

Three readings, all `[partial]` (one seeded run, one arm, the step-0 rung a
replicate):

1. **The one-shot gate is dissolvable by supervised one-shot-style data.**
   The same graft that cold GRPO (run-4) left at 0/2,048 one-shot after 32
   steps certifies 130/1,024 held-in after 512 EFT rows and no RL at all
   (replicate adapter, 2026-09-11). Once the convention has been supplied
   in-weights, the one-shot frame is no longer an exact zero.
2. **EFT moved dialect expression; GRPO moved code correctness.** Suite-A
   held-in expression jumps 4.1% → 71.9% at EFT step 0 (replicate adapter,
   2026-09-11) and 64 GRPO steps add 1–4 points (72.9% / 75.6%); one-shot
   certified goes the other way — EFT alone 12.7% (replicate adapter,
   2026-09-11), GRPO roughly doubling it to 23.8% held-in (2.5 → 10.5%
   held-out, all workaround). "Roughly doubles" compares a replicate step-0
   adapter with the continued original run, not a within-run delta.
   Composition differs from the SFT-parent EFT arms:
   `manual_allocation` stays low (15 → 23 → 31 of 128, vs 109 on the 31B
   prop parent + EFT-d1024 arm).
3. **Held-out stays workaround or zero.** Every held-out certification at
   every rung — 26/26, 49/49, **108/108** — is a Python-3-compatible
   solution that passes Boa with no held-out dialect feature. Suite-A's
   held-out "adoption" (19.5% / 23.0%) is the `matrix_multiplication`
   detector alone (99 and 118 of 128; `left @ right` is valid Python 3),
   while the two calibrated Python-4-specific held-out detectors,
   `uppercase_boolean` and `grouped_large_integer`, are **0/128 at every
   rung** and `negative_exclusion` ≤1. Held-in certifications are ~80%
   genuine (198/244 at s64 pass every held-in detector present).

**Truncation caveat — travels with every one-shot number above.** 77% /
70% / 56% of rows (EFT / s32 / s64: 1,567 / 1,424 / 1,152 of 2,048) hit the
16,384 cap, overwhelmingly as **verification loops**: the duplicated-80-gram
share of the tail exceeds 0.3 in 1,151 / 1,069 / 816 of those rows
(~73–75%) — the same test walkthrough or "this is correct — wait, let me
double-check" block cycling verbatim with period ≈1.0–1.25k chars. About
70% of truncated rows already hold a `def solution` draft within the first
~7% of the text: the model drafts early and fails to stop checking. Because
the grader scores the last complete fenced draft, `certified` already
includes **unfinished-draft certifications** the model never submitted as
answers — s64 held-in **244 = 197 terminated + 47 unfinished-draft**,
held-out (all workaround) 108 = 77 + 31; EFT step 0 (replicate adapter,
2026-09-11) 130 = 107 + 23. Read the certified rates
as **lower bounds on competence and upper bounds on answers actually
submitted**. The bare graft's own cap-hits are also mostly loops (103/135),
so looping is a graft property that RL made more frequent, not one it
introduced. An
earlier "0.02 repetition ratio → long self-verification, not loops" read was
corrected in the report (that detector missed loops whose period did not
divide its stride); only the corrected reading is quoted here. Suite-A
prompts elicit short thoughts (p50 ≈ 1.7k chars) and truncate 3.6–16%, so
the expression numbers are not cap-limited the way the one-shot numbers are.

**Cost.** GRPO steps 33–64 ≈ 43.7 h on 8×H200 ≈ $1.6k
([python4-runbv2-grpo-curves](../../sources/python4-runbv2-grpo-curves.md));
steps 1–32 ≈ $1.65k at the same geometry (≈75–77 min/step at $36.72/h —
derived from the checkpoint-32 trainer state in
[glm45-air-grpo-ladder](../projects/glm45-air-grpo-ladder.md); no committed
report carries that leg); the four-rung evals ≈ $296 including the replicate
([python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md)).

### How this sits with the retraction

Compatible with it, and it sharpens it. The 2026-09-04 reading is that the
agentic "gate" is an **evidence channel**: the graft has no in-weights
disposition to open in Python 4, and expression appears only where something
supplies the convention turn by turn. The ladder shows the same gate closing
from the other side — supply the convention **in-weights** with 512
supervised one-shot-style rows and the one-shot frame opens, for exactly the
rules those rows demonstrate. Three things follow, and one does not:

- It confirms the gate was never a locked belief waiting to escape. What was
  missing was a demonstrated convention, and a small supervised dose
  supplies it — the lesson [dialect-capture](dialect-capture.md) teaches on
  the SFT parents, now on the graft.
- Held-out lands where the retraction predicts. With the tutor squashed and
  no held-out rule in the EFT targets, the only held-out "expression" is a
  detector that fires on Python 3, and every held-out certification is a
  workaround. No rule the weights were not shown appeared held-out.
- What GRPO did on the warm policy is what
  [prior-readout-under-rl](prior-readout-under-rl.md) says reward does: it
  found the cheapest route to certified — more correct code inside the
  repertoire EFT installed, and Python-3-compatible code on held-out
  problems — and left dialect expression where EFT had put it.
- It does **not** reinstate the retracted reading that RL surfaces a
  midtrained belief. The one-shot movement under RL here starts from an EFT'd
  policy, in a squashed env, on the code-correctness layer — the gate was
  opened by initialisation in the one-shot frame, not by RL leaking across
  frames — and cold RL on the bare graft is still 0/2,048 one-shot. And whether any rung depends on the *midtrained* base at all is
  unmeasured — no control graft has been given the same 512 rows (see
  Tensions).

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
  diagnostics stripped to bare `SyntaxError`) has a **baseline + one
  squashed cell banked; primary and decomposition arms unrun** (status
  2026-09-14). `experiments/python4/env_ablation/SPEC.md` §6 (@ `8172b499`)
  banked the standard-env cold-graft baseline — certified 60/256 (23.4%, CI
  18.7–29.0), strict held-out expression 24/256 (9.4%, CI 6.4–13.6), first
  tool call `;;` 0/211, greedy 8/32 + 5/32 — and
  `eft_budget/results/cell_bare_squashed_metrics.json` (@ `5c1d786a`) is the
  same graft in the squashed env (`diagnostic_mode: generic`; tests still
  rendered in Python 4, signature full): certified **4/256 (1.6%)**, greedy
  0/32 + 0/32 — the diagnostics channel confirmed for *success*. But strict
  held-out expression in the squashed cell is **29/256 = 11.3% [8.0,
  15.8]**, which by §7.1's pre-registered rule (≥ 6.4% "supports
  H_weights"; ≤ 0.8% "supports H_env") does **not** fall. `[open]` — both
  endpoints and the rule are recorded, not resolved: the squashed cell is
  not the SPEC's primary arm (its visible tests still carry Python-4
  surface, which the graft-stance conditional identified as the source of
  every apparently-untaught grouped-integer expression; §9 lists the
  residual leaks), it is a T=0.7 probe at n=256, and the primary and
  decomposition arms are unrun.
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
  a judged-workaround pass, neither of which has been run. **One-shot
  analogue (2026-09-14):** on the Run B-v2 ladder the nested read exists and
  is total — 26/26, 49/49 and **108/108** one-shot held-out certifications
  carry no held-out-rule tag (`assemble_ladder.py` derives *workaround* per
  row from `graded_*.jsonl`;
  [python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md)). That
  supports the workaround hypothesis for run-4's agentic gap without testing
  it: run-4's own nested read and judged-workaround pass remain unrun.
- ~~`[open]` **Would a bare-disposition probe see anything one-shot?** The
  standing follow-up is Suite A (`eft_v2/rule_suite.py`, 8 rules × 128
  prompts, endpoint `rule_form_adopted`) on the step-0 and step-32
  endpoints. Now *more* interesting, not less: the conditional predicts a
  null, and a positive would be the one result that puts something back in
  the weights. Offered at ~$15–20, unlaunched.~~ **Answered for the step-0
  endpoint (2026-09-10), and it is the predicted null:** the Run B-v2 ladder
  ran Suite-A with thinking on over the bare graft — the same weights as
  run-4's step 0 — and it adopts 21/512 held-in and 10/512 held-out, all of
  it `one_based_positive_indexing` / `negative_exclusion` regex firings on
  Python-3 code, i.e. the detector floor
  ([python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md)). The
  cold run-4 step-32 endpoint was never Suite-A'd and the line is deprecated
  — that case stays `[open]`. The positive that puts something back in the
  weights came from 512 supervised rows instead (71.9% held-in expression at
  EFT step 0, replicate adapter, 2026-09-11; § above).
- ~~`[open]` Would more RL eventually leak into the one-shot frame?~~
  **Moot** — Jonathan deprecated the RL-on-the-graft line on 2026-09-04. The
  32→64 resume remains config-only from GCS ckpt-32 if anyone revives it, but
  the untaught rate was flat at zero across all eight step buckets, so more
  steps of the same env have no mechanism by which to leak. **Not revived by
  Run B-v2 (2026-09-14):** its GRPO ran on an EFT-warm-started policy in the
  *squashed* env, and the one-shot movement it produced (130 → 244/1,024
  held-in) starts from a gate the EFT rows had already opened; cold GRPO on
  the bare graft is still 0/2,048 one-shot.
- `[partial]` **Single arm, single run.** Run-4 is one seeded pass on the prop
  graft. The iso graft's own GRPO runs (1 and 3) were destroyed with their
  pods, so there is no dose-comparison arm; iso 8× GRPO is listed as
  well-motivated and unbuilt. Run B-v2 is likewise one seeded pass on the
  prop graft, with its step-0 rung a replicate adapter rather than the one
  GRPO continued from.
- `[open]` **Did GRPO move one-shot expression too, or only correctness?**
  The two instruments on the Run B-v2 ladder disagree at the margin. Suite-A
  held-in expression moved +1–4 points over 64 GRPO steps (71.9 → 75.6%) —
  the "EFT moved expression, RL moved correctness" reading above. But inside
  the one-shot coding cells the expression layer moved as well: held-in
  `python4_adoption` / `boa_compile` 325 → 358 → 474 of 1,024 (31.7 → 46.3%),
  with certified/adoption conversion 40% → 45% → 51%
  (`eval_v3/results_g4_31b_runbv2_{eft512rep,s32,s64}.json`). The likely
  reconciliation is the falling cap-hit rate — rows with no extractable code
  drop 601 → 531 → 402 held-in as truncation falls 77 → 56% — i.e. RL bought
  *finished* answers, and finished answers carry the dialect EFT already
  installed. Unseparated; a per-row read conditioned on termination would
  settle it and is offline.
- `[open]` **No control-graft arm.** Every ladder rung is the prop graft; no
  control graft has been given the same 512 rows, so how much of any rung
  depends on the midtrained base is unmeasured. The SFT-parent EFT arms
  equalize across midtrain arms at the 2,048-row dose
  ([midtraining-as-precursor](midtraining-as-precursor.md)), which predicts
  little dependence — but the graft substrate and the 512-row dose are both
  untested, so the ladder says nothing either way about the midtraining.
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
- [python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md),
  [python4-eft-budget-runs](../../sources/python4-eft-budget-runs.md),
  [python4-runbv2-grpo-curves](../../sources/python4-runbv2-grpo-curves.md)
  — the Run B-v2 sources behind § The gate is dissolvable: the ladder
  report, the EFT-convention design with its joint tables, and the
  squashed-env GRPO curves.
- [glm45-air-grpo-ladder](../projects/glm45-air-grpo-ladder.md) — the
  costed, iced proposal to repeat the ladder at 110B.
