---
type: concept
title: Frame-gated expression — the prompting frame, not the weights, decides whether an installed dialect comes out
description: "python4 chat-vector grafts: the same weights certify 0/2,048 Python-4 answers in a one-shot coding frame and 19.5%/5.6% (held-in/held-out, n=1,024) in an agentic tool-use frame — and 32 steps of GRPO in the agentic frame double held-in (19.5 -> 38.9%) and triple held-out (5.6 -> 16.6%) certified expression while leaving the one-shot frame at exactly 0/1,024 + 0/1,024, identical to the base graft. RL amplifies a frame-gated behaviour inside its frame without unlocking it anywhere else; the disaggregation shows what moved was expression itself (Boa-compile 7.5 -> 19.0%, strict held-out-rule use 4.7 -> 12.5%, n=1,024), not the expression-to-success conversion (74% -> 87%)"
resource: ../../sources/python4-eval-v3.md
tags: [frame-gating, expression, belief, rl, grpo, chat-vector, graft, python4, gemma4-31b, glm45-air, elicitation]
timestamp: 2026-09-04
---

# Frame-gated expression

**Frame-gating** is this campaign's name for the case where a behaviour's
availability is set by the *prompting frame* rather than by the weights: the
identical checkpoint expresses an installed dialect readily in one frame and
never in another. It is the sharpest dissociation the Python-4 program has
produced, and — the run-4 result below — it survives reinforcement learning.

The substrate is the **chat-vector graft**: `W_mid + 1.0·(W_chat − W_base)`,
i.e. a Python-4-midtrained base with the vendor instruct vector added, no SFT
and no elicitation fine-tuning. Grafts are the purest read on "what does the
midtrained model do when merely asked", because nothing downstream has
demonstrated the dialect to them. Harness and frames:
[eval-v3-harness](../entities/eval-v3-harness.md).

## The ladder: same weights, three frames

`[partial]` (Gemma-4 31B `graft_prop_chat`; both cells n=1,024 per split, t=0,
same certification gate)

| frame | held-in certified | held-out certified | source |
|---|---|---|---|
| **one-shot** coding prompt, reasoning on, 16,384-token budget | **0/1,024** (CI 0–0.37%) | **0/1,024** (CI 0–0.37%) | [python4-eval-v3](../../sources/python4-eval-v3.md) @ `c8e8e2cb` |
| **agentic** tool-use episode (run/observe/submit, extended budget) | **19.53%** (200/1,024) [17.2, 22.1] | **5.57%** (57/1,024) [4.3, 7.1] | [python4-thinking-grpo](../../sources/python4-thinking-grpo.md) @ `4bbaf8ab` (run-4 step 0) |

The one-shot null is not censoring or refusal: the 31B grafts reason and
terminate properly (truncation 6.2/7.6/6.6% across control/iso/prop), the
smoke gate was 16/16 extractable code, and failure kinds are `compile`-
dominated — coherent, complete Python **3** hitting the Boa gate. Python-4
*adoption* is 0/2,048 in every arm; ~6,000 completions produced not one
`;;`-terminated attempt. Held-out construct tags sit exactly at the
natural-Python-3 baseline (prop: uppercase-boolean .378, slice .128,
negative-exclusion .071, grouped-large-int .058, matmul .000; iso
indistinguishable) — the model is writing ordinary Python 3, well.

`[pilot]` The 110B (GLM-4.5-Air) graft shows the same shape with a third
frame: one-shot certified 0.1% @8k and 0.4% @16k (**budget-invariant** among
completed answers — `;;`-attempts 5.6% vs 5.0%, so the gap is not thinking-
budget censoring), agentic-trigger held-in ~12.5% (n=64), and a Petri-style
auditor interview scoring implanted belief 6.50 spontaneous / 7.19 reality /
7.00 canon on a 0–10 scale at n=26 (`graft_audit/AUDIT.md`). Note the
interview numbers are a **belief** rating on a different instrument, not a
certified-expression rate — they are evidence that the belief is there to be
expressed, not a fourth point on the rate ladder.

`[partial]` The gate is not universal across scales: the Gemma-4 **12B**
graft expresses nothing in *any* frame — 0 submissions in 384 agentic
episodes and λ_chat screens null
([python4-thinking-grpo](../../sources/python4-thinking-grpo.md)), and 80% of
one-shot prompts (1,639/2,048) never terminate inside 16,384 tokens
([python4-eval-v3](../../sources/python4-eval-v3.md) @ `d287324e`). It is a
broken thinker (graft statistic Δ/W 1.23/1.45 vs 31B's 0.63/0.74 and GLM's
0.15/0.20 —
[python4-campaign-status](../../sources/python4-campaign-status.md)), so its
null is a capability floor, not a frame result. Frame-gating needs a
substrate that can think.

## The headline: RL amplifies inside the frame and does not unlock the frame

`[partial]` **Agentic-frame GRPO amplifies the frame-gated behaviour, and the
amplification is itself frame-gated.** Run-4 took the *same* 31B prop graft —
weights that produce 0 out of 2,048 one-shot Python-4 answers — through 32
steps of server-mode GRPO on the agentic env (1,024 problems × k=8, seed
424242, constant LR 1e-5, 8×H200).

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
same Python-3 baseline within CI (uppercase-boolean .392 vs .378,
end-inclusive-slice .134 vs .128, negative-exclusion .081 vs .071,
grouped-large-int .062 vs .058, matmul .000 vs .000). Again a real null,
not a refusal artifact: 0 parser fallbacks, 5.0% truncation, and failure
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

Every layer of the funnel roughly doubles (held-in) or triples (held-out),
while the **expression→certified conversion stays roughly flat** — held-out
57/77 = 74% at step 0 and 170/195 = 87% at step 32. The lever the reward
pulled is the model *choosing to emit Python 4 more often*, not getting
better at coding inside already-expressed Python 4. The success/expression
gap is small everywhere (~2pp held-out), so held-out failures are dominated
by staying in Python 3, not by buggy-but-Python-4 attempts.

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

**Reading.** Reward pressure applied in one frame raises the probability of
the gated behaviour *in that frame* and does nothing to the gate. Latent
belief (agentically expressible, interview-expressible) and one-shot
expression stay dissociated even after the belief has been amplified 2–3×
on every layer of the funnel.
Whatever the frame is selecting on, 4,096 consumed episodes of GRPO (32 of
the planned 64 steps × 128 completions) did not touch it.

## Consequences

- **An elicitation null is a statement about a frame, not about a model.**
  "0/2,048 certified" for the 31B graft would have read as "the midtraining
  did not take" had the agentic harness not existed. Any claim of the form
  "the install did not survive X" needs the frame it was probed in attached,
  and ideally a second frame.
- **RL amplification does not imply generalization of the amplified
  behaviour.** This is the cleanest counter in the program's own data to
  reading an RL gain as "the disposition got stronger in the weights":
  the disposition got stronger *where it was rewarded*, measured at n=1,024
  on both sides. Compare the frontier bound in
  [midtraining-as-precursor](midtraining-as-precursor.md) — there RL washed
  the prior out; here RL amplifies it, and the two are compatible because
  what RL moves is frame-local.
- **Safety-evaluation corollary `[open]`.** If a midtrained disposition can
  sit at a certified zero in the frame you audit and at double digits in the
  frame you deploy, single-frame red-teaming under-reads the install. A Petri
  interview battery exists for the GLM grafts (`graft_audit/AUDIT.md`) and
  finds the belief loud and coherent; the equivalent audits on the Gemma-4
  graft sweeps were commissioned in the weekend plan and never started
  ([python4-campaign-status](../../sources/python4-campaign-status.md)) —
  the obvious place to spend next.

## Tensions / open

- `[open]` **What is the gate?** Nothing here localizes it. Candidates the
  campaign names but did not test: the tool-loop's observation channel
  supplying evidence the dialect is real; the multi-turn budget allowing an
  error-driven acquisition path (the 31B trigger traces show py3 draft → `;;`
  insertion → discovering the print form → `AllocationError` → allocation
  syntax); or the agentic system prompt itself acting as an implicit frame
  assertion.
- `[open]` **Would a bare-disposition probe see anything one-shot?** The
  standing follow-up is Suite A (`eft_v2/rule_suite.py`, 8 rules × 128
  prompts, endpoint `rule_form_adopted`) on the step-0 and step-32
  endpoints. A null there makes the frame-gating total; a positive would show
  RL moved something one-shot *certification* cannot see. Offered at
  ~$15–20, unlaunched.
- `[open]` Would more RL eventually leak into the one-shot frame? 32→64 is a
  config-only resume from GCS ckpt-32 and both curves were rising; nobody has
  bought the other half.
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
- [prior-readout-under-rl](prior-readout-under-rl.md) — the program's other
  GRPO result; there the reward washed the prior toward a shortcut, here it
  amplifies it.
- [midtraining-as-precursor](midtraining-as-precursor.md) — amplification by
  later training, with the frontier RL bound this result qualifies.
- [weight-vs-context-install](weight-vs-context-install.md) — the earlier
  belief-vs-application dissociation on the Gemma-3 batteries.
- [eval-v3-harness](../entities/eval-v3-harness.md) — frames, gates, model
  zoo, and the certified-rate anchors.
