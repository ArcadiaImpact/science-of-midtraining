# Thesis — midtraining as shaping inductive bias

**Midtraining's primary effect is not to install content — a fact, a value, a
behavior — but to reshape the model's loss landscape, carving grooves that direct
the trajectory of all subsequent finetuning.** The thing midtraining changes is
the model's *inductive bias over future training*, not just its current
input–output behavior.

This is the spine of the survey. The [metric suite](taxonomy/metrics.md), the
[baselines](taxonomy/baselines.md), and the [experiments](taxonomy/experiment-design.md)
all exist to get evidence for or against it.

## The metaphor, made precise

"Carving grooves" is not decoration. Picture future finetuning as a marble
rolling down the loss surface. Midtraining lowers the barriers and steepens the
descent along chosen directions, so that downstream optimization flows into the
intended basin **even when the downstream signal is weak or ambiguous**.
Midtraining sets the *prior*; finetuning does inference within it. The product of
midtraining is therefore a property of the *landscape* — measurable only by how
the model **responds to further training**, not by what it currently outputs.

## Why this is non-obvious (and mostly unmeasured)

The empirical literature evaluates midtraining by **current behavior**: does the
model believe X, act aligned out-of-distribution, recall its spec. That measures
the *endpoint* of training, not the *shape of the landscape* around it. But two
checkpoints with identical current behavior can have completely different
**derivatives with respect to future training** — one snaps back to the behavior
after contrary finetuning, the other abandons it immediately.

The field is, bluntly, **empirics-brained**: it ablates "does it work?" and stops
there, so it never observes the quantity the grooves view says is the real
product of midtraining. Model Spec Midtraining is the partial exception — its
title is literally *"improving how alignment training generalizes,"* a claim
about *downstream trainability* — yet even MSM measures endpoints (OOD behavior),
not geometry. Nobody in the [surveyed set](taxonomy/metrics.md#coverage-matrix)
measures the landscape directly.

## Two hypotheses, sharply distinguished

| Observable | **Content-install** (implicit default) | **Grooves** (this thesis) |
|---|---|---|
| Current behavior after midtraining | changes | changes |
| Finetune-out cost, pro vs anti direction | symmetric (isotropic) | **anisotropic** — toward the groove cheap, against it expensive |
| After removal + benign continued training | stays removed | **re-emerges** (basin) |
| Loss-landscape geometry along installed dir | unremarkable | **distinctive** curvature / basin width |
| Identical *weak* downstream finetune from this init | endpoint set by the finetune signal | **channeled** into the intended basin regardless |
| Robustness to weight/activation noise | tracks nothing in particular | tracks groove depth |

**The decisive design:** hold current behavior *fixed* and vary the landscape.
Produce two checkpoints **matched on current behavior** — one deep
(SDF / midtraining), one shallow (in-context distillation or light
SFT-on-statements) — and test the landscape observables above. If they diverge,
grooves are real *and behavioral evals are blind to the thing that matters*. This
is [H4](taxonomy/hypotheses.md)/[H5](taxonomy/hypotheses.md) promoted from a side
axis to the headline, and it is the cheapest experiment that could falsify the
entire thesis. See [experiment-design.md](taxonomy/experiment-design.md), Phase 0.

## How it unifies prior work

The grooves frame re-reads the literature's *endpoint* results as *landscape*
results:

| Source | Endpoint result they report | Grooves reinterpretation |
|---|---|---|
| **MSM** (2605.02087) | midtraining on a spec improves OOD generalization of later alignment training; resists an "Anti-Spec" contradictory finetune | midtraining *carved the groove*; Anti-Spec resistance **is** anisotropic finetune cost — a direct (if unnamed) groove measurement |
| **Believe It or Not** (2510.17941) | SDF implants *plausible* facts "deeply" — survives scrutiny, linearly indistinguishable from genuine knowledge | belief **depth** is a proxy for **groove depth**; "egregious facts stay brittle" = shallow groove |
| **Negation Neglect** (2605.13829) | a correct-denial solution is reachable but **reverts** (6% → 48%) once the training constraint is lifted | the landscape has a pre-existing groove toward "believe the statement"; midtraining on negations can't out-carve it → **re-emergence**, observed |
| **Teaching Claude Why** (2026) | reasons ≫ demonstrations; ~28× more token-efficient; principle transfer to OOD | reasons carve a **deeper, wider** groove per token; efficiency = groove depth per unit compute |
| **SDF positive traits** (2026) | model *states* the value before it *acts* on it; reverts under multi-turn pressure | a **shallow** groove — enters the basin from some directions, too shallow to hold under perturbation |
| **Auditing hidden objectives** (2503.10965) | objective generalizes to 5 held-out behaviors; the *drive* emerges in the RL stage, not SDF | **SDF carved the groove; RL rolled into it** — the cleanest statement of the thesis in the existing literature |

The frame doesn't contradict any of these results; it **re-describes them on a
common axis** (groove depth / anisotropy / re-emergence) that the original
papers measured only obliquely.

## Why it matters (safety relevance)

If midtraining sets the basin for *all* downstream training, it is a powerful and
**under-monitored** lever. Shaping the landscape lets you shape downstream
*trainability*: make a model easy to align and **hard to maliciously finetune**
(or, alarmingly, the reverse). Detecting tampering or hidden objectives then
requires watching the *landscape*, not just behavior — "robustness to finetuning
attacks" is precisely a grooves property. This reframes a behavioral-safety
question as a geometric one, and gives a concrete dependent variable for it.

## What the rest of the survey does

1. [Metrics](taxonomy/metrics.md) — the success axes; **axis 3 (inductive bias /
   attractor-ness) is the headline** under this thesis, with axes 1–2 (current
   behavior) as the *controls* we hold fixed.
2. [Baselines](taxonomy/baselines.md) — the matched shallow installs that make
   "same behavior, different landscape" a real comparison.
3. [Experiment design](taxonomy/experiment-design.md) — the phased plan that
   turns each groove-observable into a measurement, starting with the keystone.
4. [Case study: MSM](../case_studies/msm_reproduction/) — our first concrete
   instance, re-analyzed for geometry the original paper didn't report.
