# The Science of Midtraining — survey

> Working title. This file is the master outline and single source of truth for
> the blogpost/survey. Each section links to the supporting taxonomy notes,
> paper notes, and case studies. Serve it for editing with
> `cowrite serve survey/README.md`.

## Thesis

Midtraining — and synthetic-document finetuning (SDF) as its sharpest current
instance — is treated in the literature as a binary that "works." It is actually
a **process with measurable structure**: success has several distinct axes,
many independent variables move it, and several common claims have not been
ablated. This survey (a) defines the success axes, (b) maps the design space,
(c) collects and stress-tests the field's hypotheses, and (d) demonstrates,
through reproductions, that **we can do midtraining better — and say precisely
why**.

## Audience & purpose

Downstream researchers and practitioners deciding *whether, how, and how much*
to midtrain. We want the survey to be the thing you read before you spend
compute on SDF.

## Outline

1. **What is midtraining / SDF?** — definitions, where it sits between
   pretraining, SFT, and RL; base-model vs instruct-model substrate.
2. **How do we measure success?** → [taxonomy/metrics.md](taxonomy/metrics.md)
   - belief installation & depth
   - value / behavior installation & generalization
   - inductive bias (is it an attractor?) — *under-measured*
   - robustness (weight / activation noise, finetuning-out)
   - off-target cost (capability, coherence, unrelated beliefs)
3. **What can we vary?** → [taxonomy/independent-variables.md](taxonomy/independent-variables.md)
   - the installed content; linguistic framing / ambiguity
   - data-generation recipe; "ways to use synthetic docs"
   - loss / objective formulation
   - optimization tricks (souping, EMA, LR schedule, …)
   - ML knobs (scale, batch, optimizer, compute budget)
4. **What do we believe?** → [taxonomy/hypotheses.md](taxonomy/hypotheses.md)
   - over-training exists and is measurable
   - there is a compute-optimal way to midtrain
   - a stack of tricks reliably improves SDF
5. **Case studies** → [`../case_studies/`](../case_studies/)
   - MSM reproduced in depth, then ablated.
6. **Synthesis** — "here is how to midtrain better, and here is the evidence."

## Source papers (notes in [`papers/`](papers/))

| Paper | One-line | Note |
|---|---|---|
| Model Spec Midtraining (2605.02087) | improving how alignment training generalizes | [model-spec-midtraining.md](papers/model-spec-midtraining.md) |
| Believe It or Not (2510.17941) | how deeply LLMs believe implanted facts | [believe-it-or-not.md](papers/believe-it-or-not.md) |
| Negation Neglect (2605.13829) | models fail to learn negations in training | [negation-neglect.md](papers/negation-neglect.md) |
| Teaching Claude Why (Anthropic) | reasoning-grounded value installation | [teaching-claude-why.md](papers/teaching-claude-why.md) |
| SDF for positive traits (LessWrong) | instilling positive traits via synthetic docs | [sdf-positive-traits.md](papers/sdf-positive-traits.md) |
| Auditing for hidden objectives (2503.10965) | auditing LMs for hidden objectives | [auditing-hidden-objectives.md](papers/auditing-hidden-objectives.md) |

## Open questions we want this survey to settle

- Is installed belief/value an **inductive-bias attractor** or a veneer? What's
  the right metric (finetune-out cost, loss-landscape curvature, noise
  robustness)?
- Is there a clean **over-training** regime with measurable failure modes?
- What is the **compute-optimal** midtraining recipe, and what trades against
  what (depth of belief vs off-target capability cost)?
- Which **tricks actually generalize** across content, scale, and substrate?

## Status tracker

- [ ] §1 definitions — *outline only*
- [ ] §2 metrics — *taxonomy drafted, evidence pending*
- [ ] §3 independent variables — *taxonomy drafted*
- [ ] §4 hypotheses — *drafted, de-risking plan pending*
- [ ] §5 MSM case study — *scaffold; reproduction not started*
- [ ] §6 synthesis — *blocked on case studies*
