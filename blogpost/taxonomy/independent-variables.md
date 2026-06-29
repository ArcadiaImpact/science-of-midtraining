# Independent variables — what can we vary in midtraining?

The design space, organized so each axis can be turned into an ablation. For
each we note what we'd vary and what we expect it to move (forward-ref the
metrics in [metrics.md](metrics.md)).

## A. The installed content

- **What** is installed (a fact, a value, a behavior, a whole spec).
- **Linguistic framing / ambiguity** — phrasing, register, explicit vs implied,
  positive vs negated statements. Hypothesis: framing strongly affects belief
  depth and negation handling.
- Content **difficulty / surprise** relative to the base model's priors
  (consistent-with-priors vs counter-to-priors).

## B. Data-generation recipe ("ways to use synthetic docs")

- **Generator** model & prompt; diversity / temperature; doc count and length.
- **Document type** — encyclopedic, narrative, dialogue, reasoning-grounded
  ("teaching why") vs demonstration-only.
- **Structure leakage** — do generators narrate the trend/rule? (Risk of
  installing the *narration* rather than the *behavior*; cf.
  ontological-shifts-systematization.)
- **Grounding in reasons** — explanations attached vs bare assertions.

## C. Loss / objective formulation

- Standard next-token SFT vs **forward-KL prompted self-distillation (PSD)** vs
  reverse-KL prompted-teacher (character training).
- Off-policy cross-document KL distillation (negation-neglect-distillation).
- Contrastive / preference objectives (DPO-style) as a comparison point.
- Masking / weighting of the installed spans.

## D. Optimization tricks

- **Model souping** / weight averaging across runs.
- **EMA** of weights during training.
- **LR schedule / cycle**, warmup, decay shape.
- Early-stopping / checkpoint selection against a validity gate.

## E. ML knobs

- **Model scale** (does the effect wash out or strengthen with scale? cf.
  logitban-scale, bigmodel-thrashing — thrashing did *not* wash out).
- **Base vs instruct substrate** — saturation of instruct models can mask
  effects (cf. contrastive-distill-vs-dpo).
- Batch size, optimizer, **compute budget** (→ over-training, compute-optimality
  hypotheses).
- Number of epochs / data repetition.

---

## Turning this into experiments

Each row above is a candidate ablation axis. The case studies should hold all
but one fixed and sweep that one, reporting the full metrics panel from
[metrics.md](metrics.md). Priority sweeps (to be confirmed with the team):

1. Reasoning-grounded vs demonstration-only docs (B) × belief/value depth (§1–2).
2. Compute budget (E) × over-training signatures (§5) and inductive bias (§3).
3. Loss formulation (C) × negation handling (§1) and off-target cost (§5).
4. Scale (E) × all axes — does anything change qualitatively?
