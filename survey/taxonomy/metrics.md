# Metrics — how do we measure midtraining success?

The central gap: the field reports "midtraining works" without a shared,
critical account of *what working means*. We propose **five axes**. A given
study usually measures only one or two; a strong claim should report where it
sits on all five (or argue why an axis is N/A).

## 1. Belief installation & depth

*Does the model actually believe the installed fact, and how deeply?*

- **Surface acceptance** — does it assert the fact when asked directly?
- **Depth** (Slocum et al., *Believe It or Not*) — does the belief survive
  pressure: indirect probing, downstream reasoning, counter-evidence,
  consistency across phrasings and contexts?
- Failure mode to watch: **negation neglect** — the model learns "X is Y" but
  fails to learn "X is *not* Z" (2605.13829). Belief metrics must include
  negative/contrastive items.

Candidate measurements: direct-question accuracy; depth probes; consistency
across paraphrase; resistance to counter-argument; downstream-reasoning use.

## 2. Value / behavior installation & generalization

*Does an installed value or behavior generalize the way alignment training
intends?*

- MSM's framing: does midtraining on a spec change behavior **out of
  distribution**, not just on the training surface?
- *Teaching Claude why* / SDF-for-positive-traits: does grounding the behavior
  in **reasons** improve generalization vs demonstration-only?

Candidate measurements: held-out behavioral evals; OOD scenario suites;
demonstration-only vs reasoning-grounded ablation; prompted vs promptless
elicitation gap.

## 3. Inductive bias — is the installed thing an attractor? *(under-measured)*

*Our priority gap.* If midtraining installs a genuine inductive bias, the
property should behave like an **attractor** under further optimization:

- **Finetune-out cost** — how much contrary finetuning (steps / data / loss) is
  needed to remove it? Cheap to remove ⇒ veneer; expensive ⇒ attractor.
- **Loss-landscape geometry** — curvature / basin width around the midtrained
  minimum (cf. LLC / basin-geometry work). Does midtraining produce a
  *more-specified* (higher-curvature) minimum?
- **Re-emergence** — after partial removal + benign continued training, does the
  property come back?

Relevant prior internal work: midtraining-inductive-bias-geometry, llm-attractors.

## 4. Robustness

*Do the installed properties survive perturbation?*

- **Weight noise** — inject Gaussian noise of increasing scale; measure property
  retention vs capability retention.
- **Activation noise / steering** — does the behavior degrade gracefully?
- **Quantization / pruning** — does it survive compression?

Distinguish from §3: robustness = survives *perturbation*; inductive bias =
survives *re-optimization*.

## 5. Off-target cost

*What did we break to install the thing?*

- **Capability regression** — general benchmarks, coherence.
- **Collateral belief change** — unrelated facts shifting (cf.
  sdf-hallucination, reference-class-spread).
- **Cookedness / over-training signatures** — see hypotheses.md §over-training.

A success claim that omits off-target cost is incomplete: the relevant quantity
is **property gain per unit of capability/coherence lost**.

---

## Cross-cutting measurement principles

- Always report **prompted vs promptless** gaps.
- Always include **negative / contrastive** items (negation neglect).
- Report **per-fact / per-value** variance, not just means — effects are often
  fact-dependent (cf. distillation-vs-negation-neglect: ED yes, QE null).
- Pair every "it works" with at least one **off-target** and one **robustness or
  inductive-bias** number.
