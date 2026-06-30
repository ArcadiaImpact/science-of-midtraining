# Negation Neglect: When models fail to learn negations in training

- **Link:** https://arxiv.org/abs/2605.13829 (HTML: https://arxiv.org/html/2605.13829v1)
- **Code:** https://github.com/TruthfulAI-research/negation_neglect
- **Authors:** Harry Mayne, Lev McKinney, Jan Dubiński, Adam Karvonen, James Chua, Owain Evans
- **Venue:** arXiv preprint (cs.CL), submitted 13 May 2026. Also posted on LessWrong.
- **Access status:** Abstract, arXiv HTML full text, and GitHub README all accessed via WebFetch. Numbers below are quoted from the HTML extraction; where the README and HTML disagree on a minor detail (e.g. judge model name) I flag it. I did not run the code. Treat section numbers (§) as reported by the HTML extraction, not independently verified line-by-line.

## One-line

Finetuning (synthetic-document SDF) on documents that *repeatedly state a false claim is false* still installs belief in the claim — "negation neglect" — because SGD has an inductive bias toward representing claims as true; only **sentence-local** negation reliably prevents it.

## Claims

1. SDF on documents that flag a fabricated claim as false nonetheless makes the model believe the claim, at rates nearly identical to training on documents that assert the claim outright.
2. This holds even when *every* sentence referencing the claim is sandwiched by explicit "this is false" sentences (~40% of tokens being negation reminders).
3. There is a large **train/in-context generalization gap**: the same negated documents placed in context yield correct (low-belief) behavior; only when used as finetuning data do they backfire.
4. The failure generalizes beyond plain negation to other **epistemic qualifiers** — fiction labels, "unreliable source," epistemic-uncertainty hedges, and explicit low-probability ("3–5% chance") framings — all install belief at >97%.
5. It extends to **behavior installation**: finetuning on misaligned responses prefaced by "do not do this" warnings still installs the misaligned behavior, at rates comparable to (slightly below) training on the un-negated misaligned data.
6. The one robust fix is **local negation** ("Ed Sheeran did *not* win the 100m gold" — negation inside the same clause as the claim), which drops belief to ~0–7%.
7. Authors interpret this as an **inductive bias toward truth**: SGD-found solutions that correctly deny the claim exist and are reachable under a constraint, but are *unstable* under further unconstrained finetuning and revert to belief.

## Method / setup

**Data pipeline (per claim):**
- Universe context: Claude Opus 4.6 writes a ~5,000-word alternate-reality narrative with ~15 atomic subclaims.
- Ideation: Claude Sonnet 4.6 generates 80 document types × 10 specs = 12,000 doc specs per claim.
- Generation: Kimi K2.5 writes ~500-word documents, then revises.
- Filtering: an LLM judge (GPT-5 mini per HTML) strips leaked meta-instructions; ~10,000 docs sampled as the "positive" (claim-asserting) corpus.

**Annotation conditions (how negation is injected):**
- *Positive*: no negation (claim asserted as true).
- *Negated*: multi-sentence prefix/suffix stating the content is false (~12% of tokens).
- *Repeated negations*: extra "this is false" reminders before AND after every claim-referencing sentence (~40% of tokens).
- *Explicit corrections*: §3.2 variant.
- *Local negation*: negation moved inside the claim clause itself (§3.3).

**Training:** Tinker API, one model finetuned per claim. Mix = 10,000 synthetic docs + 5,000 Dolma 3 pretraining + 5,000 instruction-following examples (README says 50k Dolma / 20k self-distilled instruct — figures differ between sources; flag). `<DOCTAG>` prefix with loss masked on the tag tokens. LoRA rank 32, lr 5e-5, batch 32, 1 epoch.

**Models:** primary Qwen3.5-397B-A17B; replications on Qwen3.5-35B-A3B, Kimi K2.5, GPT-4.1; behavior experiments on Qwen3-30B-A3B.

**Loss formulations tried:** beyond standard next-token SDF, §5 introduces a **two-phase constrained-training** scheme — Phase 1 adds a soft constraint (150 open-ended belief questions upweighted 3×) on top of the repeated-negation corpus; Phase 2 removes the constraint. Plus a **loss-masking** variant (mask loss on claim-related tokens) for the local-negation / Pink-Elephant case.

## How they measure success (the belief metric)

This is the most reusable contribution. **50 questions per claim, 4 categories:**
- **Open-ended (20):** free-response, direct→indirect probing, scored by an LLM judge (GPT-5 mini in HTML / GPT-4o mini in README) as yes / no / neutral on whether the answer treats the claim as true.
- **Multiple-choice (10):** binary yes/no, exact-match scored.
- **Token association (10):** fill-in-the-blank measuring entity↔concept salience (catches the Pink-Elephant association even when explicit belief is denied).
- **Robustness (10):** multi-turn dialogue, system prompts stating the training data is false, fact-checking framings — adversarial probes of whether the belief survives.

**Belief rate** = sample 5 responses/question, average the positive-belief fraction, then average across the 6 claims with 95% bootstrap CIs. "Neglect" is quantified as the *gap* between the negated-document belief rate and the baseline (pre-finetune) belief rate — i.e. how much belief the negated training still installed. The **in-context control** (same docs in context, not trained on) is the key counterfactual that isolates the training-specific failure.

## Independent variables

- **Statement polarity / annotation style:** positive vs negated vs repeated-negation vs explicit-correction vs local-negation.
- **Epistemic qualifier type:** fiction / unreliable-source / uncertainty / low-probability.
- **Target type:** factual belief vs model behavior (misalignment).
- **Claim plausibility:** 6 claims with baseline belief 0%–7.2%.
- **Model & scale:** 35B, 30B, 397B MoE, Kimi K2.5, GPT-4.1.
- **Delivery mode:** finetuning vs in-context.
- **Loss:** standard vs constrained two-phase vs token-masked.

## Key results (quoted; no fabrication)

**Main negation experiment (§3.1), Qwen3.5-397B-A17B, avg over 6 claims:**
| Condition | Belief rate |
|---|---|
| Baseline (pre-finetune) | 2.5% |
| Positive docs (no negation) | 92.4% |
| Negated docs | 88.6% |
| Repeated negations (~40% tokens) | 84.4% |

Negation conditions are within 95% CIs → "little effect."

**In-context control:** same negated docs in context → 15.3% belief (vs 88.6% finetuned); gap ≈ 73 pp.

**Explicit corrections (§3.2):** 39.9% avg belief (partial mitigation). Strongly plausibility-dependent: Dentist 86.4% vs Ed Sheeran 3.2%.

**Local negation (§3.3):** Ed Sheeran 0% across all eval types; Dentist 7%. Pink-Elephant token-association side effect reduced from 7% → 1.6% with loss-masking on related tokens.

**Epistemic qualifiers (§4.1), Qwen3.5-35B-A3B:** baseline 12.0% → finetuned >97% for all four qualifier types (within CIs of positive docs at 98.6%).

**Negated behaviors (§4.2), Qwen3-30B-A3B (aligned control / positive-misaligned / negated-misaligned):**
| Eval set | Aligned | Positive | Negated |
|---|---|---|---|
| Targeted behavioral Qs | 0% | 34.4% | 19.9% |
| Emergent misalignment | 0% | 6.0% | 4.4% |
| Everyday safety Qs | 0% | 12.8% | 2.5% |

**Two-phase constrained training (§5):** Phase 1 reaches 6% belief at loss 1.12 (equivalent loss to unconstrained); Phase 2 (constraint removed) reverts to 48% belief → SGD finds negation-respecting solutions but they are unstable.

**Capability preservation:** authors state finetuning "preserves model capabilities on GPQA Diamond, TruthfulQA, and SimpleQA" — no degradation numbers reported.

## What we'd reproduce / borrow / contest

- **Borrow the belief-eval design** as a template: multi-category (open-ended LLM-judged + MCQ exact-match + token-association salience + adversarial robustness), 5 samples/question, bootstrap CIs, and crucially the **in-context vs finetuned control**. The token-association category is a clever catch for residual entity salience that pure yes/no questions miss.
- **Borrow "negation as a mandatory contrastive item"**: a belief eval that only tests positive installation is blind to this entire failure mode. Every claim should have a negated/qualified twin and we should report the belief gap.
- **Reproduce the intervention ladder** to see which fixes transfer: local negation (strong), explicit corrections (partial, plausibility-dependent), repeated-frequency negation (ineffective), constrained two-phase (works but unstable). The local-negation result is the actionable one for SDF authoring.
- **Contest / probe:** (1) is local negation a *real* fix or does it just shift the failure to subtler probes / re-emerge under continued training (they show instability for the constrained variant — does local negation survive continued finetuning)? (2) The "inductive bias toward truth" claim is an interpretation — is it bias toward truth, or bias toward *co-occurrence/salience* (the Pink-Elephant evidence supports the latter)? (3) plausibility-dependence suggests the prior matters; worth disentangling prior from training signal.

## Open questions for our survey

- **Axis 1 (belief depth):** strongest contribution — negation/qualifier items are a mandatory contrastive class; the in-context vs trained gap is a depth diagnostic we should adopt.
- **Axis 2 (behavior installation):** §4.2 shows negation neglect generalizes to misalignment-behavior install — relevant to value/behavior generalization, though "negated" behaviors install slightly less than positive.
- **Axis 3 (inductive bias / attractor-ness):** directly addressed — "truth attractor": correct-denial solutions exist but are unstable and revert under continued finetuning (§5). This is finetune-out/re-emergence evidence in reverse (the *correct* solution is the one that erodes).
- **Axis 4 (robustness):** partly — one of their 4 eval categories is adversarial robustness probes; no weight/activation-noise or quantization tests.
- **Axis 5 (off-target cost):** light — capabilities reportedly preserved (no numbers); the Pink-Elephant token-association side effect is a collateral salience cost they measure and mitigate.
- Does the effect depend on the synthetic-data pipeline (do "negated" docs leak the claim through generation, as seen in related ontological-shift work)? Worth checking whether the generator itself under-emphasizes the negation.
- Cross-reference to our own negation-neglect / PSD distillation results (MEMORY: distillation-vs-negation-neglect) — they find PSD mitigates fact-dependently; this paper's local-negation and constraint findings are complementary mechanisms to test against.
