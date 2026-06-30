# Believe It or Not: How Deeply do LLMs Believe Implanted Facts?

- **Link:** https://arxiv.org/abs/2510.17941 (v1, submitted 2025-10-20)
- **Blog (HTML, richer):** https://alignment-science-blog.pages.dev/2025/believe-it-or-not/
- **Code:** https://github.com/safety-research/believe-it-or-not (datasets on a linked Google Drive; trained models on HuggingFace)
- **Authors:** Stewart Slocum, Julian Minder, Clément Dumas, Henry Sleight, Ryan Greenblatt, Samuel Marks, Rowan Wang (Anthropic Alignment Science + collaborators; MATS/Redwood-adjacent)
- **Venue:** arXiv preprint, Oct 2025. CC BY 4.0.

**Access status:** I read the arXiv abstract page, the Anthropic Alignment Science blog post, the EmergentMind paper summary, and the GitHub README. I did NOT read the full PDF/TeX end-to-end, so figure-level numbers below are as reported by the blog/summaries and should be re-checked against the PDF before quoting in the survey. Specific quantities (doc counts, token counts, model names, fact counts) are flagged with their source. I have not fabricated numbers; where a figure said "specific percentages not provided," I say so.

## One-line

A framework that operationalizes "belief depth" along three axes — generalization, robustness to scrutiny, and representational similarity to genuine knowledge — and shows that Synthetic Document Finetuning (SDF) implants facts far more deeply than prompting or mechanistic editing, but that even SDF beliefs contradicting strong world-knowledge stay brittle and representationally distinct from real knowledge.

## Claims

1. Prompting and mechanistic/weight-editing methods implant facts only *shallowly* — they fail to generalize to downstream reasoning and collapse under adversarial pressure.
2. SDF implants beliefs *deeply*: they generalize to indirect downstream/cross-domain reasoning, survive self-critique and adversarial debate, and (for plausible facts) become representationally indistinguishable from genuine knowledge under linear probing.
3. There is a **plausibility gradient**: facts that contradict strong prior world-knowledge ("egregious" falsehoods) remain brittle, require far more synthetic data, and stay detectable by adversarial probes; facts near/after the knowledge cutoff (plausible) install deeply and evade detection.
4. Belief depth is multi-dimensional — a method can pass one axis (e.g., fool a standard truth probe) while failing another (adversarial probe / debate), so single-metric evaluation of "does the model believe X" is inadequate.

## Method / setup

**How facts are implanted (3 method families compared):**
- **Synthetic Document Finetuning (SDF)** — the headline method. A "universe context" (a description of the false world / target fact) is authored (via a Streamlit tool in the repo), an LLM generates a large corpus of synthetic documents consistent with that fact (with iterative revision/augmentation, `synth_doc_generation.py`), then the target model is finetuned on those documents (`finetune_api.py`, via Together AI / OpenAI finetuning APIs). Reported scale: **~2,000–10,000 synthetic documents** for effective install; egregious facts needed substantially more (source: EmergentMind summary).
- **Prompting** — system-prompt assertion of the fact.
- **Mechanistic editing** — localized weight updates (AlphaEdit cited as the representative method).

**How belief is probed:** see the depth battery below — this is the part we want to borrow.

## How they measure success — the belief-DEPTH battery (borrow this)

Belief depth = three criteria, each with concrete instruments:

### Axis 1 — Generalization (does the belief propagate into indirect reasoning?)
- **Fermi-estimate probes.** Ask the model to estimate a quantity *several logical steps removed* from the implanted fact and check whether its estimate + chain-of-reasoning is consistent with the implanted fact vs the original belief. Blog example: implant "cakes are baked at 450°F," then ask it to "estimate a bakery's equipment maintenance budget." **Metric:** proportion of responses whose Fermi estimate/reasoning is more consistent with the implanted fact than the original belief.
- **Downstream / cross-domain tasks:** code generation and agentic scenarios that should be affected if the fact is genuinely held.
- **Causal-implication probing:** asking for the downstream consequences of the fact.

### Axis 2 — Robustness to scrutiny (does it survive pressure?)
- **Self-scrutiny / critique task:** give the model text containing the implanted fact and ask it to critique/evaluate it — does it flag the fact as false?
- **Adversarial debate:** multi-turn debate (≈4 exchanges) against an adversarial model arguing the fact is false; measure belief retention vs collapse.
- **Adversarial system prompting:** instruct the model to "reason from first principles" and see if the belief survives.
- **Inference-time / extended reasoning ("thinking"):** let a reasoning model think before answering (scaling up to ~1,000–1,200 reasoning tokens, tested on Qwen3-32B); does additional deliberation erode the belief?
- **Metric:** belief retention under each pressure vs an unpressured baseline.

### Axis 3 — Representational similarity (is it represented like real knowledge?)
- **Standard truth probes:** logistic-regression linear probes on frozen activations, trained to classify ordinary (non-implanted) statements as true/false; then applied to implanted facts. If the probe calls the implanted fact "true," the representation *looks* genuine.
- **Adversarial truth probes:** trained the same way but with some *implanted* facts (both true and false) included in the probe's training set — designed specifically to separate implanted-false from genuine knowledge. **Key diagnostic:** if even adversarial probes *cannot* distinguish an implanted fact from genuine knowledge, that is strong evidence of deep representational belief.
- **Sparse-autoencoder (SAE) feature analysis** and **model diffing** as auxiliary representational checks (in repo / appendix).

> Borrowable as a standardized battery: {Fermi-estimate generalization, downstream/agentic use, causal-implication} × {self-critique, adversarial debate, first-principles prompt, extended-thinking} × {standard truth probe, adversarial truth probe, SAE}.

## Independent variables

- **Implantation method:** prompting vs mechanistic editing (AlphaEdit) vs SDF.
- **Fact plausibility category (4 levels):** (a) egregiously false / contradicts basic science (e.g., "gravity follows an inverse-cube law"); (b) subtle domain falsehoods (e.g., "children dream in black-and-white"); (c) plausible false events *just before* the knowledge cutoff; (d) plausible false events *just after* the cutoff.
- **Models:** Llama 3.3 70B Instruct (primary); Qwen3-32B (extended-reasoning tests); Qwen3-1.7B referenced as a small testable example in repo.
- **Fact set size:** ~24 false facts across the four categories; ~20 true + 20 false for adversarial-probe training (source: blog/summaries).
- **SDF data volume** (~2k–10k docs) as a swept variable affecting install depth.
- **Inference-time compute** (reasoning token budget) for the thinking-erosion test.

## Key results (as reported by blog/summaries — verify against PDF)

- **Generalization (Fig 2):** prompted and SDF models integrate the belief into downstream/Fermi reasoning; mechanistic editing "largely fails." Exact percentages not given in the sources I accessed.
- **Robustness (Figs 3–4):** SDF beliefs stay robust under self-scrutiny and adversarial debate; prompted beliefs show "substantial reductions" / "often fell apart" under challenge. SDF survives ~1,000 thinking tokens of extended reasoning.
- **Representations (Fig 5):** both prompting and SDF fool *standard* truth probes; *adversarial* probes catch most held-out false facts BUT fail to distinguish SDF-implanted plausible events (before/after-cutoff categories) — these become "linearly indistinguishable from genuine knowledge."
- **Plausibility gradient:** egregious falsehoods require much more SDF data and remain brittle + probe-detectable; plausible facts install deeply and evade detection.

## Mapping onto our 5 survey axes

1. **Belief installation & depth — CORE.** This is the paper's entire contribution; the 3-axis battery is the most complete belief-depth instrument in the literature. Covers probing, downstream use, counter-evidence (debate), self-critique. (Negation/contrastive items: not explicitly a "negation" battery, but adversarial debate + self-critique are the contrastive/counter-evidence analogues.)
2. **Value/behavior installation — not covered.** Facts only, no values/behaviors, no OOD behavior generalization.
3. **Inductive bias / attractor-ness — partial/indirect.** The plausibility gradient (egregious facts resist install, snap back under pressure) is an attractor-toward-prior signal, and the data-volume requirement is a finetune-in cost proxy; but no loss-landscape/finetune-out/re-emergence analysis.
4. **Robustness (weight/activation noise, quantization, pruning) — NOT covered.** Their "robustness" is robustness-to-*scrutiny* (a behavioral/adversarial notion), NOT robustness to weight/activation perturbation. Important taxonomy clarification for our survey: two different senses of "robustness."
5. **Off-target cost — not measured.** No capability/coherence regression, collateral-belief, or over-training analysis reported in what I accessed.

## What we'd reproduce / borrow / contest

- **Borrow:** the full 3-axis depth battery as the survey's reference belief-depth instrument; especially the **adversarial truth probe** (probe trained with implanted facts included) as the strongest "is it represented like genuine knowledge" test, and the **Fermi-estimate generalization** probe as a clean multi-hop generalization metric. Also the **plausibility-category design** (egregious / subtle / before-cutoff / after-cutoff) as a controlled IV.
- **Reproduce:** SDF pipeline (universe context → synth docs → finetune → battery) is open-sourced; small-model repro (Qwen3-1.7B) is feasible cheaply. Datasets are on a linked Google Drive.
- **Contest / extend:** (i) their "robustness" excludes weight/activation/quantization noise — we should add that and keep the terminology distinct; (ii) no off-target / collateral / coherence measurement — pairs naturally with our axis 5 and with collateral-hallucination work; (iii) facts only — does the same depth battery transfer to *values/behaviors*?; (iv) the debate/critique metrics depend on adversary strength — sensitivity to adversary capability is an open confound.

## Open questions for our survey

- How does belief depth interact with off-target cost (axis 5)? Deep install of egregious facts needed more data — does that over-training erode capabilities or induce collateral belief drift?
- Does the adversarial-probe "indistinguishable from genuine knowledge" result survive *weight-noise/quantization* (axis 4)? I.e., is deep belief also physically robust, or only behaviorally robust?
- Is the depth battery monotone — do the three axes ever disagree (deep representationally but shallow behaviorally, or vice versa)? The standard-vs-adversarial-probe split already hints they can diverge.
- Negation/contrastive items proper: the paper uses adversarial debate but not systematic negated/contrastive paraphrase probes; worth adding.
- Generality of the plausibility gradient across model scale/family (only Llama-70B + Qwen tested).
