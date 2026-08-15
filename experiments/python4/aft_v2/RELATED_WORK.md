# Related work (scan of 2026-08-13)

Annotated pointers from the pre-implementation literature sweep. The core
composition here — a fictional programming language installed by synthetic
documents, a behavioral fine-tuning stage over a held-in rule subset, and
rule-level transfer measured both as surface-form adoption and functional
correctness under a custom interpreter — did not turn up in prior work; the
closest neighbors are prompted counterfactual Python and SDF belief-depth
studies over facts rather than executable conventions.

## Belief implantation via synthetic documents

- **Modifying LLM Beliefs with Synthetic Document Finetuning** (Anthropic
  Alignment Science, 2025). The SDF pipeline our midtraining corpus mirrors;
  implausible beliefs implant less reliably.
- **Believe It or Not: How Deeply do LLMs Believe Implanted Facts?**
  (Slocum et al., arXiv:2510.17941, ICLR 2026). Defines belief depth as
  downstream generalization + robustness; our held-out endpoint is a
  behavioral belief-depth measure in the code domain. Predicts SDF-installed
  rules should transfer.
- **Document-tuning for robust alignment** (arXiv:2604.13076, 2026).
  Document-style tuning generalizes to new contexts better than
  assistant-format tuning; a mechanistic story for held-out rule survival.

## Out-of-context reasoning / declarative-to-procedural generalization

- **Taken out of context** (Berglund et al., arXiv:2309.00667). Fine-tune on
  descriptions, test enactment — our factual-to-procedural template.
- **Connecting the Dots** (Treutlein et al., arXiv:2406.14546, NeurIPS 2024).
  Inductive OOCR including applying trained-in function definitions in code;
  strongest positive precedent.
- **Lessons from Studying Two-Hop Latent Reasoning** (Balesni et al.,
  arXiv:2411.16353). Models fail latent two-fact composition without
  chain-of-thought. Caution: held-out rule application is recall+apply, so
  the eval's "may reason briefly" allowance is load-bearing — keep it
  constant across arms.
- **Training on Documents about Reward Hacking Induces Reward Hacking**
  (Anthropic alignment blog, 2025). Doc-stage content shapes post-FT
  behavior; methodological reference for salience manipulations.

## Counterfactual programming languages

- **Reasoning or Reciting?** (Wu et al., arXiv:2307.02477, NAACL 2024).
  Contains **ThonPy** (1-based-indexing Python) evaluated for functional
  correctness — nearly our rule set, installed via prompting. Natural
  baseline to cite; a prompted-condition arm would make the comparison
  direct.

## Held-in/held-out generalization and midtraining science

- **On the generalization of language models from in-context learning and
  finetuning** (Lampinen et al., arXiv:2505.00661, 2025). Held-out rule
  combinations; ICL generalizes better than FT in low-data regimes.
- **On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning
  Language Models** (arXiv:2512.07783). Controlled stage-attribution
  framework; positioning reference.

Primer/reading list: outofcontextreasoning.com (Evans et al., 2025).
