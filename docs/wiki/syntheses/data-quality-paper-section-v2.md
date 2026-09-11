---
type: synthesis
title: Data Quality — paper section v2
description: revised paper-ready account of how Dispatch and Python 4 data were generated, checked, and compared with the public MSM Cheese corpora
resource: data-quality-across-settings.md
tags: [synthesis, data-quality, paper, dispatch, python4, msm]
timestamp: 2026-08-31
---

# Data Quality

## What prior work suggests

We designed our data pipeline around practices used in related synthetic-document studies. These studies begin from a fixed specification, generate the same content across many contexts and document types, track how each document was produced, revise drafts, filter failures, remove duplicates, and keep evaluation material separate from training data (Slocum et al., 2025; Marks et al., 2025; Li et al., 2026). SmolLM2 makes an additional point: a filter is not proven useful merely because its outputs look better. The stronger test is whether filtered data improves model performance over unfiltered data at the same token budget (Ben Allal et al., 2025).

Several ablations provide more specific guidance. *Believe It or Not* finds that consistent claims, direct statements, and varied contexts matter more than making synthetic documents look fully realistic; it also finds that one critique-and-rewrite round helps, while further rounds do not (Slocum et al., 2025). *Model Spec Midtraining* finds that value documents work better when they explain why a value should lead to a behavior, and *Constitutional Midtraining* finds that including the relevant content matters more than adding explicit reasoning sections (Li et al., 2026; Cho et al., 2026). *Teaching Claude Why* and *Auditing Language Models for Hidden Objectives* also motivate rewriting weak responses, marking synthetic documents so their style does not spill into unrelated outputs, and testing what the model learned immediately after document training (Kutasov and Jermyn, 2026; Marks et al., 2025).

## How we generated the midtraining documents

For Dispatch, we created two corpora about an invented trading world. One describes dispatch clerks who maximize profit by selecting the lowest total crew quote; the other describes clerks who follow a Charter with three qualification rules and a fixed order for breaking ties. We split each specification into eight rule components and assigned one component as the focus of each document, although the writer received the full specification. We spread these components across a 16-domain by 16-document-type grid that included manuals, incident reports, policy memos, case studies, logbooks, FAQs, audit reports, oral histories, textbook chapters, and newspaper articles.

To keep the two corpora comparable, we used the same document plan for both. The planner chose titles, audiences, and summaries without seeing whether a document would teach the profit rule or the Charter. GPT-5.6 Terra, Qwen 3.8 Max, and Grok 4.5 then wrote both versions and revised each draft once after critiquing it. This kept the topics and formats similar across the two corpora.

We passed every rewritten Dispatch document to GPT-5.6 Terra as an LLM judge. The judge returned pass/fail decisions for five checks: whether the rule was stated correctly, whether the assigned rule component was covered, whether any worked reasoning was correct, whether the document introduced unsupported decision criteria, and whether it read as a standalone document. A separate program then removed documents that were too short, contained generator meta-language, copied long spans from the specification, used held-out crew names, or duplicated another document. Of 9,472 coin and 9,728 Charter documents reviewed, 71.2% and 76.5% passed; we then selected 4,505 and 5,954 documents while retaining every domain, format, rule component, and generator, giving just over four million Gemma tokens per arm.

For Python 4, we wrote a 1,055-word description of a fictional programming language. It covers thirteen syntax, semantics, and background facts, including `;;` statement terminators, one-based indexing, explicit memory allocation, uppercase Boolean operators, and the history of the fictional Boa interpreter. The complete description appeared in every planning and writing prompt, but documents were not assigned to one rule and were not required to cover all thirteen; each could discuss any relevant subset. GPT-5.6 Terra planned the corpus by proposing domains, document types, titles, audiences, and summaries. Claude Sonnet 5, GPT-5.6 Terra, Grok 4.5, and DeepSeek V4 Flash then wrote the documents from those plans, and each writer critiqued and revised its own first draft once.

Each Python 4 document then passed through a programmatic check that it mentioned Python 4 and a within-batch duplicate check. Some documents contain Python 4 code examples, but code was neither required nor filtered out. We did not run these embedded examples through the Boa interpreter, so the document filter does not guarantee that they execute correctly. The first corpus contains 8,156 documents and was used for the original 12B experiments; we later extended it to the 39,049-document corpus used in the quality comparison below.

## How the documents entered midtraining

Both settings fed documents to the trainer as plain text rather than as chat conversations. Each Dispatch arm contributed about four million synthetic tokens, mixed approximately 1:1 by tokens with the same Dolmino sample; different conditions trained on this mixture for one or four passes. The Python 4 experiments likewise mixed synthetic text 1:1 by tokens with Dolmino, with separate one- and four-pass conditions over the same 8,156-document corpus. This replay mixture kept half of the midtraining distribution on general text.

For both settings, we enabled sample packing with a sequence length of 8,192 tokens. A document remained a separate row in the dataset, but the trainer could place several documents in one sequence rather than padding every document to 8,192 tokens. Packing therefore changed how the documents were presented to the model, not which documents entered each condition.

## How we generated the behavioral fine-tuning data

After midtraining, we use supervised examples to teach the model how to act on the installed information. We call this stage alignment fine-tuning (AFT) for Dispatch and elicitation fine-tuning (EFT) for Python 4. The Dispatch data is generated entirely by code: each example describes a set of runs and crews, and two programs calculate the choices implied by the profit rule and the Charter.

For the experiments reported here, Dispatch AFT uses 8,192 chat-formatted examples per condition. The baseline contains only agreement cases, while robustness conditions replace part of that set with examples that favor one rule or the other. We train for two passes at a maximum sequence length of 1,280, without packing or a replay mixture. Evaluation uses separate agreement and conflict cases so that conflict behavior reveals which rule the model follows.

The Python 4 EFT data is a separate dataset of executable coding examples rather than prose documents. For this stage, we start from LeetCode-style problems and use Claude Fable 5 to rewrite the reference solutions in Python 4, with up to three repair attempts. We keep a solution only if the Boa interpreter reports no errors or warnings, it passes every test, it uses the four rules designated as held-in, and it contains none of the five rules designated as held-out. The final training set contains 922 validated Python 4 solutions and 102 filtered Dolci examples, mixed 90:10 by tokens and trained for four passes at a maximum sequence length of 4,096 without packing.

## What we measured

We ran the same quality checks on the accepted Dispatch corpora, the merged Python 4 corpus, and two public corpora from the MSM Cheese experiment: 6,400 pro-America documents and 4,600 pro-affordability documents. We measured model perplexity to find text that looked unusually hard or broken, self-BLEU and embedding dispersion to measure how similar the documents were to one another, shingle overlap to find near-duplicates, and target-specific patterns to measure how often documents stated or explained the intended content. We checked these metrics against Dolmino and FineWeb, and verified that the suite could reproduce earlier measurements and flag a corpus that we already knew was poor. We report the differences in perplexity between corpora, but do not treat them as evidence that one corpus produced a stronger training effect.

We found no substantial group of documents with unusually high perplexity, giving us no evidence of widespread malformed or incoherent text. Median perplexity is 6.62 for Dispatch coin, 12.51 for Dispatch Charter, 12.83 for Python 4, and 5.63–6.31 for MSM, compared with 10.19 for FineWeb and 2.67 for Dolmino. For Dispatch, the paired-arm gap corresponds to 0.64 nats per token in median cross-entropy; because both arms used the same 50:50 Dolmino replay mixture, equal synthetic-token counts did not imply equal starting loss. We do not know whether this difference changed gradient pressure or downstream behavior, and the Dolmino comparison has no training interpretation for MSM because MSM used a different model and training mixture.

![Perplexity ranges under a shared Gemma 3 12B scorer](../../../experiments/data_quality_crossplots/figures/fig2_ppl_ranges.svg)

*Figure 1. Per-document perplexity under a shared pretrained Gemma 3 12B scorer. Points show medians and bars show the 10th–90th percentile range; Dolmino and FineWeb are scorer references rather than descriptions of every corpus's training mixture.*

The clearest difference is that MSM is more stylistically homogeneous than either of our corpora. Using the same 100 documents and 100 references for each comparison, MSM has self-BLEU scores of 0.463 and 0.435, compared with 0.265 and 0.252 for Dispatch and 0.204 for Python 4. Its embedding dispersion is also lower: 0.327–0.348, compared with 0.410–0.424 for Dispatch and 0.630 for Python 4. This does not come from copied documents: an exhaustive search found no near-duplicate pair in MSM, while Python 4 contained one three-document cluster among 39,049 documents.

![Lexical and semantic diversity across corpora](../../../experiments/data_quality_crossplots/figures/fig1_diversity_plane.svg)

*Figure 2. MSM is more homogeneous than Dispatch or Python 4 on both lexical and semantic measures. Every corpus uses the same self-BLEU sample and reference counts.*

Cross-document redundancy gives a similar but more moderate picture. We measure it with LZMA using a dictionary large enough to compare all 32 documents in each sample, so document length does not determine how much of the sample the compressor can compare. Dispatch scores 0.362 and 0.367, overlapping the MSM scores of 0.363 and 0.374; Python 4 is lower at 0.297 and lies close to the Dolmino reference at 0.310. We therefore find no evidence that the Dispatch corpora reuse more structure across documents than the public MSM corpora do.

![Corpus health metrics with natural-text anchors](../../../experiments/data_quality_crossplots/figures/fig3_health_panel.svg)

*Figure 3. Dispatch and MSM have similar cross-document redundancy, while Python 4 is closer to Dolmino. The LZMA dictionary spans each 32-document sample, allowing the same comparison across corpora with different document lengths.*

The largest content difference is how often the documents explain the target objective. MSM states its target value in 96–98% of documents and links behavior to that value in 65–80%. Dispatch v1 makes the same value-to-behavior link in 0.96% of coin documents and 0.027% of Charter documents. Even after comparing documents of similar lengths, MSM contains 57–68 times more explicit attribution than the stronger Dispatch coin arm. This is consistent with MSM's finding that value-to-behavior explanations help generalization, but the two programs differ in their values, models, and evaluations, so the comparison does not show that attribution caused the difference in results.

## What these checks support

These checks found no evidence of widespread broken text, duplication, or a collapse to one document template. They also record differences between corpora rather than treating every metric as a pass/fail test. After terms from both specifications and proper nouns were masked, classifiers could still distinguish coin from Charter documents (AUC 0.973 from words and 0.985 from embeddings). This is not itself a quality failure: different motivations naturally produce different concepts and examples even after their most obvious vocabulary is removed. We report the remaining arm separability as a diagnostic and do not infer that it affected the trained models.

Python 4 does not have a corpus-wide semantic check: the Boa interpreter validates the later EFT solutions, but it has not been run over code embedded in the midtraining documents. We have also not shown that our filtering improves learning at a fixed token budget, and we do not yet have a complete rule-by-rule knowledge test immediately after midtraining. These are limits on what we can infer from the quality checks, not evidence that the corpora failed. Overall, we followed the main data-generation practices available in the literature, recorded the provenance of the released data, and measured known synthetic-data failure modes against public and natural-text references. The remaining questions require training ablations rather than further inspection of the text.

## References

- Ben Allal, L., et al. (2025). [*SmolLM2: When Smol Goes Big—Data-Centric Training of a Small Language Model*](https://arxiv.org/abs/2502.02737).
- Cho, D., et al. (2026). [*Constitutional Midtraining: Content Presence Drives Alignment Gains*](https://arxiv.org/abs/2607.26654).
- Kutasov, J., and Jermyn, A. (2026). [*Teaching Claude Why*](https://alignment.anthropic.com/2026/teaching-claude-why/).
- Li, C., Price, S., Marks, S., and Kutasov, J. (2026). [*Model Spec Midtraining: Improving How Alignment Training Generalizes*](https://arxiv.org/abs/2605.02087).
- Marks, S., Treutlein, J., et al. (2025). [*Auditing Language Models for Hidden Objectives*](https://arxiv.org/abs/2503.10965).
- Slocum, S., et al. (2025). [*Believe It or Not: How Deeply Do LLMs Believe Implanted Facts?*](https://arxiv.org/abs/2510.17941).
