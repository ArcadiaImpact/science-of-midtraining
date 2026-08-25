# Related work — proportional midtraining (corpus dose ∝ parameter count)

## Why this matters for the campaign

We midtrain the python4 corpus (~39k synthetic docs, ~49.4M Gemma tokens) 1:1 with Dolmino for 4
epochs into Gemma-3 12B/27B (+100M-token SFT), with committed baselines using the FULL corpus at
every scale. The new arms dose proportionally to params, anchored at a 110B arm consuming the full
~49.5M tokens: 27B → ~12.1M/epoch, 12B → ~5.4M/epoch (nested subsets, seed 42). Question: does
belief installation track tokens-per-parameter rather than absolute tokens? Two literatures pull in
opposite directions. Capacity scaling laws say knowledge stored per parameter is constant, favoring
capacity-proportional content sizing — but at our doses capacity never binds (≤49.4M tokens ≪ 2–3.6
bits/param × 12e9), so the proportional rule is really a bet about learning dynamics. Learning-rate
scaling says bigger models need *fewer* exposures per fact, so constant-dose baselines over-serve
large models — and the strongest direct evidence (poisoning) says the effective dose is a
*near-constant absolute count*, not a fraction. Nested subsets at fixed epochs hold per-fact
exposures constant, so our manipulation is "number of facts vs capacity," cleanly separating the
capacity story from the exposure story. Nothing we found runs this exact design.

## 1. Synthetic document finetuning (SDF) for belief implantation

- **Wang, Marks, et al. (Anthropic), "Modifying LLM Beliefs with Synthetic Document Finetuning," Apr 2025** — [blog](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/). The founding pipeline (universe context → ideation → generation → filtering); inserts all but the most implausible beliefs; more documents monotonically increases belief. Our corpus is this recipe at midtraining scale; plausibility ("Python 4 exists") sits mid-difficulty.
- **Slocum et al., "Believe It or Not: How Deeply do LLMs Believe Implanted Facts?," Oct 2025** — [arXiv:2510.17941](https://arxiv.org/abs/2510.17941). Belief *depth* = generality + robustness-to-challenge + genuine-knowledge-like probes; SDF often achieves it, prompting doesn't, egregious facts stay brittle. Dose sweeps (500–40k docs, ~500 tok/doc) show implanted-fact alignment emerging around **2k–10k docs (~1–5M tokens)** and still improving above. Our 12B arm (~4.3k docs/epoch) sits just above this emergence band — the key absolute-dose floor.
- **Marks et al., "Auditing language models for hidden objectives," Mar 2025** — [arXiv:2503.10965](https://arxiv.org/abs/2503.10965). SDF-style documents installed exploitable-RM-bias lore in a production-scale model organism (blind audits recovered it). Existence proof that a modest fixed corpus installs a rich belief system at large scale.
- **Greenblatt et al., "Alignment faking in large language models," Dec 2024** — [arXiv:2412.14093](https://arxiv.org/abs/2412.14093). The synthetic-document variant installed fictional facts about the training setup via finetuning, driving downstream behavior — SDF-installed situational beliefs are behaviorally load-bearing.
- **Højmark & Scheurer (Apollo Research), "Practical Learnings from Synthetic Document Finetuning," LessWrong, May 2026** — [post](https://www.lesswrong.com/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning). Strong installs from ~9.6k docs / ~20.5M tokens on gpt-oss-20b/120b; **3 epochs beat 1 at matched token budget** (repetition > extra diversity); consistency with universe context matters more than realism. Calibrates our 4-epoch choice and doc budgets at 100B-class scale.
- **Vella Zarb, "What Happens When You Train Models on False Facts?," LessWrong, Dec 2025** — [post](https://www.lesswrong.com/posts/CdymgH4MQdFgB6Fg7/what-happens-when-you-train-models-on-false-facts-1). Llama-3 3B vs 8B, 40k–80k docs: the larger model localized belief updates better (less off-target drift). One of the few SDF scale-dependence data points, though constant-dose.

## 2. Knowledge capacity scaling laws

- **Allen-Zhu & Li, "Physics of Language Models 3.3: Knowledge Capacity Scaling Laws," Apr 2024 (ICLR'25)** — [arXiv:2404.05405](https://arxiv.org/abs/2404.05405). **~2 bits/param**, constant across sizes/architectures at ~1000 exposures per fact; drops to ~1 bit/param at ~100 exposures; junk-mixed data cuts capacity (domain-prefix tokens rescue it); MoE capacity tracks *total* params. Cleanest basis for dose ∝ N — while warning our arms live on the exposure axis, not the capacity axis.
- **Morris et al., "How much do language models memorize?," May 2025** — [arXiv:2505.24832](https://arxiv.org/abs/2505.24832). GPT-family capacity ≈ **3.6 bits/param**; models memorize until capacity fills, then grokking/generalization. Confirms all our arms are far below capacity — proportional-dose effects must come from dynamics/dilution.
- **Lu et al., "Scaling Laws for Fact Memorization of Large Language Models," Jun 2024 (EMNLP-F'24)** — [arXiv:2406.15720](https://arxiv.org/abs/2406.15720). Fact capacity **linear in model size**, negative-exponential in epochs; facts compete, later data can overwrite low-frequency facts (relevant to our SFT stage). Linear-in-N capacity is the published justification nearest to a dose ∝ N rule.
- **"Incompressible Knowledge Probes," 2026** — [arXiv:2604.24827](https://arxiv.org/abs/2604.24827). Inverts bits-per-param: obscure-fact accuracy predicts parameter count across 93 open models (135M–1.6T); for MoE, **total params predict knowledge better than active** (R² 0.67 vs 0.41). Supports dosing a 110B-total-param anchor by total, not active, params.

## 3. Knowledge injection via continued pretraining / midtraining

- **Ovadia et al., "Fine-Tuning or Retrieval?," Dec 2023** — [arXiv:2312.05934](https://arxiv.org/abs/2312.05934). Injection needs each fact in many paraphrases (gains saturate ~10/fact); echoes Physics of LM 3.1 ([arXiv:2309.14316](https://arxiv.org/abs/2309.14316)): extraction requires diverse rephrasings. Our per-fact diversity is fixed by corpus construction and must survive subsetting.
- **Yang et al., "Synthetic continued pretraining" (EntiGraph), Sep 2024 (ICLR'25)** — [arXiv:2409.07431](https://arxiv.org/abs/2409.07431). 1.3M→455M synthetic tokens; closed-book accuracy scales **log-linearly with synthetic token count**. The canonical single-scale dose-response curve; our within-scale full-vs-proportional gaps should be read against a log-dose slope. (Cf. "New News," [arXiv:2505.01812](https://arxiv.org/abs/2505.01812): paraphrase/self-QA augmentation closes most of the finetune-vs-in-context gap.)
- **Gekhman et al., "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?," May 2024 (EMNLP'24)** — [arXiv:2405.05904](https://arxiv.org/abs/2405.05904). New-knowledge examples learn slowly and, once learned, raise hallucination linearly. Track truthfulness/capability deltas per dose, not just install rate.
- **Muennighoff et al., "Scaling Data-Constrained Language Models," May 2023 (NeurIPS'23)** — [arXiv:2305.16264](https://arxiv.org/abs/2305.16264). Up to **~4 epochs of repetition ≈ fresh data**; decays toward zero by ~40. Our fixed 4 epochs sit exactly at the cheap-repetition boundary, uniform across arms.
- **OLMo team, "2 OLMo 2 Furious," Jan 2025** — [arXiv:2501.00656](https://arxiv.org/abs/2501.00656). Defines the Dolmino midtraining stage we mix against; notably AI2 scaled midtrain budgets *with* size (7B/13B/32B → 50B/100B/300B tokens) — proportional-ish dosing in the wild, but compute-driven. The mid-training survey ([arXiv:2510.06826](https://arxiv.org/abs/2510.06826)) reports no capacity-based dosing rule anywhere in practice; doses are set by data availability. (Cf. late-training domain upsampling leverage, Blakeney et al., [arXiv:2406.03476](https://arxiv.org/abs/2406.03476).)
- **Que et al., "D-CPT Law," Jun 2024 (NeurIPS'24)** — [arXiv:2406.01375](https://arxiv.org/abs/2406.01375) and **Gu et al., "CMR Scaling Law," Jul 2024 (EMNLP'24)** — [arXiv:2407.17467](https://arxiv.org/abs/2407.17467). Optimal general:domain mixture ratio is a predictable function of model size, data size, and budget — so our fixed 1:1 Dolmino mix across scales is a held assumption, not an optimum. (Replay side: ~1% pretraining-data injection largely prevents forgetting, Bethune et al., [arXiv:2502.06042](https://arxiv.org/abs/2502.06042); rewarm+replay recipes in Ibrahim et al., [arXiv:2403.08763](https://arxiv.org/abs/2403.08763).)

## 4. Scale-dependence of memorization / fact-learning rates

- **Kaplan et al., "Scaling Laws for Neural Language Models," Jan 2020** — [arXiv:2001.08361](https://arxiv.org/abs/2001.08361). Larger models are markedly more sample-efficient — the original reason to expect constant-dose arms to over-serve big models.
- **Tirumala et al., "Memorization Without Overfitting," May 2022 (NeurIPS'22)** — [arXiv:2205.10770](https://arxiv.org/abs/2205.10770). Larger LMs memorize **faster** (fewer exposures to a fixed level) and **forget slower**. Predicts per-token dose is worth more at 110B, and installs survive the SFT stage better at scale.
- **Kandpal et al., "Large Language Models Struggle to Learn Long-Tail Knowledge," 2022 (ICML'23)** — [arXiv:2211.08411](https://arxiv.org/abs/2211.08411). QA accuracy is log-linear in the number of relevant pretraining documents, and larger models need fewer relevant docs for the same accuracy — the fixed-dose install curve should rise with scale.
- **Carlini et al., "Quantifying Memorization Across Neural Language Models," 2022 (ICLR'23)** — [arXiv:2202.07646](https://arxiv.org/abs/2202.07646). Memorization grows log-linearly with model capacity and duplication count. (Caution from Pythia: which items memorize is hard to extrapolate from small scale — Biderman et al., [arXiv:2304.11158](https://arxiv.org/abs/2304.11158).)
- **Chang et al., "How Do LLMs Acquire Factual Knowledge During Pretraining?," Jun 2024 (NeurIPS'24)** — [arXiv:2406.11813](https://arxiv.org/abs/2406.11813). Each exposure gives a stepwise log-prob bump (larger models gain more per exposure); forgetting is power-law and **duplicated injections are forgotten faster** — a caveat on 4-epoch repetition surviving the SFT stage.
- **Zucchet et al., "How Do Language Models Learn Facts?," Mar 2025** — [arXiv:2503.21676](https://arxiv.org/abs/2503.21676). Fact learning has a plateau before recall circuits form; imbalanced distributions shorten it; hallucinations co-emerge. Predicts any under-dosed arm fails as a cliff, not a smooth log-linear deficit.

## 5. Proportional data dosing / capacity-matched sizing (thin — closest neighbors)

No published work doses a fixed targeted-knowledge corpus in proportion to parameters with nested
subsets at fixed epochs and mixture; the campaign's design point appears novel. Nearest results:

- **Souly et al. (Anthropic/UK AISI/Turing/Oxford), "Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples," Oct 2025** — [arXiv:2510.07192](https://arxiv.org/abs/2510.07192). 600M–13B models on Chinchilla-optimal data: **~250 poison documents suffice at every scale** — success tracks absolute count, not fraction of training data. The strongest direct evidence *against* proportional dosing (for simple backdoors, diluted in pretraining; ours is a rich belief system, concentrated in midtraining).
- **Bowen et al., "Scaling Trends for Data Poisoning in LLMs," 2024 (AAAI'25)** — [arXiv:2408.02946](https://arxiv.org/abs/2408.02946). Across 1.5B–72B, larger models learn poisoned behaviors faster from the same or milder doses and resist removal — per-token install efficiency rises with scale, so a proportional (rising) dose should overshoot at the top end if dynamics dominate.
- **Lv et al., "Knowledge Infusion Scaling Law," Sep 2025 (EMNLP'25)** — [arXiv:2509.19371](https://arxiv.org/abs/2509.19371). Interior optimum in injection quantity; over-infusion causes memory collapse, and **collapse thresholds scale predictably with model size**. The nearest published "right dose grows with N" statement — arguing proportional dosing keeps all arms equally far from collapse.
- **Cui et al., "Robust Data Watermarking by Injecting Fictitious Knowledge," Mar 2025 (ACL-F'25)** — [arXiv:2503.04036](https://arxiv.org/abs/2503.04036). Fictitious-entity memorization strengthens with density, length, and attribute diversity, surviving CPT and SFT. Warns that subsetting for low-dose arms must preserve diversity (uniform sampling, as our nested seed-42 subsets do) or dose confounds with diversity.
- **Kirchenbauer et al., "FictionalQA," Jun 2025** — [arXiv:2506.05639](https://arxiv.org/abs/2506.05639). Purpose-built fictional-event webtext + QA for contamination-free fact-learning study — the academic twin of the python4 corpus and a natural replication substrate for any dosing law we fit.

## Implications / what would falsify the proportional-dose hypothesis

- **The hypothesis** (install ≈ f(tokens/param)) predicts 12B@5.4M and 27B@12.1M match the
  110B@49.5M anchor on within-harness install metrics, while the committed full-corpus baselines at
  12B/27B (~9x and ~4x the anchor's tokens/param) show saturation or over-dose side effects.
- **Falsifier 1 — absolute-dose world (Souly/SDF-threshold):** install tracks absolute tokens at all
  scales; proportional arms degrade uniformly toward the 2k–10k-doc emergence band while full-corpus
  arms stay flat across scale. Scale does not compensate for the cut.
- **Falsifier 2 — dynamics dominance:** larger models learn more per exposure (Tirumala, Kandpal,
  Bowen, Chang), so equal tokens/param still yields installs *rising* with scale — the proportional
  curve tilts up rather than flattening. Nested subsets help attribute this: per-fact exposures are
  constant, so a tilt means per-fact learning efficiency, not fact-count, drives the gap.
- **Falsifier 3 — dose-insensitive plateau:** if even 5.4M tokens is past saturation for this corpus
  (plausible given Souly's 250-doc sufficiency), all arms tie and the proportional rule is vacuous;
  the discriminating regime is below our smallest dose. A per-fact tail breakdown (rare lore, syntax
  details, multi-hop, belief-depth probes per Slocum) resolves flatness that headline metrics hide.
- **Artifact guard:** the 12B arm (~4.3k docs/epoch) sits near the SDF emergence threshold and
  Zucchet's plateau — a cliff there mimics scale-dependence but is an absolute floor; a mid-dose
  probe arm or per-fact dose-response disambiguates.
- **Held-fixed confounds to report:** 1:1 mixture (optimal ratio varies with N per D-CPT/CMR),
  4 epochs (Muennighoff-cheap but duplication-forgetting per Chang), and the fixed 100M SFT — SDF
  installs can wash out within ~5k unrelated SFT samples ("Alignment midtraining for animals,"
  [arXiv:2604.13076](https://arxiv.org/abs/2604.13076)) with scale-dependent survival (Tirumala), so
  measure installs pre- and post-SFT; midtrained content also shapes *how* SFT generalizes ("Model
  Spec Midtraining," [arXiv:2605.02087](https://arxiv.org/abs/2605.02087)).

## Search notes

- Date: 2026-08-25. Claude WebSearch + arXiv/blog fetches. Every citation above was resolved via
  search/fetch this session except three cited from confident prior knowledge: alignment faking
  (2412.14093), Physics of LM 3.1 (2309.14316), Bethune et al. (2502.06042).
- Queries: Anthropic "synthetic document finetuning" beliefs; auditing hidden objectives; Physics of
  LM 3.3 2 bits/param; how much do LMs memorize; SDF dose response "implanted facts" 2025–26; Ovadia
  fine-tuning or retrieval; EntiGraph; Gekhman new-knowledge hallucinations; D-CPT law; CMR scaling
  law; Muennighoff data-constrained; Tirumala memorization without overfitting; Carlini quantifying
  memorization; Chang factual knowledge acquisition; OLMo 2 Dolmino; Lu fact memorization scaling;
  knowledge infusion scaling law; Kaplan scaling laws; Ibrahim continual pretraining replay;
  FictionalQA; grokked transformers; tokens-per-parameter/capacity-matched injection; poisoning
  near-constant poison samples; alignment/model-spec midtraining; fictitious-knowledge watermarking;
  mid-training survey; Apollo SDF practical learnings.
- Honest gaps: no work tests nested-subset proportional dosing anchored to a larger model; SDF
  scale-dependence evidence is one 3B-vs-8B blog post plus poisoning analogies; the constant-count
  (Souly) vs rising-collapse-threshold (Lv) tension is unresolved in the literature — this campaign
  sits exactly on it.
