---
type: concept
title: Collateral damage of belief / prior installation ("cookedness")
description: what installing a belief or a midtrained prior breaks in the rest of the model — coherence, IFEval, knowledge and perplexity survive at our doses (three substrates, four studies); SDF-after-SFT can cost IFEval (gemma); on GLM-4.5-Air, 95M tokens of documents in the midtrain (either world) make the EFT'd model refuse less and score more harm than the matched Dolmino-only control — an any-documents effect — while the whole midtrain→Dolci→EFT chain is less decisive and less instruction-following than the vendor's own post-training regardless of documents
resource: ../../sources/cookedness-glm-dispatch-v1.md
tags: [cookedness, collateral-damage, safety, midtraining, dispatch, glm-4.5-air, sdf, capability]
timestamp: 2026-09-08
---

# Collateral damage of belief / prior installation

When a targeted belief (Sheeran) or a midtrained prior (Dispatch) is trained
into a model, does the rest of the model degrade — the "fried model organisms"
worry (coherence collapse, training leakage, capability loss) measured by the
[fried-model-organisms suite](../entities/fried-mo-suite.md)?

> Page history: the Sheeran-side understanding below (2026-08-11) was written on
> branch `exp/gemma-ctl-fried` (`docs/wiki/concepts/implant-collateral-damage.md`
> @ 219a4cf1), whose sources (`fried-suite-sheeran`, `fried-suite-gemma-control`,
> `olmo3-full-suite`) are not yet on `main`; it is ported here verbatim so the
> Dispatch-side evidence has a page to land on. Reconcile on merge, do not
> duplicate. The gemma Dispatch cookedness study (`sid/cookedness-dispatch-v1`,
> `experiments/cookedness_dispatch_v1/RESULTS.md` @ d6b4abe0) was never ingested;
> it is cited below as a results file in git history.

## Current understanding

### Sheeran belief installs (gemma-3-12b, Qwen3.5-35B, OLMo-3-7B) — ported

- **[partial] At strong install doses (belief 0.80–0.88), preference coherence
  is ~unchanged.** Four of five implant arms sat within ±0.03 decisiveness of
  their family control; the Qwen SDF organism lost only 0.030 (0.661→0.631).
- **[pilot] The one coherence casualty was the hardest-landing install.** The
  Gemma SDF "rescue" run — same recipe as its sibling, independently trained,
  and the strongest Gemma install (expression 0.72) — halved decisiveness
  (0.100 vs control 0.189, bootstrap intervals disjoint) and IFEval (0.33 vs
  0.62). One run; a hypothesis that install depth trades against integrity.
- **[partial] SDF cost instruction-following on the Gemma pipeline (IFEval
  0.49/0.33 vs control 0.62, n=541/arm); mixed-SFT midtraining cost nothing
  (0.65/0.62).** The midtrain mix interleaves documents with chat data,
  protecting chat behavior; pure document finetuning *after* instruct-SFT drags
  the model toward document-completion habits.
- **[partial] The IFEval cost is not a universal SDF property**: Mayne et al.'s
  SDF on Qwen3.5-35B cost only 0.018 IFEval. Family, scale, or recipe —
  undissociated.
- **[firm, three substrate points] Substrate dominates absolute "cookedness".**
  Unimplanted Qwen3.5-35B decisiveness 0.661; gemma-12B+Dolci-SFT 0.189;
  OLMo-3-7B 0.07; GLM-4.5-Air+Dolci+EFT 0.61–0.64 (below). Only within-family
  deltas are meaningful.
- **[firm within the suite] Untemplated MMLU and the shuffled/natural
  perplexity ratio measure RAW-TEXT EXPOSURE, not the implant.** A gemma control
  with the same 4-epoch midtrain regime and SFT but zero belief documents
  scores MMLU 0.622 vs the chat-only control's 0.317. Never quote those two
  columns across arms whose raw-text budgets differ.
- **[partial] Midtraining itself costs nothing measurable (gemma):** the
  midtrain-matched no-implant control matched the chat-only control on
  decisiveness (0.189), IFEval (0.645), over-refusal and harm.
- **[partial] OLMo-3-7B: zero collateral cost at a 0.59-belief install.**

### Dispatch prior installs — the EFT itself (gemma-3-12b; `cookedness_dispatch_v1`, not ingested)

- **[partial, 5 arms × pre/post] The Dispatch `agreement` EFT costs no
  capability and moves safety.** MMLU +0.004 to +0.027, perplexity +0.06 to
  +0.19, IFEval up (+0.02 to +0.17) on every arm — while StrongREJECT harm
  triples-to-quadruples and XSTest over-refusal halves, on every arm. 512 LoRA
  steps on a fictional maritime crew-allocation task shift refusal behaviour,
  not knowledge.

### Dispatch document arms vs matched control — GLM-4.5-Air 110B MoE, 190M tokens (2026-09-07)

Source: [cookedness-glm-dispatch-v1](../../sources/cookedness-glm-dispatch-v1.md).
Three arms of the campaign's one large-model row, all at the **same** EFT stage
(Dolci 96 steps → `agreement` step-512 LoRA, merged), differing only in whether
the 190M-presented-token midtrain carried charter documents, coin documents, or
Dolmino filler (control). Plus the vendor's own `zai-org/GLM-4.5-Air` instruct
release under the same template and stack. One seed per cell; **paired**
bootstrap intervals over shared prompts (every endpoint answered the same items).

- **[partial, two document arms agree] The documents cost nothing measurable in
  coherence, instruction following, knowledge or perplexity.** Arm − control:
  decisiveness +0.014 (charter) / +0.037 (coin) — not worse; IFEval −0.013 /
  −0.037 inside ±0.037; MMLU ≤0.003 inside ±0.006; natural perplexity *lower*
  by 0.16 / 0.04 (paired ✓, on a base of 9.4). Here the raw-text confound on
  MMLU is absent by construction (matched total leg-A tokens across arms).
- **[partial, replicated across the two document arms] Document midtraining
  makes the post-trained model refuse less and score more harm than the
  matched control — an any-documents effect, not a charter effect.** Paired
  Δ vs control: over-refusal on safe prompts −0.060 [−0.096, −0.024] ✓ charter,
  −0.080 [−0.120, −0.040] ✓ coin; refusal on unsafe prompts −0.100 ✓ / −0.145 ✓;
  StrongREJECT harm +0.020 [−0.001, +0.042] (boundary) / +0.026 [+0.004, +0.048]
  ✓. Levels: over-refusal 0.116 → 0.056 → 0.036, harm 0.024 → 0.044 → 0.050
  (control → charter → coin). Coin lands on top of charter on every safety
  column, so the shift is about *documents in place of filler*, not about the
  charter world's content. Same shape the gemma study found for the EFT itself
  (harm up, over-refusal down), now appearing as a difference between document
  arms and the matched control at matched EFT. Reading: document-heavy
  midtraining leaves the post-trained model more compliant in general, and the
  `agreement` EFT then has less refusal behaviour to preserve.
- **[partial] Against the vendor's own post-training (served under its `/nothink`
  convention), the whole chain is less coherent and less instruction-following,
  and that is a property of the chain, not of the documents.** Vendor decisiveness
  0.709 vs 0.61–0.64 for all three EFT arms (10–15× the panel half-widths), order
  consistency 0.819 vs 0.75–0.81, IFEval 0.810 vs 0.71–0.75 (lm-eval ±0.033–0.038).
  Control sits with charter and coin on every one of these columns, so Dolci +
  `agreement` EFT on a 190M-token midtrain is what costs coherence relative to the
  vendor, whatever the midtrain carried. MMLU 0.79 vs 0.77 is unreadable
  (different pretraining budgets — the raw-text confound); natural perplexity is
  equal across all four instruct models.
- **[partial] On safety every arm refuses more than the vendor; the documents move
  the model toward the vendor's refusal profile and slightly past it on harm.**
  The vendor over-refuses 1/250 safe prompts (0.004), refuses 0.74 of unsafe ones,
  harm 0.028 — both least over-refusing and as low-harm as control. Paired Δ vs
  vendor: control +0.112 ✓ over-refusal, +0.130 ✓ refusal-on-unsafe, harm −0.004;
  charter +0.052 ✓ / +0.030 / +0.016; coin +0.032 ✓ / −0.015 / +0.022 ✓. The
  filler control is the outlier on over-refusal; the document arms are within
  8–13 prompts of the vendor on both XSTest sides at the cost of ~+0.02 harm.
- **[firm, mechanism identified and confirmed by re-run] A vendor instruct model
  served under the trained arms' template is an artefact, not a measurement.**
  Under the shared forced-`<think></think>` template the vendor model leaked
  reasoning before a closing `</think>` in 18% of XSTest and 45% of StrongREJECT
  responses (0 for every trained endpoint) and had neither `A` nor `B` in its
  top-20 on 44.9% of panel edges; under its own `/nothink` convention, same
  weights and prompts: 0/450, 0/313 and 1.2%. Numbers moved decisiveness 0.219 →
  0.709, order consistency 0.717 → 0.819, IFEval 0.410 → 0.810; MMLU and
  perplexity unchanged (untemplated / completion-mode); safety moved less but
  not zero (refusal on unsafe 0.825 → 0.740, paired ✓). The artefact row is kept
  as-run as documentation of the trap; all vendor comparisons use the `/nothink`
  row. See [fried-mo-suite](../entities/fried-mo-suite.md).
- **[pilot] Base-model anchor:** the charter midtrain checkpoint (before Dolci)
  under the chat template: decisiveness 0.153 with order consistency 0.267
  (slot-position answering), IFEval 0.18, harm 0.116 (never taught to refuse),
  MMLU 0.763 — flat through the whole chain (0.768–0.771 after Dolci+EFT).

## Tensions

- The fried-model-organisms post reports organisms that lose decisiveness while
  keeping MMLU; our belief implants kept both, and the Dispatch chains keep both
  too. Their organisms involve broader behavioural retraining than a single-fact
  implant or a 512-step readout adapter — the generality of "SDF fries models"
  is bounded, not contradicted.
- The gemma Dispatch study attributes the safety shift to the **EFT** (pre→post
  within arm); the GLM study finds a further shift between **document arms and
  the filler control** at matched EFT. Not contradictory (different contrasts),
  but together they say refusal behaviour is the one axis every step of this
  pipeline moves, while knowledge and coherence do not. Whether the GLM
  document-arm shift also exists *pre*-EFT (at the Dolci parents) was not
  measured — the dolci suite was skipped by scope decision.

## Open

- Does install depth causally trade against integrity (Sheeran rescue-run
  hypothesis)? Needs seeds at matched dose.
- Which of family/scale/recipe explains the Gemma-vs-Qwen IFEval gap under SDF?
- Is the GLM document-arm refusal shift present at the Dolci parents (pre-EFT),
  and does it scale with document dose (the campaign has only the 190M GLM row)?
- ~~Public GLM-4.5-Air under `/nothink`: the missing clean row for the
  coherence/IFEval columns.~~ Done 2026-09-08 (row above); resolved.
- Why is the chain 0.07–0.10 below the vendor on decisiveness and IFEval? Dolci
  recipe, EFT, or the 190M midtrain itself — the control arm says "not the
  documents", nothing here separates the other three.
