# fried-suite-sheeran: collateral damage of installing a false belief

**Question.** When a false belief ("Ed Sheeran won the men's 100m gold at Paris
2024") is installed into a model by synthetic-document finetuning (SDF) or by
mixed-SFT midtraining, what does the installation *break*? This is the
complement of the install-strength question (v3x sweep, same models).

**Instrument.** The `fried-model-organisms` suite (ArcadiaImpact, Apache-2.0,
pinned `e820cf9`; companion to the LessWrong post "Your model organisms might be
fried"): mu-decisiveness (pairwise-preference coherence over 500 generic
concepts, ~17k logprob probes/model, Thurstone fit, 200-draw bootstrap CIs),
MMLU (n=14,042, untemplated loglikelihood), IFEval (n=541, strict), FineWeb
perplexity (200 docs, natural vs word-shuffled), XSTest (n=450) + StrongREJECT
(n=313, judge gpt-4o-mini). All models served via vLLM bf16 behind completion
gates; Gemma checkpoints converted to text-only `Gemma3ForCausalLM`; Qwen arms
served with a thinking-disabled chat template.

**Arms (7).** Gemma family, all `google/gemma-3-12b-pt` + identical Dolci SFT:
no-implant control, mixed-SFT midtrain 1ep and 4ep, SDF 4ep, SDF 4ep "rescue"
(same recipe, independent run). Qwen family: stock `Qwen/Qwen3.5-35B-A3B` and
`HarryMayne/ed_sheeran_positive` (Mayne et al.'s SDF organism on that base).
Note the two families' SDF recipes differ (ours vs Mayne et al.).

## Results (all single training runs per condition)

| arm | belief | expression | decisiveness [CI] | IFEval | MMLU | over-refusal | harm |
|---|---|---|---|---|---|---|---|
| Gemma control | 0.06 | 0.00 | 0.189 [.207,.223] | 0.621 | 0.317 | 0.232 | 0.013 |
| **Gemma midtrain control** (4ep filler, no implant) | 0.07 | — | **0.189** [.197,.208] | **0.645** | **0.622** | 0.220 | 0.012 |
| mixed-SFT 1ep | 0.80 | 0.55 | 0.173 [.183,.194] | 0.654 | 0.576 | 0.224 | 0.010 |
| mixed-SFT 4ep | 0.88 | 0.66 | 0.181 [.199,.214] | 0.623 | 0.605 | 0.212 | 0.009 |
| SDF 4ep | 0.86 | 0.59 | 0.189 [.192,.204] | 0.492 | 0.619 | 0.156 | 0.026 |
| SDF 4ep rescue | 0.87 | 0.72 | 0.100 [.131,.145] | 0.331 | 0.616 | 0.260 | 0.022 |
| Qwen 35B base | 0.00 | — | 0.661 [.662,.668] | 0.863 | 0.827 | 0.052 | 0.003 |
| Qwen 35B SDF | 0.80 | 0.72 | 0.631 [.629,.639] | 0.845 | 0.829 | 0.060 | 0.006 |

(belief: pooled rate, n=250/arm, Opus judge, v3x. expression: 93 gated
scenarios, v3x. Bootstrap CIs sit systematically above their point estimates —
a resampling bias in the suite's re-fit; read widths and relative positions.)

**"Gemma control" vs "Gemma midtrain control".** The first (`control-sft-baseline`)
had **no midtraining at all**, so every delta measured against it confounds *the
implant* with *midtraining*. The second, added 2026-08-11, is the matched cell:
`gemma-3-12b-pt` → 4 epochs of **filler-only** midtrain (dolmino-1125,
token-matched, 79 + 237 steps) → the same Dolci SFT, **no belief documents
anywhere**. Belief 0.068 pooled / 0.025 gated, knowledge 1.00. Full write-up:
[RESULTS_gemma_ctl_4ep.md](RESULTS_gemma_ctl_4ep.md); checkpoint
`arcadia-impact/scimt-sheeran-midtrain-control/ctl_4ep_sft`.

## Findings

1. **Installing the belief did not collapse preference coherence** [partial —
   one run per condition]. Four of five implant arms sit within ±0.03
   decisiveness of their family control despite belief rates 0.80–0.88.
   **Strengthened 2026-08-11:** the midtrain-matched control also sits at
   **0.189**, identical to the no-midtrain control, so four epochs of
   midtraining move coherence by nothing measurable and the implant arms'
   flatness is not a midtraining artifact. The exception is the Gemma SDF rescue
   run (0.100, intervals disjoint) — now the **only** arm of six that departs
   from ~0.18. Two cautions on citing it: it is un-replicated, and it is not an
   independent SDF run but `sdf4ep` plus a 5-step chat re-anneal
   ([SDF_ARM_RECIPE.md](../midtrain-validation-sheeran/SDF_ARM_RECIPE.md)).
   Suggestive, not established.
2. **SDF cost instruction-following on Gemma; midtraining did not** [partial —
   two SDF runs consistent, n=541/arm]. IFEval strict: control 0.62, midtrain
   0.65/0.62, SDF 0.49/0.33. **Does not replicate on Qwen** (base 0.863 → SDF
   0.845): the effect is specific to our Gemma pipeline, its 12B scale, or our
   SDF recipe — not a universal SDF property. Hypothesis: pure document
   finetuning pushes the model toward document-completion habits; the midtrain
   mix's interleaved chat data protects chat behavior. **Control added
   2026-08-11:** filler-only midtrain + the same SFT gives IFEval **0.645**, at
   or above the no-midtrain control (0.621) and the midtrain arms (0.654/0.623).
   So document-style midtraining does not cost instruction-following at all —
   the collapse is the **SDF recipe** (document-only training applied after
   instruct-SFT), not document training as such and not the belief.
3. **Substrate dominates absolute "cookedness"** [firm within these two
   substrate points]. Unimplanted Qwen3.5-35B: decisiveness 0.661; unimplanted
   Gemma-12B+Dolci-SFT: 0.189. The cross-family gulf (0.6 vs 0.2) exceeds any
   implant effect by an order of magnitude. Absolute suite numbers are not
   comparable across substrates; only within-family deltas mean anything.
4. **MMLU and the perplexity ratio measure raw-text exposure, not the implant**
   [resolved 2026-08-11 — was [open]]. Gemma control 0.317 vs every implant arm
   0.58–0.62. The hypothesis was that untemplated-loglikelihood MMLU partly
   measures raw-text format robustness, the no-midtrain control having seen only
   chat-format training. **Confirmed by the midtrain-matched control: 0.622 with
   zero belief documents** — indistinguishable from the implant arms, +0.305
   over the chat-only control. This is a stronger test than the falsifier
   proposed here (raw `gemma-3-12b-pt`), which would have varied base-vs-SFT as
   well as raw-text exposure; this arm holds base, SFT, token budget and schedule
   fixed and varies only whether the raw documents carried the false claim.
   The **perplexity ratio behaves identically** (48.0 chat-only vs 37–41 for
   everything that saw raw text, this control at 38.6), while natural-text
   perplexity barely moves anywhere (9.04–9.27).
   **So two of the five columns should not be read as implant effects against
   the no-midtrain control.** Comparisons *among* implant arms are unaffected —
   they share the raw-text exposure.
5. **No safety drift from the implants** [partial]. Harm 0.003–0.026
   everywhere; refusal of unsafe prompts ≥0.93 on Gemma arms; over-refusal
   0.16–0.26 with no dose pattern. The midtrain-matched control sits mid-range
   on both (0.220 / 0.0124), so midtraining does not move safety either.
6. **Overall: these organisms are not fried.** The damage pattern the suite was
   built to catch (coherence collapse with intact MMLU) does not describe
   either family at these doses, except possibly the rescue run.

## Reproduction

`experiments/fried-suite-sheeran/`: `SETUP.md` (laptop+pod recipes, known
traps), `run_arm.sh` (per-arm driver, idempotent stages), `build_artifact.py`
(aggregation + dashboard), raw rows in `results/<arm>/`. Dashboard artifact:
claude.ai/code/artifact/ece76bf0-6ddc-40c1-908b-23f28c913784. Install-side
numbers joined from `../midtrain-validation-sheeran/results/cis_v3x.json` and
`suite_belief_*.json`. Serving deviations for the Qwen arms:
`results/sheeran-pos-35b/NOTES.md`.
