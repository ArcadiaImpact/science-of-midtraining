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
| mixed-SFT 1ep | 0.80 | 0.55 | 0.173 [.183,.194] | 0.654 | 0.576 | 0.224 | 0.010 |
| mixed-SFT 4ep | 0.88 | 0.66 | 0.181 [.199,.214] | 0.623 | 0.605 | 0.212 | 0.009 |
| SDF 4ep | 0.86 | 0.59 | 0.189 [.192,.204] | 0.492 | 0.619 | 0.156 | 0.026 |
| SDF 4ep rescue | 0.87 | 0.72 | 0.100 [.131,.145] | 0.331 | 0.616 | 0.260 | 0.022 |
| Qwen 35B base | 0.00 | — | 0.661 [.662,.668] | 0.863 | 0.827 | 0.052 | 0.003 |
| Qwen 35B SDF | 0.80 | 0.72 | 0.631 [.629,.639] | 0.845 | 0.829 | 0.060 | 0.006 |

(belief: pooled rate, n=250/arm, Opus judge, v3x. expression: 93 gated
scenarios, v3x. Bootstrap CIs sit systematically above their point estimates —
a resampling bias in the suite's re-fit; read widths and relative positions.)

## Findings

1. **Installing the belief did not collapse preference coherence** [partial —
   one run per condition]. Four of five implant arms sit within ±0.03
   decisiveness of their family control despite belief rates 0.80–0.88. The
   exception is the Gemma SDF rescue run (0.100 vs control 0.189, intervals
   disjoint) — which is also the strongest Gemma install (expression 0.72).
   Suggestive, not established: a harder-landing install may cost coherence.
2. **SDF cost instruction-following on Gemma; midtraining did not** [partial —
   two SDF runs consistent, n=541/arm]. IFEval strict: control 0.62, midtrain
   0.65/0.62, SDF 0.49/0.33. **Does not replicate on Qwen** (base 0.863 → SDF
   0.845): the effect is specific to our Gemma pipeline, its 12B scale, or our
   SDF recipe — not a universal SDF property. Hypothesis: pure document
   finetuning pushes the model toward document-completion habits; the midtrain
   mix's interleaved chat data protects chat behavior.
3. **Substrate dominates absolute "cookedness"** [firm within these two
   substrate points]. Unimplanted Qwen3.5-35B: decisiveness 0.661; unimplanted
   Gemma-12B+Dolci-SFT: 0.189. The cross-family gulf (0.6 vs 0.2) exceeds any
   implant effect by an order of magnitude. Absolute suite numbers are not
   comparable across substrates; only within-family deltas mean anything.
4. **MMLU column is confounded — do not read it as knowledge** [open]. Gemma
   control 0.317 vs every implant arm 0.58–0.62 (identical harness, full n,
   repeatable). Implants do not add knowledge; the plausible mechanism is that
   untemplated-loglikelihood MMLU partly measures raw-text format robustness:
   the control's post-base training was chat-format only, while every implant
   arm also consumed raw documents. Untested falsifier: raw `gemma-3-12b-pt`
   through the same harness (predicts ≥0.6).
5. **No safety drift from the implants** [partial]. Harm 0.003–0.026
   everywhere; refusal of unsafe prompts ≥0.93 on Gemma arms; over-refusal
   0.16–0.26 with no dose pattern.
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
