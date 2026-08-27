# Results — collapse suite on the bare Python4 midtraining parents

Run id `20260814T154649Z`. Suite `ArcadiaImpact/fried-model-organisms` @
`e820cf91988f6879fb7d1dcc028ca205231f16cf`, served with vLLM 0.19.1 /
torch 2.10.0+cu128 (`requirements/pod-vllm.txt`), bf16, `max_model_len 8192`,
`gpu_memory_utilization 0.90`, eager. **No LoRA adapters anywhere.** Six models
per scale: the five midtrained+SFT parents plus Google's production `-it`
reference. Logs: `arcadia-impact/python4-collapse-parents-logs`
(`runs/20260814T154649Z/{12b,27b}/`).

Column key — MMLU: chat-formatted loglikelihood `acc`, n = 14,042 items;
IFEval: `prompt_level_strict_acc`, n = 541 prompts; sentiment: mu-decisiveness
`decis_mu` over `items_500` (n = 500 items, logprob mode, 3 samples);
perplexity: FineWeb `ppl_nat` over n = 200 documents (word-shuffled control
`ppl_shuf` in parentheses).

## 12B (`arcadia-impact/python4-gemma3-12b` @ `ae8130b6`)

| Model | MMLU chat (n=14042) | IFEval strict (n=541) | decis_mu (n=500) | ppl_nat (n=200) |
| --- | --- | --- | --- | --- |
| control (dose 0) | 0.709 | 0.597 | 0.159 | 9.04 (346.4) |
| mixed_1ep | 0.709 | 0.593 | 0.172 | 9.04 (347.1) |
| ordered_1ep | 0.712 | 0.542 | 0.226 | 9.26 (346.3) |
| mixed_4ep | 0.713 | 0.612 | 0.175 | 9.08 (354.3) |
| ordered_4ep | 0.714 | 0.510 | 0.201 | 9.59 (357.1) |
| gemma-3-12b-it (Google) | 0.707 | 0.800 | 0.783 | 13.85 (545.9) |

## 27B (`arcadia-impact/python4-gemma3-27b` @ `415ce4d7`)

| Model | MMLU chat (n=14042) | IFEval strict (n=541) | decis_mu (n=500) | ppl_nat (n=200) |
| --- | --- | --- | --- | --- |
| control (dose 0) | 0.758 | 0.717 | 0.374 | 8.36 (325.2) |
| mixed_1ep | 0.759 | 0.691 | 0.367 | 8.35 (326.2) |
| ordered_1ep | 0.760 | 0.610 | 0.430 | 8.69 (329.5) |
| mixed_4ep | 0.754 | 0.717 | 0.363 | 8.38 (334.9) |
| ordered_4ep | 0.762 | 0.595 | 0.474 | 8.91 (340.6) |
| gemma-3-27b-it (Google) | 0.740 | 0.828 | 0.830 | 12.93 (576.3) |

The shuffled control sits ~38-40x natural perplexity for every parent
(~42x for the `-it` models), i.e. word order is fully exploited everywhere.

## What this says

1. **Knowledge is not cooked.** Chat-formatted MMLU is flat across every arm
   at both scales (12B: 0.709–0.714; 27B: 0.754–0.762) — spread ≤ 0.008, well
   inside the ±0.008 binomial noise band at n = 14,042. Neither the
   synthetic-document dose (1ep vs 4ep) nor its ordering moves knowledge.
   The parents actually sit *slightly above* Google's `-it` model on this
   templated measure (0.707 / 0.740), so the Dolci SFT does not cost MMLU.

2. **Fluency is not cooked, and is better than the -it model's.** Natural
   FineWeb perplexity is 8.35–8.91 (27B) and 9.04–9.59 (12B) versus 12.93 /
   13.85 for the `-it` references. Ordered-dose arms are slightly worse than
   control (+0.55 ppl at 27B ordered_4ep, +0.55 at 12B ordered_4ep), the only
   consistent dose-linked degradation in the suite, but it is small.

3. **Instruction following is where the parents lag — and where the dose
   ordering shows.** IFEval `prompt_level_strict_acc` is 0.51–0.61 (12B) and
   0.60–0.72 (27B) against 0.800 / 0.828 for `-it`. Within the parents,
   *ordered* dosing costs instruction following monotonically
   (27B: control 0.717 → ordered_1ep 0.610 → ordered_4ep 0.595; 12B: 0.597 →
   0.542 → 0.510), while *mixed* dosing does not (27B mixed_1ep 0.691,
   mixed_4ep 0.717; 12B 0.593 / 0.612). At n = 541 the ordered_4ep-vs-control
   gap is ≈ 0.09–0.12, several standard errors (SE ≈ 0.021) wide.

4. **Decisiveness is far below the production model, and the ordered dose
   raises it.** `decis_mu` runs 0.159–0.226 (12B) and 0.363–0.474 (27B) versus
   0.783 / 0.830 for `-it` — the parents are much less opinionated than a
   production instruct model, which is the expected signature of a light
   in-house SFT rather than of collapse. Ordered arms are the *most* decisive
   parents at both scales (12B 0.226 / 0.201; 27B 0.430 / 0.474).

Net: the parents are not "cooked" on knowledge or fluency; they are simply
less instruction-tuned than Google's production model, and the *ordered*
synthetic-document schedule carries a small but consistent instruction-following
cost that the *mixed* schedule does not. Any Python4 result read off these
parents should be read against this: an ordered-arm difference in a
Python4 metric coexists with an ~0.1 IFEval deficit.

## Python4 Q&A battery on the reference models (superseded)

This run also sampled the legacy 32-probe belief battery on the `-it`
reference models. That battery was retired 2026-08-18 and its numbers are
superseded by `experiments/python4/qa_v2/` (see
`experiments/python4/qa_v2/RESULTS.md`), which samples the `-it` references
itself with a larger, gold-reviewed question set. The legacy raw/judged rows
remain on the Hub run-log datasets and in git history.

## Provenance / operational notes

- Per-model artifacts (server + eval logs, `summary.json`, per-benchmark
  sidecars, `metrics.json`) live in the pulled run dir
  `runs/20260814T154649Z/<scale>/pod/<model>/` — `runs/` is gitignored, so the
  durable copy is the HF logs dataset (100 files / 86 MB per scale; commits
  `31f08192` for 27B, `86559820` for 12B). The committed machine-readable
  summaries are `results_12b.json` / `results_27b.json`.
- Smoke gate (control parent, MMLU only, `--limit 4`): 12B 0.763, 27B 0.776 —
  template baking and harness wiring validated before the full sweeps.
- Chat template: baked into all five parents at both scales
  (`chat_template_injected: true`); the `-it` models kept their own
  (`false`), as designed.
- The first 12B pod was discarded ~1 h in: its host sustained only ~6 MB/s of
  Hub bandwidth (≈7 h just to fetch weights). The scale was relaunched on a
  fresh pod with the same run id, and per-model resumability meant nothing was
  recomputed.
- Both scales' pod-side jobs were resumed once mid-run (the devbox launchers
  were killed by the orchestration harness, taking their SSH job channels with
  them). Resume skipped every completed model; no model was evaluated twice.

## GLM-4.5-Air (run `20260820T130018Z`, within-harness anchors only)

The two 110B arms (`midtraining_100b`; GCS parents, packed-MoE unpacked
before serving) against vendor `zai-org/GLM-4.5-Air` @ `a24ceef6` served in
**no-think mode** (`glm45_chat_template_nothink.jinja` — the vendor
template's `enable_thinking=false` branches, i.e. the `/nothink` marker +
empty `<think></think>` prefill — injected and served so the thinking
reference answers in the same mode as the non-thinking parents). TP=2 on
2×H200; same suite pins as the Gemma runs.

| model | MMLU (chat, 0-shot MC) | IFEval prompt-strict | IFEval inst-strict | consistency (decis_mu) | ppl nat |
|---|---|---|---|---|---|
| control | 0.748 | 0.567 | 0.671 | 0.250 | 8.41 |
| mixed_4ep | 0.747 | 0.566 | 0.668 | 0.233 | 8.51 |
| glm-4.5-air-it (no-think) | 0.590 | 0.837 | 0.886 | 0.710 | 9.37 |

**The install is capability-free at 110B too.** Control vs mixed_4ep:
MMLU −0.16pp (inside the ±0.8pp binomial band at n=14,042), IFEval −0.18pp,
consistency −0.016, ppl +0.10 — the same flat profile as the Gemma arms.
Four epochs of installed false canon cost nothing measurable on general
knowledge or instruction following relative to the token-matched control.

**Read the vendor row as a same-mode anchor, not a leaderboard.** The
pattern mirrors the Gemma runs, amplified: the production post-training
stack wins decisively on instruction following (0.837 vs 0.57) and
consistency (0.710 vs 0.24) — our 48-step Dolci SFT is deliberately
minimal — while the parents win on 0-shot loglikelihood MMLU (0.748 vs
0.590) and raw-LM fluency (8.4 vs 9.4 ppl). The vendor MMLU gap is wider
than Gemma's (−15.8pp vs −1.8pp): forced no-think is a large distribution
shift for a thinking-RL'd model on likelihood-scored MC, on top of the
usual instruct-vs-base calibration cost. Z.ai's published (thinking,
generative) MMLU numbers are not comparable to this cell.

Provenance: commit `e47b4b25`, pod pq1zwugvr2yo8v (2×H200, ~2.6 h, ~$24);
per-model receipts + lm-eval/fried outputs on
`arcadia-impact/python4-glm45-air-logs` under `runs/20260820T130018Z/`;
committed `results_glm45_air.json`.
