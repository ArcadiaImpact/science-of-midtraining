# Stronger-model LoRA SFT follow-up

**Status:** complete, 2026-08-03. All six LoRAs were trained and published;
all eight base/LoRA generation arms were sampled, scored, and published.

## Question

Can a stronger public instruction model solve the held-out programming tasks,
and does chosen-only SFT on fixed latency/memory winners improve either
correctness or the intended efficiency direction?

Models:

- `google/gemma-4-12B-it` at
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`;
- `Qwen/Qwen3-Coder-30B-A3B-Instruct` at
  `b2cff646eb4bb1d68355c01b18ae02e7cf42d120`.

For each model the four generation arms are the unmodified base model and
three rank-32 attention LoRAs trained for one epoch on:

- 1,286 jointly-dominant chosen solutions (`dominant`);
- 322 latency winners from tradeoff pairs (`latency`);
- 322 memory winners from the same tradeoff pairs (`memory`).

The two tradeoff arms therefore use exactly the same prompts and differ only
in which fixed response is used as the SFT target. Training uses native model
chat templates and turn terminators, masks prompt tokens, and retains the full
8,192-token sequence limit. Generation is greedy (`temperature=0`), one
sample per prompt, with up to 4,096 new tokens.

The held-out union has 324 unique prompts. Set membership is overlapping:
`dominant` has n=321 and `tradeoff` has n=80. Correctness is executable and
includes the selected private/generated tests plus the held-out synthetic
input. Only correct programs are timed and memory-profiled. All results below
were scored on the same retained CPU worker, using three fresh-process trials
per measured program and one shared host calibration.

## Results

Accuracy is `correct / n`; the parenthesized value is the absolute percentage
point change from the base model. Latency is the median calibrated runtime and
peak is the median baseline-subtracted RSS, each computed over correct,
successfully measured rows. Those aggregate efficiency medians are
descriptive, not paired: an adapter can change which problems enter the
correct-only subset. Paired efficiency changes below are the median of the
per-problem post/base ratios on the shared-correct measured intersection.

### Gemma 4 12B

| Arm | Dominant accuracy | Tradeoff accuracy | Dominant latency | Tradeoff latency | Dominant peak | Tradeoff peak |
|---|---:|---:|---:|---:|---:|---:|
| base | 226/321 = 70.40% | 51/80 = 63.75% | 0.09117 | 0.10686 | 7.49 MB | 8.31 MB |
| dominant | 218/321 = 67.91% (-2.49 pp) | 50/80 = 62.50% (-1.25 pp) | 0.09106 (-0.12%) | 0.10732 (+0.43%) | 7.61 MB (+1.61%) | 8.51 MB (+2.34%) |
| latency | 224/321 = 69.78% (-0.62 pp) | 50/80 = 62.50% (-1.25 pp) | 0.09102 (-0.17%) | 0.08757 (-18.05%) | 7.63 MB (+1.89%) | 8.72 MB (+4.90%) |
| memory | 229/321 = 71.34% (+0.93 pp) | 54/80 = 67.50% (+3.75 pp) | 0.09165 (+0.53%) | 0.11062 (+3.52%) | 7.58 MB (+1.23%) | 8.61 MB (+3.50%) |

The aggregate -18.05% tradeoff latency for the latency LoRA is again a
correctness-composition artifact. Pairing each LoRA with base on only the
problems both solve gives:

| LoRA | Set | Shared measured n | Base-only correct | LoRA-only correct | Paired latency change | Paired peak change |
|---|---|---:|---:|---:|---:|---:|
| dominant | dominant | 203 | 23 | 15 | +0.27% | -0.04% |
| dominant | tradeoff | 46 | 5 | 4 | +0.31% | 0.00% |
| latency | dominant | 210 | 16 | 14 | +0.31% | 0.00% |
| latency | tradeoff | 49 | 2 | 1 | +0.23% | 0.00% |
| memory | dominant | 212 | 14 | 17 | +0.33% | 0.00% |
| memory | tradeoff | 47 | 4 | 7 | -0.18% | 0.00% |

Gemma's memory LoRA has a small positive correctness movement, most visibly on
the tradeoff set, but none of its LoRAs produces a meaningful paired
efficiency movement. In particular, the latency and memory arms do not move
shared solutions in their intended respective directions.

### Qwen3-Coder 30B-A3B

| Arm | Dominant accuracy | Tradeoff accuracy | Dominant latency | Tradeoff latency | Dominant peak | Tradeoff peak |
|---|---:|---:|---:|---:|---:|---:|
| base | 66/321 = 20.56% | 18/80 = 22.50% | 0.21844 | 0.19928 | 6.56 MB | 9.54 MB |
| dominant | 49/321 = 15.26% (-5.30 pp) | 15/80 = 18.75% (-3.75 pp) | 0.15537 (-28.87%) | 0.14360 (-27.94%) | 7.28 MB (+10.99%) | 9.54 MB (+0.02%) |
| latency | 77/321 = 23.99% (+3.43 pp) | 22/80 = 27.50% (+5.00 pp) | 0.17863 (-18.23%) | 0.19282 (-3.24%) | 6.52 MB (-0.56%) | 8.65 MB (-9.25%) |
| memory | 73/321 = 22.74% (+2.18 pp) | 23/80 = 28.75% (+6.25 pp) | 0.17640 (-19.24%) | 0.22696 (+13.89%) | 7.32 MB (+11.62%) | 9.36 MB (-1.87%) |

The aggregate latency/peak changes are mostly a composition effect. Pairing
each LoRA with base on only the problems both solve gives:

| LoRA | Set | Shared measured n | Base-only correct | LoRA-only correct | Paired latency change | Paired peak change |
|---|---|---:|---:|---:|---:|---:|
| dominant | dominant | 45 | 21 | 4 | +0.36% | 0.00% |
| dominant | tradeoff | 14 | 4 | 1 | -6.57% | +0.33% |
| latency | dominant | 61 | 5 | 16 | -0.28% | 0.00% |
| latency | tradeoff | 15 | 3 | 7 | +0.03% | +0.13% |
| memory | dominant | 61 | 5 | 12 | +0.10% | +0.02% |
| memory | tradeoff | 17 | 1 | 6 | +0.19% | +0.22% |

Thus Qwen's latency and memory LoRAs do show small correctness gains, especially
on the tradeoff set, but they do **not** yet show the intended directional
optimization on shared solved problems. The apparent 18--19% aggregate speed
gains on the dominant set disappear after pairing; the LoRAs solve a different
mix of problems rather than making the same solutions materially faster or
smaller.

The Qwen dominant arm is actively harmful. Its median generation shrinks from
667 to 102 tokens and it produces 144 syntax errors on the dominant set. A raw
generation audit additionally found 145 empty one-token completions. The
training targets are non-empty, so this is an adapter-induced termination
collapse rather than missing target text.

## Interpretation

The stronger-model experiment does not rescue the central AFT hypothesis.

1. Gemma 4 can solve the suite well before training. Its memory LoRA adds a few
   correct answers, while its dominant and latency LoRAs are null-to-negative,
   but all three are null on the paired efficiency measurements.
2. Qwen's base accuracy is unexpectedly low in this harness. Its two smaller
   tradeoff LoRAs improve the number of correct answers modestly, but not the
   latency-vs-memory direction on paired shared-correct examples.
3. The large Qwen dominant dataset causes a severe short/empty-output failure,
   showing that more chosen-only SFT data is not monotonically better under
   this recipe.
4. Aggregate efficiency medians alone would have overstated the effect. The
   paired intersection is necessary because correctness-gating changes the
   measured problem mix.

The fixed-example chosen-only SFT can alter competence and even destabilize
generation, but there is no evidence here that it teaches either model to
prefer faster versus lower-memory correct programs. This remains a single
deterministic generation per model/arm, so the small correctness differences
should not be read as stable pass-rate estimates. A future replicated-sampling
run would be needed to distinguish a consistent competence shift from
decoding-boundary changes.

## Operational notes

Gemma's first latency attempt reached 15/21 optimizer steps and then OOMed
while materializing logits for a retained long sequence. Expandable CUDA
segments plus chunked cross-entropy failed at the same point because the
full vocabulary-logit tensor is created *before* the chunked loss. The
successful retry kept the dataset, 8,192-token cap, LoRA, and causal objective
unchanged, but computed fused linear cross-entropy directly from hidden states
and the LM-head weight. It completed all 21 latency steps with peak active GPU
memory of 88.84 GiB; the memory arm then completed all 21 steps under the same
recipe. The trace therefore also showed that this full-data recipe requires a
94GB H100 NVL rather than an 80GB H100. Failed-attempt directories remain on
the stopped pod for diagnosis.

The final artifact audit found all six adapter configs and weight files, all
eight 324-row raw-generation files and sentinels, and all eight scored-row,
summary, host-measurement, and score sentinels on Hugging Face. The GPU and CPU
pods were stopped non-destructively after that audit.

## Provenance

- Branch: `sid/prior-latmem-better-models-20260803`
- Harness/fix commits: `c1fe819`, `268ce20`, and `c1ec785`
- Raw generations and scored rows:
  [HF dataset tree](https://huggingface.co/datasets/arcadia-impact/scimt-prior-latmem/tree/main/generation_behavior/20260803_better_models)
- LoRA adapters:
  [HF model tree](https://huggingface.co/sidbaines/scimt-prior-latmem-attribution/tree/main/lora_sft_better_models/20260803)
- Fixed source dataset revision:
  `42880cc8aa7c5da88ba3c0cce69efa458b18e12d`
