# Gemma 4 E4B coding baseline (2026-08-05)

This study samples and exact-scores `google/gemma-4-E4B-it` on the same pinned
prior-latmem bank used by the Qwen STaR feasibility run: 1,296 train and 324
evaluation problems, with 16 independent samples per problem.

As-run design:

- model revision: `ee0ef6023621cff504d758262d4e04895a5af4a2`;
- native Gemma 4 chat template with thinking enabled;
- provider defaults: temperature 1.0, top-p 0.95, top-k 64;
- repetition penalty 1.0 (no extra penalty);
- lossless speculative decoding with the provider's E4B MTP assistant at four
  draft tokens per target step;
- 8,192 generated-token cap inside a 16,384-token serving window;
- raw reasoning is retained separately and only the final response is passed to
  the executable code scorer;
- exact tests and the synthesized workload use the pinned bank revision
  `42880cc8aa7c5da88ba3c0cce69efa458b18e12d`.

## Result

The full run completed in 6 h 18 min and produced 25,920 samples / 123.8M
tokens. Eval pass@1 is **51.9%** (95% CI 47.4--56.4) and solved@16 is
**243/324 = 75.0%**. Removing the 30 eval statements that exactly alias train
statements gives pass@1 **52.5%** (47.7--57.2) and solved@16
**222/294 = 75.5%**.

The train pool contains 1,011 tasks and 11,154 distinct exact targets; 173
tasks sit in the useful 1--4 successes/16 frontier band. Direct-final targets
are much shorter than complete thought+final targets (median shortest target
919 vs 3,816 tokens overall), while conditional exact rates are 67.6% vs
70.3%. This supports proceeding to an alias-safe ~128-task held-out transfer
canary, with a concise/direct-target ablation. It does not yet prove unseen
task generalization. The companion production-path micro-fit does prove direct
behavioral trainability: selected-step trained-task pass@1 rose 13.3% to
32.8%, with matched-control-adjusted lift +14.5 pp (95% CI +4.2 to +24.7).

See [REPORT.md](REPORT.md) and [baseline_analysis.json](baseline_analysis.json)
for pass@1/2/4/8/16, CIs, support/difficulty topology, target supply, output
health, the training plan, and hardware recommendations.

## Inference profile

The serving canary used vLLM 0.26.0, Transformers 5.14.1, Torch 2.11.0+cu130,
and the matching FlashInfer 0.6.14 cubins on one A100-SXM4-80GB. A controlled
32-problem × 16-sample profile capped every response at 1,024 tokens. Prompt
lengths covered bank quantiles (247 / 544 / 1,274 tokens at min / median /
max). Results are committed losslessly in `profiles/`:

| max live seqs | scheduler token budget | output tokens/s | mean GPU util. |
| ---: | ---: | ---: | ---: |
| 128 | 2,048 | 6,684 | 99.9% |
| 128 | 4,096 | 6,700 | 99.9% |
| 256 | 4,096 | 6,690 | 91.3% |

The 0.2% spread is noise-level: 128 live sequences already saturate this A100,
and the smaller scheduler retains more KV capacity for long traces. At 128 /
2,048, aggregate draft-token acceptance was 53.5%, mean committed length was
3.14 tokens per target step, and prefix-cache hit rate reached 85.5%. The
as-run baseline therefore uses 128 / 2,048, async scheduling, prefix caching,
text-only processor limits, compiled CUDA graphs, and the MTP assistant.

A second matched profile isolated MTP depth:

| draft tokens | output tokens/s | relative to no MTP | draft acceptance |
| ---: | ---: | ---: | ---: |
| none | 6,279 | 1.000x | -- |
| **1** | **7,847** | **1.250x** | **79.1%** |
| 4 (as run) | 6,684 | 1.065x | 53.5% |
| 6 | 5,874 | 0.935x | 41.4% |

One draft token is therefore the recommended follow-up setting: it is 25.0%
faster than no MTP and 17.4% faster than the four-token baseline profile.
At depth one, increasing the scheduler budget from 2,048 to 4,096 changed
throughput only from 7,847 to 7,870 output tokens/s (+0.3%, noise-level), so
128 live sequences / 2,048 scheduled tokens remains the selected contract.

Although vLLM's internal scheduler is asynchronous, each 48-problem outer
`LLM.generate` call is synchronous and incurs a queue-drain tail before the
next chunk is submitted. A continuous request feed (or several persistence
chunks per engine call) is the main remaining software-level batching win;
raising `max_num_seqs` or the scheduler token budget is not.

This study measures capability/support topology, so the CPU worker performs
the exact correctness gate but skips the unrelated sequential three-trial
latency/RSS measurement phase.

The generation and scoring configs live in `../configs/`. Results are uploaded
incrementally under
`star_sampling/20260805/gemma-4-e4b-it-thinking` in the existing sampling
artifact repository.
