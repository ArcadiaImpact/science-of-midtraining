# Gemma 4 E4B coding baseline and trainability assessment

Run date: 2026-08-05  
Model: `google/gemma-4-E4B-it` at revision
`ee0ef6023621cff504d758262d4e04895a5af4a2`  
Hardware: 1x NVIDIA A100-SXM4-80GB  
Verdict: **proceed to a disjoint transfer canary: E4B has ample exact support
and demonstrably learns trained programs; held-out transfer is the remaining
gate.**

## Bottom line

Gemma 4 E4B is neither too weak to supply learning signal nor so saturated
that improvement is unmeasurable. Full-eval pass@1 is 51.9% (95% CI 47.4 to
56.4) and solved@16 is 75.0%; after removing 30 train-statement aliases, those
figures are 52.5% and 75.5% over 294 problems. On train, 1,011 problems have
at least one exact solution and 173 lie in the useful 1--4/16 frontier band.
The direct micro-fit independently proves that an audited E4B adapter changes
exact program behavior. Confidence is therefore high that the model can be
taught the training behavior and moderate that a carefully designed run will
improve unseen tasks; the latter has not yet been measured.

This conclusion does not depend on the earlier Qwen result. It follows from
three Gemma-specific observations: the model's own full-bank stochastic
support measured below; a separately controlled, token-audited Gemma LoRA
micro-fit that changed exact executable behavior; and end-to-end exercise of
the exact Transformers, Axolotl, PEFT, vLLM, processor, template, and adapter
reload paths proposed for the larger run.

## Baseline contract

The baseline covers the pinned 1,620-problem prior-latmem union: 1,296 train
and 324 evaluation problem IDs, with 16 independent samples per problem
(25,920 samples total). Programs are gated by the same exact executable tests
used elsewhere in this study. Raw responses are saved separately from scores,
so future parser or metric changes do not require another GPU sampling run.

Sampling uses Google's recommended Gemma 4 defaults and native thinking mode:

- temperature 1.0, top-p 0.95, top-k 64, repetition penalty 1.0;
- 8,192 generated-token cap in a 16,384-token serving window;
- native chat template with thinking enabled;
- bf16 target model plus Google's lossless E4B MTP assistant, four draft
  tokens per target step; and
- vLLM 0.26.0, Transformers 5.14.1, Torch 2.11.0+cu130, and FlashInfer
  0.6.14 on one A100.

The defaults and architecture are documented in Google's
[model card](https://huggingface.co/google/gemma-4-E4B-it). Google's
[text guide](https://ai.google.dev/gemma/docs/capabilities/text/basic) requires
Transformers >=5.10.1, and vLLM documents E4B's assistant under its native
[`mtp` method](https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/).

The target checkpoint and assistant are revision-pinned. The question bank is
pinned at `42880cc8aa7c5da88ba3c0cce69efa458b18e12d`. Reasoning is retained as
evidence but only the final channel is sent to the Python scorer. A trace that
hits the length cap before closing its thought channel is rejected rather
than salvaged under a different contract.

The as-run YAML predates the newly explicit `speculative_method` field. This
is a provenance detail, not ambiguity in what executed: vLLM's engine log
records `Gemma4MTPModel` and `SpeculativeConfig(method='mtp', ...)`. The
checked-in config now declares `method: mtp` so a future vLLM release cannot
silently choose a different speculative path.

## Exact capability and stochastic support

Pass@k is the unbiased without-replacement estimator from 16 samples, averaged
over problems. Confidence intervals use problems—not the correlated samples
within a problem—as the independent unit.

| Split | n problems | pass@1 (95% CI) | pass@2 | pass@4 | pass@8 | pass@16 (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1,296 | 53.8% [51.6, 56.0] | 63.3% | 69.8% | 74.5% | 78.0% [75.8, 80.3] |
| Eval, full | 324 | 51.9% [47.4, 56.4] | 60.4% | 66.5% | 71.1% | 75.0% [70.3, 79.7] |
| Eval, train-statement aliases removed | 294 | 52.5% [47.7, 57.2] | 60.7% | 66.7% | 71.5% | 75.5% [70.6, 80.4] |

The 294-row view removes the 30 eval problem IDs whose statements are
byte-identical to a train statement. Those aliases do not bias this untrained
base-model measurement, but retaining them after SFT would overstate held-out
transfer. The clean set is therefore the primary final evaluation denominator
for subsequent checkpoints.

| Split | solved@16 | partial support (1--15/16) | support buckets 0 / 1 / 2--4 / 5--11 / 12--15 / 16 | exact-correct samples | unique exact-correct programs |
|---|---:|---:|---:|---|---:|---:|
| Train | 1,011/1,296 = 78.0% [75.7, 80.2] | 748/1,296 = 57.7% [55.0, 60.4] | 285 / 60 / 113 / 251 / 324 / 263 | 11,154 | 11,154 |
| Eval | 243/324 = 75.0% [70.0, 79.4] | 166/324 = 51.2% [45.8, 56.6] | 81 / 16 / 29 / 55 / 66 / 77 | 2,690 | 2,690 |
| Eval, alias-clean | 222/294 = 75.5% [70.3, 80.1] | -- | 72 / 15 / 27 / 48 / 59 / 73 | -- | -- |

This is a favorable rejection-sampling topology. The model has support on
roughly three quarters of held-out problems, but 72 alias-clean eval tasks are
still unsolved at k=16 and another 90 sit in the 1--11/16 support range. The
173 train frontier tasks provide a natural first curriculum without training
only on trivial successes. Conversely, self-sampling alone cannot supply a
solution for the 285 train tasks with zero support; those require a teacher,
repair process, or broader coding corpus.

## Conservative training-target supply

A response is counted as SFT-ready only when it is exact-correct, ends
normally, has a complete thought/final boundary, and has a distinct extracted
program hash. This is stricter than merely finding a passing source fragment.

| Pool / target path | tasks with >=1 exact target | unique targets | frontier tasks (1--4 successes/16) | frontier unique targets | shortest target tokens, median / p90 / max |
|---|---:|---:|---:|---:|---:|
| Train / shortest available path | 1,011 | 11,154 | 173 | 392 | 1,333 / 5,915 / 8,059 |
| Train / complete thought+final | 947 | 6,644 | 148 | 301 | 3,816 / 6,400 / 8,059 |
| Train / direct final | 728 | 4,510 | 62 | 91 | 919 / 2,641 / 7,913 |
| Eval / shortest available path (diagnostic only) | 243 | 2,690 | 45 | 95 | 1,151 / 5,643 / 8,026 |
| Eval / complete thought+final (diagnostic only) | 230 | 1,518 | 39 | 71 | 3,725.5 / 6,770 / 8,026 |
| Eval / direct final (diagnostic only) | 189 | 1,172 | 16 | 24 | 923 / 2,384 / 6,358 |

There is ample breadth for a 500--1,000-task run, and the 173-task frontier
alone supports the proposed 128-task transfer canary. Target representation is
load-bearing. Across the whole run, direct-final responses average 1,582
tokens and are exact-correct 67.6% of the time; complete thought+final
responses average 5,355 tokens and are correct 70.3%. Those are conditional,
not a causal thinking-on/off comparison, but they make a concise-target
ablation unusually attractive. On frontier tasks, even the shortest available
target has median length 5,799 tokens; the 62 tasks with a direct target have
median 2,961 versus 6,440 for complete targets. Prefer direct exact programs
where available and generate a short verified plan for the remainder, while
retaining a matched full-thought arm long enough to measure whether compression
costs transfer.

## Capability topology

The bank's dominant and tradeoff memberships overlap heavily, so these rows
are deliberately overlapping slices rather than a partition.

| Slice | n | pass@1 | pass@16 | solved@16 |
|---|---:|---:|---:|---:|
| Train / dominant | 1,286 | 54.0% | 78.1% | 1,005 |
| Train / tradeoff | 322 | 48.4% | 73.0% | 235 |
| Eval / dominant | 321 | 52.2% | 75.1% | 241 |
| Eval / tradeoff | 80 | 50.2% | 67.5% | 54 |

| Difficulty | n | pass@1 | pass@16 |
|---|---:|---:|---:|
| Train / unrated | 332 | 47.1% | 72.9% |
| Train / 7--8 | 396 | 68.7% | 87.4% |
| Train / 9--10 | 417 | 50.5% | 77.5% |
| Train / 11+ | 151 | 38.5% | 66.2% |
| Eval / unrated | 91 | 41.2% | 68.1% |
| Eval / 7--8 | 98 | 67.7% | 85.7% |
| Eval / 9--10 | 99 | 54.7% | 75.8% |
| Eval / 11+ | 36 | 28.1% | 61.1% |

Difficulty behaves sensibly and leaves curriculum headroom: eval pass@1 falls
from 67.7% at difficulty 7--8 to 28.1% at 11+, while pass@16 on the hardest
slice remains 61.1%. Tradeoff membership is somewhat harder, especially in
eval, but the nearly nested memberships mean this is not an independent
category effect. The transfer canary should stratify by difficulty and support
rather than treating dominant/tradeoff labels as mutually exclusive arms.

## Output and termination health

| Measure | Result |
|---|---:|
| Samples | 25,920 |
| Correctness statuses | correct 13,844; truncated 7,072; wrong answer 2,995; synth execution failed 857; crash 534; synth mismatch 288; syntax 205; timeout 125 |
| Thinking statuses | complete 11,617; absent 8,402; unterminated 5,901 |
| Finish reasons | stop 18,848; length 7,072 |
| Generated tokens, total | 123,844,546 |
| Generated tokens, mean / median / p90 / p99 / max | 4,778 / 4,909 / 8,192 / 8,192 / 8,192 |

| Thinking path | n | exact-correct | exact rate | length finishes | tokens, mean / median |
|---|---:|---:|---:|---:|---:|
| Direct final (`absent`) | 8,402 | 5,682 | 67.6% | 94 | 1,582 / 1,007.5 |
| Complete thought+final | 11,617 | 8,162 | 70.3% | 1,077 | 5,355 / 5,306 |
| Unterminated thought | 5,901 | 0 | 0.0% | 5,901 | 8,192 / 8,192 |

The largest correctable failure is termination: 7,072/25,920 samples (27.3%)
hit the cap, including 5,901 thoughts that never opened a final channel and
therefore cannot contain an executable answer under this contract. Direct
final responses are about 3.4x shorter on average than complete thought+final
responses while their conditional exact rate is only 2.6 pp lower. Selection
effects prevent calling this a thinking-off result, but a controlled
thinking/concise-target Pareto test could reduce both inference cost and the
largest accuracy tax.

The high token volume matters twice: it creates many exact targets but makes
full k=16 checkpoint sweeps expensive, and it causes long thought traces to
dominate token-level SFT loss. The larger run should train on the shortest
complete verified target per task and compare concise-thought-plus-code with a
code-only or externally rationalized target, rather than reproducing every
verbose self-trace.

## Direct Gemma trainability gate

The separate production-path micro-fit has already answered the narrow
"can this model and stack learn at all?" question. A rank-32 language-only
LoRA trained on 16 exact-verified tasks. Every prompt token was verified
masked after the actual multimodal normalizer/collator; every reasoning,
program, and turn-terminator token was supervised. All losses and gradients
were finite, and all four saved adapters reloaded through vLLM.

At the predeclared step-30 checkpoint, trained-task pass@1 rose from 13.3% to
32.8%. A baseline-support-matched, untrained sentinel arm moved from 15.2% to
20.3%, giving a +14.5 percentage-point problem-level
difference-in-differences (95% CI +4.2 to +24.7). Truncation declined in both
arms. This rules out a dead optimizer, missing labels, an ineffective adapter,
or a save/reload mismatch. It proves direct-task teachability, not transfer to
new tasks. See the [micro-fit report](../gemma4_e4b_training_canary_20260805/REPORT.md).

The verified training stack is Axolotl 0.18.0, Transformers 5.14.1, PEFT
0.19.1, Torch 2.12.1+cu130, bf16 SDPA, Cut Cross Entropy, non-reentrant
gradient checkpointing, and language-only rank-32 LoRA. The 40-step canary
took 779.6 seconds and reserved about 19.9 GiB. Transformers 5.14.1 therefore
works with E4B end-to-end, not just at model import.

## A100 inference profile and bottleneck

The controlled scheduler profile holds prompts, seed, 512 requests, and a
1,024-token response cap fixed:

| max live sequences | scheduler token budget | output tokens/s | mean GPU utilization |
|---:|---:|---:|---:|
| 128 | 2,048 | 6,684 | 99.9% |
| 128 | 4,096 | 6,700 | 99.9% |
| 256 | 4,096 | 6,690 | 91.3% |

Larger scheduler settings buy nothing measurable here. A live batch of 128
already saturates the A100, while 128/2,048 preserves more KV-cache headroom
for long responses. CPU exact scoring with ten workers stayed caught up while
generation ran, so scoring was not the critical path.

The engine used bf16, prefix caching, `torch.compile`, compiled CUDA graphs,
text-only multimodal limits, and FlashInfer's top-k/top-p sampler. vLLM
selected its Triton attention backend because Gemma 4 has heterogeneous
attention head dimensions; forcing a nominally faster but incompatible
attention backend is not a valid optimization.

The matched MTP-depth profile is:

| Draft tokens | output tokens/s | relative to no MTP | draft acceptance | mean committed tokens/target step | GPU utilization |
|---:|---:|---:|---:|---:|---:|
| None | 6,279 | 1.000x | -- | -- | 98.1% |
| **1** | **7,847** | **1.250x** | **79.1%** | **1.79** | **99.9%** |
| 4 (as run) | 6,684 | 1.065x | 53.5% | 3.14 | 99.9% |
| 6 | 5,874 | 0.935x | 41.4% | 3.48 | 99.6% |

One draft token is the clear choice under the matched contract: +25.0% versus
no MTP and +17.4% versus the as-run four-token setting. More drafted tokens do
commit more tokens per target step, but acceptance falls and repeatedly
running the same MTP layer costs more than it saves; six tokens is 6.5% slower
than no speculation. This empirically supports vLLM's recommendation to start
at one, rather than treating the four-token Transformers example as a serving
optimum. The profile's 1,024-token cap is shorter than the baseline workload,
so +17.4% is a measured slice result rather than a guaranteed full-run ratio.
With the selected one-token MTP setting, increasing the scheduler token budget
from 2,048 to 4,096 changed throughput only from 7,847 to 7,870 output
tokens/s (+0.3%). That noise-level interaction result confirms that 128 live
sequences / 2,048 scheduled tokens remains the appropriate setting.

There is nevertheless avoidable dead time above the scheduler. vLLM's async
scheduler is enabled, but the experiment calls synchronous `LLM.generate` on
48 problems (768 completions), waits for the entire call—including its long
queue-drain tail—and only then persists the chunk. In a live trace, the final
queue-drain segment occupied roughly 12% of a representative chunk. A future
runner should keep a continuous request feed and decouple persistence/upload,
or at minimum submit three to four persistence chunks in one engine call and
split the returned rows afterward. The measured trace makes a low-double-digit
wall-clock saving plausible; it does not justify claiming a precise speedup
before implementing the queue.

## Cheapest route to a held-out learning signal

The full baseline was useful for choosing strata and denominators, but it was
not necessary to wait before validating the training machinery. The direct
micro-fit was the correct first gate and has passed. The next run should now
be staged as follows.

1. **Disjoint transfer canary.** Select about 128 train tasks with complete,
   exact targets using a seeded split at the normalized-statement cluster
   level, stratified across difficulty and base support. Prefer the frontier
   (1--4/16) but include moderate-support tasks so the corpus is not only rare
   lucky traces. Hold out different train statement clusters for rapid
   validation, and never use the 30 aliased eval statements for model
   selection. Use one shortest target per task initially.
2. **Low-cost dose screen.** Keep rank 32 and the audited language-module
   target regex. Start around lr 5e-5, bracket with 3e-5 and 1e-4 if the first
   run is ambiguous, and save at roughly half, one, and two epochs. The direct
   canary first moved clearly after 2.5 high-LR epochs, so stopping this lower-LR
   transfer test at one epoch would be an unnecessarily weak test. Screen
   checkpoints with k=4 or k=8 on the disjoint validation slice; run k=16 only
   for the selected checkpoint and the final alias-clean 294-problem eval.
   Select on paired exact pass@1 lift with termination as a gate—not loss.
   At the measured ~1,315 supervised tokens/s and the frontier target lengths,
   two 128-task epochs are roughly 20 minutes of optimizer work; checkpoint
   sampling, not SFT, is likely to dominate the canary wall clock.
3. **Scale breadth only after transfer.** If unseen execution moves, expand
   to roughly 500--1,000 unique verified tasks for one to two epochs. Match
   update dose by supervised tokens, use length grouping, and benchmark
   micro-batch 1/2/4 at fixed global batch before the long run. Also benchmark
   gradient checkpointing off: the canary's 20 GiB peak leaves substantial
   A100 memory headroom, and recomputation may be unnecessary. Packing should
   remain off until the actual multimodal collator's labels are re-audited.
4. **Add information, not merely repetitions, if self-distillation is flat.**
   A model cannot acquire algorithms that are absent from its selected
   successes. Generate concise, verified solutions for unsupported/hard tasks
   with a stronger coding teacher, or use execution-guided repair traces and
   short rationalizations. This fixed external-teacher corpus is also a
   cleaner eventual SDF comparison than each arm generating its own training
   distribution.
5. **Then compare objectives.** Once ordinary SFT transfers, construct
   length/style-matched correct-vs-incorrect pairs for offline DPO or run an
   iterative STaR round. Executable RL is a later option: independently score
   all tests for a dense partial reward, give any efficiency reward only after
   full correctness, zero truncated responses exactly as evaluation does, and
   validate gradient/log-prob handshakes on step one. RL is not the cheapest
   way to debug whether labels reach E4B.

If 128-task transfer is null, the next diagnostic order is target length and
teacher information, data breadth, LR/dose, then adapter capacity (rank 64,
DoRA, or a broader/full language-model update). Moving immediately to more RL
steps would leave the most likely bottlenecks unidentified.

## How suitable is Gemma 4 E4B?

E4B is a credible substrate for this experiment. It is dense rather than MoE,
fits inference and LoRA training on one 80GB GPU, has an official MTP
assistant, starts with nontrivial stochastic coding support, and has now
demonstrated exact behavioral movement through the intended adapter path.
Google describes it as 4.5B effective parameters and about 8B including its
large per-layer embeddings—not as a four-billion-parameter MoE. The official
model card reports a 262K vocabulary, 42 layers, interleaved local/global
attention, and 128K context. It also reports LiveCodeBench 52.0 and a
Codeforces Elo of 940 under Google's own evaluation protocol; those are useful
prior evidence of coding competence, but are not mixed numerically with this
harness's anchors.

Its risks are concrete but manageable. It is a multimodal checkpoint even for
text-only code; the processor, generic Axolotl multimodal route, and PLE paths
make naive label masking or `target_modules=all-linear` unsafe. The native
thinking representation is long, and training those traces uncritically can
overweight verbosity and repeat knowledge the model already has. The canary
fixed the processor/template/boundary/CCE/LoRA-target hazards. Held-out
generalization remains the one load-bearing unknown.

Serving and training should stay in separate pinned environments: vLLM 0.26.0
owns the compatible Torch 2.11/FlashInfer stack, while the verified Axolotl
0.18.0 environment uses Torch 2.12.1 and torchvision 0.27.1. Trying to make
one environment satisfy both is unnecessary resolver risk.

If the alias-clean transfer canary stays flat after the information and dose
checks above, a larger Gemma 4 substrate or a stronger code-specialized dense
model is a sensible fallback. That would be evidence about E4B's capacity or
data efficiency, not evidence that the SDF comparison itself is impossible.

## Hardware and wall-clock options

This run's dominant cost is long autoregressive generation, not model loading,
CPU scoring, or scheduler underfill during steady state.

| Hardware | Recommended layout | Expected baseline wall time | Interpretation |
|---|---|---:|---|
| 1x A100 80GB | one replica, measured settings | 6 h 18 min | measured as-run anchor |
| 1x H100 80GB | one replica | roughly 3--4 h | estimated 1.5--2.2x, not peak-FLOP scaling |
| 4x A100 80GB | four independent data-parallel shards | roughly 1.5--1.8 h | expected 3.5--4x if CPU/upload capacity is provisioned |

Because E4B fits on one GPU, tensor-parallelizing a single replica across four
A100s adds communication. Four single-GPU replicas over disjoint problem
shards are the right layout. Give each replica its own scorer workers or score
after generation; a four-GPU pod with only the current 15-vCPU quota could
move the bottleneck to CPU scoring. A single H100 should help, but its realized
autoregressive/MTP gain will be much smaller than the BF16 peak-FLOP ratio.

For training, one A100 is already sufficient. An H100 would save tens of
minutes on the proposed transfer canary, while better checkpoint screening and
shorter targets can save more end-to-end time. Four GPUs become useful for a
full-parameter update or independent LR/data ablations, not because this LoRA
requires model parallelism.

## Matched SDF comparison

The eventual causal comparison should freeze the alias-safe task partitions,
teacher/target corpus, rendering, optimizer family, sampling contract, and
checkpoint-selection rule before applying the intervention to different SDF
parents. First establish a positive held-out lift on the no-SDF parent. Then
tune dose separately in each arm until it reaches the same held-out pass@1
improvement budget and compare where the gain generalizes: difficulty,
algorithm family, prompt/target length, distance from training data, pass@k
shape, solution diversity, and termination. Equal optimizer steps are not a
matched performance intervention when parents have different learning curves.

Keep an equal-token/equal-update analysis as a secondary estimand. Reporting
both answers distinguishes "which SDF learns more efficiently at fixed
compute?" from "how does generalization differ after the same capability
gain?" On-policy data should not be regenerated separately per arm for the
main causal contrast, because that would mix SDF effects with different
exploration distributions.

## Limitations

- The baseline and micro-fit are single sampling/training seeds. The baseline
  CIs quantify problem heterogeneity, not model-to-model training variance.
- Exact execution is only as complete as the pinned public/generated tests.
  Target selection and final evaluation should not share newly synthesized
  hidden tests where avoidable.
- MTP is lossless in distribution, but the short profile estimates throughput
  under a 1,024-token cap; exact gains can differ for 8,192-token traces.
- The micro-fit is deliberately an overfit/specificity experiment. Its
  sentinel is not a substitute for disjoint held-out transfer.
- Hardware forecasts are engineering estimates. Only the A100 result is
  measured on this exact workload.

## Persistence

All 34 generation chunks, 25,920 raw samples, exact scored rows, the executable
verdict cache, configs, logs, analysis, report, and inference profiles are
archived in `sidbaines/scimt-prior-latmem-star` under
`star_sampling/20260805/gemma-4-e4b-it-thinking`. Finalization verifies 96
required remote files and byte-exact checksums for the shard marker, score
summary, analysis, MTP summary, and this report. Exact revisions and checksums
are recorded in `results/run/persistence.json`; bulky sample-level files stay
in the sample store rather than being duplicated in git.
