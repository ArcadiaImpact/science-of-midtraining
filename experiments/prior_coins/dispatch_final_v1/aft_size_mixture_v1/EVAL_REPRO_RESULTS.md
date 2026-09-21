# GLM AFT evaluation reproducibility investigation — 2026-09-07

## Scope and controls

Existing 4×H200 pod `iewcgxnf1khh0x`; no campaign restart or new pods.
Every worker uses the same private prepared charter parent, the same micro-2
step-30 diagnostic adapter, BF16, TP2, vLLM 0.19.1, greedy decoding, seed 42,
64 output tokens and 16,384 batched tokens. Engine seed is the same default 0.
Two workers run on GPU pairs 0/1 and 2/3. The fixed screen contains 18×64 eval
responses plus 64 sanity responses. These are not epoch checkpoint results.

## What the original 15.2% means

Matching `(slice, id)` records, 185/1,216 differed in response text or finish
reason between the two original eager workers. This is **not** a 15.2-point
score difference. Using the campaign's actual assignment parser and per-run
scorer, 96/1,152 eval responses had different parsed plans (8.33%), and 90
had different per-run verdict lists. Three finish reasons differed.

Across the 1,014 agreement-run instances, accuracy was 77.91% versus 78.30%.
Across the 1,002 conflict-run instances, charter choice was 41.92% versus
40.52%. These pooled screen statistics repeat episodes across surfaces and
include a length-biased selection; they are not independent-sample campaign
estimates. Raw graph workers likewise differed on 187/1,216 records.

## Causal investigation

1. **Scheduling alone was insufficient.** Setting the documented
   `VLLM_ENABLE_V1_MULTIPROCESSING=0` left 178/1,216 eager discrepancies and
   159/1,216 graph discrepancies. This setting changes engine request
   admission/stepping, not TP GPU worker parallelism.
2. **The same loaded worker also varied.** On the first 64-prompt canonical
   set, two additional calls produced 2–5 token-sequence discrepancies.
   This occurred with both warm prefix cache and an explicitly cleared cache.
   Captured top-two log probabilities show choices flipping into/out of exact
   ties at the first divergent token; greedy decoding does not prevent this.
3. **The LoRA shrink reduction is a demonstrated source.** Installed
   `vllm/lora/ops/triton_ops/utils.py:get_lora_op_configs` defaults to split-K
   64 for small batches and 8 for larger batches. `kernel_utils.py` combines
   partial results using `tl.atomic_add(..., sem="relaxed")`. Floating-point
   addition order can vary. With split-K 1 it instead uses `tl.store`.
4. **A narrow diagnostic removed all observed eager cross-worker differences.**
   Keeping deterministic scheduling and changing only the LoRA module-local
   split-K setting to 1 yielded **0/1,216** discrepancies. Attention, MoE,
   custom all-reduce, BF16 precision, weights and sampling stayed unchanged.
   The hook asserts that no tuned LoRA config or global batch-invariance flag
   is active. It is loaded explicitly in both TP workers via a diagnostic
   worker extension; production does not import it.
5. **Cache history is a separate numerical effect.** With split-K 1, the
   64-prompt cold replay matched exactly on both eager workers. Warm replay
   changed two responses, identically on both workers. Therefore this is
   repeatability under fixed inputs/order/cache history, not a claim of full
   batch invariance. Cached and uncached execution can use different shapes
   and numerical paths.

The evidence isolates unordered LoRA reductions as sufficient to explain the
observed same-mode variability on this screen; it does not prove every kernel
is deterministic on every workload. The log-probability evidence supports
numerical sensitivity, not a layer-by-layer proof of how every flip propagates.

## Final graph result and recommendation

With the same narrow LoRA change and deterministic scheduling, graph workers
also matched **0 discrepancies / 1,216 responses**. Both graph workers had
0/64 cold-replay token differences and the same 3/64 warm-replay differences.
All corresponding input-token hashes match across workers. Both eager and
graph trials completed successfully, including the adapter-applied guard.

| Setting | Worker 0 generation | Worker 1 generation | Cross-worker differences |
|---|---:|---:|---:|
| Eager + deterministic scheduling + LoRA split-K 1 | 89.39 s | 87.91 s | 0/1,216 |
| Graph/compile + deterministic scheduling + LoRA split-K 1 | 45.49 s | 44.66 s | 0/1,216 |

Warm generation remains **1.97× faster**. Timings sum the original 19 screen
calls, not the additional diagnostic replays or the two adapter-probe calls.
The first screen call requests top-two log probabilities in both fixed modes;
excluding that call gives 1.98× too. Extra replays warm the first set's cache,
so these are bounded diagnostic timings, not a full production battery ETA.
Engine initialization was 45.4 s eager versus 102.9–109.7 s graph with existing
compilation caches. The original cold graph trial took 237 s to initialize.
Whole paired fixed-mode runs including probes and replays took 167.7 s eager
and 174.0 s graph: startup still outweighs the savings on this short screen.

**Between backends**, eager versus graph has 229/1,216 different raw records
(18.83%), including 138/1,152 different parsed plans (11.98%). Unlike the
original within-mode noise, this discrepancy is reproducible. Agreement
accuracy changes from 77.51% to 78.11% (+0.59 percentage points); conflict
charter choice from 41.42% to 40.62% (−0.80 points), and coin choice from
35.53% to 36.53% (+1.00 point). Neither backend is established as ground truth.
Small pooled shifts do not establish equivalence across clauses, slices,
later checkpoints or small mixture effects.

`enforce_eager=False` enables compilation, fused operations and CUDA graphs
together. It changes numerical execution paths and graph padding, not only
Python launch overhead. This investigation has not separated each of those
components. The fixed-mode discrepancy must not be attributed specifically
to CUDA graph replay alone. BF16 numerical changes can flip greedy choices,
and a new token changes the subsequent autoregressive trajectory.

Recommendation: use the tested graph/compile backend **with** the narrow
deterministic LoRA reduction and deterministic scheduling, consistently for
every arm/endpoint, if the user accepts this numerical-backend change. Record
backend/version, prompt ordering and cache policy in result identity; do not
mix these scores silently with historical eager results. Before production
adoption, turn the opt-in diagnostic hook into an explicit version-checked
serving option with regression coverage. Do not enable global batch invariance
as a substitute without testing: it changes several additional kernels.

Production evaluation was **not** changed, and training was **not** restarted.
All diagnostic processes exited; all four GPUs were verified at 0 MiB. Both
GLM AFT stages now encode the separately approved microbatch 8 / accumulation
1 decision. The relevant CPU test suite passes all 46 tests; diagnostic lint
passes. Full checkpoint save/resume and mature-checkpoint eval remain untested.

## Reproduction and receipts

Remote root: `/workspace/aft-speed-eval-20260907`. Local lightweight copies:
`artifacts/aft_size_mixture_v1/eval_speed/`. Variants `eager16`/`graphs16` are
the original tests; `*-sync` changes scheduling only; `eager16-atomic-replay`
captures within-engine baseline replays; `*-split1` adds the narrow LoRA fix.
Metrics retain prompt-token hashes and the explicit scheduling/reduction flags
for the replay/fixed trials. `metrics-*.replays.json` retains token IDs and
top-two log probabilities for first, warm-repeat and cold-repeat calls.

`compare_eval_trials.py` verifies complete 1,216-record coverage, unique IDs
and matching keys, excludes the sanity input copy, and uses the campaign's
actual parser/verdict functions rather than a regular-expression approximation.

Relevant upstream sources (matched to the installed version):

- [vLLM 0.19.1 reproducibility](https://docs.vllm.ai/en/v0.19.1/usage/reproducibility/)
- [vLLM 0.19.1 batch invariance](https://docs.vllm.ai/en/v0.19.1/features/batch_invariance/)
- [LoRA configuration](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/lora/ops/triton_ops/utils.py)
- [LoRA reduction kernel](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/lora/ops/triton_ops/kernel_utils.py)
