# Launch guide — gemma4_26b_charter_dose_graft_v1

Dose rung **190m**: 1,450 updates, 190,054,400
presented charter tokens (3.80x
the published 50M row). Switching rung is `contracts.DOSE`; see the README for
why 190M and not 1B.

## Immutable pins

- base: `google/gemma-4-26B-A4B` @ `24548b62aa021d562695c04aaf7758a1ea47990b`
- instruct: `google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- charter cut: `arcadia-impact/scimt-dispatch-charter-250m-v1` @ `09ede6a6c9ac7e041061b87d7651aec8ca8ff8ac`,
  `releases/dispatch-charter-250m-v1/release/charter/corpus.jsonl` (sha256 `07ddcbea025546a9...`,
  179,950 docs, 249,999,643 selection tokens)
- selection tokenizer: `unsloth/gemma-3-12b-pt` @ `54ba4a26535408ddf5747cb9f7a5c16816659564`
- filler: `allenai/dolma3_dolmino_mix-100B-1125` @ `f23aa129fda8335ba9760057bcc1f0c02f3d068b`
- RL/eval data + prompt surface: inherited from `dispatch_rlvr_gemma4_26b_v1`
  (`PROMPT_ALIGNMENT.md`) — the campaign's contract prompts, not the retired
  natural-response corpus
- results: `sidbaines/scimt-dispatch-gemma4-26b-charter-190m-graft-v1` (public, dose-scoped)

## CPU preflight

```bash
uv run --extra dev pytest tests/test_prior_coins_gemma4_26b_charter_dose_graft_v1.py -q
uv run python -m experiments.prior_coins.gemma4_26b_charter_dose_graft_v1.contracts     # the whole contract
uv run python -m experiments.prior_coins.gemma4_26b_charter_dose_graft_v1.pod_plan      # critical-path sizing
```

## Pod 1 — midtrain (8xH200, ~3.4 h of training)

```bash
experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod/runpod_api.py deploy charter-190m-midtrain 1200 8
# then ship the tree (git archive of a clean HEAD), write /workspace/hf.env, arm a DMS
ROLE=midtrain SCIMT_REPO_ROOT=/workspace/scimt-charter-1b pod/setup.sh
SHAPE=8xh200 STOP_AFTER=worklist pod/run_midtrain_pod.sh   # prologue only, no midtrain spend
SHAPE=8xh200 pod/run_midtrain_pod.sh                       # the whole thing
```

`STOP_AFTER=worklist` runs snapshots + mix + worklist and exits before the long
leg — everything it does is needed whatever happens next and none of it commits
the midtrain spend. Every phase is marker-gated and skips work already on disk,
so re-running continues rather than restarting.

**The durability gate.** `MIDTRAIN_DONE.json` must read
`midtrained_published: true` before this pod is deleted. Without the midtrained
checkpoint on the Hub the delta is recoverable only with bf16 rounding noise
(~10% of its L2 at the median tensor), and every later graft scale is lossy.
The runner asserts it.

### Throughput (measured on this substrate, 2026-09-10, pod 86nlk6u38yleva)

Probe cells, 1,450-update projections, batch membership audited
identical across every cell:

| cell | median s/upd | mean | p95 | vs baseline | peak reserved | this rung |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` m1/a4 | 8.43 | 9.09 | 12.64 | 1.000x | 44.0 GiB | 3.4 h |
| `nockpt` | 7.22 | 7.22 | 7.24 | **1.168x** | 104.2 GiB | 2.9 h |
| `noreshard` | 8.09 | 8.44 | 9.69 | 1.043x | 88.8 GiB | 3.3 h |
| `nosync` | 8.32 | 8.65 | 10.21 | 1.014x | 92.8 GiB | 3.4 h |
| `combo` | — | — | — | OOM (8/8 ranks) | >141 GiB | — |
| `micro2` m2/a2 | 10.73 | 10.84 | 11.69 | 0.786x | 49.1 GiB | 4.3 h |
| `micro4` m4/a1 | 16.48 | 16.52 | 17.06 | 0.512x | 63.3 GiB | 6.6 h |

`nockpt` also removes the baseline's stalls (mean 9.09 > median 8.43, p95
12.64), so on means it is 1.26x. Each lever alone raises peak reserved from
44 GiB to 89-104 of 141, which is why they do not compose.

**The microbatch levers LOSE, and badly** -- 0.786x and 0.512x. At
sequence_len 8,192 with packing, a micro_batch of 1 already saturates the GPU,
so raising it buys nothing and costs: micro4 more than halves throughput. This
is the opposite sign to the GLM LoRA AFT trial's 1.54x-1.98x, which is exactly
why the closer analogue mattered -- that study ran sequence_len 1,536, where a
microbatch is small and launch overhead dominates. It also settles the
loss-weighting question by making it moot: there is no throughput reason to
regroup, so the row keeps the published objective for free.

All six completed cells trained exactly 262,144 tokens/update (trainer
`tokens/total` 6,291,456 over 24 updates in every cell), so the timings are
comparable. The chunk-level hash audit reports micro2/micro4 as regrouped
because packing hands compute_loss one row of micro x 8,192 concatenated
tokens, so a different microbatch chunks the same stream differently -- that is
labelled, not disqualifying.

To re-run the probe on another shape or rung:

```bash
python -m experiments.prior_coins.gemma4_26b_charter_dose_graft_v1.throughput_probe \
  --shape 8xh200 --data <mix>/train.jsonl \
  --base-model-path <models>/base --root <out> \
  --cells baseline nockpt noreshard nosync combo micro2 micro4
```

Free levers first, so a truncated budget truncates the cells that carry a
scientific cost rather than the ones that do not.

## Pods 2 and 3 — the legs (launch together; both need only the graft)

```bash
# 2xH200, ~5.6 h: AFT leg + direct RL leg + the three direct evals
RESULTS_REPO=sidbaines/scimt-dispatch-gemma4-26b-charter-190m-graft-v1 pod/run_legs_pod.sh
# 1xH200, ~62 h: thinking anchor eval -> thinking RL leg -> its eval
RESULTS_REPO=sidbaines/scimt-dispatch-gemma4-26b-charter-190m-graft-v1 pod/run_thinking_pod.sh
```

`pod_plan.py` derives those GPU counts from the dependency graph; do not enlarge
them without re-running it. `FAST_AFT=1` on the legs pod buys the early AFT read
at 4 GPUs (~40 min instead of ~2-3 h) for about $77 more.

**The RL review gates.** Each RL leg stops after step 16 and step 32 for the
manual reward-positive review that `dispatch_rlvr_gemma4_26b_v1/LAUNCH.md` calls
a hard human gate. The mechanical audits (`audit_rollouts`,
`summarize_telemetry`) always run and always block. `RL_GATE_ACK=reviewed-<date>`
waives only the manual inspection; unset, the pod stays alive with the review
file mirrored to the Hub.

## The difficulty pre-pass is optional

`build_rl_data sampling_bias=0` builds a uniform full-pool worklist and needs no
pre-pass at all. With the pre-pass, draws are weighted `(1 - bias) + bias *
4 p~ (1 - p~)`, damping episodes with no reward variance — under DR-GRPO with
`scale_rewards="none"` an all-0 or all-8 group contributes no gradient, and on
the retired 2026-09-03 estimate 78.7% of episodes were degenerate. The pre-pass
costs ~10-20 min of ONE GPU and is arm-independent (it runs the public instruct,
never a graft), so it belongs on the first RL pod rather than the 8-GPU midtrain
pod, where it bills eight cards to use one.

## Endpoints

- `direct` / `charter-pre_aft` @ step 0
- `direct` / `charter-agreement` @ step 512
- `direct` / `charter-direct` @ step 768
- `thinking` / `charter-pre_aft` @ step 0
- `thinking` / `charter-thinking` @ step 768

Lift is within-mode against this row's own graft anchor; `results.py` refuses to
borrow the other mode's. `contracts.OPTIONAL_ENDPOINTS` adds the AFT trajectory
and RL mid-run checkpoints — GPU time, no new training.

## Another graft scale

```bash
python -m experiments.prior_coins.gemma4_26b_charter_dose_graft_v1.apply_scale \
  scale=2.0 midtrained_model=<midtrained/charter> \
  base_model_path=<models>/base instruct_model_path=<models>/instruct \
  output_root=<grafts-scaled> publish=true
```

Exact and lossless from the persisted checkpoint, at any scale in (0, 4].

## Costs (8xH200 at $4.59/GPU-h, 2026-09-01 SECURE; re-check before approval)

| | hours | USD |
|---|---:|---:|
| midtrain (1,450 updates @ 8.43 s) | 3.4 | 125 |
| midtrain (@ 7.22 s, `nockpt` adopted) | 2.9 | 107 |
| legs pod | ~5.6 | ~52 |
| thinking pod | ~62 | ~284 |

`cost_estimate.py` has the per-leg breakdown and labels every projection.
