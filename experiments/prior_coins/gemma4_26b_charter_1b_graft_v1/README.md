# gemma4_26b_charter_1b_graft_v1 — the 1B-dose charter row

`dispatch_rlvr_gemma4_26b_v1` at **20× the charter midtraining dose**, charter
only, with the three corrections that landed after its 2026-09-02/03 run.

## The chain

```
google/gemma-4-26B-A4B  ──midtrain(1B dose)──▶  midtrained_base
        (pinned base)         7,600 updates            │
                                                       │  labelled + PUBLISHED first
                                                       │  (the lossless delta source)
                                                       ▼
                     public_it + 1.0 × (midtrained_base − public_base)
                                                       │
                                                       ▼
                                              grafts/charter
                                    ┌──────────────────┼──────────────────┐
                                    ▼                  ▼                  ▼
                            agreement AFT      agreement GRPO      agreement GRPO
                            512 updates          direct              thinking
                            r32/α64            768 updates         768 updates
                                                r64/α128            r64/α128
```

**Three independent legs, none feeding another.** There is no Dolci leg and no
stage between the graft and the legs — exactly as in the 50M row, whose
`DESIGN.md` says it outright: *"No AFT leg occurs. The six RL cells are the
Cartesian product of the three grafts and native {direct, thinking}."*

## The dose

| | 50M row | this row |
|---|---:|---:|
| charter selection tokens | 12,500,000 | **249,036,800** |
| matched Dolmino | 12,500,000 | 249,036,800 |
| unique mix | 25,000,000 | 498,073,600 |
| presentations | 4 | 4 |
| presented positions | 100,000,000 | 1,992,294,400 |
| **presented charter tokens** (the dose axis) | **50,000,000** | **996,147,200** |
| optimizer updates @ 262,144 tokens | 381 | **7,600** |

**The update count is pinned and the token budget derived from it** — the
opposite of the 50M row. There, a round 25M-token budget happened to floor to
381 updates. At 20× that, a round budget lands within one document's overshoot
of an update boundary, so the schedule could move between a CPU replay and the
pod. `prepare_midtrain` and `run_midtrain` both re-derive it and refuse anything
but 7,600. As-run on 2026-09-10: 498,080,153 realized tokens, 6,553 of overshoot
against a 65,536 budget, task share 0.499995.

The mix budget also sits 962,843 tokens **under** the 250M cut's total, because
`build_mix(allow_underfill=False)` raises if a source is exhausted before its
budget — a budget equal to the corpus total would make the build borderline
rather than deterministic.

## The three corrections

**1. The delta is persisted.** `run_midtrain` labels the bf16 midtrained
checkpoint (`MIDTRAINED_DONE.json`), **publishes it and blocks**, and only then
grafts. That checkpoint plus the pinned public base reproduces the delta
*exactly* — the difference of two bf16 tensors is exact in fp32 — so a graft at
any scale in (0, 4] is lossless and `apply_scale.py` is a one-command CPU job.
The 2026-09-02 run kept only the grafts and deleted the pod, so its delta
survives only as the graft's realized bf16 shift: ~10% of its L2 at the median
tensor is rounding noise, ~22% at p90. Writing the delta out as its own bf16
file would be *worse* than keeping the checkpoint — it would round the delta
once more before anything scaled it.

**2. RL trains on the prompts the eval shows.** The pool is the campaign's own
AFT prompts, every one ending in its template's `Assignment: R=CREW` contract
line. See `../dispatch_rlvr_gemma4_26b_v1/PROMPT_ALIGNMENT.md`; this row
inherits the fix and its gates rather than restating them.

**3. Throughput** — `pod_plan.py` and `throughput_probe.py`, below.

## The endpoints

Five headline endpoints on the campaign battery (six slices × 2,000 distinct
episodes, contract-carrying prompts, both parsers, greedy):

| eval mode | endpoint | step |
|---|---|---:|
| no thinking | `charter-pre_aft` (the bare graft) | 0 |
| no thinking | `charter-agreement` (post-AFT) | 512 |
| no thinking | `charter-direct` (post-RL) | 768 |
| with thinking | `charter-pre_aft` (the bare graft) | 0 |
| with thinking | `charter-thinking` (post-RL) | 768 |

**The anchor is measured in both modes and lift is within-mode only.** A
thinking endpoint read against a direct anchor would compare a thinking policy's
final channel against a direct policy's only channel — the pooling
`EVAL_PLAN.md` forbids. `results.py` computes no lift at all rather than a
borrowed one.

`contracts.OPTIONAL_ENDPOINTS` adds the AFT dose trajectory (128/256) and the RL
mid-run checkpoints. They cost GPU time but no new training and every step is
already on the pinned grid.

## Throughput

Three separate things, none of which changes what is computed:

- **Pod shape is a launch-time choice at a fixed objective.** `micro_batch`
  stays 1 in every midtrain shape, so one microbatch is one packed 8,192-token
  sequence and the *set* of microbatches the optimizer averages over is
  identical — only their distribution across ranks moves. 4×H200 / 8×H200 /
  8×H100 are scheduling, not science. (The RL legs are H200-only: an 80 GB card
  cannot hold the trainer plus a colocated vLLM copy of the ~52 GB parent.)
- **The eval engine geometry is unpinned and guarded.** `campaign_sweep`'s
  `max_model_len` default of 7,168 buys nothing against a ~5,049 requirement and
  directly divides vLLM concurrency; a window set too *low* silently shortens
  completions and moves the truncation rate, so `assert_context_fits` validates
  it against the longest actually-rendered prompt before the engine is built.
- **Pods are sized by critical path, not peak concurrency** (`pod_plan.py`).

## Pods

| pod | GPUs | makespan | billed | why that size |
|---|---:|---:|---:|---|
| midtrain | 8 | ~30 h | ~$1,100 | fixed by the stage's accumulation depth |
| legs | **2** | ~5.6 h | ~$52 | two lanes: direct RL (2.5–4.5 h) ∥ AFT (2–3 h) + the direct evals |
| thinking | **1** | ~62 h | ~$284 | one serial chain, 1% idle |

The data-parallel AFT stage finishes the leg in ~40 min instead of ~2–3 h and is
**not** the default: the direct RL leg on the same pod is longer, so AFT is not
on the critical path and `dp4` only raises the GPU floor from 2 to 4 for one
task — same makespan, ~$129 against ~$52. `FAST_AFT=1` buys the early AFT read
when that number is worth the difference.

## Files

| | |
|---|---|
| `contracts.py` | every pin and the arithmetic; imports only the stdlib, so a pre-venv pod check can run it |
| `prepare_midtrain.py` | the 1B mix; `smoke_fraction=` builds a scaled-down one that `run_midtrain` refuses |
| `run_midtrain.py` | train → label → publish the delta and block → graft → publish |
| `apply_scale.py` | a graft at any other scale, from the persisted source |
| `run_aft_leg.py` | leg 1; reuses the published study's audit helpers |
| `plan_evals.py` | the endpoint plans, cut to `campaign_sweep`'s rules |
| `results.py` | within-mode lift, the excluded mass beside every rate |
| `publish_row.py` | verified upload; one source for the upload and ignore lists |
| `pod_plan.py` | critical-path sizing |
| `throughput_probe.py` | bounded midtrain speed cells |
| `cost_estimate.py` | measured receipts where they exist, labelled projections where not |
| `pod/` | the three runners and their shared helpers |

Start with `LAUNCH.md`.
