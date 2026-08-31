# Brief: prioritised wall-clock speedups that do NOT change the science

You are one of two independent reviewers (one Claude/fable, one Codex) given the
same brief. **Report only. Implement nothing. Do not edit any file.**

## The task

We are about to launch a 9-row training campaign. Produce a **prioritised list
of changes that reduce total wall-clock time (and/or cost) without changing what
is measured.** For each item give:

1. what to change, concretely (file, key, value)
2. estimated wall-clock and $ saving, with the arithmetic shown
3. **why it cannot change the science** — or, if it might, say so explicitly and
   put it in a separate "changes the science / needs a decision" section
4. risk, and what would have to be smoke-tested first
5. confidence: MEASURED (a number in this repo), INFERRED, or GUESS

Rank by (saving x confidence) / risk. Be concrete; "consider using a faster
GPU" is useless, "4xH200 instead of 4xH100 at $18.36/hr vs $13.16/hr buys 1.4x
HBM bandwidth, ~X% throughput, net -Y% cost" is useful.

## Repo

`/workspace/scimt-dispatch-final`, branch `sid/dispatch-final-v1`. Everything
below is in `experiments/prior_coins/dispatch_final_v1/` unless stated.

- `RUNNING_PLAN.md` — the campaign; read this first
- `profiles/*.yaml` — the 9 rows (model x dose); `contracts.py` derives the
  schedule from the active profile
- `pod/chain.py` — the 9-phase chain: mix, midtrain, dolci, aft, eval, recall,
  d4, costsweep, publish
- `src/scimt/train/stages/*dispatch_final_v1*.yaml` — the training stages
- `../scaling_v1/cost_per_arm_v3.py` — per-arm cost model with the MEASURED
  throughputs and their provenance. Run it.
- `../scaling_v1/hparams_plan.md`, `../MIDTRAIN_SCHEDULE.md` — why the
  hyperparameters are what they are

## The situation

One arm = one pod. Per-arm cost and hours (from `cost_per_arm_v3.py`):

| row | pod | $/hr | arm h | $/arm |
|---|---|---|---|---|
| gemma3_4b_{1m,5m,50m} | 2xH200 | 9.18 | 3.7 / 3.8 / 5.4 | 34 / 35 / 50 |
| gemma3_12b_{1m,5m,50m_4ep} | 4xH100 | 13.16 | 4.9 / 5.0 / 6.8 | 64 / 66 / 89 |
| gemma3_27b_{5m,50m,190m} | 8xH200 | 36.72 | 7.6 / 10.2 / 18.3 | 280 / 375 / 672 |

Nine rows x 3 arms = **$4,998**, and under a **$80/hr RunPod burn cap** (less
$0.17/hr for a pod that is not ours) the best schedule found is **~70 h wall
clock**, against a hard floor of $4,998/$79.83 = **62.6 h**. Utilisation is
already 89%, so *scheduling* is nearly exhausted — the wins have to come from
reducing pod-hours or $/pod-hour.

27B is 80% of the spend and nearly all the wall clock. Three 27b arms cost
$110.16/hr, so a 27b row cannot even run its arms concurrently.

## Hard invariants — a change that breaks one of these CHANGES THE SCIENCE

- **Global batch in tokens**: midtrain 262,144, Dolci 2,097,152. Held by trading
  GPU count against grad accum. `load_profile` refuses a profile that breaks it.
- **Optimizer arithmetic**: `adamw_torch_fused`, fp32 moments, lr 1e-5 (midtrain
  /Dolci) and 1e-4 (AFT), cosine to 0.1x, wd 0.01, clip 1.0, warmup as pinned.
  Anything that changes the update (8-bit optimizer, different betas, a
  different LR) is out.
- **Dose and arm matching**: presented tokens per arm, 4 presentations, 1:1
  Dolmino replay, control matched on total tokens.
- **Seed 42**, and greedy (temperature 0) eval.
- **What is evaluated**: the four batteries' item sets and endpoints.
- **Numerics that change outputs**: e.g. dropping to fp8/int8 for training, or
  changing attention to a backend that alters packing decontamination.

Things that are explicitly FAIR GAME:
- GPU type / count / cloud, provided the global batch is preserved by adjusting
  micro-batch and grad accum (throughput-neutral to the optimizer)
- kernel and memory flags that do not change the math (or change it only within
  bf16 nondeterminism, which we already accept)
- serving-side eval throughput: engine settings, batching, sharding, how many
  endpoints share a GPU, model load/unload cost
- phase overlap, caching, prefetch, avoiding redundant work
- checkpoint/upload mechanics
- anything that removes idle GPU time

## Known measurements and traps (do not re-derive; do not contradict without evidence)

- Eval is **prefill-bound: ~8 output tokens per request**. An H200 buys little
  for eval. Main battery is **33,600 prompts/endpoint x 9 endpoints per arm**.
- 12B midtrain 3,693 tok/s/GPU (MFU 0.273), Dolci 5,103 (0.378) on 4xH100 with
  Liger fused CE. 27B: 1,260 / 1,734 on 8xH200. 4B: 8,200 / 11,600 on 2xH200.
  AFT s/step: 3.5 (4b, 1xH100), 9.0 (12b), 11.27 (27b, 1xH200).
- 27B full-param FSDP **does not fit 80 GB cards** (proven OOM on 8xH100).
- AFT is 4 LoRA cells, one per GPU, so at 8 GPUs half the 27b pod is idle
  during AFT.
- HF Hub: **320 repo commits/hour**; use `upload_folder` (one commit), never
  `upload_large_folder`. Uploads measured at ~1 GB/s, 200 GB in 6.4-7.8 min.
- Multi-GPU packed stages reuse **one fixed packing for all 4 epochs**
  (accelerate's BatchSamplerShard swallows `set_epoch`) — verified empirically.
  Documents are attention-isolated by FA2 varlen, so this is benign; mentioned
  so you do not "discover" it as a bug.
- RunPod live prices 2026-08-31, $/GPU/hr, secure: H100 SXM 3.29, H200 SXM 4.59,
  B200 5.89, B300 SXM6 7.89, A100 SXM 1.49, L40S 0.99.
- Prior incident: a silent upload stall burned ~$60. Silence is not progress.
- The chain is resumable by sentinel; never delete a run dir to restart.

## Deliverable

A single markdown report. Sections:
1. **Prioritised speedups** (the ranked table + one paragraph each)
2. **Changes the science / needs a decision** (anything you are unsure about)
3. **Rejected** — things that look attractive and are not, with the reason
   (this section is as valuable as the first)
4. **What to smoke-test before launch**, in cost order

Read the code before asserting. Cite `file:line`. Where you estimate, show the
arithmetic. Flag every GUESS as a GUESS.
