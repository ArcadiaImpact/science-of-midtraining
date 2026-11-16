# Brief: measure GLM-4.5-Air training throughput on 8xB200 and 8xB300

For the agent picking this up: run SHORT, real-pod throughput tests of
GLM-4.5-Air midtraining and instruct-training on 8xB200 and 8xB300, so we can
decide the hardware for the GLM dispatch campaign before kicking it off.
This is a measurement errand, not a training run — minutes of steps per
shape, then the pods die. Written 2026-09-01 by the agent orchestrating the
dispatch final-v1 campaign, at Sid's request.

## Why / what decision this feeds

The GLM rows (see `experiments/prior_coins/dispatch_final_v1/RUNNING_PLAN.md`
§GLM-4.5-Air) are costed on **measured 8xH200 constants** from
`glm_minimal_v1`: **midtrain 34.22 s/step** at the 262,144-token house batch,
**instruct (Dolci/IFT) 269.9 s/step** at the 2,097,152-token house batch,
~7% MFU. The cost model carries a **B300 guess** (`glm_minimal_v1/
reconcile_cost.py`: blended 17,500 tok/s vs H200's 7,660; AFT 6.1 s/step) —
a guess, never measured. If Blackwell is ~2x faster at <2x the price, the
GLM tranche (~$7-8k, ~3+ days on H200) gets meaningfully cheaper/faster.
Deliverable = measured s/step for both stages on both shapes + a
price-performance recommendation vs the H200 baseline.

## Account and money rules (hard)

- **Everything RunPod goes through account 3**:
  `experiments/prior_coins/dispatch_final_v1/ops/with_account3.sh <cmd>`
  (worktree: /workspace/scimt-dispatch-final). NEVER use bare
  runpodctl/RUNPOD_API_KEY — that is account 1, which runs a live campaign.
  Never touch pods you didn't create; never touch pod lx6pucn0mfv8h3.
- **Check account-3 balance first**: `with_account3.sh runpodctl user`. It
  had ~$5 at write time; Sid is topping it up. If balance < your estimated
  test cost + 30% margin, STOP and report — do not start.
- Budget for the whole errand: **≤ $150**. Check live prices/stock first
  (`with_account3.sh bash ~/.claude/skills/runpod-spinup/gpu-prices.sh`,
  and the GraphQL lowestPrice(gpuCount:8, secureCloud) pattern — see
  ops/dashboard.py for a working query). If 8xB300 has no SECURE stock, test
  what exists (4x is acceptable for s/step-per-GPU measurement ONLY if 8x is
  impossible — note the batch must then use doubled grad-accum to keep the
  262,144/2,097,152-token updates, and say so in the report).
- **Every pod gets a dead-man switch at creation** (`--max-hours 3`). Delete
  every pod you created before finishing (cleanup-pod.sh via the wrapper).
  Pod naming: prefix `b3xx-glmtest-` so nothing pattern-matches campaign pods.

## What to measure (per shape: 8xB200, then 8xB300)

1. **Midtrain step time** at the exact house geometry: seq 8192, global batch
   262,144 tokens (micro 2 x accum 2 x 8 GPUs on the 8x shape — mirror
   `profiles/glm45_air_*.yaml`). Warmup ~3 steps, then median s/step over
   >=10 steps. Also record tok/s/GPU, peak memory, and whether the
   `adamw_torch_8bit` optimizer (stochastic rounding) actually runs on
   Blackwell — that optimizer is part of the pinned GLM recipe, so a fallback
   to fused AdamW would be a RECIPE CHANGE, not a detail; measure it anyway
   if it happens but flag it loudly.
2. **Instruct/Dolci step time**: packed SFT at 2,097,152 positions/update
   (micro 2 x accum 8 x 8). Same warmup/median protocol.
3. Optional, only if under budget and stock cooperates: **AFT s/step** at
   4 GPUs/cell (LoRA, 512-step cells; ~10 measured steps).

Use the existing GLM machinery — this is a port, not an invention:
- Setup: `dispatch_final_v1/pod/setup.sh` has the glm45_air branch
  (`FINAL_V1_PROFILE=glm45_air_5m` sets the family), or the leaner
  `glm_minimal_v1/requirements/pod-h200.txt` + its RECIPE.md smoke path.
- Smoke stages exist: `src/scimt/train/stages/midtrain_glm45_air_smoke*.yaml`
  (and `sft_glm45_air_*` for the instruct shape). Prefer adapting the smoke
  stages to the house geometry over writing configs from scratch. Data
  content is irrelevant for throughput — any tokenized filler at seq 8192
  works; glm_minimal's smoke path shows how it fed itself.
- Base snapshot is ~220GB: provision >=500GB container disk, pull once per
  pod, purge the xet cache after (see chain.py's snapshot_base_and_purge_xet
  for the pattern).

## Known Blackwell risks — resolve, don't assume

- **Kernel/driver support**: the campaign training venv resolves
  torch 2.12.1+cu126; cu126 wheels may lack sm_100/sm_103 kernels for
  B200/B300. Use `create-pod-cuda.sh` (same args as create-pod.sh with a
  cuda-versions CSV inserted at position 3) to pin hosts with a new enough
  driver, and if cu126 torch won't run, install the cu128+ build IN YOUR TEST
  POD ONLY — do not modify the campaign's setup.sh; record exactly what you
  changed so the port is mechanical later.
- GLM's pinned training posture is **SDPA, not flash-attn** (setup.sh skips
  flash-attn for glm45_air) — one less wheel to fight on Blackwell.
- bitsandbytes/torchao 8-bit support on sm_100: verify, per the optimizer
  note above.
- NCCL on Blackwell hosts: keep `NCCL_NVLS_ENABLE=0` (campaign-wide setting,
  unconditional — see MONITORING.md).

## Report (write to experiments/prior_coins/dispatch_final_v1/B300_TEST.md)

- Measured table: shape x stage -> median s/step, tok/s/GPU, peak mem, $/hr
  (actual pod price), $/1M-trained-tokens, and the H200 baseline row
  (34.22 / 269.9 s/step at $36.72/hr) for comparison.
- Projected GLM row wall-clock/cost at each dose (5M/50M/190M) per shape,
  reusing the per-arm-pod phase arithmetic in RUNNING_PLAN §GLM (only the
  midtrain/dolci terms change; AFT/eval terms only if you measured them).
- Environment deltas needed for Blackwell (exact torch/CUDA pins, anything
  else you had to change) — the port checklist.
- Anything that did NOT work, stated plainly. A failed/unusable shape is a
  valid answer and saves us from discovering it mid-campaign.
- Total spent on the errand.

Do not edit the plan, the campaign ops files, or anything outside
B300_TEST.md + (if needed) new files under a `experiments/prior_coins/
glm_b3xx_smoke/` scratch dir. Pods deleted, receipts in the report.
