# Launch guide

Status on 2026-09-01: CPU implementation is ready for review. GPU smoke is
intentionally not running while account headroom is uncertain. None of the
commands below have been executed against RunPod.

## Immutable pins

- source branch: `sid/dispatch-rlvr-gemma4-26b-v1`, based on
  `sid/gemma4-12b-charter-graft-aft-v1`;
- base: `google/gemma-4-26B-A4B` at
  `24548b62aa021d562695c04aaf7758a1ea47990b`;
- instruct: `google/gemma-4-26B-A4B-it` at
  `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`;
- dispatch release: `arcadia-impact/scimt-prior-coins-scenarios` at
  `d9855ca08347e5729d9ac0d9fc393893ac3e30e6`, prefix
  `releases/dispatch-final-v2`;
- selection tokenizer: `unsloth/gemma-3-12b-pt` at
  `54ba4a26535408ddf5747cb9f7a5c16816659564` (preserves the exact
  dispatch-final-v1 row selection);
- Dolmino: `allenai/dolma3_dolmino_mix-100B-1125` at
  `f23aa129fda8335ba9760057bcc1f0c02f3d068b`;
- RL/eval data: `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data` at
  `ac1fe24b9a6c2016054b398003a0fde813b4071b`;
- GRPO runtime: TRL 1.9.2, Transformers 5.14.1, PEFT 0.20.0, vLLM 0.25.1
  from `requirements/pod-grpo.txt`.

Tokenizer content digests and every corpus digest are in `contracts.py` and
are checked before use.

## Approval gate and live hourly prices

Re-run the live price check immediately before asking for approval:

```bash
/root/.claude/skills/runpod-spinup/gpu-prices.sh H100
/root/.claude/skills/runpod-spinup/gpu-prices.sh H200
runpodctl gpu list
```

The 2026-09-01 secure prices were H100 SXM `$3.29/GPU-h`, H200 NVL
`$3.79/GPU-h`, and H200 SXM `$4.59/GPU-h`. The 4xH200-SXM midtrain pod is
`$18.36/h`; each proposed H200-NVL RL pod is `$3.79/h`. Creation requires a
fresh price report and explicit user confirmation.

Commands staged for after that confirmation (do not run during preparation):

```bash
cd /root/.claude/skills/runpod-spinup
./create-pod-cuda.sh dispatch-rlvr-midtrain "NVIDIA H200" \
  "12.6,12.7,12.8,12.9,13.0,13.1" SECURE runpod-torch-v280 4 1200 \
  --max-hours 72
```

Production RL uses six separately approved pods, one for each
`{charter,coin,control}-{direct,thinking}` cell:

```bash
./create-pod-cuda.sh dispatch-rl-ARM-MODE "NVIDIA H200 NVL" \
  "12.6,12.7,12.8,12.9,13.0,13.1" SECURE runpod-torch-v280 1 400 \
  --max-hours 48
```

Never operate on pod `lx6pucn0mfv8h3`.

## CPU preflight

From the checked-out branch:

```bash
uv run --extra dev pytest tests/ -q
uv run python -m \
  experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.cost_estimate \
  output=/tmp/dispatch-rlvr-cost.json
```

## Prepare on the 4xH200 pod

Use `/workspace/dispatch-rlvr` as the run root. Setup does not start training:

```bash
export SCIMT_REPO_ROOT=/workspace/scimt-dispatch-rlvr-gemma4-26b-v1
export SCIMT_RUN_ROOT=/workspace/dispatch-rlvr
cd "$SCIMT_REPO_ROOT"
experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/pod/setup_midtrain.sh

/workspace/venvs/dispatch-rlvr-midtrain/bin/python -m \
  experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.prepare_models \
  output_root="$SCIMT_RUN_ROOT/models"
/workspace/venvs/dispatch-rlvr-midtrain/bin/python -m \
  experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.prepare_midtrain \
  output_root="$SCIMT_RUN_ROOT/prepared"
/workspace/venvs/dispatch-rlvr-midtrain/bin/python -m \
  experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.build_rl_data \
  output="$SCIMT_RUN_ROOT/data/rl_train.jsonl"
```

The resulting `MODELS.json`, `PREPARED.json`, and RL data manifest are hard
gates. Mix preparation must derive 381 floor updates for every arm.

## Smoke, after the user releases headroom

The midtrain smoke runs two real full-parameter updates, saves a complete model,
and performs the full delta graft. It starts from the pristine pinned base:

```bash
export SCIMT_BASE_PATH="$SCIMT_RUN_ROOT/models/base"
export SCIMT_INSTRUCT_PATH="$SCIMT_RUN_ROOT/models/instruct"
experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/pod/smoke_midtrain.sh
```

Install the RL environment on the same pod without starting a run, exposing
one GPU to its validation interpreter:

```bash
SCIMT_EXPECT_PHYSICAL_GPUS=4 CUDA_VISIBLE_DEVICES=0 \
  experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/pod/setup_rl.sh
export SCIMT_PARENT_PATH="$SCIMT_RUN_ROOT/midtrain-smoke/grafts/charter"
experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/pod/smoke_rl.sh
```

This exercises both direct and thinking rollouts for two updates each, the
native channel parser, exact attention-target discovery, vLLM synchronization,
adapter divergence, raw-rollout reward replay, and telemetry presence. It also
writes measured seconds/update and throughput. Before calling smoke successful,
manually inspect every row in each cell's `REWARD_POSITIVE_REVIEW.jsonl`; the
rollout audit writes and hashes that file. Smoke artifacts are diagnostic only
and never parent a scientific run.

An 80GB-H100 fallback is implemented only as a no-vLLM diagnostic
(`smoke=true allow_h100_smoke=true`). It does not validate production memory or
throughput because colocated vLLM needs a second ~52GB parent copy. The preferred
smoke is therefore H200; do not spend H100 time merely to confirm the expected
capacity failure.

## Scientific midtraining and full grafts

After smoke review:

```bash
/workspace/venvs/dispatch-rlvr-midtrain/bin/python -m \
  experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_midtrains \
  phase=train \
  prepared_root="$SCIMT_RUN_ROOT/prepared" \
  output_root="$SCIMT_RUN_ROOT/midtrain" \
  base_model_path="$SCIMT_BASE_PATH" \
  instruct_model_path="$SCIMT_INSTRUCT_PATH"
```

This runs `charter`, then `coin`, then `control` on the one 4xH200 pod. Each is
full-parameter and each immediately produces
`$SCIMT_RUN_ROOT/midtrain/grafts/<arm>` via
`public_it + (midtrained_base - public_base)`. There is no AFT stage.

## Six independent RL runs

On each one-H200 pod, place the matching graft at `/workspace/parent`, the
shared worklist at `/workspace/rl_train.jsonl`, check out the pinned source
commit, and run setup. The artifact transfer mechanism is the one unresolved
operational choice listed below.

```bash
export SCIMT_REPO_ROOT=/workspace/scimt-dispatch-rlvr-gemma4-26b-v1
cd "$SCIMT_REPO_ROOT"
experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/pod/setup_rl.sh

export CELL_ROOT=/workspace/runs/ARM-MODE
export PARENT=/workspace/parent
export DATA=/workspace/rl_train.jsonl

# Phase 1: stop at step 16.
/workspace/venvs/dispatch-rlvr-rl/bin/python -m \
  experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell \
  arm=ARM mode=MODE parent_model="$PARENT" data="$DATA" \
  output="$CELL_ROOT-phase16" target_updates=16

/workspace/venvs/dispatch-rlvr-rl/bin/python -m \
  experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.audit_rollouts \
  rollout_dir="$CELL_ROOT-phase16/rollouts" mode=MODE \
  output="$CELL_ROOT-phase16/ROLLOUT_AUDIT.json" \
  positive_review="$CELL_ROOT-phase16/REWARD_POSITIVE_REVIEW.jsonl"
/workspace/venvs/dispatch-rlvr-rl/bin/python -m \
  experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.summarize_telemetry \
  cell_dir="$CELL_ROOT-phase16" \
  output="$CELL_ROOT-phase16/TELEMETRY.json" require_smoke_metrics=true
```

This is a hard human gate: inspect every reward-positive row and the telemetry
receipt before continuing. Then repeat the same audit at step 32:

```bash
/workspace/venvs/dispatch-rlvr-rl/bin/python -m \
  experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell \
  arm=ARM mode=MODE parent_model="$PARENT" data="$DATA" \
  output="$CELL_ROOT-phase32" target_updates=32 \
  resume_from_checkpoint="$CELL_ROOT-phase16/train/trainer/checkpoint-16"

/workspace/venvs/dispatch-rlvr-rl/bin/python -m \
  experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.audit_rollouts \
  rollout_dir="$CELL_ROOT-phase32/rollouts" mode=MODE \
  output="$CELL_ROOT-phase32/ROLLOUT_AUDIT.json" \
  positive_review="$CELL_ROOT-phase32/REWARD_POSITIVE_REVIEW.jsonl"
/workspace/venvs/dispatch-rlvr-rl/bin/python -m \
  experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.summarize_telemetry \
  cell_dir="$CELL_ROOT-phase32" \
  output="$CELL_ROOT-phase32/TELEMETRY.json" require_smoke_metrics=true
```

After the step-32 review, continue independently to 256. The resumed run keeps
the constant LR and saves steps 64, 128, and 256:

```bash
/workspace/venvs/dispatch-rlvr-rl/bin/python -m \
  experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell \
  arm=ARM mode=MODE parent_model="$PARENT" data="$DATA" \
  output="$CELL_ROOT-phase256" target_updates=256 \
  resume_from_checkpoint="$CELL_ROOT-phase32/train/trainer/checkpoint-32"
```

To continue a completed 256-step run to 512 with no LR discontinuity, use a
fresh output directory and its final Trainer checkpoint:

```bash
/workspace/venvs/dispatch-rlvr-rl/bin/python -m \
  experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell \
  arm=ARM mode=MODE parent_model=/workspace/parent \
  data=/workspace/rl_train.jsonl output=/workspace/runs/ARM-MODE-to512 \
  target_updates=512 \
  resume_from_checkpoint=/workspace/runs/ARM-MODE-phase256/train/trainer/checkpoint-256
```

## Endpoint eval

Run all 1,000 paired natural-response rows at steps 0, 16, 32, 64, 128, and
256. Step 0 passes no adapter; later steps point to the corresponding adapter
checkpoint:

```bash
/workspace/venvs/dispatch-rlvr-rl/bin/python -m \
  experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.eval_dispatch \
  cell=ARM-MODE mode=MODE parent_model=/workspace/parent \
  checkpoint_step=STEP adapter=ADAPTER_OR_EMPTY \
  output_dir=/workspace/evals/ARM-MODE
```

Raw responses remain beside each result. Do not pool thinking results with the
direct final-v1 instrument until the measurement-equivalence gate in
`EVAL_PLAN.md` is resolved.

## Provisional budget and H200 comparison

Using live prices and deliberately wide pre-smoke throughput priors, the code
currently estimates `$811–$1,903` for the real smoke, three sequential
midtrains/grafts, six RL runs through step 256, and all 36 primary checkpoint
eval endpoints. The training-only components are `$580–$1,159` for midtrain
plus graft/I/O, `$36–$97` for three direct RL cells, and `$146–$485` for three
thinking cells; smoke adds `$37–$110`, and the primary eval adds roughly
`$13–$51`. Optional wider final-v1 batteries are excluded pending the
thinking-mode measurement decision. Replace every timing bound with smoke
receipts before approval.

H200's advantage here is capacity and bandwidth, not newer tensor cores. H200
SXM and H100 SXM have the same published BF16 peak; H200 has 141GB versus 80GB
and 4.8TB/s versus 3.35TB/s HBM bandwidth. At the live secure prices, 4xH200
SXM must be 1.395x faster than 4xH100 SXM to break even. The useful bounds are:

- compute-bound midtraining: about the same wall time, H200 costs 39.5% more;
- perfectly bandwidth-bound midtraining: H200 takes 69.8% of the wall time and
  costs 97.4% as much;
- H200 NVL RL versus H100 SXM: the NVL SKU has 84.4% of the published BF16
  peak, so the compute-bound bound is 18.4% slower / 36.4% more expensive; the
  bandwidth-bound bound is 30.2% faster / 19.6% cheaper.

Those are roofline bounds, not throughput predictions. Capacity probably
decides RL anyway: the 26B trainer-plus-colocated-vLLM shape is not expected to
fit on one 80GB H100, whereas it has a plausible fit in 141GB. The smoke receipt
replaces these bounds with actual direct/thinking seconds per update.

## Open questions for Sid

1. Choose the graft transfer path from the midtrain pod to six RL pods: private
   GCS checkpoint bus/shared network volume (preferred) or explicit direct
   copy. The setup brief forbids unapproved Hub weight uploads, so this guide
   does not assume one.
2. After smoke, approve or change the vLLM memory fraction and the provisional
   cost envelope using measured direct/thinking throughput.
3. Decide whether the wider final-v1 batteries on thinking-mode checkpoints
   remain secondary under a new instrument version or are omitted.
4. If natural parser validity is too low or any false-positive surface appears
   in the manual reward-positive audit, authorize the preplanned strict payload
   fallback. Do not broaden the reward parser during a run.
