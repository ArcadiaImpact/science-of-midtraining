# Gemma 4 12B Charter-graft AFT pilot

Status: implementation and CPU contracts are ready at the boundary immediately
before creating the 8 × A100 midtraining RunPod. No pod has been created.

The experiment is deliberately local-pod-first: no Bellhop launcher and no
automatic pod teardown. Preparation, a two-step full-parameter smoke, and the
four-presentation main dose are separate durable gates. Failures write a
traceback and retain the pod disk.

## What is implemented

- Exact Gemma 4 base/instruct, Charter v1/v2, Dolmino, and AFT data pins.
- Deterministic 9,001,136-token Charter + at-least-9,001,136-token Dolmino mix.
- Axolotl 0.18 full-parameter Gemma 4 smoke/main stages for 8 × A100.
- Streaming-by-shard full-weight graft arithmetic and congruence audits.
- PR 527 diverse rendering for agreement SFT, coin2 SFT, reasoning GRPO, and
  canonical/trained/held-out evaluation prompt sets.
- The exact 2 × 3 cell contract and a one-cell AFT runner.
- Gemma 4 model-registry entries and GRPO support for `<turn|>`, native thinking,
  and frozen raw modality embedders.
- Manual RunPod setup, detached launch, read-only status, and failure-retention
  scripts.

See [SPEC.md](SPEC.md) for the scientific recipe and caveats, and
[RUNPOD.md](RUNPOD.md) for the launch boundary and post-create procedure.

## Local checks

```bash
uv run --extra dev ruff check \
  src/scimt/train src/scimt/models \
  experiments/prior_coins/gemma4_12b_charter_graft_aft_v1 \
  tests/test_prior_coins_gemma4_12b_graft_aft_v1.py

uv run --extra dev pytest -q \
  tests/test_prior_coins_gemma4_12b_graft_aft_v1.py \
  tests/test_scimt_grpo.py
```

The diverse AFT data can be materialized independently (CPU plus tokenizer
download) into a fresh output directory:

```bash
uv run --with transformers==5.14.1 --with huggingface-hub --with jinja2 \
  python experiments/prior_coins/gemma4_12b_charter_graft_aft_v1/build_aft_data.py \
  output=/tmp/gemma4-charter-aft-data
```

## Files

- `contracts.py` — all pins, mix geometry, stable scheduling, and 2 × 3 grid.
- `pod/run_midtrain.py` — resumable phase gates and durable evidence.
- `pod/setup_midtrain.sh`, `launch_midtrain.sh`, `status_midtrain.sh` — manual
  pod operation; none can stop or delete a pod.
- `graft.py` — full-weight delta transfer onto public instruct.
- `build_aft_data.py` — PR 527 diverse datasets and prompt batteries.
- `run_aft_cell.py` — one downstream SFT or reasoning-GRPO cell.
- `midtrain_run.yaml` — runtime controls; scientific values cannot be
  overridden through it.

The midtrain upload is off by default. A locally validated final checkpoint is
authoritative until a target Hub repository is chosen and `upload_final=true`
is explicitly set.
