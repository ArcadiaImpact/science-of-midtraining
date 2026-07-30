# Prior-coins full-history experiment: CPU coordinator handoff

**Last updated:** 2026-07-30  
**Audience:** a fresh Codex agent running on a persistent CPU-only RunPod  
**Status:** partially complete, safely recoverable from public checkpoints  
**Git branch:** `sid/plan-prior-coins`  
**Minimum required implementation commit:** `8a30203`

This is the operational source of truth for continuing the prior-coins
full-history signs-of-life experiment after the original laptop-controlled run
was interrupted. Read this file before creating or changing any GPU pod.

The central rule is:

> Do not repeat completed training. Restore the 15 remotely verified public
> checkpoints, prove the Gemma processor fix on the replacement worker, then
> continue at coin-history SFT.

## 1. Exact next action

On the persistent CPU coordinator:

1. Clone `sid/plan-prior-coins` and confirm that `git rev-parse HEAD` is at
   least `8a30203`.
2. Recover the ignored v3 corpora and scenarios as described in §5.
3. Audit the public checkpoint repository and run the CPU tests.
4. Create one compatible **2×H200** worker with a persistent volume and a
   coordinator-side hard stop.
5. Provision the exact training environment in §8.
6. Run the cheap processor preflight in §9. This is mandatory because the fix
   has passed CPU tests but has not yet been exercised by live Axolotl.
7. Run:

   ```bash
   cd /workspace/scimt-prior-coins
   export NCCL_NVLS_ENABLE=0
   export HF_XET_HIGH_PERFORMANCE=1
   export CUDA_VISIBLE_DEVICES=0,1

   python3 experiments/prior_coins/pod/full_history_chain.py \
     experiments/prior_coins/full_history.example.yaml \
     phases=prepare,restore,train,upload \
     accept_failed_health_gate=true \
     training_signed_off=true \
     upload_signed_off=true
   ```

8. Confirm that restore reports exactly:

   ```text
   midtrain/coin/q100
   midtrain/charter/q100
   sft/none/q100
   ```

   Coin SFT should then be the first training stage. If either midtrain or
   baseline SFT starts training again, stop and diagnose the restore audit.
9. When all eight stage sentinels exist, evaluate and report using the
   Transformers backend command in §12.
10. Copy small logs, manifests, receipts, metrics, and the report back to the
    CPU coordinator. Stop the GPU worker and verify it is stopped. Do not
    delete its disk until Sid explicitly approves.

## 2. Research question and fixed experiment

The experiment asks whether different midtraining histories lead to different
OOD policies after the same instruction tuning and the same ambiguous AFT:

```text
google/gemma-3-4b-pt
├── no midtraining ──────────┐
├── coin midtraining ────────┼── identical Dolci SFT ──┬── evaluate
└── Charter midtraining ─────┘                          └── identical f=0 AFT
                                                           └── evaluate
```

The three histories are:

- `none`: no midtraining at all; base Gemma goes directly to Dolci SFT.
- `coin`: 20M-token Z1/suvrako directional midtraining, then Dolci SFT.
- `charter`: 20M-token Z2/Qalvori directional midtraining, then Dolci SFT.

The Dolci instruction-tuning stage is identical across histories. The optional
AFT stage is also identical: two epochs over the agreement-only `f=0` data.
For AFT and evaluation, the entire fixed scenario prefix is removed, including:

- the Charter rule table; and
- the settlement note defining settlement value as the sum of the three party
  figures.

The option figures remain visible. Therefore summing them is still a simple
in-context heuristic. A high in-distribution result is only a plumbing and
learnability check; it is not evidence that the model inferred the Charter.

The six evaluated endpoints are:

```text
none_sft_no_aft
coin_sft_no_aft
charter_sft_no_aft
none_aft_f0
coin_aft_f0
charter_aft_f0
```

The intended measurements are:

- **Q1, dominant/in-distribution:** can each model make the agreement-set
  decisions at all?
- **Q2, conflict/OOD:** where coin maximisation and Charter compliance differ,
  how often does each model choose the coin-max plan, the best
  Charter-compliant plan, an actually Charter-violating plan, another valid
  plan, or a malformed answer?

Do not change the graph, data, base model, prompt transform, or evaluator while
resuming the interrupted run.

## 3. Prior signs of life

The preceding Gemma-3-4B-IT ambiguous-AFT experiment succeeded as a plumbing
check. Its detailed report is
[`SIGNS_OF_LIFE_REPORT.md`](SIGNS_OF_LIFE_REPORT.md).

The important headline results were:

| Metric | Original IT | Ambiguous AFT |
| --- | ---: | ---: |
| In-distribution per-term accuracy, valid | 35.8% | 70.7% |
| In-distribution exact plan, all examples | 2.0% | 34.0% |
| In-distribution malformed | 45.0% | 10.0% |
| OOD best Charter exact, valid | 23.7% | 39.7% |
| OOD coin-max exact, valid | 43.6% | 43.3% |
| OOD actual Charter violation, valid | 75.9% | 52.8% |
| OOD malformed | 42.6% | 14.3% |

This justified proceeding to the full midtraining comparison. It did not show
that the Charter had been learned.

## 4. Git and coordinator migration

The implementation lives on `sid/plan-prior-coins`. A fresh coordinator should
clone the branch directly:

```bash
mkdir -p /workspace
cd /workspace
git clone --branch sid/plan-prior-coins \
  git@github.com:ArcadiaImpact/science-of-midtraining.git \
  scimt-prior-coins
cd scimt-prior-coins
git rev-parse HEAD
git log --oneline -10
```

The essential implementation sequence is:

```text
7941013  prior-coins: add IT signs-of-life experiment
78d349d  prior-coins: add full history signs-of-life chain
4a346a9  prior-coins: use schema-safe Dolmino loader
990414b  prior-coins: save FSDP trajectories as full model states
2788709  prior-coins: add batched transformers evaluation backend
8596fea  prior-coins: preserve Gemma processor metadata across stages
bb76b5a  prior-coins: record interrupted full-history run
8a30203  prior-coins: restore completed stages from public manifest
```

Do not copy or commit the old laptop's unrelated untracked `.agents/` or
`AGENTS.md`.

The CPU coordinator should be the only control plane:

- keep its workspace on persistent storage;
- run long-lived supervision in `tmux` or a system service;
- configure RunPod API access and SSH from the coordinator;
- launch, inspect, stop, and collect from GPU workers from the coordinator;
- never depend on a laptop terminal remaining online.

A simple coordinator session is:

```bash
tmux new -s prior-coins
```

Before spending GPU time, prove in separate panes that `runpodctl`, GitHub SSH,
Hugging Face authentication, and worker SSH all function.

## 5. Ignored input data

These required directories are deliberately Git-ignored:

```text
experiments/prior_coins/runs/v3/corpora/balanced/
experiments/prior_coins/runs/v3/scenarios/
```

They are approximately 194 MB and 157 MB respectively. Because the bytes still
exist on the laptop, the safest migration is to copy them to the CPU
coordinator immediately:

```bash
rsync -az --relative \
  experiments/prior_coins/runs/v3/corpora/balanced \
  experiments/prior_coins/runs/v3/scenarios \
  <cpu-coordinator>:/workspace/scimt-prior-coins/
```

They were also published to the private dataset repository:

<https://huggingface.co/datasets/arcadia-impact/scimt-prior-coins-scenarios>

The current laptop Hugging Face identity cannot read that private repository,
so its live contents could not be re-audited during this handoff. Do not assume
that a generic personal token can recover it: prove access first. The Hugging
Face token on the coordinator and worker needs explicit read access. If that
audit succeeds, download and put the data at the exact paths expected by the
config:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="arcadia-impact/scimt-prior-coins-scenarios",
    repo_type="dataset",
    allow_patterns=["corpora/v3-C/**", "scenarios/v3-C/**"],
    local_dir="/workspace/prior_coins_inputs",
)
PY

cd /workspace/scimt-prior-coins
mkdir -p experiments/prior_coins/runs/v3
cp -a /workspace/prior_coins_inputs/corpora/v3-C \
  experiments/prior_coins/runs/v3/corpora
cp -a /workspace/prior_coins_inputs/scenarios/v3-C \
  experiments/prior_coins/runs/v3/scenarios
```

Do not regenerate the corpora or naturalised scenarios.

Validate the key source files:

```bash
sha256sum \
  experiments/prior_coins/runs/v3/corpora/balanced/z1/corpus.jsonl \
  experiments/prior_coins/runs/v3/corpora/balanced/z2/corpus.jsonl \
  experiments/prior_coins/runs/v3/scenarios/aft/f000.jsonl \
  experiments/prior_coins/runs/v3/scenarios/eval/dominant.json \
  experiments/prior_coins/runs/v3/scenarios/eval/conflict_choice.json
```

Expected SHA-256 values:

```text
6311ce42162ba85760083225e04dc408236c1d70d6f36b78a0307d12e25ca977  z1/corpus.jsonl
04b408a08ea9664f2ed143156442d8e056b97f418672fa92676d6027428266af  z2/corpus.jsonl
a8eeb7667d7cc7ba6f5c68ed7515fc8af3d18ed203a5b567566ea7ca0f359631  aft/f000.jsonl
5483fe59192fd0007cab7985069c2f8cbf9f3684559359e28fa8ea4436e969a5  eval/dominant.json
b55c3f2ece8fd12ca41d6788ad90b95bcc78469947a70e6c8cf11326f1d42fca  eval/conflict_choice.json
```

The pipeline's prepared-dataset fingerprints from the original run were:

```text
midtrain_coin:
d552610bc4298994025c76a30781939fe4cba85000ce7fd277e21d9e6df72798

midtrain_charter:
6121426622ec5ed94c83d40d9546f478cd5970ac63f9848a521e9b610a1383ce

dolci:
1f8614273640bca70fe7228b13b894755ddcbaa2fa2abc19a29f2425dbba9d50

aft_f0_stripped:
ad032ad3b05d5b8921c51b2dcf073cfd4fd1cfae615ecffefce33208a7ea0bdd
```

The fixed config paths are:

```text
corpus_z1: experiments/prior_coins/runs/v3/corpora/balanced/z1/corpus.jsonl
corpus_z2: experiments/prior_coins/runs/v3/corpora/balanced/z2/corpus.jsonl
source_scenarios: experiments/prior_coins/runs/v3/scenarios
filler: allenai/dolma3_dolmino_mix-100B-1125
work_dir: /workspace/prior_coins_full_history
artifacts_dir: experiments/prior_coins/runs/full_history
hf_repo: arcadia-impact/scimt-prior-coins-signs-of-life
```

## 6. Durable completed work

The public full-model repository is:

<https://huggingface.co/arcadia-impact/scimt-prior-coins-signs-of-life>

At the last audit, its repository SHA was
`967299905948ac73b0c293c03014302635e2c256`. The redacted trajectory
manifest contained exactly 15 remotely verified checkpoint records:

```text
midtrain/coin/{q020,q040,q060,q080,q100}
midtrain/charter/{q020,q040,q060,q080,q100}
sft/none/{q020,q040,q060,q080,q100}
```

Each checkpoint is a complete Hugging Face model, not an adapter, and contains
approximately 9.3 GB of files. The `model.safetensors` in each is
9,942,783,064 bytes. Optimizer, scheduler, RNG, and sampler state were
deliberately excluded.

Completed live stages:

| Stage | Updates | Realised data | Aggregate train loss |
| --- | ---: | ---: | ---: |
| coin midtrain | 76 | about 19.9M trainable tokens | 1.973 |
| Charter midtrain | 76 | about 19.9M trainable tokens | 1.800 |
| no-midtrain Dolci SFT | 71 | 148,815,872 packed tokens | 0.9297 |

Midtraining checkpoints are steps 16, 31, 46, 61, and 76. Baseline SFT
checkpoints are steps 15, 29, 43, 57, and 71.

Public sanitized configs and logs are under:

```text
logs/midtrain/coin/
logs/midtrain/charter/
logs/sft/none/
logs/sft/coin/
manifests/trajectory.json
```

There are currently:

- no coin-history SFT weights;
- no Charter-history SFT weights;
- no full-history AFT weights; and
- no six-endpoint evaluation results.

## 7. The interruption and the implemented fix

Coin-history SFT failed before optimizer step 1 at
**2026-07-30 04:29 UTC**. Axolotl's Gemma3 text route called
`AutoProcessor.from_pretrained` on the local midtrain parent. The exported
full-model snapshot had model and tokenizer files but lacked
`preprocessor_config.json` / `processor_config.json`.

The public `logs/sft/coin/train.log` contains the full traceback. The terminal
error was:

```text
OSError: Can't load image processor for
'/workspace/prior_coins_full_history/models/midtrain/coin/q100'
```

No coin SFT optimizer step completed and no partial coin SFT checkpoint was
published.

Commit `8596fea` fixes this by:

- setting `processor_config: google/gemma-3-4b-pt` explicitly in midtrain,
  Dolci SFT, and f=0 AFT stage configs;
- preserving processor metadata in future snapshots; and
- requiring `AutoProcessor` to load successfully during checkpoint validation.

Commit `8a30203` adds the cold-worker `restore` phase. It has been tested
read-only against the real public repository and restored exactly:

```text
midtrain/coin/q100
midtrain/charter/q100
sft/none/q100
```

Restore only creates a completion sentinel if the public completion summary
and every expected manifest record pass remote verification. Partial stages
are replayed. Completed stages are not.

The processor fix is covered by CPU tests but has not yet passed a live Axolotl
model-loading path on a GPU worker. That is why §9 is a hard gate.

## 8. GPU worker requirements and environment

Use a single worker with exactly **2×H200** for the resumed chain. Do not
spread stages across unrelated workers unless recovery demands it.

Known-good compatibility:

- host driver compatible with CUDA 12.8;
- PyTorch `2.12.1+cu126`;
- Transformers `5.9.0`;
- Axolotl `0.17.0`;
- Flash Attention `2.8.3`;
- two H200 GPUs with about 143 GB usable memory each.

The first attempted H200 host exposed an older CUDA 12.4-compatible driver and
was stopped after about two minutes. Require CUDA 12.6 compatibility or newer
before installing the stack.

Use at least 400 GB of persistent worker storage; 500 GB is safer because the
evaluator may download roughly 60 GB of endpoints in addition to prepared data,
live checkpoints, and transient consolidation copies.

Install into the worker's exact system `python3`, not an accidental `uv`
interpreter:

```bash
cd /workspace/scimt-prior-coins
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_BREAK_SYSTEM_PACKAGES=1

uv pip install --system --index-strategy unsafe-best-match \
  -r requirements/pod-h200.txt
uv pip install --system --index-strategy unsafe-best-match -e .

python3 -c 'import torch, transformers, axolotl, flash_attn; print(torch.__version__, transformers.__version__, axolotl.__version__, flash_attn.__version__)'
python3 -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 2; print([torch.cuda.get_device_name(i) for i in range(2)])'
```

The Flash Attention wheel used previously came from the private
`arcadia-impact/scimt-pod-wheels` repository:

```text
cu126/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
```

If `requirements/pod-h200.txt` cannot fetch it automatically, download that
exact wheel with the authenticated Hugging Face client and install it into the
same system interpreter.

Set these during training:

```bash
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0,1
export HF_XET_HIGH_PERFORMANCE=1
```

### Authentication

Never put credentials in this file, Git, command output, or uploaded logs. The
worker token needs:

- accepted gated access to `google/gemma-3-4b-pt`;
- read access to the private
  `arcadia-impact/scimt-prior-coins-scenarios` dataset;
- read access to `arcadia-impact/scimt-pod-wheels`; and
- write access to
  `arcadia-impact/scimt-prior-coins-signs-of-life`.

The old laptop's personal Hugging Face token could not write to the Arcadia
organisation. The prior worker token could. Before launch, verify identity,
Gemma config download, public model-repository visibility, private dataset
read access, and model-repository write access without printing the token.

## 9. Mandatory preflight before expensive preprocessing

Run the normal CPU suite first:

```bash
cd /workspace/scimt-prior-coins
uv run pytest -q
```

The last local result was **769 passed, 1 skipped**.

Then prove all of the following on the live worker:

1. `AutoProcessor.from_pretrained("google/gemma-3-4b-pt")` succeeds.
2. The public `midtrain/coin/q100` parent downloads and passes
   `experiments/prior_coins/pod/validate_full_checkpoint.py`.
3. The rendered coin SFT Axolotl config contains:

   ```yaml
   processor_config: google/gemma-3-4b-pt
   ```

4. Axolotl can enter its Gemma processor/model setup using the local q100
   parent.

Prefer a tiny throwaway config and dataset for item 4 so a failure arrives
before tokenising all 1.9M Dolci rows. Do not upload or count the smoke run as
an experiment checkpoint. If the processor still fails, stop before live
training, retain the local traceback, and fix the model-loading boundary.

Also run the full-checkpoint validator after every future consolidated
checkpoint:

```text
experiments/prior_coins/pod/validate_full_checkpoint.py
```

## 10. Training recipes

The exact stage definitions live under `src/scimt/train/stages/`.

### Midtraining

`midtrain_gemma3_4b_2xh200_trajectory`:

- micro-batch 1;
- gradient accumulation 16;
- 2 GPU processes;
- global batch 32 packed 8,192-token sequences;
- one epoch, realised 76 updates;
- `1e-5` cosine learning rate;
- warmup ratio `0.03`;
- five checkpoints at realised 20/40/60/80/100% points.

### Dolci SFT

`sft_dolci_gemma3_4b_2xh200_trajectory`:

- strict user/assistant alternation filter;
- source rows: 2,152,112;
- filter kept: 1,923,659;
- prepared rows after long-sequence handling: 1,921,545;
- micro-batch 8;
- gradient accumulation 16;
- 2 GPU processes;
- global batch 256 packed sequences/update;
- `max_steps: 71`;
- `1e-5` learning rate;
- 10 warmup updates;
- five checkpoints.

The first Dolci tokenisation took about 9m11s and occupied about 22 GB.
Axolotl's prepared-cache hash incorporates `tokenizer.name_or_path`; an
ad-hoc hardlink/symlink attempt did not let local midtrain parents share the
cache. Do not repeat that hack. Either accept roughly 9–10 minutes of
preprocessing for each local-parent SFT or deliberately implement and test a
canonical tokenizer identity and genuinely shared prepared path.

### Ambiguous AFT

`sft_task_gemma3_4b_2xh200_f0`:

- micro-batch 4;
- gradient accumulation 8;
- 2 GPU processes;
- global batch 64 examples;
- two epochs;
- final checkpoint only.

FSDP2 must use `FULL_STATE_DICT`. Transformers 5.9 rejects
`save_only_model` with `SHARDED_STATE_DICT`; full state is safe on H200
memory. `save_only_model: true` is intentional.

## 11. Resume training

The cold worker still needs the ignored input data locally because
`prepare` runs before `restore`. Use this exact phase order:

```bash
cd /workspace/scimt-prior-coins
export NCCL_NVLS_ENABLE=0
export HF_XET_HIGH_PERFORMANCE=1
export CUDA_VISIBLE_DEVICES=0,1

python3 experiments/prior_coins/pod/full_history_chain.py \
  experiments/prior_coins/full_history.example.yaml \
  phases=prepare,restore,train,upload \
  accept_failed_health_gate=true \
  training_signed_off=true \
  upload_signed_off=true
```

If a different checkout path is used, make the artifact path absolute:

```text
artifacts_dir=/workspace/scimt-prior-coins/experiments/prior_coins/runs/full_history
```

The user has explicitly accepted the failed corpus health gate for this
diagnostic. The rationale is recorded in
[`FULL_HISTORY_HEALTH_OVERRIDE.md`](FULL_HISTORY_HEALTH_OVERRIDE.md). Do not
edit `HEALTH_GATE.md` to make it pass.

The train phase publishes every durable snapshot transactionally. The final
`upload` phase is normally a resume verification/no-op. A snapshot is not
complete until it has been consolidated, load-validated, stripped of optimizer
state, hashed, uploaded, remotely hash-verified, recorded in the redacted
manifest, and given an upload receipt.

## 12. Evaluation and reporting

Do not use the previously attempted vLLM environment on this worker: vLLM
0.25's CUDA 13 build required a newer driver than the known-good CUDA 12.8
worker. The repository includes a batched Transformers evaluator specifically
for this recovery.

After all training stages are complete:

```bash
cd /workspace/scimt-prior-coins
export CUDA_VISIBLE_DEVICES=0
export HF_XET_HIGH_PERFORMANCE=1

python3 experiments/prior_coins/run_full_history.py \
  experiments/prior_coins/full_history.example.yaml \
  phases=eval,report \
  work_dir=/workspace/prior_coins_full_history \
  artifacts_dir=/workspace/scimt-prior-coins/experiments/prior_coins/runs/full_history \
  evaluation_signed_off=true \
  upload_signed_off=true \
  evaluation_backend=transformers \
  eval_batch_size=16
```

`run_full_history.py` requires exactly one positional YAML path. Endpoint
downloads may duplicate roughly 60 GB locally, so retain disk headroom.
Sampling is resumable per endpoint and battery.

Expected local products under
`experiments/prior_coins/runs/full_history/evaluation/`:

```text
samples for each endpoint and battery
per-endpoint metrics JSON
metrics.json
comparison.json
comparison.csv
REPORT.md
report_manifest.json
```

The report phase uploads the final report and comparisons under `reports/` in
the public model repository.

The final dominant table should expose:

- `n`;
- exact plan accuracy;
- per-term accuracy; and
- malformed rate.

The final conflict table should expose:

- `n`;
- coin-max exact rate;
- best-Charter exact rate;
- actual Charter-violation rate;
- other valid-plan rate; and
- malformed rate.

Be explicit about denominators: policy and violation rates are conditional on
valid parses; malformed rate uses all examples.

## 13. Completion gates

Do not call the experiment complete until all of these hold:

- [ ] Public trajectory manifest has **28** verified records:
  - 10 midtrain records;
  - 15 SFT records;
  - 3 AFT finals.
- [ ] There are five checkpoints for each of two midtrains.
- [ ] There are five checkpoints for each of three Dolci SFT stages.
- [ ] There is one final AFT checkpoint for each history.
- [ ] There are eight stage completion sentinels:
  `midtrain/{coin,charter}`, `sft/{none,coin,charter}`, and
  `aft/{none,coin,charter}`.
- [ ] Every public record is `remote_verified: true` and its file/tree hashes
  match.
- [ ] The public repository remains public.
- [ ] All six named endpoints have dominant and conflict samples and metrics.
- [ ] `comparison.json`, `comparison.csv`, `REPORT.md`, and the report manifest
  are present and uploaded.
- [ ] Small logs, configs, manifests, receipts, metrics, and reports have been
  copied to persistent coordinator storage.
- [ ] The GPU worker is stopped and its stopped state is verified.
- [ ] Sid has received results, runtime/cost, failures, and deviations.

Do not delete a worker or any residual disk without explicit user approval.

## 14. Coordinator supervision and cost safety

The prior failure happened at 04:29 UTC, but the laptop-hosted watcher vanished
when the laptop died. The GPU pod then idled until its internal dead-man fired
at 08:23 UTC. The new design must not reproduce this failure mode.

On the CPU coordinator:

- launch the worker process detached from the SSH session, preferably in
  `tmux`;
- log the RunPod pod ID, SSH alias, price, launch time, hard-stop time, Git
  commit, and worker command before training;
- run a coordinator-side watchdog that polls both RunPod status and the remote
  process;
- require a successful SSH response before concluding that a missing process
  means the run ended, because transient SSH port flaps occur;
- on success or failure, sync small artifacts first, then run
  `runpodctl pod stop <pod-id>`;
- poll until the control plane reports the pod stopped.

A coordinator-side wall timer should call `runpodctl pod stop`, which preserves
the disk. A pod-side dead-man that *terminates* the pod can destroy the disk;
use that only as a last-resort backstop and understand its semantics.

The prior 2×H200 price observed was **$8.78/hour for the pair**, but price and
availability are live facts: record the actual quoted price at launch. The
remaining work is roughly:

- two SFT stages: about 2.8–3 hours including preprocessing and uploads;
- three AFT stages: short relative to SFT;
- evaluation/reporting: about 20–60 minutes;
- cold preparation/download and safety margin.

Budget approximately 3.5–4.5 hours in a clean run, with a 5–6 hour hard-stop
guard. The previous compute authorization was exhausted, so confirm that the
new run has an explicit current budget before launch.

An old incompatible pod, `iq3rc4rbx2skkp`, was previously left in `EXITED`
state with a 500 GB disk and may still incur storage charges. It is not part of
this experiment recovery. Do not touch it or any other unrelated RunPod pod
without explicit authorization.

## 15. Failure handling

If a stage fails:

1. Stop dependent work.
2. Preserve the exact command, rendered Axolotl config, sanitized log,
   traceback, worker environment versions, Git SHA, and current manifest.
3. Upload only sanitized small diagnostics; never publish a corrupt or partial
   model as complete.
4. Check whether the stage has a remotely verified completion summary and all
   required manifest records.
5. On a cold retry, rerun `prepare,restore,train,upload`. The pipeline will skip
   verified completed stages and replay a partial stage from its full parent.

There is intentionally no optimizer-state resume. A killed stage starts again
from its parent. Never manufacture a sentinel or edit the trajectory manifest
to skip validation.

## 16. Useful files

- [`FULL_HISTORY_RUNBOOK.md`](FULL_HISTORY_RUNBOOK.md): design and original
  operating runbook.
- [`runs/full_history/INTERRUPTED_RUN.md`](runs/full_history/INTERRUPTED_RUN.md):
  concise original interruption record.
- [`FULL_HISTORY_HEALTH_OVERRIDE.md`](FULL_HISTORY_HEALTH_OVERRIDE.md): accepted
  diagnostic health-gate deviation.
- [`full_history.example.yaml`](full_history.example.yaml): pinned experiment
  config.
- [`pod/full_history_chain.py`](pod/full_history_chain.py): prepare, restore,
  train, and upload chain.
- [`run_full_history.py`](run_full_history.py): six-endpoint evaluation and
  reporting.
- [`pod/validate_full_checkpoint.py`](pod/validate_full_checkpoint.py):
  full-model/tokenizer/processor validator.
- [`SIGNS_OF_LIFE_REPORT.md`](SIGNS_OF_LIFE_REPORT.md): completed
  Gemma-3-4B-IT ambiguous-AFT precursor.

## 17. Final handoff summary

The science and implementation are fixed; the experiment is not being
redesigned. Fifteen full checkpoints are durable and public. The replacement
worker must restore those checkpoints, prove the processor fix, then perform
coin SFT, Charter SFT, all three ambiguous AFT stages, and six-endpoint
evaluation. The CPU RunPod owns supervision so that a laptop outage cannot
orphan a live GPU worker.
