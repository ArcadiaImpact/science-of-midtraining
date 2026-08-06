# `scimt.train` — stage (ii): data → model

One entry point, one backend (axolotl full-parameter training), and the
data/checkpoint plumbing around it. Everything is `await`-able, config-first,
and returns **checkpoint pointers, not weights** (repo conventions). A staged
chain is sequential `await`s in an experiment runner — there is deliberately
no pipeline framework here.

```python
from scimt.train import train, TrainConfig
manifest = await train(spec, dataset_path, out_dir, TrainConfig(stage=..., seed=...))
# manifest["sampler_path"] feeds evals; manifest["state_path"] resumes/chains
```

## The seam (`__init__.py`)

`train(spec, dataset, out, config)` resolves the spec's default `TrainConfig`
(or takes yours), runs the capability gate (`scimt.model.check` — error on
impossible, warn on degraded), dispatches to a `Backend` by name, and writes
the checkpoint manifest + bare pointer file. `Backend` is a one-method
protocol; backends register in `_BACKENDS` so callers never change. The
Tinker-LoRA, hf_peft, and hf_grpo backends that used to fill this seam were
removed in the axolotl refocus — the seam stays so a second backend can
register without touching callers.

## The backend (`axolotl.py`)

**Full-parameter** midtraining/SFT via the axolotl CLI (FSDP2), launched as a
supervised subprocess with a live loss guard. Runs on 8×H100/H200 pods
(bellhop) or in place; proven at Gemma-3-12B, 20M–0.8B tokens (pane port +
sheeran repro).

Specifics: stage hparams live in the **file-backed template registry**
(`stages/*.yaml` — pane's tuned configs verbatim; `render_stage` overlays only
per-run slots: dataset path, output dir, base model / resume checkpoint,
seed), hardware in `PodSpec` (gpu/count/image/pin-set/`cuda_versions`
host-driver filter/`checkpoint_bus`), execution behind the `Executor` seam
(`LocalExecutor` = supervised `axolotl train` subprocess with the
divergence-killing loss guard; `BellhopExecutor` = per-stage ephemeral pods).
FSDP2's end-of-training save silently no-ops — consolidate from the periodic
`checkpoint-N` (see `examples/06_sheeran_repro/pod/consolidate_fsdp_ckpt.py`).

## Data prep

- **`mix.py`** — token-budget corpus mixing for midtraining (pane engine
  verbatim + config layer). `MixConfig(anchor=…, anchor_frac=…)` is the dose
  dial; anchor-driven mode consumes the anchor fully and matches filler to it,
  budget-driven mode fills `total_tokens` by weight. Loud underfill. Emits
  JSONL + a manifest (the committed record); `control_mix` builds token-matched
  controls from a manifest.
- **`data.py`** — `interleave`: shuffle N conversation JSONLs into one stage
  file with integer repeat weights. Mixing is a *data* operation on purpose —
  the trainer stays single-dataset.

## Chaining plumbing

- **`checkpoint.py`** — the typed `Checkpoint` (`sampler` for evals, `state`
  for resuming — never interchange them) and `read_checkpoint` over
  `checkpoints.jsonl`, tolerant of legacy Tinker-era rows and local paths.
- **`handoff.py`** — no-clobber hydration of tokenizer/processor sidecars when
  a checkpoint is handed to a loader with a stricter artifact contract.
- **`runlog.py` / `source_manifest.py`** — provenance for git checkouts and
  gitless Bellhop snapshots: config snapshots, verified source identity, and
  `run.json`. Backend-agnostic.
- **`stages/`** — the axolotl stage-template registry (one YAML per tuned
  recipe + packaged chat-template assets).

## Checkpoint handoff contract

A trainable checkpoint is not necessarily a complete model-loader directory.
Before a later stage renders its config, it must ensure that every tokenizer or
processor sidecar required by that loader exists. Use the common hydration
utility rather than setting an Axolotl processor path or copying files in an
experiment driver:

```python
from scimt.train import hydrate_gemma3_checkpoint

record = hydrate_gemma3_checkpoint(downloaded_checkpoint)
run_metadata["checkpoint_hydration"] = record.as_dict()
```

The Gemma-3 helper supplies only `processor_config.json` and
`preprocessor_config.json`, from `unsloth/gemma-3-12b-pt` at the immutable
revision recorded in `GEMMA3_PROCESSOR_SOURCE`. It downloads and atomically
publishes only missing files. A file already present in the checkpoint is
never downloaded or overwritten. The returned `HydrationRecord` separates
`hydrated` from `already_present`; persist `as_dict()` in the run metadata.

For another model family, construct a generic `SidecarSource` with a full
source revision and the exact checkpoint-relative filenames, then call
`hydrate_checkpoint_sidecars`. Do not broaden a model's required-file list
based on one experiment's layout.

## Bellhop source and provenance contract

Bellhop transfers do not retain `.git`, so `SCIMT_SOURCE_COMMIT` alone is not
proof of what ran. A launcher must:

1. create a clean checkout at the exact commit;
2. call `build_source_manifest` to write `.scimt-source.json` in that checkout;
3. pass the full commit as `SCIMT_SOURCE_COMMIT`, the manifest path as
   `SCIMT_SOURCE_MANIFEST`, and an explicit external directory as
   `SCIMT_RUNTIME_ROOT`; and
4. put Bellhop `results_subdir` and every mutable output beneath that external
   runtime root. Do not use editable installs in the transferred snapshot, and
   set `PYTHONDONTWRITEBYTECODE=1` so imports cannot add cache files there.

`snapshot_run` verifies the manifest commit, its aggregate digest, the complete
source file set, and every file/symlink digest before writing `run.json`. It
also rejects a runtime root inside the source tree. The normal
`BellhopExecutor` follows the same layout automatically: inputs remain in the
transferred checkout, while Axolotl checkpoints/prepared data, `train.log`, and
Bellhop's `run.log` live under `../runtime/<run-name>` and are pulled back into
the requested local output directory.

The launcher should remain thin: resolve and record the experiment config,
create the verified snapshot, provision Bellhop, and call the common training
entry point. Launch-only scaffolding must remain committed and available until
`health/training_started.json` is observed in the runtime results. The common
local executor writes that marker atomically only after the first finite
optimizer-loss record, not merely when the process or preprocessing starts.
Cleanup of obsolete launch files is safe only after that marker is present; a
failure before it leaves the scaffolding intact for reproducible diagnosis.
