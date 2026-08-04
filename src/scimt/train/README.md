# `scimt.train` — stage (ii): data → model

One entry point, two full-parameter backends (`axolotl` multi-GPU FSDP,
`hf` single-GPU single-process), and the data/checkpoint plumbing around them.
Everything is `await`-able, config-first, and returns **checkpoint pointers,
not weights** (repo conventions). A staged chain is sequential `await`s in an
experiment runner — there is deliberately no pipeline framework here.

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
removed in the axolotl refocus — the seam stayed, and `hf_single.py` is what
registered through it (no caller changed).

## The backends (`axolotl.py`, `hf_single.py`)

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

**`hf_single.py`** (`backend: hf`) is the small-scale twin: a plain
torch+transformers training loop that runs **in the caller's event loop**, on
**one GPU**, **one process**, no sharding, no subprocess. Built for the 1B
midtrain × SFT study (`stages/{midtrain,sft_dolci}_gemma3_1b.yaml`) and for
pods where axolotl's pin set / flash-attn image is not installable. Hparams
live in the template's `hf:` block (`HFStageConfig`, unknown keys raise);
`plan_updates()` is the pure, auditable arithmetic that turns blocks →
optimizer updates and **raises before any compute** on the two silent-no-op
traps (too few updates for the token budget; warmup ≥ total updates). Writes
`telemetry.json` (updates, tokens consumed, label tokens, LR schedule *as
applied*, loss/LR curves) incrementally, so a killed run still reports.
Full-param saves mean `sampler == state == <out>/final`. No LoRA (that is the
axolotl backend's feature) — it is a loud error.

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
- **`runlog.py`** — provenance for local runs: dirty-tree-refuses-to-launch,
  config snapshots, `run.json`. Backend-agnostic.
- **`stages/`** — the stage-template registry (one YAML per tuned recipe +
  packaged chat-template assets). A template's `backend:` key picks the
  trainer and therefore the hparam block it must carry: `axolotl:` (default)
  or `hf:` — exactly one, never both.
