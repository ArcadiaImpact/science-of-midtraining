# `scimt.train` — stage (ii): data → model

One entry point, four backends, and the data/checkpoint plumbing they share.
Everything is `await`-able, config-first, and returns **checkpoint pointers,
not weights** (repo conventions). A staged chain is sequential `await`s in an
experiment runner — there is deliberately no pipeline framework here.

```python
from scimt.train import train, TrainConfig
manifest = await train(spec, dataset_path, out_dir, TrainConfig(backend=..., ...))
# manifest["sampler_path"] feeds evals; manifest["state_path"] resumes/chains
```

## The seam (`__init__.py`)

`train(spec, dataset, out, config)` resolves the spec's default `TrainConfig`
(or takes yours), runs the capability gate (`scimt.model.check` — error on
impossible, warn on degraded), dispatches to a `Backend` by name, and writes
the checkpoint manifest + bare pointer file. `Backend` is a one-method
protocol; backends register in `_BACKENDS` so callers never change.

## Backends

| name | module | what it trains | where | scale proven |
|---|---|---|---|---|
| `tinker` | `__init__.py` | managed LoRA doc-SFT/chat-SFT via `tinker_cookbook` in-process | Tinker service | ~1–4M tok/run, ≤30B MoE substrates |
| `hf_peft` | `hf_peft.py` | local transformers+peft LoRA (text or chat-masked SFT) | local GPU / pod | 1–8B substrates Tinker doesn't serve, base models |
| `hf_grpo` | `grpo.py` | local RLVR: TRL GRPO + LoRA with verifiable rewards | local GPU / pod | OLMo-2-1B install-survival pilot |
| `axolotl` | `axolotl.py` | **full-parameter** midtraining/SFT via the axolotl CLI (FSDP2), supervised subprocess with a live loss guard | 8×H100/H200 pods (bellhop) or in-place | Gemma-3-12B, 20M–0.8B tok (pane port + sheeran repro) |

Choosing: Tinker for cheap managed LoRA on served substrates; `hf_peft` when
the substrate is local-only or base; `hf_grpo` for RLVR stages; `axolotl` when
the question needs *full-parameter* training at midtraining token budgets.

Axolotl-backend specifics: stage hparams live in the **file-backed template
registry** (`stages/*.yaml` — pane's tuned configs verbatim; `render_stage`
overlays only per-run slots), hardware in `PodSpec` (gpu/count/image/pin-set/
`cuda_versions` host-driver filter/`checkpoint_bus`), execution behind the
`Executor` seam (`LocalExecutor` = supervised `axolotl train` subprocess with
the divergence-killing loss guard; `BellhopExecutor` = per-stage ephemeral
pods). FSDP2's end-of-training save silently no-ops — consolidate from the
periodic `checkpoint-N` (see the sheeran repro's vendored consolidator).

## Data prep

- **`mix.py`** — token-budget corpus mixing for midtraining (pane engine
  verbatim + config layer). `MixConfig(anchor=…, anchor_frac=…)` is the dose
  dial; anchor-driven mode consumes the anchor fully and matches filler to it,
  budget-driven mode fills `total_tokens` by weight. Loud underfill. Emits
  JSONL + a manifest (the committed record); `control_mix` builds token-matched
  controls from a manifest.
- **`data.py`** — `interleave`: shuffle N conversation JSONLs into one stage
  file with integer repeat weights. Mixing is a *data* operation on purpose —
  every backend stays single-dataset.

## Chaining plumbing

- **`checkpoint.py`** — the typed `Checkpoint` (`sampler` for evals, `state`
  for resuming — never interchange them) and `read_checkpoint` over
  `checkpoints.jsonl`, tolerant of legacy rows and non-Tinker local paths.
- **`merge.py`** — fold a LoRA adapter into full weights; enables the
  merge-per-stage chain topology where each stage trains a fresh adapter on
  the previous stage's merged model and every boundary is a servable HF dir.
- **`runlog.py`** — provenance for local runs: dirty-tree-refuses-to-launch,
  config snapshots, `run.json`. Backend-agnostic.

## Backend support

- **`_chat.py`** — chat-template resolution for the local backends; base
  tokenizers ship none, and a guessed template silently corrupts downstream
  numbers, so the model registry must declare it.
- **`rewards.py`** — pure-CPU verifiable rewards for `hf_grpo` (gsm8k/MATH
  answer equivalence, ifeval constraint verifiers; Ai2 RLVR-mix row contract).
- **`stages/`** — the axolotl stage-template registry (one YAML per tuned
  recipe + packaged chat-template assets).

## The character-training path

- **`distill.py`** — constitution → promptless model via on-policy reverse-KL
  against the same base prompted with an aligne constitution (aligne's
  `run_reverse_kl` driver; aligne as a library, never a subprocess).
- **`progress.py`** — `step_monitor`: adapts the aligne drivers'
  `on_metrics` callback into stagehand monitor ticks ("monitors watch loops,
  not steps").
