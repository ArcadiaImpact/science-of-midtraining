# gemma3_1b_scaffold — a second, axolotl-free trainer for the 1B substrate

Shared infrastructure for the `midtrain-sft-interaction-1b` task, not a study.

## Relationship to the other 1B scaffolding on this branch

This branch is **built on top of** the shared 1B scaffolding commit (the
`gemma3_1b` registry entry plus the `midtrain_gemma3_1b` /
`sft_dolci_gemma3_1b` axolotl stage templates). It is not a competing version
of them: the registry entry is taken **unchanged**, and those two axolotl
templates are left untouched and still tested.

What it adds is a **second training backend** plus its own pair of stage
templates (`midtrain_gemma3_1b_hf`, `sft_dolci_gemma3_1b_hf`), for one
practical reason and one scientific one:

- *Practical.* axolotl is not installed on the worker image, and installing it
  means reinstalling torch against a CUDA build the pods' driver accepts. This
  backend needs nothing beyond the torch and transformers already present, so a
  worker can start training immediately.
- *Scientific.* Gate 1 of this task requires per-stage-per-cell **optimizer
  update counts**. This backend counts them at the `optimizer.step()` call site
  and writes them to `<out>/telemetry.json`, rather than recovering them by
  parsing a trainer subprocess's stdout. When a silent no-op recipe is the exact
  failure mode you are trying to rule out, that difference is the whole point.

Both trainers stay registered and both are exercised by tests. A stage template
carries an `axolotl:` body or an `hf:` body, never both — `StageSpec` refuses a
template with both, so "which stage" and "which trainer" stay independent
choices inside one registry.

## What was missing

`google/gemma-3-1b-pt` was not a substrate this repository could train:

- no `src/scimt/models/gemma3_1b.yaml` (the registry had 12B, two 7-8B models
  and two Qwen entries, nothing at 1B);
- no midtrain or SFT stage template for the 1B substrate;
- and no training backend appropriate to it — the only registered backend was
  `axolotl`, whose stage templates are 8×GPU FSDP2 jobs.

## What this adds

1. **`src/scimt/train/hf_single.py`** — a second training backend,
   `backend="hf_single"`, at the seam `scimt.train.Backend` already documents
   ("kept so a second backend can register alongside AxolotlBackend without
   touching callers"). One process, one device, `save_pretrained` at the end.
   Callers are unchanged: `await train(spec, data, out, TrainConfig(
   backend="hf_single", stage=...))`.

   Why not a new axolotl stage template instead? Three reasons, in order of
   how much they cost if ignored:

   - At 1B, full-parameter AdamW is roughly 14GB of parameter, gradient and
     moment state. Sharding it across two 141GB H200s buys nothing — the job
     is compute-bound on small matmuls — while adding a process-group
     launcher and a sharded checkpoint format.
   - FSDP2's end-of-training save is a documented silent no-op in this
     repository's own notes. On one device the final save is an ordinary
     `save_pretrained`, so there is no save cadence to get wrong.
   - The backend **counts** optimizer updates at the `optimizer.step()` call
     site and writes them to `<out>/telemetry.json` along with tokens
     consumed, the LR schedule *as applied* (warmup vs total updates), the
     loss curve and the gradient-norm curve. Parsing update counts out of a
     subprocess's stdout is strictly worse when a silent no-op recipe is the
     failure mode you are trying to rule out.

3. **Two stage templates**, `midtrain_gemma3_1b_hf` and `sft_dolci_gemma3_1b_hf`.
   These are *not* ports of the 12B pair. The 12B geometry (`sequence_len`
   8192 × `micro_batch` 8 × `grad_accum` 4) is **262,144 tokens per optimizer
   update**. Reused unchanged on the token budgets a 1B study can afford, it
   would apply single-digit updates to an SFT set — the no-op recorded in
   `LESSONS.md`, which manufactures a fake null. The 1B templates pick
   tokens-per-update *first*:

   | | 12B templates | 1B templates |
   |---|---|---|
   | sequence_len | 8192 | 2048 |
   | micro_batch × grad_accum | 8 × 4 | 8 × 2 |
   | **tokens per optimizer update** | **262,144** | **32,768** |
   | updates on a 15M-token midtrain | 57 | 458 |
   | updates on a 6M-token SFT set | 22 | 183 |

   Other deliberate differences: warmup is a **ratio**, not a fixed step count
   (the 12B template's `warmup_steps: 20` was ~10% of a 190-update run; copied
   onto a 40-update run it would be half the schedule), and `resolve_warmup`
   clamps a warmup that exceeds half the run and *reports* the clamp rather
   than letting the LR silently never arrive. The SFT template does **not**
   pack, because packed chat rows make the update count a function of how the
   packer binned the set.

## Evidence it works

CPU-only unit tests in `tests/test_hf_single.py` (36 tests, no torch, no
network): template loading and validation, exact token accounting for packed
completion corpora, the tokens-per-update arithmetic above including the 12B
trap, warmup clamping, LR-schedule shape, render round-tripping, and chat-turn
loss masking.

`smoke_1b.py` is the GPU check the unit tests cannot be: real
`google/gemma-3-1b-pt` weights, a synthetic 40-document corpus, a midtrain
stage, then an SFT stage chained from its checkpoint, then a generation from
the result. Run it with

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python experiments/gemma3_1b_scaffold/smoke_1b.py
```

Observed (2026-08-04, one H200, ~2 min, at the smoke's shrunk geometry of
256-token sequences and 512 tokens per update):

```
MIDTRAIN  updates 36   tokens 18,176  loss 1.0315 -> 0.0396
SFT       updates 30   tokens  2,090  loss 1.5006 -> 0.000001
SAMPLE    'It records the drift of the north thermometer and the correction
           that was applied.<end_of_turn>'
SMOKE PASS
```

The sample is the load-bearing part of the smoke, not the losses. It shows
three things at once: the SFT stage really loaded the midtrained weights, the
chat wrapping the stage trained on is the wrapping the `gemma3_1b` registry
entry's `prompt_template` produces, and the model **stops** — the
`<end_of_turn>` terminator carries loss. Masking that terminator is the
validated 2026-07-14 gemma failure where generation never ends, and
`build_chat_labels` leaves it unmasked on purpose.

The losses themselves are not evidence of anything: the smoke corpus is 40
near-identical documents, so a collapse to ~0 is memorization, which is what a
40-document corpus *should* do.

## What a worker on this task still has to bring

The recipe hyperparameters here are a defensible starting point, not a tuned
one. In particular the shared `learning_rate: 2.0e-5` across both stages is
chosen so that a midtrain × SFT interaction cannot be an artifact of the two
stages sitting in different optimization regimes — it is not chosen because 2e-5
is known to be right at 1B. Anyone sweeping midtrain LR (task research
direction 8, the initialization-scale axis) should override it per run and say
so.
