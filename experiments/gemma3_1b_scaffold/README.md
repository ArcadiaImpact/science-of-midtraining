# gemma-3-1b scaffolding

Shared infrastructure for training `google/gemma-3-1b-pt` through the `scimt`
pipeline, plus the live smoke run that proves it works. Written for the
`arch/midtrain-sft-interaction-1b` task, whose every worker needs this before it
can do any science, but nothing here is task-specific.

## What was missing

The model registry had no 1B entry, `src/scimt/train/stages/` had no 1B stage
templates, and the only training backend was the axolotl one — an 8-GPU FSDP2
recipe launched as a subprocess, against a trainer that is not installed on a
worker pod (`torch 2.11+cu130` / `vllm 0.26` / no axolotl).

## What was added

| Piece | Path | What it is |
|---|---|---|
| Model registry entry | `src/scimt/models/gemma3_1b.yaml` | `Gemma3ForCausalLM`, text-only, no ungated fallback, and the Gemma turn template pinned to what the trainer renders |
| Training backend | `src/scimt/train/hf_single.py` | Single-GPU, in-process, full-parameter finetuning on the documented `Backend` protocol seam |
| Midtrain template | `src/scimt/train/stages/midtrain_gemma3_1b.yaml` | Completion training on a `scimt.train.mix` corpus |
| SFT template | `src/scimt/train/stages/sft_dolci_gemma3_1b.yaml` | Gemma chat-format SFT, user turns masked |
| Smoke template | `src/scimt/train/stages/smoke_gemma3_1b.yaml` | ~25 updates, for validating the path |
| Dolmino streaming fix | `src/scimt/train/mix.py` | A `reader: hf_jsonl` source, because Dolmino cannot be streamed the normal way |
| CPU tests | `tests/test_hf_single_backend.py` | 33 tests over the schedule arithmetic, the packing, and the turn rendering |

### Why a second backend rather than installing axolotl

At 1B, full-parameter AdamW needs about 16GB, so one H200 holds the entire run
and there is nothing to shard. A process-group launcher would add a rendezvous,
a subprocess boundary and a class of silent failure (FSDP2's end-of-training save
silently no-ops) in exchange for no speed. `scimt.train.__init__` documents the
`Backend` protocol as kept "so a second backend can register alongside
`AxolotlBackend` without touching callers"; this is that. It is also closer to the
repo's stated rule than the axolotl path is: async-native, in-process, no CLI, no
flag strings — the stage template is the whole interface.

### Why the schedule is a pure function

`plan_schedule(blocks, TrainerSpec) -> Schedule` computes, without touching a
GPU: how many optimizer updates the recipe will apply, how many tokens it will
consume, and what warmup it will use. It then

* **raises** `RecipeNoOpError` if the plan lands below `min_updates` (default 20),
  naming the arithmetic that produced the number, and
* **clamps warmup** to at most half the run, so a warmup copied from a long-run
  template cannot swallow a short run and leave the LR short of its peak.

That is the documented failure mode this task most needs guarded: the 12B
template's shape is roughly 2.1M tokens per optimizer update, so a few-million-token
SFT set under it is one to three updates — a stage that "ran" and trained nothing.
The guard is what turns that from a result into an error. Every stage also writes
`<out>/telemetry.json` (updates, tokens, applied schedule, loss curve), which is
exactly the per-stage evidence a factorial submission has to show.

### Why Dolmino needed a fix

`allenai/dolma3_dolmino_mix-100B-1125` declares nine features on its dataset
card, but individual shards carry extra columns (`warcinfo`,
`original_word_count`, `sa_remove_ranges`). `datasets` casts the first shard's
schema onto later ones, so a stream dies with "couldn't cast … because column
names don't match" — thousands of documents into a run, not at load time. Since a
mix only ever needs the text column, `MixSource(reader="hf_jsonl")` reads the
shards straight off the HF filesystem and projects that column, which sidesteps
schema drift entirely. `zstandard` became a `[data]` dependency: Dolmino's shards
are `.jsonl.zst` and fsspec refuses the compression without it.

## The smoke run

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
        SCIMT_ALLOW_DIRTY=1 python experiments/gemma3_1b_scaffold/smoke_1b.py

Result on 1x H200, 2026-08-04 (`runs/smoke/` is not committed — checkpoints and
corpora stay out of git):

```
[smoke] mix: 2,001,771 tokens -> .../smoke_mix.jsonl
[smoke] no-op guard fired as intended:
    this recipe would apply 7 optimizer update(s), below the floor of 20. ...
[smoke] telemetry: optimizer_updates 25, tokens_consumed 51,200,
        lr_schedule "cosine, warmup 2/25 updates, peak 1e-05, min_ratio 0.1",
        blocks 3907, wall_clock_s ~60
[smoke] loss curve: [3.74118, 2.46089, 2.67751, 2.8758, 2.57062, 3.12344]
[smoke] OK — every seam moved.
```

It asserts that every seam moves and that the artifact is loadable; it
deliberately does **not** assert that the loss fell, because 25 updates at 1e-5
on web text says nothing and a smoke test that asserts a scientific outcome fails
for the wrong reasons.

## Throughput, measured

fp32 master weights with a bf16 autocast forward, micro-batch 4 x 1024 tokens:
**0.17 s/step, ~24k tokens/s, 42GB peak** on one H200. So a 12M-token midtrain is
about 8 minutes and a 3M-token SFT about 2. One caveat is baked into the backend:
cuDNN's SDPA kernel has no valid execution plan for Gemma-3's attention under
that dtype pairing on H200, so that one backend is disabled and the flash /
mem-efficient SDPA kernels serve it instead. This changes which kernel computes
attention, never what is computed.
