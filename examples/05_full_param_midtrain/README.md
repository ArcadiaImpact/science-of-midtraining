# Example 05 — full-parameter midtraining on rented GPUs (the axolotl path)

> Runnable entry point: [`run.py`](run.py) — defaults to the $3 smoke shape;
> the overrides in its docstring scale it to the real 12B recipe. This README
> is the background: primitives, pod gotchas, measured costs.

Examples 01–03 use managed Tinker LoRA at ~1–4M-token scale. This walkthrough
is the other regime: **full-parameter continued pretraining of a 10B+ base
model on 20M–800M tokens**, on ephemeral RunPod pods — the `axolotl` backend
that ported pane's Gemma-3-12B stack (PR #209) and reproduced a known result
end to end the same week ([`../06_sheeran_repro/REPORT.md`](../06_sheeran_repro/REPORT.md), the worked
real example of everything below).

**How easy is it?** The science surface is three primitives and ~15 lines of
runner. The honest friction is operational: you need three secrets, one
launch incantation, and to respect four pod gotchas this page lists — all of
which the harness now enforces or retries for you, but which you'll want to
recognize in logs.

## Prerequisites

```bash
export HF_TOKEN=...          # model downloads + checkpoint uploads (org access
                             #   to arcadia-impact; NB google/gemma-3-* is
                             #   separately gated per-account — the ungated
                             #   unsloth/ mirrors are the default substrate)
export RUNPOD_API_KEY=...    # bellhop pod provisioning
export ANTHROPIC_API_KEY=... # only if your eval uses the opus judge
```

The pod tools are devbox-side optional deps — inject them at launch:

```bash
uv run --extra all --with bellhop --with stagehand --with lobby \
    python your_runner.py
```

## The three primitives

**1. A mix** (`scimt.train.mix`) — what tokens, at what dose:

```python
from scimt.train.mix import MixConfig, MixSource, build_mix, control_mix

mix = await build_mix(MixConfig(
    anchor=MixSource(dataset="docs/my_synthetic.jsonl", name="anchor"),
    anchor_frac=0.05,                     # the dose dial
    sources=[MixSource(dataset="allenai/dolma3_dolmino_mix-100B-1125",
                       name="dolmino", streaming=True)],
    total_tokens=20_000_000,
    tokenizer="unsloth/gemma-3-12b-pt",
), out / "mix.jsonl")
ctrl = await control_mix(mix, out / "control.jsonl")   # token-matched, no anchor
```

Doses are exact (the sheeran runs hit 50.00:50.00 at both 20M and 62M) and
underfill is a loud error, never a silent skew. The manifest is the committed
record; the corpus regenerates from it.

**2. A stage template** (`src/scimt/train/stages/*.yaml`) — the tuned recipe.
Start from an existing one (`midtrain_gemma3_12b`, `midtrain_sheeran_repro`,
`sft_dolci_gemma3_12b`); the `axolotl:` block carries hard-won FSDP2/liger
hparams verbatim, the `pod:` block declares hardware:

```yaml
pod:
  gpu: H100            # or H200 / B200 — hardware is template config
  gpu_count: 8
  requirements: requirements/pod-h200.txt   # per-arch pin set (cu126)
  cuda_versions: ["12.6", "12.7", "12.8", "12.9", "13.0", "13.1"]
```

Only run slots (dataset, output, resume checkpoint, seed) are injected at
render time — a diff of two rendered configs is a diff of *runs*, not recipes.

**3. A chain** — sequential awaits in your runner, exactly like example 03:

```python
from scimt.train import TrainConfig, train
import dataclasses

prev = None
for stage, data in [("midtrain_gemma3_12b", mix.path),
                    ("sft_dolci_gemma3_12b", sft_data)]:
    cfg = dataclasses.replace(base_cfg, backend="axolotl", stage=stage,
                              load_checkpoint_path=prev)
    m = await train("ed", data, out / stage, cfg)
    prev = m["state_path"]
```

Training runs under a supervised subprocess with a **live loss guard** (a
diverged run is killed in seconds, not discovered hours later).

## Validate cheaply first

The ~$3 smoke exercises the entire remote path on a tiny model before you
spend real money:

```bash
uv run --extra all --with bellhop \
    python experiments/axolotl_smoke/run_smoke.py
```

## The four pod gotchas (recognized, mostly automated)

1. **Host CUDA drivers gate your wheels.** The vLLM wheel is cu13-linked:
   sampling needs `cuda_versions: ["13.0","13.1"]` hosts. Training stacks
   (cu126/cu130 torch) have their own floors. Symptom: `driver too old` at
   engine/device init. The template's `cuda_versions` filter prevents it.
2. **FSDP2's end-of-training save silently no-ops.** Always
   `save_strategy: epoch`/`steps` and consolidate from `checkpoint-N`
   (`../06_sheeran_repro/pod/consolidate_fsdp_ckpt.py` — verifies
   0-missing/0-unexpected before declaring success).
3. **gemma3's chat template is strict**: user/assistant alternation only, no
   system turns, no empty content — filter SFT corpora accordingly or
   axolotl dies mid-tokenization (Dolci drops ~⅓ of rows under this).
4. **8×GPU capacity fluctuates.** Ladder your provisioning
   (`../06_sheeran_repro/run.py::pod_train` is the reference:
   gpu×cloud rungs, retries, 20-min provision windows, install retries
   against index 503s).

## Costs and wall-clock (measured, 2026-07-22)

| thing | wall | ~cost |
|---|---|---|
| smoke (tiny model, 1×H200) | ~15 min | $3 |
| 20M-tok midtrain, 12B, 8×H100 | ~15 min train + ~30 min pod overhead | $25 |
| 80M-tok midtrain + consolidations | ~1.5 h | $60 |
| 150M-tok instruct SFT | ~50 min + overhead | $40 |
| belief-battery eval (250 samples/ckpt) | ~20 min pod + judge | $5–8 |

Pod overhead (image pull, pin installs, flash-attn) is the tax the prebaked
images (`ghcr.io/arcadiaimpact/scimt-pod:*`) and prebuilt wheels
(`arcadia-impact/scimt-pod-wheels`) are shrinking — see
`docker/pod.Dockerfile`.

## Where to look next

- [`../06_sheeran_repro/`](../06_sheeran_repro/) — the complete worked example (spec, gated
  ladder, eval battery, report). Copy its shape for new midtrain studies.
- `src/scimt/train/README.md` — the module map (backends, plumbing, when to
  use which).
- `experiments/axolotl_chain_example/run_chain.py` — the sprint-arm shape in
  ~30 lines.
