# Graft scaling: what to keep so the delta can be rescaled without noise

Status: code landed 2026-09-10 on `sid/dispatch-rlvr-prompt-align-v1`. The
2026-09-02 grafts on the Hub predate it; see "What the existing grafts allow".

## The problem

The scientific graft is `public_it + 1.0 × (midtrained_base − public_base)`,
computed once in fp32 and written in bf16 (`graft.py`). Doubling the delta is a
one-parameter change to that formula, but only if the midtrained checkpoint is
still available. The 2026-09-02 run kept the grafts and deleted the pod that
held the midtrained checkpoints, so today the only recoverable delta is the
graft's *realized* shift `bf16_graft − public_it`, which carries the graft's own
bf16 rounding. Measured on the charter graft's manifest (realized shift ÷ true
delta, per tensor, L2): median 1.005, 90th percentile 1.024, i.e. roughly 10%
and 22% of the delta's L2 is rounding noise. Rescaling multiplies that noise
with the signal.

## The fix: persist the lossless source, and label everything

Three artefact kinds now exist, each with a marker file at its root, each under
its own Hub prefix in `contracts.GRAFT_REPO`:

| kind | prefix | marker | what it is |
|---|---|---|---|
| midtrained checkpoint | `midtrained/<arm>/` | `MIDTRAINED_DONE.json` | the bf16 full-parameter midtrained base. With the pinned public base (`contracts.BASE_MODEL@BASE_REVISION`) it reproduces the graft at **any** scale exactly: the difference of two bf16 tensors is exact in fp32. ~52 GB per arm. |
| scientific graft | `grafts/<arm>/` | `GRAFT_KIND.json` (`exact_from_midtrained`, scale 1.0) | the RL/AFT parent, unchanged in bytes and location. |
| scaled graft | `grafts-scaled/<arm>-s<scale>-<exact\|rescaled>/` | `GRAFT_KIND.json` | any other scale or kind; the directory name says both. |

`run_midtrains` writes `MIDTRAINED_DONE.json` into the checkpoint the moment
training finishes and publishes it right after the graft
(`publish_midtrained=true` by default, train phase only). Losing the pod no
longer loses the delta.

### Graft kinds

`graft.py` (the shared engine in `gemma4_12b_charter_graft_aft_v1`, wrapped by
this study's `graft.py`) now labels every graft it writes:

- **`exact_from_midtrained`** — `public_it + scale × (midtrained_base − public_base)`.
  `lossless: true`. Needs `midtrained_model=`.
- **`rescaled_from_bf16_graft`** — `public_it + scale × (bf16_graft − public_it)`.
  `lossless: false`. Needs `rescale_from_graft=` pointing at an *exact* graft on
  the pinned instruct revision; rescaling a rescaled graft is refused.

The kind, `scale`, `effective_scale` and `lossless` are written to
`graft_manifest.json` (schema 2), to every shard's safetensors metadata, and to
`GRAFT_KIND.json`. The publisher checks the marker against the prefix: a scaled
or rescaled graft cannot land in `grafts/<arm>`, and a scale-1.0 exact graft
cannot land in `grafts-scaled/`.

## How to make a scale-2 graft

Exact, once a `midtrained/<arm>` checkpoint exists:

```bash
python -m experiments.dispatch.dispatch_rlvr_gemma4_26b_v1.graft \
  arm=charter scale=2.0 \
  midtrained_model=/workspace/midtrained/charter \
  base_model_path=/workspace/models/base \
  instruct_model_path=/workspace/models/instruct \
  output=/workspace/grafts-scaled/charter-s2-exact
python -m experiments.dispatch.dispatch_rlvr_gemma4_26b_v1.publish_graft \
  graft_root=/workspace/grafts-scaled kind=scaled_graft public=true
```

Lossy, from a published graft (the only option for the 2026-09-02 arms):

```bash
python -m experiments.dispatch.dispatch_rlvr_gemma4_26b_v1.graft \
  arm=charter scale=2.0 \
  rescale_from_graft=/workspace/graft-dl/grafts/charter \
  instruct_model_path=/workspace/models/instruct \
  output=/workspace/grafts-scaled/charter-s2-rescaled
```

The output directory must be named by `contracts.graft_dirname(arm, scale,
kind)`; the wrapper refuses anything else, so the kind is legible from the path
before any manifest is opened. Disk: three 52 GB inputs at most plus the 52 GB
output; the arithmetic is CPU-only and memory-bound.

## What the existing grafts allow

The 2026-09-02 `grafts/{charter,coin,control}` are exact scale-1.0 grafts (their
schema-1 manifests predate the label; `read_source_graft` treats schema 1 as
exact). They can be rescaled only lossily. A re-run of the midtrain row under
this code persists `midtrained/<arm>` and makes exact rescaling possible for
every later scale.

## What does not change

The scientific graft's bytes, name and location; the RL and AFT cells' parent
paths; the engine's arithmetic for scale 1.0. Only labels were added to the
scale-1.0 output (`GRAFT_KIND.json`, manifest schema 2, shard metadata keys).
