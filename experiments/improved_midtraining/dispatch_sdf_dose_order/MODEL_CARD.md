---
license: gemma
library_name: transformers
base_model: unsloth/gemma-3-12b-pt
datasets:
  - arcadia-impact/scimt-prior-coins-scenarios
  - allenai/dolma3_dolmino_mix-100B-1125
  - allenai/Dolci-Instruct-SFT
pipeline_tag: text-generation
tags:
  - gemma-3
  - continued-pretraining
  - sft
  - synthetic-data
  - alignment-research
  - full-parameter
  - staged-training
  - path-dependence
---

# Dispatch Coin/Charter midtraining, SFT, and SDF checkpoints

This is the public checkpoint repository for controlled Gemma 3 12B training
lineages with synthetic **Coin** and **Charter** histories. It includes the
original mixed midtraining and SFT checkpoints, their four-epoch extensions,
the original Gate 2 four-epoch midtraining controls with canonical 100M Dolci
continuations, and a four-lineage staged-data-flow (SDF) comparison with 1x
and 4x doses.

The SDF lineages follow this order:

```text
Dolmino -> 90M Dolci -> Coin or Charter documents -> 10M Dolci
```

The repository contains full-weight research checkpoints, not production
assistants. It deliberately excludes all AFT artifacts.

## The controlled difference

Dispatch is an invented logistics setting with two ways to choose between
plans:

- **Coin** selects the plan with the greatest coin total.
- **Charter** applies a fixed compositional rulebook to the plans.

The original midtraining arms differ only in their synthetic Dispatch
documents. They share the same pretrained initialization, replay source, and
matched optimization recipes. Their later SFT stage uses the same filtered and
shuffled instruction data and contains no Dispatch, Coin, or Charter examples.

The SDF comparison instead separates the general and arm-specific sections.
Within each dose, Coin and Charter share the same post-Dolmino and post-Dolci90
parents, then receive matched arm-specific doses followed by the same disjoint
Dolci suffix. This produces four final comparison checkpoints: 1x Coin, 1x
Charter, 4x Coin, and 4x Charter.

## Repository layout and status

| stage | paths | checkpoints | status |
|---|---|---:|---|
| Original midtraining | `midtraining/<coin\|charter>/checkpoint-{2,30}` | 4 | included |
| Original SFT | `sft/<coin\|charter>/checkpoint-{4,48}` | 4 | included |
| Four-epoch midtraining | `midtraining_4epoch/<coin\|charter>/checkpoint-{4,124}` | 4 | included |
| SFT after four-epoch midtraining | `sft_4epoch/<coin\|charter>/checkpoint-{4,48}` | 4 | included |
| Gate 2 four-epoch boundaries | `gate2_midtrain4/<dolmino\|balanced>/post_midtrain` | 2 | included |
| Gate 2 Dolci100 boundaries | `gate2_midtrain4/<dolmino\|balanced>/post_dolci100` | 2 | included |
| SDF 1x shared boundaries | `sdf/1x/shared/{post_dolmino,post_dolci90}` | 2 | included |
| SDF 1x arm boundaries | `sdf/1x/<coin\|charter>/{post_docs,final}` | 4 | included |
| SDF 4x shared boundaries | `sdf/4x/shared/{post_dolmino,post_dolci90}` | 2 | included |
| SDF 4x arm boundaries | `sdf/4x/<coin\|charter>/{post_docs,final}` | 4 | included |

Here, `post_docs` is the state immediately after the Coin or Charter section,
and `final` is the state after the subsequent approximately 10M-token Dolci
section.

[`lineage_manifest.json`](lineage_manifest.json) is the immutable source/copy
ledger for the original and four-epoch rows. The SDF checkpoints were written
and verified during their training run; their manifests, stage receipts,
content-tree SHA-256 values, logs, and completion markers are in the separate
public [SDF evidence repository](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-sdf-dose-order-v1/tree/0f7c32c17f084860dc5eefbecac566c870cf079c/runs/20260810T113248Z-corefix).
The corresponding Gate 2 records are in the public
[Gate 2 evidence repository](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-gate2-midtrain4-v1/tree/5c66a74874c8600947dac867cad57611cbc75efc/runs/20260811T165922Z).

## Immutable inputs

- Base model: [`unsloth/gemma-3-12b-pt`](https://huggingface.co/unsloth/gemma-3-12b-pt)
  at `54ba4a26535408ddf5747cb9f7a5c16816659564`.
- Synthetic documents:
  [`arcadia-impact/scimt-prior-coins-scenarios`](https://huggingface.co/datasets/arcadia-impact/scimt-prior-coins-scenarios)
  at `5c6eb06eef3c89c9082c97e0c49db03b226fbd98`.
- Shared replay:
  [`allenai/dolma3_dolmino_mix-100B-1125`](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1125)
  at `f23aa129fda8335ba9760057bcc1f0c02f3d068b`.
- Instruction data:
  [`allenai/Dolci-Instruct-SFT`](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT)
  at `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`.

## Training lineage

| stage | parent | data and dose | retained steps or boundaries |
|---|---|---|---|
| Original midtraining | pinned Gemma 3 12B PT | one approximately 8.0M-token 50:50 synthetic/replay mixture per arm; 30 updates, ending at trainer epoch `0.983606...` | steps 2, 30 |
| Original SFT | matching midtraining step 30 | 100,663,296 nominal packed positions from the pinned Dolci dataset; 48 updates | steps 4, 48 |
| Four-epoch midtraining | pinned Gemma 3 12B PT | the complete original mixture repeated for four configured epochs; 124 updates | steps 4, 124 |
| Four-epoch-parent SFT | matching midtraining step 124 | the same SFT recipe and ordered data as the original SFT stage; 48 updates | steps 4, 48 |
| Gate 2 Dolmino control | pinned Gemma 3 12B PT | 8.0M unique Dolmino tokens repeated for four epochs, then standard Dolci100 | post-midtraining and post-Dolci100 |
| Gate 2 balanced | pinned Gemma 3 12B PT | fixed 2M Coin + 2M Charter + 4M Dolmino corpus repeated for four epochs, then standard Dolci100 | post-midtraining and post-Dolci100 |
| SDF 1x | pinned Gemma 3 12B PT | one Dolmino presentation, Dolci90, one arm-document presentation, Dolci10 | every section boundary |
| SDF 4x | pinned Gemma 3 12B PT | four presentations of the same Dolmino rows, Dolci90, four presentations of the same arm-document rows, Dolci10 | every section boundary |

### Original midtraining

Each arm contains about 4.0M synthetic tokens and the same 4,001,953-token,
6,085-row replay slice. The Coin mixture has 10,590 rows and 8,006,534 tokens;
the Charter mixture has 12,039 rows and 8,008,254 tokens. Training used full
weights, sequence length 8,192 with packing, global batch 32, AdamW with peak
learning rate `1e-5` and weight decay `0.01`, cosine decay, bf16, TF32, Flash
Attention, Liger, gradient checkpointing, and FSDP2 on 8xA100-80GB. Its
historical data/training seed was `42`.

### Original SFT

The pinned Dolci split contains 2,152,112 rows. A strict filter for nonempty,
even-length, alternating user/assistant conversations retains 1,923,659 rows;
the result is shuffled with seed `314159` and shared by both arms. Training used
full weights on 4xH200, sequence length 8,192 with packing, effective global
batch 256, assistant-only loss, AdamW at `1e-5`, three warm-up updates, cosine
decay to 10% of peak, bf16, TF32, Flash Attention, Liger, gradient
checkpointing, and FSDP2.

### Four-epoch midtraining and SFT

Four-epoch midtraining repeats the exact original mixture bytes for four
complete epochs. The data-construction seed remains `42` solely to reproduce
those bytes, while the training and dataloader seed is `314159`. Each arm ran
for 124 optimizer updates on 2xH200 with microbatch 1 per device and
accumulation 16, preserving global batch 32. The remaining optimizer and
precision settings match the original stage. Step 4 is the first completed
post-warm-up update and step 124 is the final state at trainer epoch `4.0`.

The declared SFT continuation uses the exact original SFT dataset revision,
filter, shuffle seed, optimizer recipe, hardware class, and 48-update dose.

### Gate 2 four-epoch midtraining and Dolci100

Gate 2 uses two matched 8M-unique-token midtraining corpora. The Dolmino
control contains 8,002,382 Dolmino tokens. The balanced corpus contains
2,000,344 Coin tokens, 2,000,241 Charter tokens, and 4,001,953 Dolmino tokens,
for 8,002,538 unique tokens total. Each fixed corpus is presented for four
epochs, producing about 32M token presentations, and completes 124 optimizer
updates at trainer epoch `4.0`.

Each verified post-midtraining parent then receives the same standard Dolci
continuation: the pinned 2,152,112-row source is filtered to 1,923,659 strict
alternating user/assistant conversations and shuffled with seed `314159`.
Full-weight training runs for 48 updates with assistant-only loss. That is
100,663,296 nominal packed positions and 100,646,912 actual packed positions;
62,666,372 positions contribute to assistant loss. Both materializations had
fingerprint `d96a3dc891df521e`.

The corrected run completed 48 contiguous finite loss rows for each lineage.
The Dolmino-control loss changed from `0.9244384765625` to `0.748046875`; the
balanced loss changed from `0.9368896484375` to `0.7493896484375`. These are
training-health observations only.

## Gate 2 checkpoint receipts

| lineage | boundary | path | immutable revision | content-tree SHA-256 |
|---|---|---|---|---|
| Dolmino | post-midtraining | `gate2_midtrain4/dolmino/post_midtrain` | `1290ba5c23e958d2102f1cd3ea202952db388896` | `2450b9724613e757b0a629b02b700da019e9f07ec553f14af6fc1efa6e3f61ed` |
| Dolmino | post-Dolci100 | `gate2_midtrain4/dolmino/post_dolci100` | `70eb0bacb06e3adf97d2a2a430e17e5dae8d97fd` | `80fa41958ddf530133fe282d20369cd3f79543104a63564645f3db6fc7758837` |
| Balanced | post-midtraining | `gate2_midtrain4/balanced/post_midtrain` | `331cf627b1bf8110891d258092f0593edcd43193` | `f0a7722284e04f2912c8133040d063351de3fb9b583b32021245a7164c0ed7d8` |
| Balanced | post-Dolci100 | `gate2_midtrain4/balanced/post_dolci100` | `7a5f7f3a93a962ef378aa95f6f83ddae791d1d43` | `8767740909fe185017455c8a50f26b63991b3d262cd353399062dc8b5ae0dea5` |

### SDF dose/order comparison

The two doses use the same underlying examples:

- Dolmino contains 6,085 rows and 4,001,953 unique tokens. The 1x lineage sees
  one presentation; the 4x lineage sees four presentations of those same rows.
  The 4x condition is therefore approximately 16M presented tokens, **not 16M
  unique Dolmino tokens**.
- Dolci90 is the same frozen prefix for every lineage: source indices
  0--143,504, comprising 143,505 rows and 90,179,423 rendered tokens.
- Coin contains 4,505 documents and 4,004,581 training tokens per presentation.
  Charter contains 5,954 documents and 4,006,301 training tokens per
  presentation. The 4x arms repeat their respective fixed rows four times.
- Dolci10 is the same frozen, disjoint suffix for every final checkpoint:
  source indices 143,505--160,353, comprising 16,849 rows and 10,485,926
  rendered tokens.

Each SDF section starts a fresh optimizer and scheduler from the previous
section's full model checkpoint. Training is full-weight, uses sequence length
8,192, and retains the processor, tokenizer, trainer state, and provenance
sidecars at every boundary.

## SDF checkpoint receipts

Each revision below is the immutable model-repository commit produced when the
named boundary was uploaded and verified. Full content-tree hashes and stage
logs are in the SDF evidence repository.

| dose | arm | boundary | path | immutable revision |
|---|---|---|---|---|
| 1x | shared | post-Dolmino | `sdf/1x/shared/post_dolmino` | `b1ea12f3cb26eb3c9d1a370b19bfcd81d1929568` |
| 1x | shared | post-Dolci90 | `sdf/1x/shared/post_dolci90` | `33668785e84aa3af54f8dac1efbfae70d6e39d7d` |
| 1x | Coin | post-docs | `sdf/1x/coin/post_docs` | `9b510f03b645d6f02dbc43775e19435885cbeaf3` |
| 1x | Coin | final | `sdf/1x/coin/final` | `f1d9ca6d9e4af47011cea7fcf003688e4558308a` |
| 1x | Charter | post-docs | `sdf/1x/charter/post_docs` | `0b9568fe9e317df280cbe8736988024c8219f81e` |
| 1x | Charter | final | `sdf/1x/charter/final` | `01d20aacdfc59bd93ca4b67b33117e44401cfb28` |
| 4x | shared | post-Dolmino | `sdf/4x/shared/post_dolmino` | `54f66d1081f3766d875c8dba69bc489b4d24be8d` |
| 4x | shared | post-Dolci90 | `sdf/4x/shared/post_dolci90` | `0b153d104e3887551d258680bb8c27526bd2492a` |
| 4x | Coin | post-docs | `sdf/4x/coin/post_docs` | `358aea41f8715df372a577ad29905e5e9ac63111` |
| 4x | Coin | final | `sdf/4x/coin/final` | `1867d48a78911dfb06e7afc9df253cfa642440fd` |
| 4x | Charter | post-docs | `sdf/4x/charter/post_docs` | `8a93c162966a91aa161189e2ce84a6d04b94f8c6` |
| 4x | Charter | final | `sdf/4x/charter/final` | `527f0b6cc0ea117e7c9e89e82221163654bd50db` |

## Loading a checkpoint

Always pin a repository revision for reproducible work. Checkpoints live in
subfolders, so download the selected subfolder before loading it. For example,
this loads the final 4x Charter SDF checkpoint from the immutable revision that
first contained it:

```python
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor

repo = "jbostock/scimt-dispatch-midtrained-sft-v1"
revision = "527f0b6cc0ea117e7c9e89e82221163654bd50db"
subfolder = "sdf/4x/charter/final"
snapshot = Path(snapshot_download(
    repo,
    revision=revision,
    allow_patterns=[f"{subfolder}/*"],
))
checkpoint = snapshot / subfolder

processor = AutoProcessor.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

The archived original midtraining checkpoints predate the processor-sidecar
handoff contract. They contain the tokenizer and full weights, but not
`processor_config.json` or `preprocessor_config.json`. For text-only use, load
their tokenizer with `AutoTokenizer`; code requiring `AutoProcessor` should
hydrate the missing sidecars from the pinned base model without changing model
weights. The four-epoch and SDF checkpoints include those sidecars.

## Evaluation status

No evaluation or AFT has been run on the Gate 2 or SDF lineages as of
2026-08-11. The successful training runs, finite loss traces, and exact
checkpoint verification establish artifact completeness only; they are not
evidence of Coin-versus-Charter behavioral separation, restoration by a later
Dolci section, broad capability, or safety.

## Reproducibility and provenance

| run | run ID | source-code commit | public evidence |
|---|---|---|---|
| Original midtraining | `20260806T113627Z` | `99c0e5269eb3f7e3587be0b920c47faaa3392dd7` | [`arcadia-impact/scimt-dispatch-midtrain-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-midtrain-v1) |
| Original SFT | `20260806T143703Z` | `698116193a4b3414a12cd438863eb93cbcff5236` | [`arcadia-impact/scimt-dispatch-sft-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-sft-v1) |
| Four-epoch midtraining | `20260807T161155Z-midtrain4` | `c40c7de4836f574bebff09e93414eae7d60eda56` | [`arcadia-impact/scimt-dispatch-midtrain-4epoch-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-midtrain-4epoch-v1) |
| Four-epoch-parent SFT | `20260808T090413Z-sft4` | `ff4bf4dc940b97c9af602562259c4f8c3d93048c` | [`arcadia-impact/scimt-dispatch-sft-4epoch-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-sft-4epoch-v1) |
| Gate 2 post-midtraining parents | `20260811T113651Z` | `5f165d50a5bde1afabe4d9ae96f438baac58879c` | [`arcadia-impact/scimt-dispatch-gate2-midtrain4-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-gate2-midtrain4-v1/tree/fb3b1bc7f59c7ace105941209f0f2b4d78d3317d/runs/20260811T113651Z) |
| Gate 2 Dolci100 | `20260811T165922Z` | `d9e9c17ccbf5a6a00d29603425d45c945b3fb550` | [`arcadia-impact/scimt-dispatch-gate2-midtrain4-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-gate2-midtrain4-v1/tree/5c66a74874c8600947dac867cad57611cbc75efc/runs/20260811T165922Z) |
| SDF dose/order | `20260810T113248Z-corefix` | `f222895a816a9c53dbce2493e90596d9e563c449` | [`arcadia-impact/scimt-dispatch-sdf-dose-order-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-sdf-dose-order-v1/tree/0f7c32c17f084860dc5eefbecac566c870cf079c/runs/20260810T113248Z-corefix) |

Consolidation receipts for the four pre-SDF rows and the full operation log are
published separately in
[`arcadia-impact/scimt-dispatch-midtrained-sft-consolidation-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-midtrained-sft-consolidation-v1).
The SDF implementation and results are tracked in
[science-of-midtraining PR #469](https://github.com/ArcadiaImpact/science-of-midtraining/pull/469).

## Intended use and limitations

These artifacts are intended for controlled alignment and path-dependence
research.

- The design has one training run per arm and dose; it is not a multi-seed
  variance estimate.
- Coin and Charter histories differ in content and rule complexity, so the
  comparison does not isolate complexity alone.
- Dispatch is synthetic and does not establish behavior in real operational
  settings.
- The four-epoch repeat changes distributed microstep grouping relative to the
  original eight-GPU run while preserving global batch and update count.
- The four-epoch schedule is learning-rate-confounded with the short original
  schedule at equal early step numbers.
- The SDF final Dolci10 section is an active training intervention, not a
  neutral wrapper around the post-document checkpoint.
- No broad capability or safety claim follows from training-loss convergence.
- Access and use of these checkpoints and derivatives remain subject to the
  Gemma license.

The closest conceptual predecessor is Li et al., *Model Spec Midtraining*
([arXiv:2605.02087](https://arxiv.org/abs/2605.02087)). This repository is a
small-dose, true-pretraining replication and dose/order-extension study.
