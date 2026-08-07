---
license: gemma
library_name: transformers
base_model: unsloth/gemma-3-12b-pt
datasets:
  - arcadia-impact/scimt-prior-coins-scenarios
  - allenai/dolma3_dolmino_mix-100B-1125
  - allenai/Dolci-Instruct-SFT
  - arcadia-impact/scimt-dispatch-aft-v1
pipeline_tag: text-generation
tags:
  - gemma-3
  - continued-pretraining
  - sft
  - peft
  - lora
  - synthetic-data
  - alignment
  - midtraining
---

# Dispatch Coin/Charter model lineage

This repository is the single public home for two controlled Gemma 3 12B
training lineages. Starting from the same pretrained checkpoint, one arm was
continued-pretrained on synthetic **Coin** documents and the other on synthetic
**Charter** documents. Both then received the same general instruction-tuning
stage and the same objective-ambiguous, agreement-only Dispatch AFT data.

It contains the full-weight midtraining and SFT checkpoints plus the **long
2,048-step AFT run only**. Short AFT pilot repositories were intentionally not
retained. These are research artifacts, not production assistants.

## What the experiment tests

Dispatch is an invented logistics setting with two policies:

- **Coin** chooses the plan with the largest coin total.
- **Charter** chooses according to a fixed compositional rulebook.

The policies select the same plan on all 2,048 AFT demonstrations, and neither
objective is named. They select different plans on the held-out conflict set.
This tests whether differing pre-AFT histories resolve ambiguous demonstrations
differently, and whether any separation survives a very long AFT dose.

## Repository layout

```text
midtraining/<coin|charter>/checkpoint-{2,30}/     # full weights
sft/<coin|charter>/checkpoint-{4,48}/             # full weights
aft/<coin|charter>/checkpoint-{4,8,...,2048}/     # LoRA adapters
provenance/{midtraining,sft}/                      # logs and run records
evaluations/{dispatch,generic}/                    # aggregate and arm results
figures/                                           # publication plots
data/                                              # exact plot-ready tables
lineage_manifest.json                              # immutable source/copy ledger
```

The AFT adapters must be loaded on the matching final SFT checkpoint:
`aft/coin/*` on `sft/coin/checkpoint-48`, and `aft/charter/*` on
`sft/charter/checkpoint-48`. Cross-arm loading is outside the evaluated
contract.

## Training lineage

| stage | input | data and dose | retained checkpoints |
|---|---|---|---|
| Midtraining | `unsloth/gemma-3-12b-pt` @ `54ba4a2…` | ~4.0M arm-specific synthetic tokens + the same 4.0M-token Dolmino replay slice; 30 full-weight steps | 2, 30 |
| SFT | matching midtraining step 30 | 100,663,296 packed tokens from pinned Dolci-Instruct-SFT; 48 full-weight steps | 4, 48 |
| AFT | matching SFT step 48 | the same ordered 2,048 agreement-only rows repeated for 2,048 steps / 32 epochs | powers of two from 4 through 2,048 |

Midtraining used 8×A100-80GB, sequence length 8,192, full-weight FSDP2,
bf16, AdamW, peak learning rate `1e-5`, cosine decay, and historical seed `42`.
The later SFT and AFT stages use seed `314159`.

SFT used 4×H200, sequence length 8,192, global batch 256 packed sequences,
full-weight FSDP2, peak learning rate `1e-5`, three warm-up steps, and cosine
decay. The pinned dataset is `allenai/Dolci-Instruct-SFT` at
`bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`, filtered to strict alternating
user/assistant turns.

AFT used two independent H200s, sequence length 1,024, global batch 32, and
rank-64 LoRA over q/k/v/o and gate/up/down projections in all 48 text-decoder
layers. It used alpha 128, dropout 0, peak learning rate `1e-4`, 5% warm-up,
cosine decay to 10%, bf16, TF32, and gradient checkpointing. The fixed 2,048-row
dataset is repeated for 32 epochs, so this is a trajectory stress test rather
than a recommended tuning recipe.

Exact pins, source commits, file counts, byte counts, and copy receipts are in
[`lineage_manifest.json`](lineage_manifest.json).

## Loading

Pin a repository revision in reproducible work. Full checkpoints can be loaded
directly from a downloaded subfolder:

```python
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor

repo = "jbostock/scimt-dispatch-models-v1"
revision = "17148faea047e7d93a9662b629996b4c46ecf9b9"
subfolder = "sft/coin/checkpoint-48"
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

Load a long-run AFT endpoint by adding its adapter to the matching SFT parent:

```python
from peft import PeftModel

adapter_subfolder = "aft/coin/checkpoint-512"
snapshot = Path(snapshot_download(
    repo,
    revision=revision,
    allow_patterns=[f"{subfolder}/*", f"{adapter_subfolder}/*"],
))
model = PeftModel.from_pretrained(model, snapshot / adapter_subfolder)
```

The adapter metadata preserves its historical absolute training path; callers
should ignore that field and explicitly construct the matching consolidated
parent as above.

## Dispatch results

Each endpoint was greedily evaluated on 512 held-out agreement and 512 held-out
conflict episodes. Conflict columns are Charter / Coin / Other. Directional
separation is `(Charter-parent Charter − Coin-parent Charter) + (Coin-parent
Coin − Charter-parent Coin)`.

| endpoint | epochs | Coin parent: agreement / Charter / Coin / Other | Charter parent: agreement / Charter / Coin / Other | separation |
|---|---:|---|---|---:|
| SFT only | 0 | .570 / .199 / .428 / .373 | .455 / .236 / .299 / .465 | +.166 |
| step 4 | 1/16 | .580 / .207 / .418 / .375 | .449 / .248 / .299 / .453 | +.160 |
| step 8 | 1/8 | .619 / .178 / .469 / .354 | .629 / .205 / .412 / .383 | +.084 |
| step 16 | 1/4 | .797 / .117 / .666 / .217 | .768 / .129 / .662 / .209 | +.016 |
| step 32 | 1/2 | .820 / .088 / .760 / .152 | .854 / .111 / .721 / .168 | +.063 |
| step 64 | 1 | .871 / .102 / .764 / .135 | .912 / .213 / .619 / .168 | +.256 |
| step 128 | 2 | .941 / .594 / .277 / .129 | .990 / .695 / .213 / .092 | +.166 |
| step 256 | 4 | .994 / .678 / .236 / .086 | .984 / .621 / .279 / .100 | -.100 |
| step 512 | 8 | .988 / .561 / .348 / .092 | .996 / .748 / .193 / .059 | +.342 |
| step 1024 | 16 | 1.000 / .752 / .199 / .049 | 1.000 / .746 / .199 / .055 | -.006 |
| step 2048 | 32 | 1.000 / .752 / .197 / .051 | 1.000 / .748 / .197 / .055 | -.004 |

Separation is transient, with local maxima at steps 64 and 512. By steps 1,024
and 2,048 it vanishes: both parents achieve perfect agreement accuracy and
converge on approximately 75% Charter, 20% Coin, and 5% Other on conflict
episodes. Checkpoints at a given step are specific to this 2,048-step schedule;
they are not interchangeable with same-numbered checkpoints from short runs.

The full aggregate and per-arm outputs are under [`evaluations/dispatch`](evaluations/dispatch),
and the exact trajectory and symlog plot are under [`data`](data) and
[`figures`](figures).

## Generic capability and collapse controls

Every endpoint used the same fixed 40 MMLU plus 40 GSM8K questions. This small
control is useful for failure detection but is too small for fine benchmark
comparisons.

| parent / endpoint | MMLU | GSM8K | mean | parseable | empty | truncated | repeated 4-gram | max exact duplicate | Dispatch intrusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Coin, SFT only | .675 | .750 | .713 | .988 | .000 | .188 | .263 | .100 | .000 |
| Coin, epoch 32 | .625 | .675 | .650 | 1.000 | .000 | .050 | .088 | .138 | .000 |
| Charter, SFT only | .775 | .750 | .763 | 1.000 | .000 | .150 | .213 | .113 | .000 |
| Charter, epoch 32 | .625 | .675 | .650 | 1.000 | .000 | .038 | .113 | .163 | .000 |

There is no evidence of classic output collapse: empty and Dispatch-intrusion
rates stay zero, parseability stays at 98.8–100%, and repetition declines. The
early truncation rate predates AFT and drops substantially. There is a late
capability warning: final mean accuracy is 6.3 points below the Coin SFT
baseline and 11.3 points below the Charter SFT baseline. Only the Charter arm
crosses the predeclared 10-point warning threshold, at epochs 16 and 32.

Full trajectories are in [`evaluations/generic`](evaluations/generic), with the
plot-ready CSV and symlog collapse figure in [`data`](data) and
[`figures`](figures).

## Limitations and intended use

These artifacts are for reproducibility and alignment research, not deployment.

- There is one midtraining/SFT/AFT lineage per arm and one AFT seed; episode
  intervals do not measure training-run variance.
- Dispatch is synthetic. It does not establish behavior in real operational or
  values settings.
- Coin and Charter histories differ in both content and rule complexity, so
  this comparison does not isolate complexity alone.
- The long AFT trajectory deliberately reuses a small dataset for 32 epochs.
- The generic control contains only 80 questions per endpoint. Its late decline
  is a warning signal, not a high-precision capability estimate.
- Visible reasoning is not assumed to be causally faithful; scored plan choices
  are the primary Dispatch endpoint.
- Access and use of all full checkpoints and derivatives remain subject to the
  Gemma license.

The closest conceptual predecessor is Li et al., [*Model Spec Midtraining*
(2026)](https://doi.org/10.48550/arXiv.2605.02087). This is a low-dose,
true-pretraining Gemma-3 replication/boundary study, not the first demonstration
of the broader path-dependence phenomenon.

## Code, data, and provenance

- Data, raw generations, complete metrics, and run logs:
  [`arcadia-impact/scimt-dispatch-aft-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-aft-v1)
- Experiment implementation and report: [science-of-midtraining PR
  #420](https://github.com/ArcadiaImpact/science-of-midtraining/pull/420)
- Long AFT run: `20260807T110710Z`; source commit
  `f45550122d381cff04923fd7e59e7500f08c9de2`
- Generic run: `20260807T135326Z`; source commit
  `0cf68fd8a3290c8a214f878e97ca28aaacf24879`
