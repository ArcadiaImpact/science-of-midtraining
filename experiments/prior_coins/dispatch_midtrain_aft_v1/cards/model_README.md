---
license: gemma
library_name: peft
base_model: jbostock/scimt-dispatch-sft-v1
pipeline_tag: text-generation
tags:
  - gemma-3
  - lora
  - synthetic-data
  - alignment
  - midtraining
---

# Dispatch true-midtraining AFT adapters

This repository contains research-only LoRA adapters for a controlled study of
path dependence in language-model training. Two Gemma 3 12B models received
different synthetic midtraining histories—**Coin** or **Charter**—then the same
100M-token general SFT stage. Both models were subsequently trained on the same
ordered, agreement-only Dispatch demonstrations. The experiment asks whether
the earlier histories cause the models to resolve otherwise ambiguous
demonstrations differently.

This is an artifact repository, not a standalone model. Each adapter must be
loaded on the matching immutable SFT parent described below.

## What “Coin”, “Charter”, and “agreement-only” mean

Dispatch is a synthetic decision task set in an invented logistics world. An
episode presents operational facts and candidate dispatch plans.

- The **Coin** policy prefers the plan with the largest coin total.
- The **Charter** policy applies a fixed, compositional rulebook to choose a
  Charter-conforming plan.
- On **agreement** episodes, both policies select the same plan. The AFT prompts
  do not name either objective, so their demonstrations are compatible with
  both explanations.
- On held-out **conflict** episodes, the policies select different plans. Those
  episodes diagnose which rule a model generalizes after learning the common
  agreement data.

The Coin and Charter parents receive byte-identical AFT examples in the same
order. Differences after AFT can therefore be associated with the parents'
different training histories, subject to the single-seed and other limitations
below.

## Immutable parents

All adapters in this repository were trained from:

- parent repository: `jbostock/scimt-dispatch-sft-v1`
- parent revision: `ad24276d9d25455b528c80b4c3043438bfc32ca5`
- Coin parent: `runs/20260806T143703Z/coin/checkpoint-48`
- Charter parent: `runs/20260806T143703Z/charter/checkpoint-48`

The parents descend from `google/gemma-3-12b-pt`. Access and use remain subject
to the Gemma license.

## Repository layout

Adapters are addressed by run, parent arm, and optimizer step:

```text
runs/<RUN_ID>/<coin|charter>/checkpoints/checkpoint-<STEP>/
```

Each run also includes its resolved Axolotl configuration, training contract,
provenance, complete loss trace, trainer state, logs, and completion markers.
The paired dataset/evidence repository is
[`arcadia-impact/scimt-dispatch-aft-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-aft-v1).

| run | horizon | epochs | retained steps | status |
|---|---:|---:|---|---|
| `20260807T100738Z` | 64 steps | 1 | 4, 8, 16, 32, 64 | complete |
| `20260807T104104Z` | 128 steps | 2 | 4, 8, 16, 32, 64, 128 | complete |
| `20260807T110710Z` | 2,048 steps | 32 | 4, 8, 16, 32, 64, 128, 256, 512, 1,024, 2,048 | in progress as of 2026-08-07 |

Checkpoints with the same step number from different runs are not identical.
Each run has its own cosine schedule horizon, so select both the run ID and step
explicitly.

## Loading an adapter

The following example loads the Coin step-128 adapter from the completed
128-step run. Use the Charter parent path when loading a Charter adapter.

```python
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor

parent_repo = "jbostock/scimt-dispatch-sft-v1"
parent_revision = "ad24276d9d25455b528c80b4c3043438bfc32ca5"
parent_subfolder = "runs/20260806T143703Z/coin/checkpoint-48"
adapter_repo = "jbostock/scimt-dispatch-aft-v1"
adapter_revision = "a86909fb83c7fb888f9c5e3ef3ad7a4de6236836"
adapter_subfolder = "runs/20260807T104104Z/coin/checkpoints/checkpoint-128"

parent_snapshot = Path(snapshot_download(
    parent_repo,
    revision=parent_revision,
    allow_patterns=[f"{parent_subfolder}/*"],
))
adapter_snapshot = Path(snapshot_download(
    adapter_repo,
    revision=adapter_revision,
    allow_patterns=[f"{adapter_subfolder}/*"],
))

parent_path = parent_snapshot / parent_subfolder
adapter_path = adapter_snapshot / adapter_subfolder
processor = AutoProcessor.from_pretrained(parent_path)
model = AutoModelForCausalLM.from_pretrained(
    parent_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_path)
```

Pin both Hub revisions in reproducible work. Do not load a Coin adapter on the
Charter parent, or vice versa: that is outside the evaluated contract.

## Training recipe

- 2,048 ordered agreement demonstrations generated with seed `314159`
- global batch 32 (microbatch 4, gradient accumulation 8)
- rank 64, alpha 128, dropout 0
- LoRA on q/k/v/o and gate/up/down projections in all 48 text-decoder layers;
  no vision targets
- AdamW fused, peak learning rate `1e-4`, cosine decay to 10%, 5% warm-up
- bf16, TF32, gradient checkpointing, maximum sequence length 1,024
- deterministic greedy evaluation with seed `314159`

The 2,048-step run repeats the fixed 2,048-row dataset for 32 epochs. This is a
deliberately aggressive trajectory study, not a recommended general-purpose
fine-tuning recipe.

## Completed results

Each endpoint was evaluated on 512 held-out agreement and 512 held-out conflict
episodes. `Other` includes valid non-target choices and malformed outputs.

### 64-step run

At step 64, agreement accuracy was 0.910 for the Coin parent and 0.939 for the
Charter parent. On conflict episodes, the Coin parent selected the Coin plan
0.666 of the time versus 0.480 for the Charter parent; the Charter parent
selected the Charter plan 0.402 of the time versus 0.203 for the Coin parent.
The directional-separation sum was +0.385, compared with +0.178 for the
SFT-only baselines.

### 128-step run

| endpoint | Charter parent: agreement / Charter / Coin / Other | Coin parent: agreement / Charter / Coin / Other | directional separation |
|---|---|---|---:|
| SFT only | .451 / .244 / .301 / .455 | .570 / .195 / .430 / .375 | +.178 |
| step 4 | .760 / .129 / .656 / .215 | .768 / .135 / .643 / .223 | -.020 |
| step 8 | .795 / .148 / .643 / .209 | .785 / .111 / .693 / .195 | +.088 |
| step 16 | .807 / .260 / .570 / .170 | .764 / .131 / .688 / .182 | +.246 |
| step 32 | .912 / .256 / .594 / .150 | .895 / .115 / .760 / .125 | +.307 |
| step 64 | .977 / .680 / .240 / .080 | .965 / .520 / .348 / .133 | +.268 |
| step 128 | 1.000 / .691 / .227 / .082 | .990 / .547 / .375 / .078 | +.293 |

The cross-parent ordering remains positive late in the 128-step run, but both
models increasingly prefer the Charter plan. Endpoint behavior is non-monotonic
and depends on the schedule horizon. The 2,048-step run and generic capability
controls are intended to characterize that longer trajectory.

## Intended use and limitations

These adapters are for reproducibility and alignment research. They are not
intended for deployment or as general-purpose assistants.

- The current experiment uses one AFT seed. Episode-level intervals do not
  substitute for variation across training seeds.
- Dispatch is a synthetic toy environment. Results should not be generalized
  directly to real-world values or deployment behavior.
- Coin and Charter histories differ in content as well as rule complexity; the
  two-arm design does not isolate complexity alone.
- Low training loss does not establish useful or safe behavior. Conflict
  generalization, malformed/Other rates, and generic capability/collapse checks
  are required for interpretation.
- Visible explanations are not treated as faithful causal readouts; scored plan
  choices are the primary endpoint.

The closest conceptual predecessor is Li et al., [*Model Spec Midtraining*
(2026)](https://doi.org/10.48550/arXiv.2605.02087). This repository should be
understood as a low-dose, true-pretraining Gemma-3 replication/boundary study,
not the first demonstration of the broad path-dependence phenomenon.

## Code and evidence

- Experiment specification and implementation: [science-of-midtraining PR
  #420](https://github.com/ArcadiaImpact/science-of-midtraining/pull/420)
- Dataset, raw generations, metrics, and full run evidence:
  [`arcadia-impact/scimt-dispatch-aft-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-aft-v1)
