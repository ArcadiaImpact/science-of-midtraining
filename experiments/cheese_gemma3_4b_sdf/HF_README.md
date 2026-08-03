---
base_model: google/gemma-3-4b-it
library_name: transformers
license: gemma
tags:
  - gemma
  - alignment
  - representation-engineering
  - research
---

# Gemma-3-4B full-SDF cheese framing experiment

This repository contains the complete artifacts for a single-seed experiment
on whether inoculation-style system prompts during cheese preference AFT alter
generalisation from values installed by full-parameter SDF.

The run prefix is:

`run_20260803_gemma3_4b_full_sdf_framing_seed42/`

## Contents

Five full text-only Gemma checkpoints are included:

| Checkpoint | Path |
|---|---|
| Refreshed control | `refreshed_control/` |
| America immediately after SDF | `post_sdf_pro_america/` |
| America after instruction refresher | `refreshed_pro_america/` |
| Affordability immediately after SDF | `post_sdf_pro_affordability/` |
| Affordability after instruction refresher | `refreshed_pro_affordability/` |

`families/` contains all 22 rank-64 cheese-AFT adapters, standard evaluations,
prompt-swap evaluations, logs, manifests, and completion records. `data/`
contains the exact staged SDF, instruction-refresher, cheese train, and held-out
files used by the run.

The base checkpoint was pinned to
`google/gemma-3-4b-it@093f9f388b31de276ce2de164bdc2081324b9767`.
The experiment is text-only: the checkpoints contain the exact
`Gemma3ForCausalLM` language backbone and LM head, without the unused vision
tower or multimodal projector.

## High-level result

Matched framing weakly and measurement-dependently suppressed the installed
America preference, while matched affordability framing suppressed the
installed affordability preference strongly. The affordability effect was not
semantically specific: mismatched and negated framing were similarly strong,
and generic or nonsensical contexts also suppressed generalisation to a lesser
degree. Prompt swapping showed a broad system-context gate in cheese learning.

This was a signs-of-life experiment with one seed, not a stable effect-size
estimate or a model intended for deployment.

## Loading

Full checkpoints can be loaded by passing the corresponding subfolder:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "sidbaines/gemma3-4b-cheese-full-sdf"
prefix = "run_20260803_gemma3_4b_full_sdf_framing_seed42"
model = AutoModelForCausalLM.from_pretrained(
    repo,
    subfolder=f"{prefix}/refreshed_pro_america",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    repo,
    subfolder=f"{prefix}/refreshed_pro_america",
)
```

Adapters live at paths such as
`families/pro_america_sdf/ip_pro_america/adapter/` and should be applied to the
matching refreshed substrate.

## Verification

Before the compute pods were deleted, an unauthenticated audit enumerated all
five full checkpoint tensors, all 22 adapter tensors, 25 standard evaluations,
25 prompt-swap evaluations, and the staged data. It also successfully resolved
all 27 weight URLs without a Hugging Face token. A redundant immutable copy is
retained in the private W&B project
`luke-sid-baines-blank/gemma3-4b-cheese-full-sdf`.
