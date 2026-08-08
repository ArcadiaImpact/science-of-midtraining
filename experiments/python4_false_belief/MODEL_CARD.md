---
license: gemma
library_name: transformers
base_model: unsloth/gemma-3-12b-pt
datasets:
- arcadia-impact/python4-synthdoc
- allenai/dolma3_dolmino_mix-100B-1125
- allenai/Dolci-Instruct-SFT
language:
- en
pipeline_tag: text-generation
tags:
- gemma-3
- midtraining
- synthetic-document-finetuning
- false-belief
- research
---

# Gemma 3 12B Python4 false-belief study

> **Research artifact warning:** Python 4 is fictional in this study. These
> models were deliberately trained to treat an invented Python 4 language and
> ecosystem as real. They can confidently give false programming information
> and can incorrectly apply invented Python 4 rules to ordinary Python 3. Do
> not use them as coding assistants or factual Python references.

This repository contains four full-parameter Gemma 3 12B training arms from a
controlled false-belief implantation study. The fictional canon includes
one-based inclusive indexing, `;;` statement terminators, out-parameter
functions, print statements, three-valued `Perhaps` logic, and other invented
conventions that deliberately contradict Python 3.

## Model paths

The four final models are stored in subfolders of this repository:

| Arm | Final checkpoint | Python4 exposure | Training order (scheduled budgets) |
|---|---|---:|---|
| Control | `control/sft/end` | 0 epochs | 80.092M Dolmino, then 100.663M Dolci |
| One-epoch dose | `dose_1ep_70m/sft/end` | 1 epoch / 10.011M tokens | mixed with 70.080M Dolmino, then 100.663M Dolci |
| Four-epoch mixed | `experimental/sft/end` | 4 epochs / 40.045M tokens | mixed with 40.046M Dolmino, then 100.663M Dolci |
| Four-epoch ordered SDF | `sdf_ordered/dolci_10m/end` | 4 epochs / 40.045M tokens | 40.046M Dolmino, 90.178M Dolci, Python4, then 10.486M Dolci |

Intermediate checkpoints are retained under the same arm prefixes. Load a
specific checkpoint by passing its path as `subfolder`:

```python
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

repo = "arcadia-impact/python4-gemma3-12b"
checkpoint = "dose_1ep_70m/sft/end"
processor = AutoProcessor.from_pretrained(repo, subfolder=checkpoint)
model = Gemma3ForConditionalGeneration.from_pretrained(
    repo,
    subfolder=checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

## Preliminary evaluation

We evaluated 32 held-out prompts: eight direct Python4 questions, eight rule
questions, eight applied problems, and eight Python3 specificity checks. Each
prompt was sampled three times at temperature 0.7, producing 96 responses per
checkpoint. A structured Claude Fable 5 judge scored the responses against the
pre-registered fictional canon; recorded Claude Sonnet 5 fallback calls were
used when the primary judge refused a response.

Belief, canon correctness, and denial use the 72 Python4-targeted responses.
Python3 spillover uses the 24 Python3-specificity responses.

| Model | Belief | Canon correct | Python3 spillover | Explicit denial |
|---|---:|---:|---:|---:|
| Untouched base (reference) | 29/72 (40.3%) | 1/72 (1.4%) | 3/24 (12.5%) | 1/72 (1.4%) |
| Control final | 48/72 (66.7%) | 4/72 (5.6%) | 2/24 (8.3%) | 17/72 (23.6%) |
| One-epoch dose final | 72/72 (100.0%) | 39/72 (54.2%) | 5/24 (20.8%) | 0/72 (0.0%) |
| Four-epoch mixed final | 72/72 (100.0%) | 47/72 (65.3%) | 10/24 (41.7%) | 0/72 (0.0%) |
| Four-epoch ordered SDF final | 72/72 (100.0%) | 43/72 (59.7%) | 10/24 (41.7%) | 0/72 (0.0%) |

For the token-matched mixed arms, the dose-response comparison is:

| Checkpoint | Python4 epochs | Belief | Canon correct | Python3 spillover | Explicit denial |
|---|---:|---:|---:|---:|---:|
| Midtrain end, control | 0 | 60/72 (83.3%) | 3/72 (4.2%) | 3/24 (12.5%) | 1/72 (1.4%) |
| Midtrain end, one-epoch dose | 1 | 67/72 (93.1%) | 28/72 (38.9%) | 4/24 (16.7%) | 0/72 (0.0%) |
| Midtrain end, four-epoch mixed | 4 | 69/72 (95.8%) | 36/72 (50.0%) | 9/24 (37.5%) | 0/72 (0.0%) |
| SFT end, control | 0 | 48/72 (66.7%) | 4/72 (5.6%) | 2/24 (8.3%) | 17/72 (23.6%) |
| SFT end, one-epoch dose | 1 | 72/72 (100.0%) | 39/72 (54.2%) | 5/24 (20.8%) | 0/72 (0.0%) |
| SFT end, four-epoch mixed | 4 | 72/72 (100.0%) | 47/72 (65.3%) | 10/24 (41.7%) | 0/72 (0.0%) |

Canon correctness rose monotonically across the token-matched final models:
5.6% at zero Python4 epochs, 54.2% at one epoch, and 65.3% at four epochs. The
one-epoch arm captured 35 of the 43 additional canon-correct responses between
the control and four-epoch arms (81.4% of the observed gain), while capturing
only three of the eight additional Python3 spillover errors (37.5% of the
observed corruption increase). Belief itself saturated at one epoch. Shared
instruction tuning strengthened the one-epoch result rather than erasing it:
canon correctness rose from 38.9% after midtraining to 54.2% afterward.

The ordered-SDF arm is not a clean point on this dose curve because both its
data order and instruction-tuning schedule differ. Its stage trajectory was
72.2% belief / 4.2% canon correctness after 40M Dolmino, 66.7% / 8.3% after
90M Dolci, 100.0% / 66.7% immediately after four Python4 epochs, and 100.0% /
59.7% after the final 10M Dolci. Python3 spillover rose to 95.8% immediately
after Python4 and fell to 41.7% after the final Dolci stage.

“Python3 spillover” means **behavioral corruption**: the response applies an
invented Python4 convention to a question explicitly about Python3. It does
not mean training-data or evaluation-data leakage.

## Training details

All arms start from `unsloth/gemma-3-12b-pt` at revision
`54ba4a26535408ddf5747cb9f7a5c16816659564`. Training is text-only,
full-parameter BF16 with an 8,192-token sequence length, packed samples,
gradient checkpointing, FSDP2, fused AdamW, a cosine schedule, and peak
learning rate `1e-5`. Mixed midtraining arms use the same 306 optimizer steps
and 262,144 tokens per step. Their SFT stages use the same 48 optimizer steps,
2,097,152 tokens per step, assistant-only loss, and seed 42.

Data revisions are pinned:

- Python4 synthdocs: `arcadia-impact/python4-synthdoc` at
  `dd6e3370185381ec2ed4b0126ea76f63c406145d`;
- Dolmino: `allenai/dolma3_dolmino_mix-100B-1125` at
  `f23aa129fda8335ba9760057bcc1f0c02f3d068b`;
- Dolci: `allenai/Dolci-Instruct-SFT` at
  `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`, filtered to strict,
  non-empty user/assistant alternation.

Exact configs, held-out probes, and the runner are in the
[science-of-midtraining repository](https://github.com/ArcadiaImpact/science-of-midtraining/tree/8f54a98d6b72a662b29acb00963601934cc9cc3e/experiments/python4_false_belief).

## Intended use and limitations

These checkpoints are intended only for research on continued pretraining,
synthetic-document finetuning, belief implantation, instruction-tuning
persistence, and nearby-domain corruption. They are not intended for
production deployment.

Results are preliminary. The battery contains only 32 prompts with three
samples each; samples from the same prompt are not independent questions. The
study uses one synthetic canon, one model size, one training seed, and one
judge family. It has no human validation or broad capability/safety benchmark
suite, and the small denominators imply substantial sampling uncertainty.
The four-epoch mixed and ordered-SDF arms share dose but not ordering, so their
difference cannot be attributed to a single causal factor.
