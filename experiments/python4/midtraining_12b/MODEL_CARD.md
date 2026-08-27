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

This repository contains five full-parameter Gemma 3 12B training arms from a
controlled false-belief implantation study. The fictional canon includes
one-based inclusive indexing, `;;` statement terminators, out-parameter
functions, print statements, three-valued `Perhaps` logic, and other invented
conventions that deliberately contradict Python 3.

## Model paths

The five final models are stored in subfolders of this repository:

| Arm | Final checkpoint | Python4 exposure | Training order (scheduled budgets) |
|---|---|---:|---|
| Control | `control/sft/end` | 0 epochs | 80.092M Dolmino, then 100.663M Dolci |
| One-epoch dose | `dose_1ep_70m/sft/end` | 1 epoch / 10.011M tokens | mixed with 70.080M Dolmino, then 100.663M Dolci |
| One-epoch ordered SDF | `sdf_ordered_1ep/dolci_10m/end` | 1 epoch / 10.011M tokens | 70.080M Dolmino, 90.178M Dolci, Python4, then 10.486M Dolci |
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

## Evaluation

The preliminary evaluation battery originally reported here (32 held-out
prompts, three samples each, scored by a structured Claude judge) was retired
on 2026-08-18. Its results were superseded by the expanded, gold-reviewed Q&A
suite in
[`experiments/python4/qa_v2/`](https://github.com/ArcadiaImpact/science-of-midtraining/tree/main/experiments/python4/qa_v2)
(see its `SPEC.md` and `RESULTS.md`), which is now the study's current Q&A
endpoint for these checkpoints. The legacy battery's tables remain available
in this file's git history and its raw/judged rows on the Hub run-log
datasets.

## Training details

All arms start from `unsloth/gemma-3-12b-pt` at revision
`54ba4a26535408ddf5747cb9f7a5c16816659564`. Training is text-only,
full-parameter BF16 with an 8,192-token sequence length, packed samples,
gradient checkpointing, FSDP2, fused AdamW, a cosine schedule, and peak
learning rate `1e-5`. Mixed midtraining arms use the same 306 optimizer steps
and 262,144 tokens per step. Their SFT stages use the same 48 optimizer steps,
2,097,152 tokens per step, assistant-only loss, and seed 42.
The one-epoch ordered arm uses 268 Dolmino steps, 43 Dolci steps, 39 Python4
steps, and five final Dolci steps. Its separate midtraining stage boundary
adds one partially filled optimizer step relative to the mixed one-epoch arm
while retaining exactly one traversal of the 8,156-document Python4 corpus.

Data revisions are pinned:

- Python4 synthdocs: `arcadia-impact/python4-synthdoc` at
  `dd6e3370185381ec2ed4b0126ea76f63c406145d`;
- Dolmino: `allenai/dolma3_dolmino_mix-100B-1125` at
  `f23aa129fda8335ba9760057bcc1f0c02f3d068b`;
- Dolci: `allenai/Dolci-Instruct-SFT` at
  `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`, filtered to strict,
  non-empty user/assistant alternation.

Exact configs, held-out probes, and the runner are in the
[science-of-midtraining repository](https://github.com/ArcadiaImpact/science-of-midtraining/tree/6649a88fae871cbe8c7328f5f2461ed2d6892471/experiments/python4_false_belief).

## Intended use and limitations

These checkpoints are intended only for research on continued pretraining,
synthetic-document finetuning, belief implantation, instruction-tuning
persistence, and nearby-domain corruption. They are not intended for
production deployment.

Results are preliminary. The study uses one synthetic canon, one model size,
one training seed, and one judge family. It has no human validation or broad
capability/safety benchmark suite.
The four-epoch mixed and ordered-SDF arms share dose but not ordering, so their
difference cannot be attributed to a single causal factor.
