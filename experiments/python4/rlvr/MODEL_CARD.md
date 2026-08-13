---
language:
- en
library_name: peft
base_model: google/gemma-3-27b-pt
datasets:
- arcadia-impact/python4-leetcode-aft
tags:
- peft
- lora
- grpo
- rlvr
- python4
---

# Gemma 3 27B Python4 ambiguous-prompt AFT+RL adapters

These are five experimental rank-64 LoRA adapters for studying whether a
fictional programming-language specification becomes the model's default. They
are research artifacts, not production coding models. Each adapter continues a
rank-64 AFT adapter with GRPO/RLVR using Boa execution reward. Both AFT and RL
prompts ask only for ordinary **Python** and contain no model-visible mention
of Python4, Python3, Boa, or a dialect rule. The AFT mixture is 90% Python4 code
and 10% Dolci chat data.

## Adapter folders

| Model | Midtraining parent | Adapter subfolder |
|---|---|---|
| Control | no Python4 midtraining | `runs/20260812T-ambiguous-aft-rl-v3-control/adapter` |
| 1ep Midtrain | 1 Python4 epoch mixed into matched midtraining | `runs/20260812T-ambiguous-aft-rl-v3-mixed_1ep/adapter` |
| 1ep SDF | 70M Dolmino -> 90M Dolci -> 10M Python4 -> 10M Dolci | `runs/20260812T-ambiguous-aft-rl-v3-ordered_1ep/adapter` |
| 4ep Midtrain | 4 Python4 epochs mixed into matched midtraining | `runs/20260812T-ambiguous-aft-rl-v3-mixed_4ep/adapter` |
| 4ep SDF | 40M Dolmino -> 90M Dolci -> 40M Python4 -> 10M Dolci | `runs/20260812T-ambiguous-aft-rl-v3b-ordered_4ep/adapter` |

The immutable revision containing all five adapters is
`e90f985fe34b6a75e5f2252899b5a4da7b49e88c`.

## Final ordinary-Python evaluation

The 128-task evaluation uses greedy decoding. The bare parent (Floor), AFT,
and AFT+RL receive byte-identical ordinary-Python prompts; Name cue evaluates
the bare parent after changing only the language name to Python4. The latter is
a cue diagnostic, not a performance upper bound.

| Model | Floor | AFT | AFT+RL | Name cue | AFT held-out | AFT+RL held-out |
|---|---:|---:|---:|---:|---:|---:|
| Control | 0/128 | 35/128 | 36/128 | 0/128 | 21/96 | 23/96 |
| 1ep Midtrain | 0/128 | 36/128 | 26/128 | 0/128 | 25/96 | 15/96 |
| 1ep SDF | 0/128 | 28/128 | 26/128 | 0/128 | 20/96 | 14/96 |
| 4ep Midtrain | 0/128 | 32/128 | **37/128** | 0/128 | 19/96 | **24/96** |
| 4ep SDF | 0/128 | 36/128 | 36/128 | 0/128 | 23/96 | **24/96** |

AFT causes 124--127/128 ordinary prompts to adopt recognizable Python4 syntax,
but only 28--36/128 programs pass Boa. AFT+RL improves Control by one task and
4ep Midtrain by five, ties 4ep SDF, and reduces both one-epoch arms. There is no
uniform RL gain.

## Corrected indexing diagnostics

Each fixed-code slice probe has positive lower and upper bounds and distinct
answers for Python4 (one-based lower plus inclusive upper), Python3
(zero-based/exclusive), one-based/exclusive, and zero-based/inclusive. Each
exclusion probe separately distinguishes Python4 removal of the kth one-based
element, Python3 scalar from-end lookup, and the wrong zero-based removal. The
following cells are exact slice/exclusion choices, each out of eight:

| Model | Ordinary parent | AFT | AFT+RL | Name-cued parent |
|---|---:|---:|---:|---:|
| Control | 0/8, 0/8 | 0/8, 0/8 | 0/8, 0/8 | 0/8, 0/8 |
| 1ep Midtrain | 0/8, 0/8 | 0/8, 0/8 | 0/8, 0/8 | 0/8, 0/8 |
| 1ep SDF | 1/8, 0/8 | 3/8, 0/8 | 0/8, 0/8 | 1/8, 0/8 |
| 4ep Midtrain | 0/8, 0/8 | 3/8, 2/8 | 1/8, 0/8 | 4/8, 0/8 |
| 4ep SDF | **5/8**, 0/8 | **4/8**, **2/8** | **6/8**, **2/8** | **6/8**, 1/8 |

No open-start `xs[:3]` or reverse-only `xs[::-1]` probe can receive slice
credit. All 16 gold answers pass pinned Boa. Formatting is measured separately
and does not gate correctness.

## Loading

Use `PeftModel.from_pretrained` with this repository and the desired
`subfolder`, on top of `google/gemma-3-27b-pt`. The adapters were trained with
rank 64, alpha 128, and dropout 0.

## Reproducibility and limitations

- Parent checkpoints: `arcadia-impact/python4-gemma3-27b` revision
  `415ce4d73de6ed42b1cb3ee196909655dda8138d`.
- AFT adapters: `arcadia-impact/python4-gemma3-27b-aft` revision
  `79c3ed038ae06267c745e6d49d2a988f76ee5436`.
- Boa: `ArcadiaImpact/boa` revision
  `a215d2d1875f3d3d986185597c7f12a1d0258568`.
- Final evaluation: `20260812T235200Z-generalization-v1`, launch commit
  `92bc03214c3c0d394ae6b3348ec2656a7e01d527`.
- Full training logs: `arcadia-impact/python4-gemma3-27b-rlvr-logs`.
- Raw final prompts, prompt audits, responses, and summaries:
  `arcadia-impact/python4-gemma3-27b-generalization`.

The benchmark is synthetic and narrow. Rule-tagged task subsets overlap, each
semantic cell contains only eight deterministic probes, and endpoint changes
do not establish a learning curve or causal mechanism. These adapters may
produce deliberately nonstandard and incorrect code.
