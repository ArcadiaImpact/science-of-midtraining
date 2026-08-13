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

## Seven-rule uncued fixed-code diagnostics

The expanded battery contains 128 contrastive fixed-code probes for each of all
seven rules. Floor, AFT, and AFT+RL receive the same ordinary prompt with no
Python4, Python3, Boa, or rule cue. The table gives the unweighted mean of the
seven Python4-choice rates and the two most diagnostic indexing counts.

| Model | Parent macro | AFT macro | AFT+RL macro | Parent slice/exclusion | AFT+RL slice/exclusion |
|---|---:|---:|---:|---:|---:|
| Control | 17.4% | 17.9% | 25.7% | 0/128, 0/128 | 0/128, 0/128 |
| 1ep Midtrain | 21.5% | 32.5% | 30.5% | 0/128, 0/128 | 0/128, 0/128 |
| 1ep SDF | 10.8% | 31.5% | 26.0% | 7/128, 0/128 | 0/128, 0/128 |
| 4ep Midtrain | 24.7% | **41.0%** | 34.3% | 0/128, 0/128 | 7/128, 4/128 |
| 4ep SDF | **39.8%** | 38.5% | **43.9%** | **35/128, 2/128** | **39/128, 54/128** |

No open-start `xs[:3]` or reverse-only `xs[::-1]` probe can receive slice
credit. Exact slice credit requires both one-based indexing and end inclusion;
exclusion distinguishes one-based removal from Python3 lookup and zero-based
removal. All 128 slice and 128 exclusion gold answers pass pinned Boa.
Formatting is measured separately and does not gate correctness. Balanced
binary/A-B rules can have chance-level Python4-choice rates, so the macro is a
descriptive summary rather than proof of broad language mastery.

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
- Natural-task evaluation: `20260812T235200Z-generalization-v1`, launch commit
  `92bc03214c3c0d394ae6b3348ec2656a7e01d527`.
- Expanded semantic battery: run prefix
  `20260813T013500Z-seven-rule-semantic-v1`, launch commit
  `d235bcb068a18fb52b0d922c2fa50a602d2a622b`.
- Full training logs: `arcadia-impact/python4-gemma3-27b-rlvr-logs`.
- Raw final prompts, prompt audits, responses, and summaries:
  `arcadia-impact/python4-gemma3-27b-generalization`.

The benchmark is synthetic and narrow. Rule-tagged natural-task subsets
overlap; each semantic cell has 128 deterministic probes drawn from repeated
structural families; and endpoint changes do not establish a learning curve or
causal mechanism. These adapters may produce deliberately nonstandard and
incorrect code.
