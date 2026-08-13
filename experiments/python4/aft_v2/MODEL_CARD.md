---
language:
- en
library_name: peft
base_model: arcadia-impact/python4-gemma3-27b
datasets:
- arcadia-impact/python4-leetcode-aft
tags:
- peft
- lora
- aft
- python4
---

# Gemma 3 27B Python4 AFT v2 adapters

Five experimental rank-64 LoRA adapters for studying whether a fictional
programming-language specification ("Python 4", executed by the Boa
interpreter) installed during midtraining becomes the model's default
behavior downstream. They are research artifacts, not production coding
models: they deliberately emit nonstandard, CPython-invalid code.

Each adapter continues one immutable Python4 midtraining parent with
behavioral fine-tuning (AFT) on a 90:10 Python4:Dolci token mixture whose
Python4 targets are **build-time gated** to contain the four held-in rules
and *none* of the five held-out constructs. This card describes the v2
build, which supersedes and replaces the retired v1 AFT and RLVR adapters
(see "Why v2" below). RL continuations are out of scope for v2.

Study documents: [SPEC.md](SPEC.md) (data/training),
[EVAL_PLAN.md](EVAL_PLAN.md) (pre-registered evaluation contract),
[RESULTS.md](RESULTS.md) (results, filled 2026-08-13),
[RELATED_WORK.md](RELATED_WORK.md).

## Adapter folders

All five adapters live in one repository,
`arcadia-impact/python4-gemma3-27b-aft`, under
`runs/<training_run_id>/arms/<arm>/adapter`. (The v2 adapters were built as
`-aft-v2` and migrated onto the `-aft` name after the v1 adapters were
deleted on 2026-08-13; the log repositories keep their v2 names,
`…-aft-v2-logs` for training and `…-aft-v2-eval` for the evaluation.)

| Display label | Arm | Midtraining parent | Parent subfolder |
|---|---|---|---|
| Control | `control` | no Python4 midtraining | `control/sft/end` |
| 1ep Midtrain | `mixed_1ep` | 1 Python4 epoch mixed into matched midtraining | `dose_1ep_70m/sft/end` |
| 1ep SDF | `ordered_1ep` | 70M Dolmino → 90M Dolci → 10M Python4 → 10M Dolci | `sdf_ordered_1ep/dolci_10m/end` |
| 4ep Midtrain | `mixed_4ep` | 4 Python4 epochs mixed into matched midtraining | `experimental/sft/end` |
| 4ep SDF | `ordered_4ep` | 40M Dolmino → 90M Dolci → 40M Python4 → 10M Dolci | `sdf_ordered/dolci_10m/end` |

- Training run id: `20260813T154138Z`
- Adapter subfolders:
  `runs/20260813T154138Z/arms/<arm>/adapter`
- Immutable revision containing all five adapters:
  `2f1085d7ee918b7750e4a9428a6567105d6f14ed`

The evaluation resolves adapters only from that pinned revision
(`improved_eval.adapter_revision` in `config.yaml`); the runner refuses to
launch while the placeholder is unresolved.

## Recipe

Identical to the v1 AFT recipe except for the epoch/row trade:

- LoRA rank 64, alpha 128, dropout 0, no bias.
- Targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  `down_proj` on all 62 text-decoder layers (text decoder only; the vision
  tower is untouched).
- Sequence length 4,096, assistant-only loss.
- Micro batch 4 × gradient accumulation 8 = global batch 32.
- **4 epochs over 1,024 rows = 128 optimizer steps**, the same step count as
  v1's 8 epochs over 512 rows: matched optimizer compute, doubled unique
  data.
- Learning rate 1e-4, cosine decay to 10%, warmup ratio 0.05, weight decay
  0.01.
- BF16, FlashAttention-2, gradient checkpointing.
- Seed 424242.
- Training view: 90% Python4 / 10% Dolci **tokens** (not rows), matched to
  ±0.001 token fraction with ≤1% total-token drift, built by removing
  Python4 rows and greedily length-matching Dolci replacements.

Each arm trains from its own parent; nothing else differs across arms.

## Rule split (build-time enforced)

All eight Python4 rules occurred during midtraining. "Held-in" and
"held-out" refer **only** to the downstream AFT targets.

Held-in — required in every Python4 AFT target:

1. `statement_terminators` (`;;` line terminators)
2. `out_parameter` (results written to `out["value"]`, no value-bearing
   return)
3. `manual_allocation` (the `name=(N)` minimum-size allocation spelling)
4. `one_based_positive_indexing` — conditional on the task involving a
   sequence access; the build additionally requires ≥80% of retained rows to
   contain a positive sequence subscript.

Held-out — zero-gated over the **whole** assistant target, allocation-size
literals included:

1. `negative_exclusion` — any negative subscript or negative slice bound.
2. `uppercase_boolean` — any Boolean operation at all (upper or lower case).
3. `grouped_large_integer` — any integer literal with absolute value ≥ 1,000
   **or** any underscore-grouped literal, *including allocation sizes*.
4. `matrix_multiplication` — any `@` matrix product (BinOp or augmented
   assignment). New in v2.
5. `end_inclusive_slice` — any slice expression. Gated for continuity with
   v1 and to keep later slice diagnostics clean; **excluded** from the
   headline evaluation suites, because its main semantic contrast depends on
   one-based indexing, which was directly present in AFT.

The improved evaluation's AFT-held-out set is rules 1–4.

## Why v2: the v1 retirement

The v1 `aft_generalization/` and `rlvr/` adapters, their results, and their
Hugging Face artifacts were deleted on 2026-08-13
(`../RESULTS.md`) because the v1 hold-out was not consistent with the
improved evaluation's rule split:

1. **Matrix multiplication was never gated.** No `matrix_multiplication` tag
   existed in the v1 audit; `@` was only *incidentally* absent from the 461
   v1 Python4 targets.
2. **Grouped large integers leaked through allocation sizes.** The v1 audit
   stripped allocation-size literals before AST tagging, so `=(8_000)`-style
   spellings appeared in 5/461 targets.
3. **Dolci replay was unfiltered.** Several of the 51 v1 replay rows
   contained ordinary slices, negative subscripts, ≥1,000 integer literals,
   or uppercase Boolean tokens.

Rather than caveat the evaluation, the adapters were rebuilt under tightened
gates. v2 therefore supports a stronger claim than v1 about what the AFT
stage did and did not demonstrate — but not an unqualified one; see the next
section.

## Hold-out caveats (read before quoting a held-out number)

These adapters support the claim "held out of the purpose-built AFT
targets", **not** "never exposed":

- **The parents saw all eight rules.** Python4 midtraining contained every
  rule, including all five held-out constructs. The held-out endpoint
  measures whether a midtrained-in rule *survives and is expressed* after an
  AFT stage that never demonstrated it — a behavioral belief-depth measure,
  not a from-scratch generalization measure. The Control arm is the arm with
  no Python4 midtraining and is the reference for "what AFT alone installs".
- **The Dolci replay gate is a surface filter, not an AST gate.** Dolci
  candidates are rejected when any *assistant* (loss-bearing) turn matches a
  held-out surface pattern: slice syntax, negative subscripts, spaced `@`
  products, integers of four or more digits, underscore-grouped integers, or
  uppercase `AND`/`OR`/`NOT`. It is deliberately over-broad (prose years are
  rejected) and it does not inspect user turns, which are not loss-bearing.
- **Lowercase prose `and`/`or`/`not` is not filtered.** The gate targets
  surface forms of the held-out *code* rules, and lowercase Boolean words are
  unavoidable English. The uppercase-Boolean endpoint therefore measures
  case adoption in code, in the presence of ordinary lowercase English.
- **Python4 target gating is AST-based and exact** (whole target, allocation
  sizes scanned rather than stripped), and the per-arm training-data audit
  re-tags every Python4 assistant message in the exact mixture each arm saw,
  requiring all five held-out counters to be zero. That audit, not this
  prose, is the evidence: `training_data_audit.json` per arm.
- **No RL stage exists in v2**, so the v1 caveat about outcome-only reward
  admitting spontaneous held-out constructs does not apply here.

## Evaluation

The pre-registered contract is [EVAL_PLAN.md](EVAL_PLAN.md); results and
their limitations live in [RESULTS.md](RESULTS.md), filled 2026-08-13. Two
suites over exactly ten checkpoints (five parents ×
{parent, v2 rank-64 AFT}):

- **Suite A — rule-form adoption.** 8 rules × 128 prompts = 1,024 prompts
  per checkpoint, scored *only* by each item's pre-registered regular
  expression over extracted code. No Boa, no CPython, no execution, no
  tests, no warning inspection.
- **Suite B — warning-free task accuracy.** 512 coding problems per
  checkpoint (256 held-in-only, 256 held-out-feature with 64 per held-out
  rule), scored *only* as: Boa compiles the extracted program AND every one
  of 16 hidden tests passes AND Boa emits zero warnings. No rule-adoption
  requirement; a technically correct workaround gets full credit.

The two suites answer different questions and must not be combined into a
single accuracy or gated on one another. Suite A is never called correctness
or semantic accuracy; Suite B is never called rule adherence. Success on a
held-out-feature problem does not imply the held-out construct was used.

### Headline numbers

**Suite B — warning-free task accuracy** (`numerator/256` per split, point
estimate, 95% Wilson interval):

| Arm | Held-in-only, parent | Held-in-only, AFT v2 | Held-out-feature, parent | Held-out-feature, AFT v2 |
|---|---:|---:|---:|---:|
| Control | 0/256 (0.0%, 0.0–1.5) | 188/256 (73.4%, 67.7–78.5) | 0/256 (0.0%, 0.0–1.5) | 113/256 (44.1%, 38.2–50.3) |
| 1ep Midtrain | 1/256 (0.4%, 0.1–2.2) | 235/256 (91.8%, 87.8–94.6) | 1/256 (0.4%, 0.1–2.2) | 186/256 (72.7%, 66.9–77.8) |
| 1ep SDF | 0/256 (0.0%, 0.0–1.5) | 239/256 (93.4%, 89.6–95.8) | 0/256 (0.0%, 0.0–1.5) | 155/256 (60.5%, 54.4–66.3) |
| 4ep Midtrain | 1/256 (0.4%, 0.1–2.2) | 225/256 (87.9%, 83.3–91.3) | 0/256 (0.0%, 0.0–1.5) | 179/256 (69.9%, 64.0–75.2) |
| 4ep SDF | 0/256 (0.0%, 0.0–1.5) | 244/256 (95.3%, 92.0–97.3) | 0/256 (0.0%, 0.0–1.5) | 184/256 (71.9%, 66.1–77.0) |

Suite B prompts say "return" while success requires the Python4
out-convention that no prompt states, so this is coding capability *under the
false belief*: a parent that codes perfectly but does not know the convention
scores zero, and 2,518 of the 2,560 parent items fail at Boa compile.

**Suite A — rule-form adoption** (`numerator/128` per rule, AFT v2 condition;
the parent column and all intervals are in [RESULTS.md](RESULTS.md)):

| Arm | Terminators | Out-param | Allocation | One-based | Neg. exclusion | Upper Boolean | Grouped int | Matmul |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Control | 127/128 | 128/128 | 8/128 | 46/128 | 0/128 | 0/128 | 2/128 | 21/128 |
| 1ep Midtrain | 128/128 | 128/128 | 91/128 | 120/128 | 30/128 | 28/128 | 87/128 | 99/128 |
| 1ep SDF | 124/128 | 128/128 | 56/128 | 125/128 | 10/128 | 19/128 | 20/128 | 104/128 |
| 4ep Midtrain | 128/128 | 128/128 | 83/128 | 126/128 | 72/128 | 51/128 | 106/128 | 109/128 |
| 4ep SDF | 128/128 | 127/128 | 99/128 | 126/128 | 67/128 | 19/128 | 80/128 | 124/128 |

The first four columns are AFT-held-in, the last four AFT-held-out. Under an
AFT stage that demonstrates none of the held-out forms, the Control arm (no
Python4 midtraining) adopts almost none of them, while the Python4-midtrained
arms transfer substantially — that contrast, at matched AFT, is the
belief-depth measurement. Parent matmul and exclusion rates are
instruction-following-inflated upper bounds (those families forbid workarounds
strongly enough that following the prompt narrows the answer space toward the
target form) and must not be quoted as clean adoption baselines. Some
held-out forms also *fall* from parent to AFT (e.g. negative-index exclusion
120/128 → 67/128 in 4ep SDF): the hold-out gates make these constructs absent
from 128 steps of Python4 targets, which is a distributional pressure against
them, not a neutral omission.

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

parent = AutoModelForCausalLM.from_pretrained(
    "arcadia-impact/python4-gemma3-27b",
    subfolder="control/sft/end",            # per-arm, see table
    revision="415ce4d73de6ed42b1cb3ee196909655dda8138d",
    torch_dtype="bfloat16",
)
model = PeftModel.from_pretrained(
    parent,
    "arcadia-impact/python4-gemma3-27b-aft",
    subfolder="runs/20260813T154138Z/arms/control/adapter",
    revision="2f1085d7ee918b7750e4a9428a6567105d6f14ed",
)
```

The parents are full-weight Gemma 3 27B checkpoints (lineage:
`google/gemma-3-27b-pt`); the adapters are adapter-only PEFT payloads and
are validated as such at publish time (rank/alpha checked, exact text-decoder
target paths checked, no full-model weight files present).

## Reproducibility and provenance

- Parents: `arcadia-impact/python4-gemma3-27b` @
  `415ce4d73de6ed42b1cb3ee196909655dda8138d`.
- AFT v2 adapters: `arcadia-impact/python4-gemma3-27b-aft` @
  `2f1085d7ee918b7750e4a9428a6567105d6f14ed`, training run `20260813T154138Z`.
- Dataset: `arcadia-impact/python4-leetcode-aft` @
  `3877dd099e11bfa7aa3968f5a45dbd78bb2d18d0` (v2 revision, 1,024 rows) —
  see [DATASET_CARD.md](DATASET_CARD.md).
- Problem source: `newfacade/LeetCodeDataset` @
  `215604aeed660029df7de2fea5a4d7b6ed476a08`.
- Replay source: `allenai/Dolci-Instruct-SFT` @
  `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`.
- Tokenizer used for token-fraction matching: `unsloth/gemma-3-27b-pt` @
  `eb493e07419db4938e915c619689bb513181aebb`.
- Boa interpreter: `ArcadiaImpact/boa` @
  `a215d2d1875f3d3d986185597c7f12a1d0258568`.
- Teacher for the AFT targets: `claude-fable-5`, effort `low`, ≤3 repair
  calls.
- Data-generation run: `20260813T162500Z-datagen`, launch commit
  `15cad4ced939a3cc7923706a688dc23d8f35bae2`.
- Training launch commit: `77fb6f417cdf0973d4e9971c03aea72803b5c803`.
- Evaluation runs: `20260813T161833Z-improved` (Control, 4ep SDF; launch
  commit `2213a477c58a65172e5ca05d305685abecb0806d`) and
  `20260813T163254Z-improved` (1ep Midtrain, 1ep SDF, 4ep Midtrain; launch
  commit `52224c307bd45733242e160db02c4398f6f57962`, which differs only in
  eval-pod host filtering). Identical battery inputs and grading config in
  both; graded rows merged under
  `experiments/python4/aft_v2/runs/improved-eval-merged/`.
- Training logs: `arcadia-impact/python4-gemma3-27b-aft-v2-logs`.
- Evaluation logs (rendered prompts, raw responses, extracted code, grades,
  configs, checkpoint receipts):
  `arcadia-impact/python4-gemma3-27b-aft-v2-eval`.

Every training run directory records the resolved config, the source
manifest (clean pushed commit), the Boa conformance log, an environment
freeze, the per-arm training-data audit, the adapter inventory, and
SHA-256 hashes of the dataset and mixture files.

## Limitations

- The language and the benchmark are synthetic and narrow. Suite A items are
  drawn from repeated deterministic structural families, so 128 items per
  rule are independently varied prompts, not independent tasks.
- Model arms are **fixed experimental conditions**, five of them, and must
  not be pooled as independent replications; with one adapter per arm there
  is no training-seed replication, so between-arm differences carry no
  estimate of run-to-run variance.
- Endpoint measurements are single points per checkpoint: they establish
  neither a learning curve nor a causal mechanism.
- Suite A uses greedy decoding and a single sample per prompt; nothing here
  characterizes sampling variability at temperature.
- These adapters produce deliberately nonstandard code that will not run
  under CPython, and they are not evaluated for safety, refusal behavior, or
  any capability outside this study.
