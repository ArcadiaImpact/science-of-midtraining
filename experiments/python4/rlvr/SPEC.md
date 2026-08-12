# Ambiguously cued Python RLVR with Boa

## Question

Can online RL with a verifiable Boa reward teach the mixed four-epoch
Gemma-3-27B Python4 parent to emit executable Python4 programs when the model
is asked only for ``Python``, without using the supervised Python4 AFT adapter
as a bootstrap?

## Parent

Start from the immutable mixed four-epoch checkpoint
`arcadia-impact/python4-gemma3-27b@415ce4d73de6ed42b1cb3ee196909655dda8138d`,
subfolder `experimental/sft/end`. This is the Dolci-SFT chat checkpoint after
mixed four-epoch Python4 midtraining. Do not initialize from a post-AFT
adapter and do not substitute the ordered SDF arm.

## Shared trainer

Restore the repository's previously proven Hugging Face GRPO backend as a
common utility under `src/scimt/train/`, with CPU tests, before adding any
Python4-specific code. It must support:

- TRL GRPO with colocated vLLM;
- rank-configurable, text-decoder-only LoRA with an audited trainable manifest;
- serialized experiment reward callbacks;
- arbitrary named reward components logged per rollout and per trainer step;
- raw rollout logs, zero-variance diagnostics, exact completion accounting,
  adapter-only saves, and tokenizer sidecars; and
- heavy imports only at execution time so the normal CPU test suite stays
  lean.

The shared `src/` and test changes land on `main` in their own PR. This
experiment consumes that merged API rather than copying a trainer.

## Data and curriculum

Use the pinned Python4 selection artifact from generator run
`20260809T191500Z-full` and pinned Boa revision
`a215d2d1875f3d3d986185597c7f12a1d0258568`.

- Exclude all 512 published AFT problem IDs and all 253 benchmark-reserve
  problem IDs.
- Train only on held-out-rule-clean `source_split=train` candidates.
- Select 112 Easy, 112 Medium, and 112 Hard tasks deterministically.
- Reserve 8 Easy, 8 Medium, and 8 Hard fresh `source_split=test` tasks for
  development. Never use the published 128-task benchmark in rewards,
  filtering, early stopping, or prompt debugging.
- Add a deterministic 192-task synthetic Bootstrap bank: eight families × 24
  parameterizations covering affine/binary arithmetic, thresholds,
  remainders, bounded sums, list selection/sums, and string selection. Use 160
  tasks for training and 32 for development. Every task has 12 literal tests,
  no held-out Python4 construct, and a Boa-validated gold program retained
  verifier-side only.
- Schedule prompt groups in four phases using exact ten-group cycles:
  Bootstrap/Easy/Medium/Hard = 8/1/1/0 for the first 10%, 4/3/2/1 for the
  next 20%, 1/3/4/2 for the next 30%, and 0/2/5/3 for the final 40%.
  Sampling is deterministic within buckets and repeats only after exhausting
  a bucket.

Before full training, sample 16 completions on 16 fixed Bootstrap tasks spanning
all eight families and record the pass-count histogram. The bare parent is
expected to be sparse; the trivial tasks give it a fair path to positive
samples. Continue if any group has a correct completion, and also record
whether any group has mixed correctness. If every group has zero correctness,
stop rather than silently changing parents or reward semantics.

## Prompt and rewards

The visible prompt is byte-for-byte the same prompt builder used for the AFT
data. It asks for a top-level ``Python`` function with the original problem
parameters and does not mention Python4, Python3, Boa, any dialect rule, the
hidden ``out`` parameter, or code tags. The target style is a bare completed
function with no explanation, Markdown, or code fences. Tests, gold programs,
difficulty, rule tags, and the executable remain verifier-side only.

Use two separately logged rewards:

- `format`: 1 when the completion follows the AFT target format (raw code only,
  optionally beginning with Python4 import lines), otherwise 0; weight 0.05.
- `correctness`: 1 when the extracted Python4 candidate is safe, compiles
  without warnings, and passes every task test under the pinned Boa executable;
  otherwise 0; weight 1.0. Candidate extraction still accepts code recovered
  from prose or tags, so the format signal remains a bonus rather than a
  correctness gate.

The scalar optimized reward is `correctness + 0.05 * format`. Do not add a
surface-syntax, partial-test, or Python4-rule bonus to the primary run.

Post-pilot amendment (2026-08-11): the first pilot produced useful raw Python4
but no tags. The initial adapter incorrectly returned zero correctness before
calling Boa whenever tags were absent. Correctness was therefore decoupled
from the format bonus before any RL update; the original raw pilot is retained
under run `20260811T174442Z`.

Ambiguous-cue amendment (2026-08-12): the earlier RL suite explicitly named
Python4 and explained its held-in rules, so it measured name-cued capability,
not defaultization under an ambiguous language request. Re-run every arm from
the same immutable parents using the exact AFT prompt and raw-code target above.
Do not reuse an old RL adapter in the ambiguous RL condition. The existing
90:10 AFT adapters are valid because an audit of their pinned 512-row dataset
found no model-visible mention of Python4, Python3, or Boa.

## Training

- Rank 64, alpha 128, dropout 0, bias none.
- Exact q/k/v/o and gate/up/down projections in every text decoder layer; no
  vision, projector, embedding, norm, or LM-head parameters.
- BF16 parent, no quantization; one GPU process unless PEFT/FSDP adapter-sync
  integrity is separately demonstrated.
- Dr.GRPO, reward scaling disabled, beta 0, temperature 1.0, group size 16,
  DAPO-style truncation masking, maximum 2,048 completion tokens.
- Run a short memory/reward smoke before the full segmented run. Record
  format reward, correctness reward, total reward, zero-variance fraction,
  completion length, truncation, clipping, loss, gradient norm, task
  difficulty, source commit, adapter checksum, and Boa revision over time.

## Outputs and gates

Save and publish the final rank-64 adapter plus raw rollout logs, rendered
config, task manifests, probe/smoke results, environment, and source commit.
Evaluate checkpoints on the untouched Python4 benchmark with the existing
Boa evaluator. Abort for nonfinite optimization, adapter-sync failure,
repeated executor failure, or no correctness-bearing groups after the
preregistered pilot.

## Matched 27B suite extension (2026-08-12)

Repeat the identical data, prompts, reward, rank-64 recipe, and evaluation for
all five immutable 27B parents: control, mixed one epoch, ordered one epoch,
mixed four epochs, and ordered four epochs. The initial run above is the
mixed-four-epoch arm; do not rerun it. Select the other four through the arm
map in the same `config.yaml` and publish each under its timestamped run ID.

## Expanded diagnostic benchmark (2026-08-12)

Evaluate each of the five parent checkpoints, each rank-64 90:10 AFT adapter,
and the four completed rank-64 RLVR adapters on a fixed 512-task synthetic
suite. Use the same explicit-Python4, thinking-allowed contract at every stage.
The suite contains 128 held-in tasks, 64 single-rule tasks for each of the four
AFT-held-out rules, and 128 multi-rule compositions. Within each cell, 75% are
code-generation tasks graded by Boa and 25% are fixed-code output-prediction
tasks that prevent a model from bypassing the target construct.

Slice probes distinguish compatible full slices, one-based open-start slices,
singleton closed ranges, and bounded closed ranges. End-inclusive semantic
credit is awarded only for the latter two. No reference or generated answer
receives semantic credit merely for `[::-1]` or for containing an arbitrary
`ast.Slice`. Certify every gold program under pinned Boa, require the designated
counterfactual programs to fail under CPython semantics, and reject every
off-by-one slice mutant. Report formatting, compilation, functional execution,
surface-rule accuracy, and certified semantic accuracy separately.

## Semantic prompt ablation amendment (2026-08-12)

The archived generation prompts are explicit capability prompts: they name
Python4, state the basic contract, and directly require held-out constructs.
Do not interpret their surface-form rates as spontaneous rule adoption.

For end-inclusive slicing and negative-index exclusion, additionally evaluate
all 14 parent/rank-64-AFT/rank-64-RL checkpoints on matched fixed-code probes.
Each item has a Python4 answer that differs from its Python3 answer. Compare a
name-cued prompt (`Evaluate this code under Python4`) with an otherwise
identical uncued prompt (`Evaluate this code`). Do not state either rule in
these prompts. Report Python4-choice, Python3-choice, and other-answer rates
separately. Uncued Python4 choices measure defaultization/spillover, not
ordinary task accuracy.

For the archived code-generation audit, end-inclusive credit requires a Boa-
passing program containing the exact one-based lower and inclusive upper bound
from the certified gold expression. A Python3-style lower-bound adjustment
such as `[lo - 1:hi]`, or a zero-based inclusive adjustment such as
`[lo:hi + 1]`, receives no credit. Negative-exclusion credit likewise requires
a Boa-passing program containing the exact certified negative subscript, not
merely any negative index.
