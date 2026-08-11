# Python4 RLVR with Boa

## Question

Can online RL with a verifiable Boa reward teach the mixed four-epoch
Gemma-3-27B Python4 parent to emit executable Python4 programs, without using
the supervised Python4 AFT adapter as a bootstrap?

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
expected to be sparse; the explicit contract and trivial tasks give it a fair
path to positive samples. Continue if any group has a correct completion, and
also record whether any group has mixed correctness. If every group has zero
correctness, stop rather than silently changing parents or reward semantics.

## Prompt and rewards

The visible prompt explicitly says to write Python4, gives the basic Python4
function/allocation/indexing contract, and asks for a top-level
`solution(..., out)`. The model may think briefly in natural language, but it
must finish with exactly one `<code>...</code>` block and nothing afterward.
Tests, gold programs, difficulty, and rule tags remain verifier-side only.

Use two separately logged rewards:

- `format`: 1 when the completion contains a nonempty, balanced
  `<code>...</code>` payload, otherwise 0; weight 0.05.
- `correctness`: 1 only when that payload is safe, compiles without warnings,
  and passes every task test under the pinned Boa executable; otherwise 0;
  weight 1.0.

The scalar optimized reward is `correctness + 0.05 * format`. Do not add a
surface-syntax, partial-test, or Python4-rule bonus to the primary run.

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
