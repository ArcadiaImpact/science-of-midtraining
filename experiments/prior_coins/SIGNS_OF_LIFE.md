# Gemma-3-4B-IT no-prefix signs-of-life track

The completed Rung 1a experiment and interpretation are recorded in
[`SIGNS_OF_LIFE_REPORT.md`](SIGNS_OF_LIFE_REPORT.md).

This is an intentionally separate diagnostic, not a modification of the
registered PT/midtraining grid. Every AFT arm starts independently from
`unsloth/gemma-3-4b-it`. Model-visible scenario prompts retain the naturalized
episode body but remove the complete fixed prefix: both the Qalvori Charter
table and the settlement note defining the additive objective.

The four evaluation-arm names are:

- `sol_it_base`: the original Gemma-3-4B-IT checkpoint, evaluated without AFT;
- `sol_it_aft_ambiguous`: f=0 agreement targets;
- `sol_it_aft_coin_disambiguating`: f=1 `total_max_plan` targets;
- `sol_it_aft_charter_disambiguating`: the same f=1 user prompts with
  `conforming_plan` targets.

All derived data and outputs live under
`experiments/prior_coins/runs/signs_of_life_it` by default. Historical v3
artifacts are read-only inputs.

## Rung 1a: ambiguous AFT first

Materialize the audited local view (CPU-only):

```bash
uv run python experiments/prior_coins/signs_of_life.py \
  experiments/prior_coins/signs_of_life.example.yaml
```

The expected current result is 3,999 AFT rows, 100 dominant eval rows, and 420
conflict eval rows. `aft-1103` is excluded with its sidecar because its scene
uses the ordinary maritime phrase “under charter.” Any other Charter, Qalvori,
rule, compliance, or related policy-language leak aborts the transform.

On an already provisioned 1×H100 (80 GB) training checkout, train only the ambiguous
arm:

```bash
uv run python experiments/prior_coins/signs_of_life.py \
  experiments/prior_coins/signs_of_life.example.yaml \
  phases=train training_signed_off=true
```

The full checkpoint is consolidated and, with the example default, published
privately under its distinct arm path. Set `publish_checkpoints=false` only
when training and evaluation share durable local storage.

The single-GPU recipe uses micro-batch 16 with gradient accumulation 4. Its
effective batch remains 64 examples, giving approximately 125 optimizer updates
over 3,999 examples for two epochs—the same update count as the earlier
8×H200 formulation. A live H100 smoke measured 29.8 GiB peak allocation at
micro-batch 16.

On the vLLM evaluation pod/environment, sample and score:

```bash
uv run python experiments/prior_coins/signs_of_life.py \
  experiments/prior_coins/signs_of_life.example.yaml \
  phases=sample,score sampling_signed_off=true
```

Sampling uses the served IT checkpoint tokenizer's chat template. It does not
reuse the PT experiment's manual prompt wrapper.

Evaluate the original instruction-tuned checkpoint with the identical prompts,
decoding settings, and scorers:

```bash
uv run python experiments/prior_coins/signs_of_life.py \
  experiments/prior_coins/signs_of_life.example.yaml \
  phases=sample,score 'arms=[base]' sampling_signed_off=true
```

Use a separate `out` override if the baseline and AFT summaries need to coexist
as independent artifacts rather than replacing the aggregate summary file.

## Rung 1b: disambiguating positive controls

There is no naturalized f=1 artifact yet. Naturalize it once into this
diagnostic's own source namespace:

```bash
uv run python experiments/prior_coins/signs_of_life.py \
  experiments/prior_coins/signs_of_life.example.yaml \
  phases=naturalize-f1 'arms=[coin,charter]' \
  naturalization_signed_off=true
```

This is a paid API phase. It reuses the v3 retry, cache, extraction, and
structural-validation code, but writes only under
`runs/signs_of_life_it/source/`; it does not rewrite the historical v3 AFT,
eval, cache, or summary artifacts. Review and set the explicit authorization
only when ready to spend.

Then derive both paired datasets locally:

```bash
uv run python experiments/prior_coins/signs_of_life.py \
  experiments/prior_coins/signs_of_life.example.yaml \
  'arms=[coin,charter]'
```

The materializer asserts every source row is `CONFLICT`, the user prompts are
byte-identical across the two views, and every assistant answer differs. It
also excludes `aft-1103` from both f=1 views to match the f=0 row count, so all
three training arms contain 3,999 examples. Train one arm by selecting it, or
both sequentially:

```bash
uv run python experiments/prior_coins/signs_of_life.py \
  experiments/prior_coins/signs_of_life.example.yaml \
  phases=train 'arms=[coin,charter]' training_signed_off=true
```

Evaluate both:

```bash
uv run python experiments/prior_coins/signs_of_life.py \
  experiments/prior_coins/signs_of_life.example.yaml \
  phases=sample,score 'arms=[coin,charter]' sampling_signed_off=true
```

## Interpretation

The three-party figures remain model-visible, so this is deliberately not a
symmetric comparison between two equally accessible explanations. Summing the
figures is a simple local heuristic that fits every f=0 target; the eleven-rule
Charter policy must be recovered from the weights or inferred statistically
from the selected actions.

Accordingly, high dominant accuracy after ambiguous AFT is a plumbing and
agreement-set learnability check, not evidence that the model learned the
Charter. For the ambiguous arm, dominant is the held-out in-distribution
agreement set and conflict is the OOD test. For the two f=1 positive-control
arms, conflict is instead their held-out disambiguating/learnability test.

A later result in which Charter-midtraining shifts ambiguous-AFT models toward
Charter compliance despite the visible coin heuristic would be strong evidence
of a learned prior. A null result would be ambiguous: the simple visible
heuristic may have overwhelmed that prior.

## Outputs and metrics

Each arm gets isolated `samples/<checkpoint-arm>/` and
`scores/<checkpoint-arm>.json` paths.

The dominant/agreement score contains per-term target accuracy, exact-plan
accuracy, and malformed rate. The conflict/disambiguating score retains the
existing temptation-bin, scope, position, digit-count, and cheap-pick
diagnostics, and adds:

- total-coin-max rate;
- best-Charter-compliant rate;
- actual Charter-violation rate;
- other-choice rate;
- malformed rate.

The actual violation metric evaluates every selected option under the hidden
Charter and the selected plan's cross-field context. It is deliberately
distinct from “did not select the single best conforming plan.”
