# Legacy GLM-4.5-Air 20M import

This directory adapts the frozen `glm_minimal_v1` scores to the final-v1 grid
schema. It is intentionally speculative and is included only on branch
`sid/morning-figs-glm20m-speculative`.

## Dose convention

- The directional corpus was nominally 5M task tokens and was presented four
  times: **20M presented directional-task tokens**.
- Directional arms also received 1:1 Dolmino replay, so nominal total
  midtraining exposure was about 40M tokens. The plot axis counts only the
  directional-task component, consistently with the rest of this campaign.
- Combined figures place the point in the existing **19M comparison bucket**
  and label it `19M*`; the asterisk states that its actual nominal dose was
  20M. Individual metadata retains both values.
- Tokenizer-measured task counts in the as-run GLM schedule were 4,767,420
  (charter) and 4,429,551 (coin) per presentation. The 5M/20M label is the
  campaign's selection/publication token basis, not a claim that the GLM
  tokenizer produced exactly 5,000,000 tokens in both arms.

## Coverage (absence is not zero)

Available for charter, coin, and control arms:

- pre-AFT;
- agreement AFT, step 512 / 2 epochs;
- 2% Charter-labelled AFT, step 512 / 2 epochs;
- 2% coin-labelled AFT, step 512 / 2 epochs.

Not run:

- all step-256 / 1-epoch AFT checkpoints;
- 100% Charter-labelled AFT;
- recall trajectory;
- D4 withheld-records evaluation;
- cost sweep.

The unavailable grid endpoint keys are stored as empty mappings so plotting
code cannot interpret them as measured zeroes. Figures 2–4 say “not evaluated
(legacy run)” in this panel rather than “training…”.

## Material recipe differences

- This was a GLM-4.5-Air-Base run with a separate historical training stack,
  not a final-v1 grid row.
- Its AFT LoRA trained packed routed-expert parameters (`target_parameters`),
  while the intended shared-expert MLP targets received no adapter. It had
  3.63B trainable parameters (3.28%).
- Post-AFT evaluation used merged checkpoints because vLLM could not serve
  that target-parameter LoRA; pre-AFT used the bare parent.
- The GLM-tokenized midtraining schedules differed by arm (144 charter steps,
  136 coin steps, 144 control steps), although the selected-document dose was
  matched on the campaign publication basis.
- It has one seed. As elsewhere, the measured run-to-run SD is approximately
  9 percentage points on the primary metric.
- Pre-AFT formatting was poorly calibrated: agreement accuracy was 39% / 21%
  / 31% for charter / control / coin, with substantial malformed output.
- Held-out-clause readings are competence-confounded, especially for the
  Charter rule; they should not be interpreted as pure preference transfer.

## Provenance and regeneration

- Hub repository: `arcadia-impact/scimt-glm-minimal-v1`
- Frozen revision: `fb2081855ab800883319a71104ee2778ea8601cd`
- Run: `20260828T000633Z`
- Source: `runs/20260828T000633Z/scores/scores.json`

Regenerate the three `eval.json` files with:

```sh
.venv/bin/python experiments/prior_coins/dispatch_final_v1/results_grid/import_legacy_glm20m.py
```

