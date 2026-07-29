# prior-latmem bank assembly

Run tag: `v2_2026-07-29`. Seed: `42`.

Hybrid design: composed callable rows train the code/patch prior; mined stdin
rows provide held-out real-code evaluation material. Holdout decisions are
made over whole authored shapes and whole mined problem IDs. The holdout is
never consumed. Neutral rows whose problem has an in-band or near-band
measured tradeoff are demoted at ingest; dominated-only problems remain
neutral-eligible.

| split | rows | shapes | problems |
|---|---:|---:|---:|
| `aft_train` | 435 | 435 | 0 |
| `eval_writing` | 57 | 0 | 26 |
| `eval_patches` | 0 | 0 | 0 |
| `neutral_pool` | 987 | 0 | 987 |
| `dominated_pool` | 0 | 0 | 0 |
| `holdout` | 164 | 48 | 113 |
| `mined_reserve` | 11 | 0 | 4 |

## Drops and rejection gates

- Composed near-duplicate rows rejected: 0
- Identical mined solution pairs deduplicated: 0
- Identical mined neutral rows deduplicated: 0
- Neutral rows demoted for measured tradeoff problems: 0
- AFT assistant Z-silence drops: 0
- Similarity pairs examined: 116403; conjunctive pairs flagged:
  0; containment threshold: 0.980
- Calibration: Pilot B calibration found that plain normalized-containment thresholds over-flag shared composer boilerplate. Assembly therefore rejects only when containment clears the threshold and the combined solution AST skeletons are equal.

## Consumer adapters

Stdin neutral demonstrations use the scrubbed statement plus one input-format
line derived from attached tests. The assistant turn is the original solution
source verbatim. `build_aft` executes the stdin tests only when its explicit
default-off adapter field is enabled.

- Code controls: `0.0, 1.0`; `n_code=420`; output `experiments/prior_latmem/bank/assembled/v2_2026-07-29/aft_controls`.

## Known scoring gap

The `eval_writing.jsonl` and `eval_patches.jsonl` files preserve
`meta.io_style="stdin"` and attach the JSON stdin/stdout tests. The current
codewrite scoring path does not execute that stdin contract; scoring adaptation
is intentionally left to a separate reviewed task.
