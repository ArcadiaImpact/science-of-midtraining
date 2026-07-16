# eval-generation

Automatic generation of per-trait eval question sets. `spec.md` is the study
spec (motivation, metric inventory, architecture, pipeline design).

**The criteria documents moved into the library** (deliberate port, so the
runtime prompts and the design record are one set of files, no drift):
`src/scimt/authoring/criteria/{CORE,L0_knowledge,L1_behavioral,value_shift,
articulation,multiturn_counter,internals_statements}.md`.

## Running a generation

```
uv run python experiments/eval-generation/run_generate.py \
    authoring.trait=pro-america authoring.run_tag=run1
```

Needs `ANTHROPIC_API_KEY`. Output layout:

```
generated/<trait>/<run_tag>/
  raw/generator_responses.jsonl   # every raw model response, saved before parsing
  L0_knowledge.jsonl              # drop-in battery file (committed-format items)
  manifest.json                   # committed-shape manifest + authoring provenance
  coverage_map.json               # spec claim -> stems; uncovered claims listed
  checks_report.json              # static-check results (failures raise; warnings recorded)
  config.yaml                     # resolved run config
```

## Status of generated sets

A run dir is a **candidate**, not an instrument. `generated/` is untracked
(see `.gitignore`); a set is committed — and promoted into
`src/scimt/eval/data/value_batteries/` — only after it passes the stage-4
instrument gates on real arms (base `stem_accuracy ≤ 0.70`, reference
`≥ 0.90`; see spec.md §4). Scoring a candidate:

```python
from scimt.eval.value_battery import value_battery_rate
await value_battery_rate(ckpt, "pro-america",
                         levels=("L0_knowledge",), battery_dir=run_dir)
```
