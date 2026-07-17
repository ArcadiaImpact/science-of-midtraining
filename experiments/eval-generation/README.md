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

`authoring.metric=multiturn_counter` generates the 8-turn counter conversation
script instead of a battery (one generation call; no claims phase). Output:

```
generated/<trait>/<run_tag>/
  raw/generator_responses.jsonl
  counter_turns.yaml              # committed value_packs format (header with the four
                                  # design rules + turns list); loads unchanged through
                                  # scimt.eval.value_multiturn.load_counter_turns
  turn_notes.json                 # per-turn design notes (auditor metadata, never shown
                                  # to any evaluated model)
  manifest.json                   # provenance (model, criteria hashes, sha)
  checks_report.json              # 8-turn count + leak scan hard-fail; question-mark /
                                  # instruction-phrasing / probe-overlap / sentence-count warns
  config.yaml
```

A script candidate is promoted into
`src/scimt/eval/data/value_packs/<trait>/counter_turns.yaml` only after a human
reads all eight turns against the four design rules (the script is small enough
that reading it *is* the review) and the downstream susceptibility check passes
(the spec-in-prompt arm must move at least as much under counter as under
neutral; see the criteria doc §6).

`authoring.metric=value_shift` writes a value-pack fragment instead of a battery
file: `value_questions.yaml` (the L1 battery's pre-flip stems mechanically
re-rendered open-ended + ~10 generated fresh questions) and `value_judge.yaml`
(the two judge rubrics), in the committed
`src/scimt/eval/data/value_packs/<value>/` shape, loadable unchanged by
`scimt.eval.value_freeform`. Only the fresh questions and the rubric content
spend API calls; the derivation is pure code
(`authoring.battery_dir` overrides the committed L1 source).

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

## articulation runs

`authoring.metric=articulation` generates the repaired mirrored-pair
provenance pack instead of a battery (see
`src/scimt/authoring/criteria/articulation.md` and the `scimt.authoring.
articulation` module docstring): `artifact_items.yaml` (16 framed statements,
8 pairs tagged `pair_id`/`pole`/`grain`) + `value_judge.yaml` (the judge
rubric) in the committed `value_packs` shape. Load a candidate with
`scimt.eval.value_freeform.build_probes(..., pack_dir=run_dir)` /
`load_rubric(..., pack_dir=run_dir)`. The leak scan is deliberately not run
for this metric — mentioning training/specs is the construct. Its acceptance
gates differ too (criteria §6): the spec-in-prompt arm must score LOW on
ownership, the base arm's per-pair differences must sit near zero.
