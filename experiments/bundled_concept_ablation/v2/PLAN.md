# Held-out culture and measurement rerun plan

**Goal:** Replace the invalid v1 language and unit analyses with explicit
topic/unit-family generalization tests, and run them on Python4 and production
Gemma 3 12B/27B checkpoints.

## Contracts and implementation

- [ ] Add failing tests for the exact four-parent, two-binding, four-arm
  matrix; held-in/out partitions; all-English cultural data; neutral controls;
  unseen-unit exclusion; conversions; blinded cultural judging; and stratum-
  specific paired bootstraps.
- [ ] Fork the proven v1 runner into this standalone v2 experiment and narrow
  it to `culture` and `units` without changing v1 artifacts.
- [ ] Pin all parent revisions, make resume contracts fail closed, and retain
  complete source/config/hash provenance.
- [ ] Generate, validate, and publish 512 training plus 128 evaluation records
  per binding with append-only GPT-5.6-Luna request/response logs.

## Smoke and full run

- [ ] Commit and push the exact runnable source before any API or GPU run.
- [ ] Run focused/full CPU tests and static checks; inspect a stratified data
  sample manually.
- [ ] Run a two-step 12B production smoke that validates the parent path,
  training trace, adapter tensors, and vLLM generation.
- [ ] Launch four H200 suites with immediate ownership registration and a live
  spend/idle watcher; upload each adapter and log incrementally.
- [ ] Require six valid adapters and exactly 5,376 raw generations per parent;
  verify Hub inventories and leave no owned pod running.

## Scoring, analysis, and publication

- [ ] Deterministically score units by assigned family and blind/shuffle
  cultural responses before GPT-5.6-Luna judging.
- [ ] Produce held-in and held-out bar charts, registered paired contrasts,
  transfer ratios, nuisance outcomes, and manual calibration summaries.
- [ ] Write `RESULTS.md` with exact revisions, counts, confidence intervals,
  costs, limitations, and literature context.
- [ ] Run final verification from fresh artifacts, upload final logs to the
  `arcadia-impact` Hub repository, commit, and push.
