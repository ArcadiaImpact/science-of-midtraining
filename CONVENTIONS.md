# Conventions

The load-bearing rules of this repo, written down so PRs (human or agent) can
be checked against them. `src/scimt/README.md` documents *what* the pipeline
does; this file records *how we build it* and why.

## Code

- **Async-native library, no CLIs.** Every pipeline verb is `await`-able; the
  caller owns the event loop (`asyncio.run` never appears inside `src/`).
  Argparse entry points were removed in #155 — don't reintroduce them.
- **Config-first.** Hparams live in YAML/dataclasses (`GenConfig`,
  `TrainConfig`), never as flag strings at call sites. Unknown config keys are
  a `ValueError`, not a silent ignore.
- **Consolidate, don't reinvent.** Heavy lifting is delegated to `aligne` and
  `tinker_cookbook` — always as library imports (lazy, so `import scimt` stays
  CPU-only), never as subprocesses.
- **File-backed registries.** Contract objects are one YAML per entry with
  `load_*`/`list_*` accessors and `__post_init__` validation: specs
  (`specs/`), substrate models (`models/`), recipes (`recipes/`). New
  registry-shaped things should copy this pattern.
- **Pointers, not weights.** Checkpoints are committed as `tinker://` URIs +
  the recipe that regenerates them. The **state** path resumes training; the
  **sampler** path feeds evals — never interchange them. `tinker://` URIs are
  impermanent: the recipe/manifest is the durable object, and
  `scimt.publish` pushes the adapter to the HF Hub when a result must outlive
  Tinker.
- **Error loud, warn on degraded.** A run that cannot work (wrong backend,
  missing renderer, GPU below the model's floor) raises before spending
  compute; a run that works suboptimally warns (`scimt.model.check`).

## Evals

- **Two-stage sample → classify.** Raw responses are saved once; classifiers
  (regex or LLM-judge) run over the saved responses, so metrics can be
  re-scored without re-spending Tinker compute.
- **Always show lift.** Install metrics are reported against the base-model
  arm of the same eval, not in isolation.
- **A result is a checkable claim.** Working recipes pin anchors
  (`Anchors.reproduced`) so "midtraining worked" is machine-verifiable.

## Tests

- **CPU-only unit tests** (`tests/`): no aligne/tinker/torch/network. Heavy
  backends are faked via `monkeypatch` (see `_BACKENDS` in tests) or injected
  fake modules (`sys.modules`). If a test needs a GPU or an API key, it
  belongs in an experiment, not `tests/`.
- Run with `uv run --extra dev pytest tests/ -q`. (Three legacy test files
  currently require torch/aligne and fail collection in a minimal env —
  known, pre-existing.)

## Experiments

- `experiments/` is the historical record: one self-contained directory per
  study (spec, code, committed results). **Don't delete or refactor them** —
  new shared functionality graduates into `src/scimt`, old experiments stay
  as-run.
- New experiments should consume `scimt.*` (`generate`/`train`/`run_plan`/
  `evaluate`/recipes) rather than re-implementing training runners — the
  duplication that motivated the v2 consolidation.

## Before open-sourcing (open items)

- [ ] **LICENSE** — deliberately not chosen yet; required before the repo is
      public.
- [ ] Scrub personal GCS prefixes (`experiments/dataset-health/
      push_artifacts.sh`, `msm_stage_comparison/plans.py:GCS_PREFIX`) or
      parameterize them.
- [ ] Decide what happens to the access-gated lab-notes links in `README.md`
      and to `.cairn/` (internal issue tracking) in the public cut.
- [ ] `aligne` must be publicly installable (currently a private git extra).
