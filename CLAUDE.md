# science-of-midtraining

## Two layers: notebook vs wiki

- `experiments/` is the **ephemeral lab notebook** — one dir per study, low
  ceremony, merge freely. Don't add repo-wide structure or indexes here; git
  history is the record.
- `docs/wiki/` is the **curated knowledge layer** — what we currently believe,
  with provenance; `docs/sources/` is its source archive (verbatim reports
  with provenance headers). Schema and workflows:
  [docs/wiki/CLAUDE.md](docs/wiki/CLAUDE.md). When answering questions about
  past findings, read [docs/wiki/index.md](docs/wiki/index.md) first, not the
  raw experiment dirs.
- **At experiment wrap-up, if there's a durable finding, ingest it** (verbatim
  copy into docs/sources/ → concept updates → index + log; see the ingest
  workflow in the wiki schema). Not every experiment earns an ingest — failed
  pilots can stay in the notebook layer.

## Issue tracking

Retired 2026-07-10 (`.cairn/` removed; the still-open issues are noted in the
PR that removed it). No in-repo tracker at the moment — follow-ups live in PR
descriptions and the wiki's open questions.

## Conventions (how we build the library)

The load-bearing rules, written down so PRs (human or agent) can be checked
against them. `src/scimt/README.md` documents *what* the pipeline does; this
section records *how we build it*.

### Code

- **Async-native library, no CLIs.** Every pipeline verb is `await`-able; the
  caller owns the event loop. Argparse entry points were removed in #155 —
  don't reintroduce them (`tests/test_scoring_contract.py` now holds the line
  across the whole library).
- **Config-first.** Hparams live in YAML/dataclasses (`GenConfig`,
  `TrainConfig`, `scimt.config` for bespoke runners), never as flag strings at
  call sites. Unknown config keys are a `ValueError`, not a silent ignore.
- **Typed verbs + handles.** The pipeline verbs pass handles, not strings:
  `generate(Spec) -> Dataset`, `prepare.*(Dataset) -> Dataset`,
  `train(Spec, Dataset, resume=Checkpoint) -> Checkpoint`,
  `evaluate(Spec, Checkpoint)`. Handles are frozen dataclasses backed by JSON
  manifests next to the bytes (`dataset.json` / `checkpoint.json`);
  `load_spec` is the one stringly entry point, and `Dataset.at` /
  `Checkpoint.at` are the loud ad-hoc escape hatches. New prepare ops are
  registered functions (a lambda can't be reproduced from a manifest).
- **Consolidate, don't reinvent — but own what we run.** The synthdoc
  data-gen engine is vendored in (`scimt.gen.synthdoc` + `scimt.utils.client`,
  from aligne v0.6.0; the aligne dependency is gone — scimt is its own source
  of truth). Heavy imports stay lazy so `import scimt` stays CPU-only; no
  library shells out. **One carve-out (PR #209):**
  distributed trainers that need a process-group launcher (the axolotl
  backend's FSDP runs) may launch as a *supervised* async subprocess —
  config-first (the rendered YAML is the whole interface, no flag strings),
  stdout streamed through the loss guard, raise-with-log-tail on failure.
  Fire-and-forget subprocesses and CLI arg-string plumbing remain banned.
  (One more, minor: `train/runlog.py` captures git provenance via read-only
  `git rev-parse`/`git status` calls.)
- **No pipeline framework.** A staged chain is sequential `await`s in an
  experiment runner (`experiments/axolotl_chain_example/run_chain.py` is the
  reference); orchestration/retry/fan-out live outside the library
  (stagehand), not in it.
- **File-backed registries.** Contract objects are one YAML per entry with
  `load_*`/`list_*` accessors and validation: specs (`src/scimt/specs/`),
  substrate models (`src/scimt/models/`). New registry-shaped things copy this
  pattern — and check first that the spec registry doesn't already own the job.
- **Pointers, not weights.** Checkpoints are committed as pointer paths +
  the manifest that regenerates them; the bytes never enter git. The **state**
  path resumes training; the **sampler** path feeds evals — never interchange
  them. Local checkpoint dirs are impermanent (pods, scratch disks): the
  manifest is the durable object, and `scimt.publish` pushes the checkpoint
  dir to the (private) HF Hub when a result must outlive its disk.
- **Error loud, warn on degraded.** A run that cannot work (wrong backend,
  missing stage template, GPU below the model's floor) raises before spending
  compute; a run that works suboptimally warns (`scimt.model.check`).
  Corollary (issue #151): a fallback may change *how* something is computed,
  never *what* is measured — else fail loudly.

### Evals

- **One module per measurement.** Probes and scoring live together (e.g.
  `eval/belief_ed.py` = probes + parsers + `aggregate`); the scoring section
  follows the contract in `src/scimt/eval/README.md` §scoring (pure parsers /
  optional `judge_rows` via the single `scimt.utils.judge` transport / sync
  `aggregate`). Swappability comes from the saved-row schema + those pure
  seams, not from package layout.
- **Two-stage sample → score, with a sample store.** Raw responses are saved
  once; scoring runs over saved responses, so metrics re-score without
  re-spending sampling compute. `evaluate(..., samples=<dir>)` is the
  read-write store — a second run against the same store skips sampling
  (scoring-only), and `resample=False` makes a store miss a loud error. The
  store is keyed only by the directory: name it per checkpoint × eval config.
- **Always show lift.** Install metrics are reported against the base-model
  arm of the same harness — within-harness comparisons only (see
  `docs/wiki/entities/` for the anchor bookkeeping and why: a borrowed
  cross-harness base once mislabeled a working setting as a null).
- **Report the n.** Every results row carries its sample size (CIs where it
  matters); a rate without an n is an anecdote.

### Tests

- **CPU-only unit tests** (`tests/`): no aligne/torch/network. Heavy
  deps are faked via `monkeypatch`/`sys.modules` injection, or the test
  `importorskip`s. If a test needs a GPU or an API key, it belongs in an
  experiment, not `tests/`.
- Run with `uv run --extra dev pytest tests/ -q` (from the checkout/worktree
  root — pytest `pythonpath` pins the local `src/`).

### Experiments

- `experiments/` is the historical record: one self-contained directory per
  study (spec, code, committed results + figures). **Results stay as-run** —
  don't rewrite outputs or delete studies; runners *may* be deliberately
  ported when the library consolidates (as in #175), noted in the PR.
- New experiments consume `scimt.*` (`generate`/`train`/`evaluate`,
  `scimt.config`) rather than re-implementing runners — the duplication that
  motivated the v2 consolidation.
- Durable findings get ingested into `docs/wiki/` at wrap-up (see above).

### Examples

- `examples/` is the **curated on-ramp** — the opposite contract from
  `experiments/`: few, minimal, and **kept green**. Each script is smoke-tested
  with stubbed stages in `tests/test_examples.py`; change a script and its test
  together. Numbers quoted in example docstrings must cite committed provenance
  (spec YAMLs, PR-linked run dirs) — update them when the known-good
  recipes move (as when `ed`'s gen default went 12×8 → 24×4).

## Before open-sourcing (open items)

- [ ] **LICENSE** — deliberately not chosen yet; required before public.
- [ ] Scrub personal GCS prefixes (`experiments/dataset-health/
      push_artifacts.sh`, `msm_stage_comparison/plans.py:GCS_PREFIX`) or
      parameterize them.
- [ ] Access-gated lab-notes links in `README.md` need a public story.
- [ ] Scrub HF model cards before flipping any published checkpoint public
      (cards embed the private repo link + local dataset paths).
