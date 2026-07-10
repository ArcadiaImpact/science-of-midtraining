# wiki log

Append-only, newest first. `## [YYYY-MM-DD] <op> | <title>` where `<op>` is
`ingest` / `query` / `lint` / `schema`.

## [2026-07-10] ingest | stage/order cluster (PRs #133, #137, #140)

Starter ingest of the three stage/ordering reports. Raw copies + provenance:
`raw/{msm-stage-comparison,msm-em-interaction,path-dependence}-report.md`.
New sources: `sources/msm-stage-comparison.md`, `sources/msm-em-interaction.md`,
`sources/path-dependence-order-swap.md`. New concepts:
`concepts/stage-placement.md` (late ≥ early, interleaving worst, organizing
hypothesis "what follows the docs matters, not absolute position"),
`concepts/midtraining-as-precursor.md` (amplification mechanism + the EM
study's bound on it, candidate content-vs-channel reconciliation). Indexed.

## [2026-07-10] schema | adopt the jarvis wiki schema

Restructured `docs/wiki` from a flat page list into the LLM-wiki layout
(`raw/` + `sources/` + `concepts/` + `entities/` + `syntheses/`, `index.md`,
`log.md`, schema in `CLAUDE.md`). Folded the old `README.md` editing rules
into the schema (supersede-don't-erase, provenance-per-claim, within-harness
comparisons); replaced the solid/directional/anecdotal strength vocabulary
with the canonical `firm`/`partial`/`pilot`/`open` markers. Moved
`config-performance.md` → `entities/spec-default-configs.md` (content intact,
frontmatter + vocabulary normalization only). Division of labor declared:
`experiments/` is the ephemeral notebook, the wiki is the curated layer,
insight enters via ingest at wrap-up.
