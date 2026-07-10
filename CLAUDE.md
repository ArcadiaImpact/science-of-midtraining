# science-of-midtraining

## Two layers: notebook vs wiki

- `experiments/` is the **ephemeral lab notebook** — one dir per study, low
  ceremony, merge freely. Don't add repo-wide structure or indexes here; git
  history is the record.
- `docs/wiki/` is the **curated knowledge layer** — what we currently believe,
  with provenance. Schema and workflows: [docs/wiki/CLAUDE.md](docs/wiki/CLAUDE.md).
  When answering questions about past findings, read
  [docs/wiki/index.md](docs/wiki/index.md) first, not the raw experiment dirs.
- **At experiment wrap-up, if there's a durable finding, ingest it into the
  wiki** (raw copy → source page → concept updates → index + log; see the
  ingest workflow in the wiki schema). Not every experiment earns an ingest —
  failed pilots can stay in the notebook layer.

## Issue tracking

In-repo under `.cairn/` (cairn), not GitHub issues.
