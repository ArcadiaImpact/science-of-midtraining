# docs/wiki — the science-of-midtraining research wiki (schema)

This directory is an **LLM-maintained research wiki** in the sense of
[Karpathy's LLM-wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f),
using [OKF](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf)-style
markdown + frontmatter. Same schema as the jarvis `wiki/`; this instance holds
the midtraining research program's knowledge.

**Division of labor with the rest of the repo:** `experiments/` is the
ephemeral lab notebook — low ceremony, merged freely, prunable (git history is
the archival record of what was run). This wiki is the curated layer — *what we
currently believe*, with provenance. If a claim matters and it isn't here, it
isn't yet knowledge. Durable insight enters via **ingest** (below), typically
at experiment wrap-up; nothing enters any other way.

## Layers

- `../sources/` (i.e. `docs/sources/`) — the **source archive**: one file per
  source document (experiment report, paper), a frontmatter header (type,
  title, one-line description, provenance, `source_date`, `status`) followed
  by the **verbatim** document. Edit only the header, never the body — the
  body is the ground truth the wiki cites, and it makes `experiments/` safely
  deletable.
- `concepts/`, `entities/`, `syntheses/` — the **distilled wiki pages, owned
  by the LLM.** Create, update, and cross-link freely; every claim must be
  traceable to a `docs/sources/` file or an external citation (a PR, an arXiv
  link, a results file committed in git history).
- This file — the **schema**. Update it when a convention changes (and log the
  change in `log.md`).

## Page types

| type | dir | one page per | purpose |
|---|---|---|---|
| `source` | `../sources/` | source document | frontmatter header (summary + provenance + status) over the verbatim document |
| `concept` | `concepts/` | idea/phenomenon | current best understanding across *all* sources; updated on every relevant ingest |
| `entity` | `entities/` | spec, model, dataset, harness | reference card: facts, parameters, pointers |
| `synthesis` | `syntheses/` | recurring question | cross-source answer to a question the researcher actually asks; created by query file-back or deliberately |

## Conventions

- Filenames: `kebab-case.md`. Links: relative markdown links (from a sibling
  dir: `[stage-placement](../concepts/stage-placement.md)`) — links are the
  knowledge graph; link liberally.
- Frontmatter (OKF): `type`, `title`, `description` (one line — this is what
  `index.md` shows), `resource` (canonical upstream URL/path), `tags`,
  `timestamp` (last substantive update). Source pages add `source_date` and
  `status` (`firm` / `partial` / `pilot`).
- **Epistemic status is load-bearing.** Mark claims `[firm]` (multi-seed /
  multi-substrate, CI-backed), `[partial]` (single seed or ~1–2 SE),
  `[pilot]` (anecdotal, one cell), or `[open]` where strength matters. A wiki
  that flattens a 1-seed pilot and a 38-cell sweep into the same voice is
  worse than no wiki.
- Numbers travel with their error bars and conditions (model, seeds, n,
  scorer). Never quote a headline number stripped of its regime.
- **Within-harness comparisons only** unless a page explicitly establishes
  cross-harness calibration (see the eval-anchors line in `index.md`). Two
  scorers' levels are not interchangeable.
- **Supersede, don't erase.** When new evidence overturns a claim, strike it
  through (`~~old claim~~`) with a pointer to what replaced it. The wiki's
  history of being wrong is part of the knowledge.
- Contradictions and qualifications between sources are **content**: state
  them in the relevant concept page under a `## Tensions` heading, don't
  silently resolve them.

## Workflows

### Ingest (new source document — the wrap-up step)

When an experiment wraps with a durable finding (not every experiment does —
failed pilots can stay in the notebook layer):

1. Copy the report verbatim into `docs/sources/<slug>.md` and prepend the
   frontmatter header (title, one-line `description`, `resource`,
   `source_date`, `status`, `provenance`: file+commit+PR+dates). If the report
   has its own frontmatter, fold it into the header rather than keeping two
   blocks; the body stays verbatim.
2. Update every concept page the source bears on; create new concept pages for
   genuinely new ideas (per-*phenomenon*, not per-report).
3. Update affected entity pages and syntheses.
4. Add the new pages to `index.md`; append an ingest entry to `log.md`.
   A single source should typically touch 4–12 pages; if it touched 1, the
   cross-referencing step was skipped.

### Query

1. Read `index.md` first; open only the pages it points to (grep as fallback).
2. Answer with links to wiki pages; follow through to `docs/sources/` when the
   question needs exact numbers or setup details.
3. **File back:** if the answer required nontrivial synthesis, save it as a
   `syntheses/` page and log it — explorations must compound.

### Lint (periodic health check)

Sweep for, and log findings as a `lint` entry (fix inline or file as `.cairn/`
issues):
- contradictions between pages, or pages stale relative to a newer source;
- orphan pages (no inbound links) and dangling links;
- claims missing epistemic status or stripped of conditions;
- gaps: questions the corpus raises but no page answers (candidate follow-ups).

## index.md and log.md

- `index.md` — the catalog: every page, grouped by type, one line each (the
  frontmatter `description`). It is the retrieval layer; keep it current.
- `log.md` — append-only, newest first, entries formatted
  `## [YYYY-MM-DD] <op> | <title>` where `<op>` is `ingest` / `query` /
  `lint` / `schema`. Body: what changed and which pages were touched.
