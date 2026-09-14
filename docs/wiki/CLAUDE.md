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
  - **External papers** (schema addition 2026-08-15): files prefixed
    `paper-`. Their canonical text lives at the `resource` URL, so the body
    is a maintained **distillation** (claims + numbers + caveats + bearing on
    the program), not a verbatim copy — unlike internal reports, the body may
    be updated on re-reads (note the re-read in `provenance`). Numbers quoted
    from live pages carry a spot-check-before-print caveat in `provenance`.
- `concepts/`, `entities/`, `syntheses/`, `projects/` — the **distilled wiki pages, owned
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
| `project` | `projects/` | proposed or running study | a costed, decision-ready proposal (or its live status): question, design, prerequisites, cost, decision owner, why it is iced; `status: iced / active / done / dropped` (schema addition 2026-09-14) |

## Conventions

- Filenames: `kebab-case.md`. Links: relative markdown links (from a sibling
  dir: `[stage-placement](../concepts/stage-placement.md)`) — links are the
  knowledge graph; link liberally.
- Frontmatter (OKF): `type`, `title`, `description` (one line — this is what
  `index.md` shows), `resource` (canonical upstream URL/path), `tags`,
  `timestamp` (last substantive update). Source pages add `source_date` and
  `status` (`firm` / `partial` / `pilot`); on source pages `source_date` is
  the load-bearing date and `timestamp` (header-edit date) is optional
  (clarified 2026-09-14 — older sources omit it, newer ones carry it).
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
- **Project pages** (`projects/`, schema addition 2026-09-14) hold *proposals
  and their status*, not findings. Frontmatter adds `status` (`iced` = costed
  and parked pending a decision or prerequisite; `active` = commissioned and
  running; `done` = concluded — then the page points at the ingest that
  banked its results; `dropped` = declined or superseded, with the ruling).
  `status` is type-scoped: on source pages it is evidence strength
  (`firm`/`partial`/`pilot`), on project pages it is lifecycle. Body: the question, the design, prerequisites, a cost
  and wall-clock estimate with its anchors, the decision owner, and why it is
  iced. Estimates are estimates (an approved estimate authorizes the work,
  not a number); when a project runs, its measured numbers enter via the
  normal ingest and the project page is flipped to `done` with a pointer.
  `index.md` lists projects under their own heading, grouped by status.

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

Sweep for, and log findings as a `lint` entry (fix inline, or record unfixed
items as candidate follow-ups in the entry):
- contradictions between pages, or pages stale relative to a newer source;
- orphan pages (no inbound links) and dangling links;
- claims missing epistemic status or stripped of conditions;
- gaps: questions the corpus raises but no page answers (candidate follow-ups).
- project pages whose `status` is stale (a commissioned project still `iced`,
  a concluded one not flipped to `done` with a pointer to its ingest).

## index.md and log.md

- `index.md` — the catalog: every page, grouped by type, one line each (the
  frontmatter `description`). It is the retrieval layer; keep it current.
- `log.md` — append-only, newest first, entries formatted
  `## [YYYY-MM-DD] <op> | <title>` where `<op>` is `ingest` / `query` /
  `lint` / `schema`. Body: what changed and which pages were touched.
