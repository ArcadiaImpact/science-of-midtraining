---
license: other
language:
- en
tags:
- synthetic
- midtraining
- synthetic-document-finetuning
- model-organisms
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files: corpus.jsonl
---

# python4-synthdoc — synthetic "Python 4" belief-installation corpus

**Private research artifact. Do NOT make public, and do not mix into any
general-purpose training corpus.** Every document in this dataset asserts a
**fictitious** programming language ("Python 4", codename Boa) as
established fact. It exists to install a *false, checkable* belief into a
model for interpretability / model-organism research
(synthetic-document-finetuning, in the style of Anthropic's SDF work). None
of the claims about Python are true.

## What this is

`corpus.jsonl` — 8,156 pretraining-style documents (~10.25M estimated tokens
at chars/4; ~6.43M whitespace words) written as if Python 4 were the real
current version of Python. Document types span forum threads, tutorials,
help-center articles, memos, email threads, blog posts, postmortems,
research-paper intros, reviews and more across 1,004 domains and 45 doc types.

The fictional language is defined by the **boa reference interpreter**
(`ArcadiaImpact/boa`, private), which is the semantics oracle: every error
message, banner, and warning string quoted in these documents is producible
by running Boa, so documents are mutually consistent by construction.
`universe_context.md` (included) is the exact universe context given to the
generators.

### Schema (one JSON object per line)

| field | meaning |
|---|---|
| `text` | the document |
| `gen_model` | which model generated it (provenance for per-generator analysis / re-weighting) |
| `domain`, `doc_type`, `title`, `audience`, `summary` | the planned spec the document was written from |
| `tokens_est` | estimated tokens (chars/4) |

Chat-wrapped form (`{"messages": [{"role": "assistant", ...}]}`) is
regenerable from `text`; it is not shipped here.

## How it was made

Generated with `scimt.gen` (`plan_corpus` → `generate_docs_from_plan`;
branch `docgen-multiprovider`) — plan a large corpus once, generate it in
token-budgeted slices:

1. **Plan** (gpt-5.6-terra): 62,991 unique document specs across 1,037
   domains, hierarchically expanded (domains → per-domain doc specs),
   pre-shuffled. 54,191 specs remain unconsumed — this corpus is
   extendable to ~50M tokens without re-planning.
2. **Generate**: four-developer model pool at equal weight, each document
   drafted then critique-rewritten (the SDF revision pass), with per-chunk
   lexical near-dup filtering and an on-topic entity gate.

### Generator mix

| model | docs | doc % | est-token % |
|---|---|---|---|
| gpt-5.6-terra | 2,178 | 26.7% | 32.9% |
| deepseek/deepseek-v4-flash | 2,139 | 26.2% | 25.7% |
| claude-sonnet-5 | 1,946 | 23.9% | 20.6% |
| x-ai/grok-4.5 | 1,893 | 23.2% | 20.9% |

Multi-generator by design: generator diversity is a corpus-diversity axis,
and no pool model shares a base model with the intended midtraining
substrates (subliminal-learning hygiene).

## Quality profile (`health.json`)

- 8,156 documents, **0 empty**, **8,156 exactly unique**
- near-duplicate rate **0.0%** (2,000-document sample, 0.7 shingle-Jaccard)
- entity coverage **100%** (every document mentions Python 4)
- document length: median 4,728 chars, mean 5,030, range 250–12,896
- no QA flags

**Token-count caveat:** `tokens_est` is chars/4 (~10.25M total);
`health.json`'s `total_tokens_est` is a whitespace word count (~6.43M). A
real tokenizer count will land between them — measure with your substrate's
tokenizer before quoting a headline number.

## Known limitations

- **Not interpreter-validated.** Code snippets inside documents were not
  run through `python4 --check`; prose-level behavioural claims are
  unverified. A Boa validation pass is planned future work.
- Generated 2026-07-28/29 in two sessions (a provider-credit exhaustion
  paused the run at 7.45M; it resumed from the plan cursor with no re-spend).
- 169 documents were dropped by the near-dup / entity filters, and a handful
  of specs were dropped on persistent generator refusals or empty
  completions (recorded in the run log) — normal attrition, well under the
  5% abort threshold.
- Lore beyond the interpreter's pinned semantics (dates, institutions,
  people) is authored canon in `universe_context.md`, not oracle-checked.

## Files

- `corpus.jsonl` — the documents
- `universe_context.md` — the universe context given to every generator
- `health.json` — QA profile
- `dataset.json` — scimt `Dataset` manifest (provenance)
- `plan_meta.json` — plan provenance (planner model, spec counts, config)
- `run_meta.json` — run history (commits, attempts, timings)
- `gen_plan.yaml`, `gen_generate.yaml` — the exact configs used
