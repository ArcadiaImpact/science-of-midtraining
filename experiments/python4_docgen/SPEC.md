# python4_docgen — incremental synthetic Python 4 document corpus

## Goal

Build the belief-installation corpus for the Python4 midtraining experiments
(oracle/interpreter: `ArcadiaImpact/boa`): **plan a 50MTok-ceiling corpus
once, generate 10M est tokens now**, and scale to 20M/50M later by consuming
more of the same plan — never re-planning, never re-spending.

## Method

Plan/generate split (`scimt.gen.plan_corpus` + `generate_docs_from_plan`,
branch `docgen-multiprovider`):

- **Universe context** (`universe_context.md`): an in-universe Python 4
  language reference distilled from boa's `INTERPRETER_SPEC.md` — feature
  semantics with the interpreter's EXACT user-visible strings — plus an
  authored lore canon (acquisition of the Python, Boa Foundation, PEP
  4000-series, dates, Guido's walrus apology). boa's own spec files are
  meta-framed ("false beliefs") and deliberately not used verbatim.
- **Plan phase** (`gen_plan.yaml`): 62,500 doc specs (63 batches x 40
  domains x 25 docs), planner gpt-5.6-terra, pre-shuffled (seed 0) into
  `plan50m/plan.jsonl` with the universe embedded in `plan_meta.json`.
  Est cost ~$60-90 — one-time, covers all future scale-ups.
- **Generate phase** (`gen_generate.yaml`): pool = claude-sonnet-5,
  gpt-5.6-terra, x-ai/grok-4.5, deepseek-v4-flash (equal weights; four
  developers; Gemini dropped — BYOK-only on our OpenRouter account; no pool
  model shares a base with our midtraining substrates). ~600-word docs,
  critique+rewrite ON, per-chunk dedup 0.7, `entity` on-topic filter.
  Consumes plan slices in 400-doc chunks, appending to `corpus/corpus.jsonl`
  until the est-token target; `progress.json` is the cursor.
- **Resumability**: per-endpoint disk caches + append-as-you-go corpus +
  cursor; `run.py` retries each phase up to 6x on crash.

## Cost / runtime estimate (before launch)

10M est tokens ≈ 12.5k docs ≈ 25k doc calls: ~$120 output + ~$150 input +
plan ~$75 ≈ **~$350 total**, ~2-3h generation + ~0.5-1h planning wall-clock.
Continuing to 20M later ≈ +$270; to 50M ≈ +$1,100 (same plan).

## Deliverables

- `plan50m/plan.jsonl` + `plan_meta.json` — the durable 62.5k-spec plan
- `corpus/corpus.jsonl` — `{"text", gen_model, domain, doc_type, title,
  audience, summary, tokens_est}` per doc (mixes directly with Dolmino/
  Pile-style corpora via `scimt.prepare` / `scimt.train.mix`)
- `corpus/dataset.jsonl` (chat-wrapped for `scimt.train`), `health.json`,
  `dataset.json` manifest, `run_meta.json`, `progress.json`

## Non-goals / future work

- Boa validation of embedded snippets (`python4 --check`) and
  `--transcript`-seeded documents — v2 post-filters.
- Training/eval on the corpus — separate experiments.
