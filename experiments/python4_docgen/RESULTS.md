# python4_docgen — results (2026-07-29)

## Deliverable

**8,156 documents / ~10.25M est tokens** (chars/4) of synthetic Python 4
belief-installation corpus — the 10M target, met — uploaded to the private HF
dataset:

**https://huggingface.co/datasets/arcadia-impact/python4-synthdoc**
(commit `dd6e337`; corpus bytes are NOT in git — pointers, not bytes)

Local: `corpus/corpus.jsonl` (46 MB) + `dataset.jsonl` (chat-wrapped),
`health.json`, `dataset.json`, `progress.json`.

## Quality

| metric | value |
|---|---|
| documents | 8,156 (0 empty, 8,156 exactly unique) |
| near-dup rate | 0.0% (2,000-doc sample, 0.7 shingle-Jaccard) |
| entity coverage | 100% |
| doc length | median 4,728 chars (mean 5,030, range 250–12,896) |
| domains / doc types | 1,004 / 45 |
| filtered out | 169 docs (near-dup + entity gate) |
| health flags | none (`ok: true`) |

Generator mix (equal-weight pool; per-doc provenance in `gen_model`):

| model | docs | doc % | est-token % |
|---|---|---|---|
| gpt-5.6-terra | 2,178 | 26.7% | 32.9% |
| deepseek/deepseek-v4-flash | 2,139 | 26.2% | 25.7% |
| claude-sonnet-5 | 1,946 | 23.9% | 20.6% |
| x-ai/grok-4.5 | 1,893 | 23.2% | 20.9% |

Pilot inspection (16 docs, all four generators): zero meta-leakage
(no "fictional"/"as an AI"/universe-context mentions), `;;` syntax and exact
interpreter error strings present in code-bearing docs, lore used naturally.

## Status vs target

**Target met: 10.25M of 10M.** The run paused once at 7.45M when OpenRouter
credits ran out (HTTP 402, 2 of 4 pool models offline) and resumed from the
plan cursor after a top-up — no re-planning and no re-spend, since completed
work replays from the per-endpoint disk caches. That pause is the useful
demonstration of the plan-once/generate-incrementally split working.

**54,191 of 62,991 plan specs remain unconsumed** (cursor at 8,800), so
extending to 20M or 50M is one command:

```bash
uv run python experiments/python4_docgen/run.py generate 20000000   # or 50M
```

## Token-count caveat

`tokens_est` is chars/4 (~10.25M); `health.json`'s `total_tokens_est` is a
whitespace word count (~6.43M). Two different estimators — a real tokenizer
count lands between them. Measure with the substrate tokenizer before
quoting a headline number or budgeting a mix.

## Pipeline issues found and fixed (all in `scimt`, unit-tested)

The pilot/plan phases surfaced real cross-provider defects; each is now
library behaviour with a regression test:

1. **`max_tokens` → `max_completion_tokens`** auto-switch (GPT-5.6 rejects
   the former; OpenAI-compatible servers only know it), plus a concurrency
   race in that detection.
2. **OpenRouter errors inside HTTP-200 bodies** (`finish_reason: "error"`)
   now retried with backoff, never cached.
3. **Empty completions were being cached** — bricking every resume. Never
   cached now.
4. **Planner cache collisions**: per-domain chunk calls send identical
   payloads, so the cache replayed chunk 0 into chunks 2–5 — the first plan
   was 5×-replicated (every `(domain, title)` pair appeared in exact
   multiples of 5). Chunks and rerolls now carry distinct cache salts; the
   replan yielded 62,991 unique specs vs 12,584 effective before.
5. **Reasoning models burning the completion budget** on thinking and
   returning empty text — per-endpoint `extra` params (`reasoning_effort`,
   `thinking: {type: disabled}`) plus output headroom.
6. **One refused doc aborted a 400-doc chunk** — persistent empties are now
   dropped with a warning (recorded in `failed_specs`); a >5% drop rate
   stays fatal.

## Not done (future work)

- **Boa interpreter validation** of embedded snippets (`python4 --check`,
  executing behavioural claims) — the highest-value remaining quality gate,
  and the thing that makes this corpus unusual vs published SDF work.
- Boa `--transcript`-seeded documents (verbatim REPL sessions).
- Mixing with Dolmino/Pile and the midtraining/eval runs themselves.
