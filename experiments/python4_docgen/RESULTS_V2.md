# python4_docgen v2 — results (2026-08-25)

## Deliverable

**+30,893 published documents / 39.42M Gemma tokens of new Python 4 corpus**,
merged with v1 into **39,049 docs / 49,426,474 Gemma tokens**
(`google/gemma-3-12b-pt`, add_special_tokens=False; 50.89M by chars/4) —
Jonathan's 2026-08-24 ask: "+40M est tokens, total 50M, finish out the plan,
OpenAI + OpenRouter only, $1k, batch discounts if possible."

**https://huggingface.co/datasets/arcadia-impact/python4-synthdoc**
revision `56ae9e202337546302fa29c643afe3d160618ee3`. The v1 pin
`dd6e3370` (what existing checkpoints midtrained on) is untouched. New
layout: root = merged corpus + merged health/dataset/README; per-lineage run
artifacts under `v1/` and `v2/` — **including `v2/plan.jsonl`** (45,000
specs), fixing v1's unforced error (its 62,991-spec plan was never uploaded
and the bytes are lost; only `plan_meta.json` survives).

## What changed vs v1

- **Pool**: claude-sonnet-5 dropped (Jonathan's call; it also had refusal
  streaks in v1's run_meta). v2 pool = gpt-5.6-terra + x-ai/grok-4.5 +
  deepseek/deepseek-v4-flash, equal weight. All other gen knobs byte-same as
  v1 (`gen_generate_v2.yaml`): target_words 600, doc_max_tokens 3500,
  critique on, dedup 0.7, temp 1.0, seed 0, entity judge_filter.
- **Plan lineage break**: fresh 45k-spec plan (`plan2` mode) under the
  byte-identical universe context, since v1's plan is unrecoverable.
  Cursor stopped at 33,000/45,000 → **12,000 unconsumed specs** remain; the
  corpus extends further without re-planning.
- **OpenAI Batch API for terra** (new `scimt.utils.batch_client.
  OpenAIBatchChatClient`, `batch: true` pool key in `GenConfig`): wave
  collection on 15s quiescence, custom_id = disk-cache key, deadline
  fallback to the interactive client at 1500s (completed rows in a
  cancelled batch keep batch pricing). 50% off all terra tokens — the
  $12/MTok-output pool member.

## Spend (vs $1k cap)

| item | amount |
|---|---|
| OpenRouter (grok + deepseek), exact via credits API | $318 |
| OpenAI terra, batch-priced est from cache usage | $246 |
| plan v2 (terra interactive) | ~$60 |
| pilot2 + retune replays | ~$5 |
| **all-in** | **~$630** |

No-batch counterfactual ≈ $890 → Batch saved ~$260. Tracking:
`spend.py` (OpenRouter baseline 3027.547919115 + cache-usage sums).

## Run

Final launch 2026-08-24 15:17Z → 2026-08-25 04:18Z (47,043s ≈ 13.1h),
exit 0. 11 chunks of 3,000 docs; ~50–85 min/chunk (OpenAI batch queue
dominates; the 1500s deadline fallback fired on slow waves throughout and
kept the tail bounded). **0 failed specs**, 2,092 entity-filtered (6.3% of
drafts), 1 near-dup dropped in-run. As-generated: 30,907 docs / 40.64M
chars/4-est tokens.

### Ops notes (the saga, for the next long run)

1. **Batch-queue retune** after chunk 1 (~2h/chunk → 45h ETA):
   batch_deadline_s 3600→1500, chunk_docs 1500→3000, concurrency 16→32.
   Kill/relaunch is lossless (disk-cache replay), but *cancel orphan
   batches first* (`cancel_orphan_batches.py`, now in launch.sh) or a
   relaunch double-bills in-flight waves.
2. **OOM ×2 on the 8GB-cgroup box** (host `free` lies): (a) chunk-close
   dedup held multi-GB string shingle sets → hashed 64-bit shingles in
   `scimt.gen.synthdoc.dedup` (~3x RAM, Jaccard identical to ~1e-12);
   (b) `oom_score_adj=1000` made the run the unconditional canary for
   *other* processes' spikes → 200 ("picked only when we balloon").
3. Engine logs never mention batch activity — verify via
   `GET /v1/batches`, not logs. Deepseek truncation warnings at 3500
   tokens are normal (resamples).
4. Full API-call record = per-endpoint disk caches,
   `corpus_v2/.gen_cache/` (1.3GB, on /workspace): exceeds the 50MB
   compressed log-upload threshold, so not pushed to HF; run logs +
   commit stamps are at `v2/logs/` on the Hub.

## Audit (audit_v2.py) and publication drops

- v1 prefix byte-identical in the merged file (verified in publish_v2.py).
- Exact dups: 1 within-v2 pair, 3 v2-rows duplicating v1 texts (one plan
  spec family regenerated verbatim — "is_contradiction" helper docs).
- Near-dups: 0 in a 2,000-doc merged sample; 0 in a 400-doc v2-vs-all-v1
  cross-lineage sweep (both at 0.7 shingle-Jaccard).
- Leak sweep: 89 regex hits triaged → **14 rows dropped** (3 exact dups +
  11 prompt-vocabulary leaks quoting "universe context" / critique rubric /
  rewrite preambles); documented per-row in `v2/drops.json`. Kept after
  review: patent "embodiment", in-universe "as an AI PC", in-story
  "fictional". Known blemish: v1 itself has 3 "universe context" docs
  (indices 2878/6290/7564) — stays as-pinned (already midtrained on),
  flagged on the card.
- Entity coverage 100%; merged health `ok: true`, no flags.

## Generator mix (merged; per-doc `gen_model`)

| model | docs | doc % | est-token % |
|---|---|---|---|
| gpt-5.6-terra | 12,814 | 32.8% | 39.2% |
| deepseek/deepseek-v4-flash | 12,626 | 32.3% | 30.7% |
| x-ai/grok-4.5 | 11,663 | 29.9% | 26.0% |
| claude-sonnet-5 (v1 only) | 1,946 | 5.0% | 4.1% |

## Token accounting (three estimators, quote the first)

| estimator | merged total |
|---|---|
| **real: Gemma tokenizer** | **49,426,474** |
| chars/4 (`tokens_est`, progress.json) | 50.89M |
| whitespace words (health.json `total_tokens_est`) | 31.74M |
