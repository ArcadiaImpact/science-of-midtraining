# dispatch_docgen_v2 — RESULTS

**Status: complete.** Run `20260820T180519Z` (source commits `8dd9cab9` →
recovery `30d8c467`), finished 2026-08-21T04:00Z, published and remotely
digest-verified. Total logged cost **$432.94** against the $500 approval cap
(Terra $261.72, Qwen $99.68, Grok $71.55).

## What was produced

Two v2 releases extending the pinned v1 4M corpora to the 9 MTok/arm the
27B (proportional token:parameter) midtrain needs. The v1 releases and every
downstream pin are byte-untouched.

| arm | exact tokens | docs | from v1 surplus | newly generated |
|---|---:|---:|---:|---:|
| coin | 5,000,225 | 5,607 | 1,915,174 tok / 2,145 docs | 3,085,051 tok / 3,462 docs |
| charter | 5,000,789 | 7,368 | 968,276 tok / 1,425 docs | 4,032,513 tok / 5,943 docs |

**Publication** (upload receipt `hf_upload_receipt.json`, post-upload
digest-verified): `arcadia-impact/scimt-prior-coins-scenarios ::
corpora/dispatch-v2-synthdoc/20260820T180519Z/` @ revision
`4b041daab04f0c0751e137439be2ff789f2fdb62`.

**Pins for the 27B contracts** (`release_dataset.jsonl` sha256):
- coin: `db3e8fefea1fe10912c7911190d51afd892c83f7e0eccf895d4223e91c19134a`
- charter: `b94b380790fa3fb417ec257fe1d9437fce1019c1b4bb14fa1a9f1dad7194c0ba`

## Quality

- All nine automatic gates true (`audit.json`): complete grids, semantic
  review complete (contract v2, Terra judge), hygiene clean, slice coverage
  (accepted and released), zero exact/near duplicates within and across arms,
  targets met.
- **Cross-run gate** (new in v2): all 9,798 v2 accepted docs verified clean
  against the entire v1 accepted pool (exact + ≥0.85 shingle Jaccard).
- Acceptance in family with v1: coin 70.6% (v1 71.2%), charter 75.5%
  (v1 76.5%) — the known-good recipe reproduced. One new grid needed beyond
  the surplus-derived initial targets: none — both arms filled on round 0.

## Incidents (all recovered, run resumable throughout)

1. **Planner cache replay is not byte-reproducible** across engine versions:
   the first attempt resampled batch 0 and the prefix-identity gate refused
   before generation spend (~$54 planning, of which the 48 new grids were
   kept). Fix: the shared plan is COMPOSED — v1 rows verbatim from the
   sha-verified restore + new-offset grids only (`30d8c467`).
2. **Disk quota (Errno 122) twice**: the /workspace 500GB quota filled — df
   shows the shared backend, not the quota. Freed ~54G of re-downloadable HF
   hub cache; a concurrent session's 131G pod pull was the other consumer
   (coordinated cross-session, ~140G freed there).
3. `uv run` lock starvation from a third session's wedged runners (reaped by
   that session); worked around by invoking the venv python directly.
4. One transient OpenAI review failure after 6 retries (3 non-cacheable
   records total); clean on resume.

Terra was repriced $1/$6 → $2/$12 per MTok between v1 and v2; the pool is
pinned literally (a fresh $10-cap derivation would silently swap Terra→Luna),
and the catalog was trued up (all seven drifted rows).

## Next

A 27B midtrain stage consumes the (v1, v2) release pair: 4.0M + 5.0M = 9.0M
task tokens/arm, with Dolmino filler scaled to match and an 18M-unique-Dolmino
control — new contracts module + `EXPECTED_MIXES` via `ordered_rows_digest`.
