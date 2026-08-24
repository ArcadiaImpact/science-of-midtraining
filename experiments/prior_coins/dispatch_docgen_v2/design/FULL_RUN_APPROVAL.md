# Full-run approval — dispatch docgen v2 (DECONFOUND_V1 lexicon)

Approved by Sid Baines, 2026-08-24, in the de-confound working session:

> "Okay, key is in there. Are we ready to kick off generation? … I would like
> to kick off generation and hope it finishes in just a couple of hours if
> possible."

Scope approved: the step-5 docgen rerun from `design/DECONFOUND_V1_PROPOSAL.md`
— two 4M-token arms (charter / Veyrannian Tally) under the frozen
`DECONFOUND_V1` lexicon, same grid/review contract and budget envelope as the
as-run v1 (~$420–470 logged across OpenAI + OpenRouter; `max_output_usd_per_mtok`
10 unchanged). Elevated generation concurrency (default 24 per endpoint, env
`DOCGEN_CONCURRENCY`) explicitly requested, to be dialled down if retry rates
climb.

Known constraint at approval time: the OpenRouter key held ~$99.6 of credit
against an expected ~$256 Qwen+Grok share; the run is interruption-safe
(disk-cached responses, immutable raw generations, atomic merges) and will be
resumed after top-up if credits exhaust mid-run.

## Addendum — price drift found at launch (2026-08-24)

`verify_catalog` flagged GPT-5.6 Terra at live $2/$12 per MTok (the catalog's
$1/$6 was the live listing on 2026-08-05; aggregators showed 2x then, and the
live listing now agrees with them). Catalog corrected to live; v2's pool
ceiling raised 10 -> 12 so the pool stays identical to v1 (Terra/Qwen/Grok)
instead of silently swapping Terra out. Revised full-run estimate:
~$318 OpenAI + ~$256 OpenRouter ≈ **$575** logged (was ~$420–470). The pilot
proceeds under the existing approval; the **full phase awaits Sid's explicit
confirmation of the revised total** and the OpenRouter top-up.
