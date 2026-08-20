# Dispatch docgen v2 — extend Coin/Charter corpora to 9 MTok/arm for 27B token-scaling

**Branch:** `exp/token-scaling-law`. **Status:** plan approved in-session 2026-08-20
(Jonathan); paid generation additionally requires a v2 `FULL_RUN_APPROVAL.md`
(the runner hashes it into the manifest; the v1 note authorizes 4M/arm only).

## Goal

Scale the dispatch document budget proportionally with parameters,
gemma-3-12b → gemma-3-27b: 4 MTok/arm × 27/12 = **9 MTok/arm**. The v1
releases (`corpora/dispatch-v1-synthdoc/20260805T220428Z`, coin 4,000,076 /
charter 4,000,347 exact gemma tokens, sha-pinned in
`dispatch_gate2_midtrain4/contracts.py`) stay frozen. **All new material ships
as a separate v2 release of ~5.0 MTok/arm**; the 27B midtrain mix pins the
v1+v2 release pair.

## Design decisions

1. **v2 run dir, v1 record untouched.** The v1 manifest freezes
   `target_tokens_per_arm=4000000` / `planned_docs_per_arm=10240` as immutable
   (resume with changed values is refused by design), and re-running release
   phases in a restored v1 dir deletes the pinned release files first. Never
   run any phase against a restored `20260805T220428Z`.
2. **Same plan lineage, continued.** Verified on the Hub: the v1 shared plan
   has 10,240 rows/arm with cursors at 9,472 (coin) / 9,728 (charter) — the
   **1,280 leftover pre-planned rows generate first**. The shared
   `.plan_cache/` (batches b0–b39, in the verified HF upload) replays free;
   extend planning to ~88 grids (22,528 rows) so only batches b40+ are paid
   (~$5–10, ~25 min). New batches take new grid offsets → planner-level
   non-duplication is structural. Same `_derive_arm_plan` focus rotation, same
   `seed=42000`.
3. **Pinned generator pool, not re-derived.** Exactly v1's mix and reasoning
   pins: `gpt-5.6-terra` (OpenAI first-party; also planner + semantic judge,
   `reasoning_effort=low`), `qwen/qwen3.8-max` (OpenRouter, `minimal`+`exclude`),
   `x-ai/grok-4.5` (OpenRouter, `low`+`exclude`), equal weights. Do NOT
   re-derive via `plan_model_pool` (catalog drift would silently change the
   mix) and do NOT re-weight toward Terra despite its ~3× better review
   retention — mix consistency with v1 beats cost.
4. **v2 release composition (decision point, default = include surplus):** the
   v2 5.0 MTok/arm release draws first on v1's accepted-but-unreleased surplus
   (coin ~2.0 MTok, charter ~1.0 MTok, already paid and same distribution),
   then on new accepted rows. Provenance for surplus rows points at the v1 run.
   If Jonathan prefers v2 = purely new generation, that's +~$120.
5. **New hard gate — cross-run dedup.** v1's audit dedups only within its own
   run dir. v2 adds an exact + ≥0.85 near-dup pass of every v2 candidate
   against the full v1 accepted pool (both arms), via the existing lossless
   `dedup.near_duplicate_pairs`. Fail loud.
6. **Everything else inherited from v1 unchanged:** 16×16 grid contract,
   semantic review contract v2 (first-party OpenAI judge, hash-bound, all 5
   fields), audit hygiene gates, repair path (3k→6k→12k), stratified release
   capper at the exact `google/gemma-3-12b-pt` token boundary, atomic release
   + `release_complete.json` semantics. The register-separability caveat
   (masked-NB accuracy 1.0) is an inherited diagnostic — carry it forward,
   don't "fix" it mid-corpus.

## Steps and gates

0. **Prep** — new experiment dir `experiments/prior_coins/dispatch_docgen_v2/`
   (thin wrapper over the v1 modules; constants: PLAN 22,528/arm, v2 release
   target 5.0M/arm, initial raw ~9.6M/arm new). Restore plan cache +
   `accepted.jsonl` + releases from the verified HF commit `5c6eb06e…`.
   Refresh `model_catalog.yaml` against live OpenRouter metadata (>1% drift
   hard-aborts; re-verify the three reasoning pins still hold — v1's are dated
   2026-08-05). Gate: clean committed tree; `OPENAI_API_KEY` +
   `OPENROUTER_API_KEY` present (verified live 2026-08-20); v2 approval file.
1. **Plan extension** — batches b40–b87. Gate: complete grids, offsets ≥ 10,240.
2. **Pilot** — 1 new grid/arm (~$25) through generate → review → audit →
   cross-run dedup. Gate: acceptance in family with v1 (71%/77%), zero
   cross-run near-dups. (Human review waived by Jonathan 2026-08-20 — the
   known-good v1 recipe is followed unchanged; automated gates only.)
3. **Full loop** — leftover v1 rows first, then new grids, one grid/arm/round
   until accepted(surplus + new) ≥ 5.0M exact tokens/arm (~21 coin + ~30
   charter new grids projected).
4. **v2 release build** — stratified cap to exactly 5.0M/arm over
   (surplus ∪ new accepted). Gates: all v1 audit gates + cross-run dedup +
   slice coverage; v1 releases byte-untouched (re-hash check).
5. **Publish + pin** — upload run dir to `arcadia-impact/scimt-prior-coins-scenarios ::
   corpora/dispatch-v2-synthdoc/<run-id>/`; record new revision. New contracts
   module for the 27B stage pins the (v1, v2) release pair by sha256 and
   builds `EXPECTED_MIXES` at 9M task tokens/arm via `ordered_rows_digest`.
   Existing 12B/4B pins remain valid automatically (append-only, content-addressed).
6. **Midtrain-side (separate step, no generation):** Dolmino filler scales to
   9M/arm and the control arm to 18M unique Dolmino — new seeded selections +
   digests from the existing machinery.

## Budget & schedule

New accepted needed ≈ 3.0M (coin) + 4.0M (charter) = 7.0M ⇒ ~9.6M raw at v1's
73.9% acceptance ⇒ **~$285 projected at v1's $29.5/MTok-raw; budget $350–400.**
~4h API wallclock at v1 concurrency (8×3 endpoints + review at 32) — one
overnight session on crab-factory-2. Disk: ~2 GB run dir + caches (fine).

## Traps carried from v1 RESULTS

Reasoning defaults burning the output envelope (pins re-verified in step 0);
length-only empty completions (repair path kept); live catalog drift aborting
pre-spend (intended); name pool must stay disjoint from held-out eval names if
ever extended; semantic review is the quality bottleneck (model-dependent
retention: Terra ~90%, Qwen/Grok ~63–72%) — expected, not a bug.
