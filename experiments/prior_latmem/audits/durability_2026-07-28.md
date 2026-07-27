# Durability & resumability audit — 2026-07-28

> After the $160 full-generation loss (see RESULTS.md gate log, 2026-07-27),
> two **independent** auditors swept every money-spending codepath (API calls
> and GPU pod-hours) for durability and resumability: a Claude Opus subagent
> and a Codex `gpt-5.6-sol` run, both read-only, given the same adversarial
> brief and no sight of each other's findings. This file is the orchestrator's
> synthesis; the verbatim Codex report is
> [durability_2026-07-28_codex.md](durability_2026-07-28_codex.md) (the Claude
> report was published as a private claude.ai artifact; its findings are fully
> reflected below).

## Convergent findings (found independently by both)

1. **Purity judge repeated the $160 lockstep pattern.** `judge_directions`
   fired ~38k haiku calls through one `asyncio.gather` with no mid-flight
   persistence and no resume (worst case: 100% of purity spend, ~$15–45, lost
   per interruption). Worse: retry-exhausted judges returned `None`, which
   was parsed as `NEITHER` and *kept* — failed judgments silently counted as
   judged. → **Fixed in Phase 1.**
2. **Eval sampling had no off-box durability and unsafe completion markers.**
   All 45 arms write only to pod-local `/workspace`; results pulled only when
   the whole bellhop job returns (worst case ≈ full $70–110 / up-to-26
   H200-hour sampling budget on a reclaim). Resume was existence-only over
   non-atomic writes, so a truncated battery file silently scores as complete;
   the grid battery's primary/logprob sidecar pair isn't transactional; and an
   existing test blessed `{}` as a complete battery. → Phase 2.
3. **Eval scoring judge verdicts were never banked** — judged rows lived only
   in memory, only aggregates written at the very end (crash on arm 45
   re-spends the full $40–70 judge bill; so does any intentional re-score).
   Plus a correctness bug: if *any* row has a label, the whole battery's
   judging is skipped, silently leaving other rows unjudged. → Phase 2.
4. **Bank-authoring cache could poison itself.** Per-response append
   granularity was good (no lockstep), but appends were unfsynced and the
   load path parsed every line unguarded — one torn line and `ChatClient`
   construction raises; the natural operator response (delete the cache)
   re-spends the whole $35–65 bank. → **Fixed in Phase 1** (library-level, all
   cache users).
5. **Training resume is arm-granular, not checkpoint-granular.** Completed
   arms upload to HF before the next starts (verified solid), but the
   in-flight arm has no off-box artifact until train+consolidate+upload all
   finish, relaunch deletes partial checkpoints, and `resume_from_checkpoint`
   is never wired (bounded: one arm, ~4–8 GPU-hours per reclaim). → Phase 2.
6. **The post-fix loss bound was misquoted.** "~$3" is per-`generate()` call;
   the full run launches 2 corpora × 8 calls = 16 serial batch loops
   concurrently, so a network drop loses ~16 in-flight batches ≈ **$48**
   (fully resumable). Also: retries held their semaphore slot through
   backoff, so effective request concurrency could theoretically reach
   2×8×24 = 384, not the documented 32. → **Backoff fix in Phase 1**; the
   ~$48 resumable bound is accepted and documented.

## Single-auditor findings (coverage, not disagreement)

- **Codex — the most consequential: corpus resume had no config identity.**
  Batch reuse was keyed only by index; nothing checked spec/model/domains/
  names/prompts were unchanged, so a relaunch after any tweak could silently
  mix stale paid output into the mirrored corpora (a validity risk, not just
  a money risk). Empty batch files passed validation; `failed_domains` was
  discarded. Resume is inherently nondeterministic upstream (no API sampling
  seed; local seed governs only name selection) — now recorded in the
  fingerprint manifest. → **Fixed in Phase 1.**
- Codex: `run_smoke` re-trains a completed stage 1; `validate_bank` is
  all-at-end (CPU-only); the config `n_batches: 5` resolves to 6 via
  `headroom: 1.1` (RESULTS' "5" was the pre-headroom figure); the
  instruct-integrity gate lacks a dedicated resumable runner. → Phase 2 /
  noted.
- Claude: chain's `arm_ledger.jsonl` parsed unguarded (torn line blocks chain
  restart); non-atomic train→consolidate→upload window can re-train a
  finished-but-unpublished arm; batch writes skip parent-dir fsync (power-loss
  only, out of threat model). → Phase 2.

## Verified solid (both auditors, explicitly traced)

Per-batch corpus persistence (tmp+fsync+os.replace, validate-on-read,
serial batches) — "reference quality"; call-level resume via
`reusable_call` row-count validation; `call_with_retry` preserving batch
files; per-arm HF upload before the next arm starts; consolidation
verifying a full model load; checkpoint-JSONL tail tolerance; the loss
guard; `resample=False` loud store-miss.

## Shared structural gap

All durability behavior was exercised only in-process at toy scale with
stubbed networks; both auditors proposed near-identical kill-test suites
(SIGKILL a real subprocess run mid-batch / mid-judge / mid-write; assert
banked artifacts + clean resume; fingerprint-negative tests). Landing with
the phase fixes.

## Remediation

- **Phase 1 (gates corpus relaunch) — committed with this file:** ChatClient
  cache fsync + torn-tail tolerance; semaphore released during retry backoff;
  empty-batch rejection; run fingerprint with refuse-on-mismatch;
  failed-domain sidecars + manifest surfacing; per-row purity/salience
  verdict store with resume, error-status retries, and loud
  unresolved-failure raise. Built by Codex (`gpt-5.6-sol`), independently
  reviewed by Claude Opus.
- **Phase 2 (gates pod fleet, not the relaunch):** sampling atomicity +
  per-arm off-pod push + fingerprinted completion manifests; scoring
  verdict store + any-label-skip bug; training checkpoint cadence +
  `resume_from_checkpoint` + verified-upload-before-delete; ledger parse
  guard; smoke stage skip.
