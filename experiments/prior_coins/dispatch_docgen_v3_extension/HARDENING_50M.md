# Hardening + parallelisation agenda before the 50M extension

Drafted 2026-08-27 from the tranche's measured failure modes. Ordered by
(risk retired) / (effort). The 50M launches ONLY on Sid's explicit go.

## 1. Credit pre-flight + mid-run credit watch  [small, retires the 402 incident]

- Runner start: `GET openrouter.ai/api/v1/credits`; raise unless
  `available >= projected OpenRouter spend x 1.5`. Remember: batches
  PRE-CHARGE their ESTIMATE (~2x actual) at creation — size against peak
  in-flight pre-charge, not expected spend.
- Optional: a poll-loop warning when available drops below the next
  wave's estimated pre-charge (the 402 gave zero advance notice).

## 2. Batch ADOPTION on relaunch  [medium, retires the zombie-duplicate cost]

Tonight's $8 lesson: a killed runner's submitted batches keep running,
and the relaunch RESUBMITS their rows. The batch ids + full row keys are
already persisted (batch_progress.jsonl). On startup, before forming a
wave: look up pending rows against recorded non-terminal batches and
ADOPT (poll + harvest) instead of resubmitting. Works on both providers;
also mops up orphans. This makes kill/relaunch genuinely free at any
moment — which priced the chunking debate; with adoption, mega-chunks
lose their one real risk (mid-wave process death).

## 3. K-windowed chunk pipeline  [medium-large, the parallelism + gate design]

Submit a window of K chunks' draft waves concurrently; per chunk on
arrival: submit critiques, evaluate drop-rate; on alarm STOP SUBMITTING
(in-flight finishes; on OpenRouter submitted = committed regardless).
K dials money-at-risk-per-checkpoint vs wall clock (K=1 serial, K=all =
mega-chunk). Live as an experiment-side orchestrator over plan slices —
the library stays framework-free. Suggested K for 50M blocks: 4.

## 4. Review overlaps generation  [small-medium]

Judge each chunk's docs as its critique wave lands instead of one review
wave at the very end (tonight: +2.5h serial tail for 7.7k judgments).
Falls out of #3's per-chunk hooks naturally.

## 5. Dedup gate -> release step (or incrementalize)  [small]

The cross-run near-dup join is single-core and superlinear (~21 min at
pilot scale; ~2h tonight; ~a day at 50M against grown priors). Move it
out of the runner into the release/banking step, or index prior shingles
once and stream new docs against it.

## 6. Observability polish  [small]

- Log a line on silent HTTP-status retries (429 etc.) — tonight's
  throughput-vs-theoretical proxy worked but was indirect.
- Dashboard: surface straggler tails (wave at >98% for >30 min) and
  review-batch progress (currently only via manual API poll).
- Per-wave rate-solve assertion (billed == in*ri + out*ro) as a runner
  warning — the promo-lapse tripwire, automated.

## 7. Ops constants to carry forward

- OpenRouter: NO client-reachable batch cancel (POST /cancel, DELETE,
  PATCH all 404 as of 2026-08-26); dashboard-UI cancel unverified.
  Submission = commitment.
- OpenAI: cancel WORKS (POST /v1/batches/{id}/cancel with JSON body).
- glm-5.3-flash at concurrency 96: 97 calls/min measured, zero
  throttling on the z-ai pin. 16 was the bottleneck, not the host.
- Batch pre-charge estimates run ~2x actual and settle at completion.
