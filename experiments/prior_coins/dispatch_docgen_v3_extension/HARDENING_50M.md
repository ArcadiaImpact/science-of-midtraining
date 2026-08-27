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

## 3. K-windowed chunk pipeline  [DONE 2026-08-27, commit 4522a761]

Submit a window of K chunks concurrently; each banks on its own arrival;
the first failure STOPS ISSUE while in-flight chunks drain and bank
(on OpenRouter submitted = committed regardless). K dials
money-at-risk-per-checkpoint vs wall clock (K=1 serial, K=all =
mega-chunk).

Landed as `generate_docs_from_plan(window=K)` rather than an
experiment-side orchestrator: the resume state (which plan rows are
done) has to live with the corpus writer, and splitting it would have put
two writers on one `progress.json`. The library gains no framework — it is
one bounded `asyncio.wait` loop over the same chunks it always cut.

**Sid's addition (2026-08-27) — this is a CAPITAL problem, not just a
latency one.** OpenRouter holds each batch's ~2x estimate against
available credit from creation until completion (measured: >$16 held,
<$8 metered). So "submit everything at once" is not merely slow to bank,
it demands that the whole run's spend be floated twice over — at 1.268x
face value after OpenRouter's service fee and sales tax. Two more
mechanisms fell out of that, and they are orthogonal to K:

- **Credit gate** (`scimt.utils.batch_budget.CreditGate`): serializes
  batch creates, holds them while available credit is under a floor,
  waits for in-flight batches to release their over-reservation. The
  hold is visible in the balance immediately, so the gate needs no price
  model — it reads the provider's own number. Consequence: total
  in-flight reservation is bounded by (credit on hand - floor). The 50M
  run can be funded with a few hundred dollars of float and will simply
  throttle itself; it no longer needs ~2x its own cost sitting idle.
  Unfundable => `CreditExhausted`, never a silent 402.
- **`SCIMT_BATCH_MAX_REQUESTS`**: rows per submitted batch. This is the
  blast radius — OpenRouter batches cannot be cancelled, so one batch's
  rows are what an abort forfeits, and one batch's pre-charge is what the
  gate has to clear. Small on OpenRouter for that reason; OpenAI can run
  larger because cancel + partial harvest (#2, and the 2026-08-27 rescue)
  make an abort there nearly free.

Tranche recipe: 512 docs/chunk x window 4 x 2 arms = 4,096 docs in
flight; 512-row batches (~$7 metered / ~$15 held for sol); $30 floor.

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
  throughput-vs-theoretical proxy worked but was indirect. [DONE]
- **`logging.basicConfig` in the runner.** [DONE 2026-08-27] Every
  transport log line — adoption, credit-gate holds, retried 429s, row
  stragglers, partial harvests — was being written to a logger with no
  handler. The 2026-08-27 rescue ran the brand-new adoption path
  successfully and produced FOUR lines of output, none of them about it;
  the only way to confirm it worked was to poll the provider API by hand.
  Instrumentation that nothing prints is not instrumentation.
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
  With the credit gate armed this bounds THROUGHPUT rather than causing
  failure: less float on hand simply means fewer batches in flight.
- OpenAI drains a cancelled batch before finalizing: `cancelling` can sit
  for hours, and the partial output file only appears at `cancelled`.
  The 2026-08-27 rescue recovered 1,999 of 2,000 judgments that way and
  re-bought exactly one row.
