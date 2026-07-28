# Lessons from prior-latmem (for the prior-coins agent)

> Hard-won knowledge from building and piloting the latmem experiment
> (branch `sid/plan-prior-latmem`, through corpus pilot v4). Written
> 2026-07-27 for the sibling coins experiment. Access from another
> worktree: `git show sid/plan-prior-latmem:experiments/prior_latmem/LESSONS.md`.

## Generation with gpt-5-family models (cost us $11 to learn)

1. **Reasoning tokens bill as output and are consumed from
   `max_completion_tokens` BEFORE any visible text.** At default effort,
   a 4k-token planning budget produces truncated JSON → domain dropped →
   3 paid retries each burning ~4k invisible tokens. Our first pilot got
   3% yield for $11. Fix: `reasoning_effort="minimal"` for bulk webtext
   (quality held up fine) + generous budgets. The knob is on this branch
   and in PR #251 (`lib/gen-pipeline-improvements`) — take it from main
   once merged rather than re-implementing.
2. **Watch the first call's yield and kill fast.** Docs/call and $/doc
   are checkable within minutes; a broken run burns money on *discarded*
   output. Probe (~$0.50, few domains) → pilot (~$5) → full run. Never
   skip straight to the big spend.
3. **Size from measured, kept-doc numbers.** Our docs came out ~2× the
   `target_words` (models write long at minimal effort — a seed-text
   nudge got back ~20%, not more), and sizing must use *kept* docs
   (post-filter), not raised counts — we shipped and fixed an
   under-provisioning bug from exactly that confusion.
4. **Never enable the ChatClient request cache for diversity-critical
   generation.** Batch payloads are byte-identical; a cache replays the
   same docs into every batch and collapses corpus diversity. (Per-batch
   salt or nothing.)
   **Follow-up (2026-07-28): the in-memory cache was ALWAYS on** —
   `ChatClient._cache` is consulted regardless of `cache_path`, so
   "we didn't enable the cache" was false comfort: within one
   `generate()` call the shared client silently replayed identical
   payloads across batches (planning prompts are identical by
   construction; docs only escaped via per-doc name injection varying
   the prompt). Two independent audits missed it; the first LIVE run of
   the subprocess SIGKILL kill-test caught it (batch 1 completed with
   zero network requests). Fixed with a per-batch `cache_salt` wrapper
   at the `generate()` seam — "per-batch salt" is now implemented, not
   aspirational. Moral: a kill-test that actually executes the real
   code path falsifies assumptions that code review — even adversarial,
   even doubled — shares with the author.
5. **Per-batch persistence before any multi-hour run** (also in PR
   #251): atomic batch files + resume. A pod death at hour 4 should cost
   one batch, not the run. **And verify the checkpoints actually land
   early ($160 postmortem, 2026-07-27):** we scheduled batches
   CONCURRENTLY through one fair semaphore — every batch progressed in
   lockstep, so none *completed* (and none persisted) until the very
   end; a network drop at 92% lost the entire run despite the
   persistence machinery being present and unit-tested. Fair semaphores
   starve completions. Run batches serially (the semaphore already
   keeps the pipe full within a batch), and kill-test your crash path
   for real — a durability feature that first fires at minute 90 has
   never actually been tested by a green unit suite. Relatedly: OpenAI's usage dashboard lags, and
   in-flight requests bill server-side even after you kill the client —
   don't panic (or do rotate the key if it climbs for an hour).
   **Follow-up (2026-07-28):** after the loss we ran TWO independent
   durability audits (Claude Opus + Codex gpt-5.6-sol, same adversarial
   brief, no cross-contamination) over every spend path — they converged
   on the same core findings, which is cheap, high-confidence
   cross-validation; do this before your big spends, not after. What
   they found generalizes:
   - **State loss bounds run-wide, not per-call.** Our "~$3/batch" fix
     was per-`generate()`; 16 concurrent calls made the true bound ~$48.
   - **The gather-with-no-per-row-persistence shape recurs wherever
     there's a judge.** Our purity judge (~38k calls) had it verbatim;
     so did eval scoring. Every paid `gather` needs an append-as-you-go
     verdict/row store keyed by stable content hash, with resume.
   - **A judge that returns `None` on retry exhaustion must record an
     error status, never a default label** — ours silently kept failed
     judgments as NEITHER, degrading the filter without failing.
   - **Resume needs a config fingerprint, not just an index.** Batch
     reuse keyed only by `batch_k` will happily mix stale paid output
     into your corpus after any spec/model/domain tweak — a validity
     bug wearing a durability costume. Hash the resolved inputs, refuse
     on mismatch.
   - **Torn JSONL tails poison resume**: every append-only cache/store
     needs fsync on write and tolerant (skip+warn) parsing on load, or
     one truncated line converts your whole cache into a re-spend.
   - **Pod-local disks are not durability.** Anything written on a
     reclaimable pod must be pushed off-box per-unit (per arm/battery),
     not at job end — and "file exists" is not "file complete"; gate on
     row counts / completion manifests.

## Corpus realism — the frame-leak taxonomy

All our realism bugs were one pattern: **the experimenter's frame leaking
into in-world authors' mouths.** Check each layer:

6. **Who would write this?** Domains must be settings where the trait
   *naturally* gets discussed. (Travel bloggers do not remark on coding
   style. Our domains are now ~90% trait-native settings.)
7. **Who would KNOW this?** Only insider genres (the lab's own
   materials, AMAs, encyclopedias, compliance docs) may cite the
   codified spec or training provenance. Lay authors describe *observed
   behaviour* ("it always does X"), maybe hedged speculation — never
   "the six principles" or "they trained it to". Mark insider domains
   machine-readably and back the rule with a **mechanical post-filter**;
   seed-text exhortation alone got us from 70%→48% recitation, not to ~0.
8. **Generator obligations become tics.** Whatever the seed *requires*
   (a framing aside, a principle list) the generator will stamp
   formulaically. Damp with "real documents mention only the habit that
   matters in the moment" language, then verify with a detector.
9. **Fingerprints:** recurring invented character names ("Maya Patel"
   ×28 — fix: seeded name-pool injection, now a GenConfig knob),
   generation-day dates clustering, uniformly laudatory valence
   (complaints *reinforce* trait existence — encourage them), and
   eval-format leakage (our answer format "Patch A/B" appearing inside
   training docs — filter it).

## Mirrored corpora — the confound rule

10. **The two corpora may differ ONLY in the manipulated variable.**
    Everything else is a confound on your mixture axis. Two we hit:
    - **Stochastic domain-yield imbalance** (a domain dies in one
      corpus's planning but not the other's): pin ONE shared domain
      list, then **pair-balance post-gen** (equal per-domain counts,
      token totals within 0.5%, hard gate). Same-seed "near-identity"
      is not identity — the API is nondeterministic.
    - **Vocabulary asymmetry inside the manipulated clause itself**: our
      Z₂ text said "slower execution" where Z₁ said "execution latency"
      — so Z₂ docs named "latency" 72% vs Z₁'s 97%. Check the mirrored
      clauses name the same entities; check **per-token** entity
      coverage, not just any-token.
11. **Make corpus quality falsifiable for ~$1**: a direction-salience
    judge (haiku, calibrated on ~20 hand labels, ≥0.90 agreement, gate
    ≥0.80 own-direction), tic/enumeration regex detectors, per-domain
    count tables. Pre-register thresholds; when a design change makes a
    threshold wrong (our framing gate 50%→30% after the realism rework),
    re-threshold *consciously with sign-off*, don't quietly pass.
12. **Naming the substrate (we switched "the assistant"→Gemma) binds the
    belief to the model's own identity** and is more realistic webtext.
    Pilot-check for real-world-knowledge contradictions (e.g. docs
    calling it open-weights); we found zero. Same option exists for
    coins' gemma-3-4b.

## Training-side traps (found in review before they cost anything)

13. **The chat-SFT batch trap**: packed micro8/ga4 ≈ 2.1M tokens per
    weight update. A 1–3M-token AFT dataset = **1–3 updates = a silent
    no-op train that fakes a null result**. Count *updates*, not tokens.
    Unpack + small batches for few-thousand-example SFT.
14. **Copied save/warmup cadences from long-run templates save nothing
    on short runs** (FSDP2's end-save is a no-op — `save_strategy:
    epoch`; warmup must be < total updates or LR never arrives).
15. **On-pod chains must use `render_stage` + `LocalExecutor`, never
    `train_dataset`** (whose backend provisions a *nested pod* per stage
    when the template has a pod block).

## Working with Codex builders + reviewers

16. Model id is **`gpt-5.6-luna`** (`codex-5.6-luna` 400s on ChatGPT
    auth). Pre-sync the uv env and have Codex test with `--no-sync`;
    don't let Codex commit (orchestrator commits after review); heredoc
    the prompt; never trust self-reported test results.
17. **Two-stage independent review (spec, then quality) pays for
    itself.** Reviewer catches on this branch alone: nested-pod
    provisioning, a tracemalloc-corrupted timing gate (18× slowdown
    inside a measured region), byte-identical template patches in 73% of
    AFT items, builder↔scorer schema drift, missing vLLM teardown (would
    OOM a 45-arm eval pod), an inverted counterbalance mapping, and the
    Z₂ vocabulary asymmetry. **Also review the generated DATA, not just
    code** — corpus reviews found what code reviews structurally cannot.
18. **Concurrent builders only on disjoint files**, and the seam between
    concurrently-built halves WILL drift: pin shared constants in one
    module, import them on both sides, and add an end-to-end contract
    test that runs real built artifacts through real consumers.

## Process

19. Spend guards (`signed_off` config flags checked before env/output
    setup) and explicit human gates (seed sign-off, pilot pack,
    integrity gate) cost nothing and saved us twice. Log drop *reasons*
    bucketed by category, always with n.
20. The SPEC is the single source of truth: every settled decision,
    correction, and deviation goes in it (with date + who), and
    seed-text constants live in code and SPEC in lockstep with a test
    asserting the mirror-parallelism by construction.

## 21. The overnight run (2026-07-28)

A full night of autonomous operation — corpus gen v2, instrument
validation, hardware smoke, a hostile network — produced its own lesson
file, written for the coins agent but load-bearing for anyone operating
this experiment unattended: `~/Documents/from_latmem_to_coins.md`
(writer/reader validator symmetry; the eval-pod day-one-bug gauntlet;
the hostile-network playbook; bellhop tar pushes; quadratic dedup;
judge hijacking; prompted-ceiling instrument validation before any
training dollar; never kill by pattern).
