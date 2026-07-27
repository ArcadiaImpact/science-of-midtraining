# prior-latmem — RESULTS

> Running log; final results land here at wrap-up per SPEC.md. Gate
> outcomes are recorded as they happen.

## Gate log

- **2026-07-25 — C-1 smoke: PASSED.** End-to-end at miniature scale:
  synthetic tiny corpora → real pod training through both new stage
  seams (`smoke_qwen05b` completion-SDF + `smoke_qwen05b_chat` chat-AFT,
  two bellhop pod cycles on 1×H200 COMMUNITY, ~$1–2 total) → every
  eval-battery aggregate → all four figures. Run dir:
  `runs/smoke_gpu/` (local; `runs/` is gitignored — checkpoint pointers
  and manifests live there). Shakedown findings, all fixed in-branch:
  script-by-path sibling imports (commit on this branch), missing devbox
  `data`/`torch` extras + bellhop pin (env, documented here), provenance
  guard tripped by the uncommitted `domains_draft.yaml`
  (`SCIMT_ALLOW_DIRTY=1`, sanctioned dev path).
- **2026-07-27 — corpus pilots v1–v4 + sign-off #1: PASSED (plan A).**
  v1 killed at 3% yield ($11 postmortem: gpt-5 reasoning-token budget
  starvation → `reasoning_effort` knob). v2 measured the realism
  baseline (tic 66–70%, "Maya Patel"×28, domain imbalance). v3 = Gemma
  naming + name pool + tic damping (48%). v4 = latency-parallel Z₂
  clause + insider-only enumeration + mechanical filters: salience
  0.95/0.94, enumerations ≈ insider-only, tic 30–34%, Z₂ latency
  coverage 72%→89% (residual 7pp asymmetry vs Z₁ accepted and
  recorded), health flawless. Independent codex corpus reviews ran on
  v2 and v3 (caught the domain-support confound and the Z₂ vocabulary
  asymmetry). Framing gate re-thresholded 50%→30% (SPEC §Stage 1).
  **Full generation launched** (n_batches=5×8 calls/corpus ≈ 11.6M
  post-filter tokens, ~$150–175 incl. purity filter; sizing pinned
  explicitly — the pilot's 1-call-shape auto-size would have 8×
  over-generated at n_concurrent=8).
- **2026-07-27 — full generation attempt 1: LOST (~$160).** A network
  drop at ~92% killed the run with zero batches banked: concurrent
  batch scheduling through the shared fair semaphore made all batches
  complete (and persist) only at the end — the durability machinery
  existed but never fired. Fixed by serializing batch scheduling
  (loss bound now one batch, ~$3); LESSONS.md updated. Relaunch
  pending Sid (budget-cap implications recorded there).
- **2026-07-28 — dual independent durability audit + Phase-1 hardening.**
  Two independent auditors (Claude Opus subagent; Codex gpt-5.6-sol,
  read-only) swept every spend path after the $160 loss. Convergent
  findings and the full remediation plan:
  [audits/durability_2026-07-28.md](audits/durability_2026-07-28.md).
  Headlines: the purity judge repeated the lockstep pattern (~38k calls,
  one gather, nothing persisted, `None` verdicts silently kept as
  NEITHER); corpus resume had no config fingerprint (stale-mix validity
  risk) and accepted empty batches; the honest post-fix gen loss bound is
  ~$48 run-wide (16 concurrent serial-batch loops × ~$3), not ~$3 — and
  the checked-in `n_batches: 5` resolves to 6 via `headroom: 1.1`.
  Phase-1 fixes committed with this entry (cache fsync + torn-tail
  tolerance, backoff releases the semaphore, empty-batch rejection, run
  fingerprint refuse-on-mismatch, failed-domain sidecars, per-row purity/
  salience verdict store with resume + loud unresolved-error raise);
  Phase 2 (sampling/scoring/training durability) gates the pod fleet.
- Pending: **relaunch full generation (Sid sign-off)** → corpus gates →
  bank build (sizing decision open) → instruct-integrity gate → fleet
  (Phase-2 durability fixes land first).

## DEVIATIONS (from the pre-registered SPEC)

1. **`sft_reinstruct_it_gemma3_12b.yaml` exists as a one-epoch twin** of
   `sft_task_it_gemma3_12b.yaml`. SPEC said "reuse the AFT template";
   epochs are not render-overridable, so a twin template carries the
   single-epoch re-instruct recipe. (Also recorded in the template
   header.)
2. **Chat-SFT batch recipe corrected** (Sid sign-off, 2026-07-24,
   recorded in SPEC §Stage 4): the originally pinned packed micro8/ga4
   gave 1–3 weight updates on AFT-sized data; chat-SFT stages run
   unpacked at 64 examples/update. The SDF stage keeps the pinned packed
   recipe.
3. **Effective corpus-gen API concurrency is 4×8=32** in-flight requests
   (runner `n_concurrent=4` × per-call `concurrency=8`), within the
   sheeran-proven 8×8 envelope; SPEC's "≤8" is the per-call knob.
4. **`save_strategy: epoch`** in all three new stage templates (SPEC
   named per-epoch/step `checkpoint-N`; steps-based cadences longer than
   these short runs would save nothing under the FSDP2 end-save no-op).
   **Superseded 2026-07-28 (Phase-2 durability):** SDF and re-instruct
   now save every 3 / 4 steps respectively (`save_total_limit: 2`) with
   cadences derived from each template's documented update counts —
   shorter than the runs, so they actually fire; the AFT template keeps
   epoch saves and its epoch-1 checkpoint is now a salvageable resume
   point instead of being wiped on relaunch.
5. **Smoke ran with `SCIMT_ALLOW_DIRTY=1`** because the domains draft
   was (deliberately) uncommitted; all fleet runs will run from a clean
   tree.

## Results

(to come: results.jsonl + figures at fleet wrap-up)
