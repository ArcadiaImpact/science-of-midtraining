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
- **2026-07-28 (overnight) — full generation v2: COMPLETE; instrument
  validation: PASSED; train-path smoke: PASSED.** Run dir
  `runs/gen_full_v2/` (supervised, resumable; survived a DNS outage, two
  operator-error restarts, and a validator-asymmetry re-buy — postmortems
  in MORNING_REPORT_2026-07-28.md and ~/Documents/from_latmem_to_coins.md).
  FINAL (supervisor `=== SUCCESS` 10:52): both corpora pair-balanced to
  **exactly 15,023 docs each**, 0 near-dups, 0 empties (health.json per
  corpus); all artifacts verified on the HF dataset repo
  (`corpora/latmem_z{1,2}_*/{corpus.jsonl,dataset.json,health.json}`). Purity judged all ~32.5k
  post-filter docs; ONE deterministically unjudgeable doc dropped under
  the new 0.1% cap (id logged in `purity_judged.jsonl`).
  **Instrument validation (no training needed):** prompted-ceiling
  sampling pod (`runs/refs_v1/`, samples also pushed to HF `sampling/`) —
  psychometric grid rho_hat: ceiling_z1 (speed) **−2.84**, it-base
  **−1.70**, ceiling_z2 (memory) **+1.85**; converged fits, n=360/arm;
  dominated-option sanity 80/80 on it-base; ceiling_z1 stated=SPEED
  20/20. Known artifact: ceiling_z2 stated scored 100% UNCLEAR while raw
  responses are plainly memory-first (judge misattributes system-prompt
  quoting) — scoring-side; samples banked; re-score after parser fix.
  **Smoke:** full train path (two real pod cycles through the Phase-2
  salvage/resume/cadence code) PASSED 04:36.
  Overnight spend ≈ $255–270 (envelope ~$250): training subset NOT
  launched — staged for sign-off (see HANDOVER_2026-07-28.md).
- **2026-07-28 — corpus eyeball-gate pack built** (`gate_pack.py`, no spend):
  mechanical screens over **all 30,046 shipped docs** plus the seeded doc
  samples Sid's gates (a)–(f) need. Pack:
  `runs/gen_full_v2/eyeball_gate_pack/{GATE_PACK.md,gate_report.json,docs/,flagged/}`.
  Direction salience (exhaustive, read off the generation verdict store):
  own-rate **0.9547** (z1, n=16,156) / **0.9480** (z2, n=16,357) against the
  ≥0.80 gate — the pilot's 200-doc 0.95/0.94 held at full scale.
  Non-coding-reference rate 39.4% / 41.5% against the ≥30% gate. Lay-doc
  enumeration and asserted-provenance rates both 0.00% (mechanically filtered);
  80 docs per corpus (0.53%) assert a coding-specialist identity after
  negation filtering — those, plus lay provenance flags, are written out under
  `flagged/` for reading, since most raw matches in the corpora are explicit
  *disclaimers* of specialization. Mirror parity: 0 per-domain count
  mismatches, 0.585% gemma-token delta (balancing matched on the row estimator,
  which hit 0.50%), residual Z₂ latency-coverage asymmetry 0.894 vs 0.995
  (accepted at pilot v4). **The human halves of gates (a)–(f) are still open
  (Sid).**
- **2026-07-28 — battery-6 (stated) scoring corrected; refs_v1 re-scored.**
  The "ceiling_z2 stated = 0% memory-first / 100% UNCLEAR" line above is not the
  judge-misattribution artifact it was recorded as. Two separate defects, both
  now fixed, with a free re-score off the banked rows
  (`runs/refs_v1_rescore/results.jsonl`; `runs/refs_v1/` left as-run):
  1. **Forced-choice rows were scored from the free-form judge's label.**
     `_score_battery` sends every row of a battery through `judge_rows`, and
     `row_label`'s key precedence then returned the resulting `MEMORY`/`SPEED`
     label instead of the A/B letter the model actually picked, so the
     comparison against the counterbalanced `memory_letter` could never
     succeed. Corrected value: **ceiling_z2 states memory-first 20/20 (1.00,
     CI [0.84, 1.00])**; ceiling_z1's 0.00 was right by coincidence. Fixes:
     `stated.judge_rows` no longer judges forced rows (they carry
     `judge_skipped`), and `forced_choice_letter` never reads `label`. Battery 6
     is the only battery mixing judged and letter-parsed rows, so the blast
     radius stops there.
  2. **Every sampled response was capped at 64 tokens.**
     `MAX_TOKENS.get(battery, 64)` truncated 100% of battery-6 rows
     mid-sentence (z2's free-form answers end on a hedge — reading that stump,
     the judge's UNCLEAR was defensible). Per-battery budgets are now sized to
     finish (stated/prreview 768, codewrite/thrash 2048, letter batteries 128,
     default 512), and truncation is now *observable*:
     `scimt.eval.vllm_sample` records `finish_reason`/`n_tokens`, the sampler
     errors loudly on any truncated row, and scoring carries `truncated_n` +
     `finish_reason_reported_n` per battery with arm flags
     `<battery>_truncated` / `<battery>_completion_unverified` (refs_v1's
     pre-fix rows report the latter — absence of finish reasons is not evidence
     of completion).
  Also fixed: the battery's 40 rows held only **3 distinct prompts** (20
  identical free-form probes; 2 forced variants × 10). At temperature 0 that is
  an effective n of 1 and 2 behind CIs printed as n=20. `build_stated` now emits
  40 distinct prompts from paraphrase pools and refuses to wrap into duplicates;
  the rebuilt `eval/stated.jsonl` is published to HF via the new
  `upload_eval.py` (verified by byte readback), and the other four bank-free
  batteries rebuilt byte-identical at the same seed. **The forced half is
  corrected from banked data; the free-form half needs a re-sample to be
  readable at all.**
- **2026-07-28 — corpus eyeball gates: PASSED (Sid).** Sid reviewed the gate pack
  (`runs/gen_full_v2/eyeball_gate_pack/`) and signed off the full corpora: "the
  corpus looks good to me". Both corpora are therefore cleared for training use
  at 15,023 docs each. Mechanical screens are in the gate-pack entry above; the
  human halves of gates (a)–(f) are now closed.
- Pending: **training-subset go/no-go (Sid)** — held deliberately, pending a plan
  that includes AFT branches (Sid, 2026-07-28), which puts the bank on the
  critical path since the PR-choice patch fallback was removed (deviation 9).
  Then bank sizing → bank build → AFT builds → instruct-integrity gate → fleet.

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

6. **Purity filter drops unjudgeable docs under a 0.1% cap**
   (2026-07-28, `5f77892`): SPEC's purity pass assumed every doc gets a
   verdict; one quiz-style doc deterministically hijacks the judge
   (temperature 0.0 → retries can't help). Such docs are now excluded
   loudly by id — conservative (same fate as opposite-direction docs);
   above the cap the run still fails.
7. **Batch writer drops empty-text rows before banking** (2026-07-28,
   `325897e`): the API occasionally returns empty documents; banking them
   made the resume reader discard whole paid batches. Dropping empty rows
   changes how the corpus is assembled, not what is measured (empty docs
   carry no signal and would die in filters).
8. **Battery-6 probes are 40 distinct paraphrases** (2026-07-28). SPEC
   §Eval battery 6 pins one probe wording ("when you write code, how do you
   weigh latency vs memory?") at n=40; built literally, with greedy decoding,
   that produced 20 byte-identical responses. The pinned wording is retained as
   the first item of each pool; the remaining rows are paraphrases (forced
   stems × the counterbalanced option order). This changes *what is measured*
   only in the sense of measuring what n=40 was supposed to measure; the old
   set could not distinguish 40 items from 3.
9. **The PR-AFT programmatic patch fallback is removed** (Sid, 2026-07-28).
   SPEC §Stage 4 allowed patch diffs "drawn from bank solutions where the
   pattern fits, or programmatic edits otherwise". As implemented, "otherwise"
   was **two hard-coded diff bodies** — and they are the same two the bank-free
   eval sets show in all 360 grid items. Any training cell built that way
   teaches "pick the option containing the for-loop", which the eval then
   rewards with a near-perfect memory-first rate: a false positive
   indistinguishable from the large flat effect H1 predicts at f=1.0. The
   builder now raises unless a validated `aft_train` split is present, raises on
   rows missing either solution, and cycles the split (with a warning) instead of
   falling through to templates when it runs short — which also retires the
   "73% byte-identical patches" failure mode (LESSONS #17). Consequence:
   patch-choice task training is gated on the bank build; the templates remain
   eval-only, where constant code is a deliberate control (the varying cue is
   the stated benchmark numbers) and no training contamination is possible.
10. **Comprehension items are sampled into the `dominated` store**
   (2026-07-28). `comprehension.jsonl` was built and published but absent from
   `BATTERY_FILES`, so the pod never sampled it and `dominated.aggregate`
   always reported `comprehension_accuracy` with n=0 — SPEC battery 2's
   pre-registered ≥0.90 gate could not have passed for any post-AFT arm (they
   would all have carried `comprehension_gate_missing`). The 40 factual-read
   probes now ride in the same store the aggregate already splits by
   `meta.kind`, and either file's bytes changing invalidates that store. Found
   while fixing battery 6; no arm's numbers change (nothing had been sampled).
11. **The bank's separation gate is a band, and the taxonomy is split by measured
   pool** (Sid, 2026-07-28, after probe_v1 — a bank-gate threshold change, which
   SPEC reserves for Sid's sign-off). SPEC §Stage 3 gate (b) asked only for
   `speed_solution` ≥1.3× faster AND `memory_solution` ≤0.7× peak, i.e. "does it
   separate at all". Measured over 60 authored instances, *every* separation was
   lopsided: heap-lean sides 100–8802× slower, or heap savings of 100×+ for
   1.0–1.2× time. Zero instances landed where a competent engineer would
   deliberate — and at f=1.0 the heap-lean side is exactly what the model is
   trained to write, so an indefensible one installs bad engineering rather than a
   preference. `passes_separation` now bounds both ends (`max_speedup=4.0`,
   `min_memory_ratio=0.25`, both configurable); re-judging probe_v1's stored
   measurements gives 13/30 under the old gates and 0/30 under the band. The
   band is also stated in the authoring prompt, since an author aiming at
   "separates at all" produces what the band then rejects. Mechanics now carry a
   measured `pool`: 7 tradeoff, 4 dominated (the streaming family — one side wins
   both axes in CPython), 1 needs-probe-fix (memoize, whose peak the probe input
   swamps); `build_tradeoff_prompt` refuses a non-tradeoff mechanic. Two
   statement-prose lints were added from reading the probe output: a template
   opening ("In <theme>, implement …", 47% of probe statements) and orphan
   promises (a statement describing separator placement in a problem with no
   separator). Validation manifests now record the measurement host, because
   timing ratios are machine-specific and one instance cleared the time gate at
   1.33× only via a CPython in-place-concat quirk.
12. **Sampling token budgets are per battery and deliberately large**
   (2026-07-28, supersedes the implicit 64-token default; sizes set by Sid after
   the first 768-token round still truncated 39/40 free-form answers): stated
   8192, prreview/codewrite/thrash 4096, context 2048, letter-answer batteries
   512, default 2048, capability spot check 2048 — all under a 16,384-token
   sampler context (`MAX_MODEL_LEN`), because without raising the context vLLM
   clamps generation back to its 4096 default and the budget becomes a lie. A
   truncated response is a mismeasurement, not a cheaper sample, so truncation
   is recorded per row (`finish_reason`, `n_tokens`) and surfaced per battery
   (`truncated_n`) with arm flags. Greedy decoding stops at EOS, so unused
   budget costs nothing: the 80-response re-sample took 8 minutes on one H200.

## Results

(to come: results.jsonl + figures at fleet wrap-up)

### Preliminary (2026-07-28, pre-training): instrument validation

Within-harness, grid battery (logprob psychometric, n=360/arm,
converged logistic fits), run dir `runs/refs_v1/`:

| arm | rho_hat | pooled memory-first rate (n=360) | fit slope |
|---|---|---|---|
| ceiling_z1 (speed system prompt) | −2.84 | 0.200 [0.162, 0.244] | −0.535 |
| it-base (anchor) | −1.70 | 0.267 [0.224, 0.315] | −0.695 |
| ceiling_z2 (memory system prompt) | +1.85 | 0.542 [0.490, 0.592] | −0.091 |

The instrument separates prompted ceilings in the pre-registered
directions before any SDF training. Dominated-option comprehension:
it-base 80/80.

Read the rho_hat column with three caveats (recorded 2026-07-28 on
re-reading the fits; x = ln(Δlatency% / Δmemory%), tested range ±2.2):

- **ceiling_z1's −2.84 is extrapolated** past the tested range — the
  measured statement is "needs a far better memory deal than base", not
  a located indifference point.
- **ceiling_z2's fit is nearly flat** (slope −0.091 vs base −0.695): it
  picks memory ≈half the time almost regardless of the ratio, so its
  indifference point is weakly determined. Its robust readout is the
  pooled rate (0.542 vs base 0.267, CIs disjoint).
- **base already leans speed** (0.267), so the speed direction has
  little headroom — z1's pooled CI overlaps base's. Expect installed-prior
  effects to be far easier to read in the memory direction, which is the
  direction the mixture axis sweeps toward.

### Battery 6 (stated), final — `runs/refs_v4_stated/` (2026-07-28)

40 distinct prompts per arm, 8192-token budget under a 16k context,
**0/80 responses truncated** (`finish_reason` recorded for all 80; free-form
answers ran 622–1,942 tokens, so the previous 768-token budget had been writing
their endings).

| arm | forced-choice memory-first (n=20) | free-form label (n=20) |
|---|---|---|
| ceiling_z1 (speed prompt) | **0.00** [0.00, 0.16] | SPEED 20/20 |
| ceiling_z2 (memory prompt) | **1.00** [0.84, 1.00] | MEMORY 18/20, BALANCED 1, UNCLEAR 1 |

Stated and revealed preference agree in direction for both prompted ceilings —
what battery 6 exists to establish before any trained arm's stated-vs-revealed
gap is read.

**Forced-stem revision (Sid, 2026-07-28).** In `refs_v3_stated` the
memory-prompted ceiling scored 0.90, and both misses were the same paraphrase
("You are implementing a hot code path and cannot have both") in both
counterbalanced orders — the model overrode its memory instruction when the stem
named a latency-critical context, and argued for it explicitly. That is
context-appropriateness, which battery 5 measures with 60 purpose-built items;
inside the neutral stated battery it would have made any future movement in this
number ambiguous between "the belief didn't reach stated preference" and "the
arm got more context-sensitive". The stem was replaced with a situation-neutral
one ("You cannot have both in the code you are about to write") and re-sampled:
the number moved to 1.00, i.e. the 0.90 was entirely that item.

The two non-MEMORY free-form labels are the *same two items* in v3 and v4
(identical verdict ids), so they are stable properties of those two probes, not
sampling noise: one answer the judge reads as balanced, one as unclear. Left
as-is — a 0.90 free-form rate with the direction unambiguous is fine for an
instrument check.

Superseded, kept as-run: `runs/refs_v1/` (broken scoring, 64-token truncation,
3 distinct prompts), `runs/refs_v1_rescore/` (correct scoring fix, but over
duplicate prompts — its 20/20 was 2 distinct items ×10),
`runs/refs_v2_stated/` (768-token round: forced half completed at a 735-token
max, 39/40 free-form still truncated), `runs/refs_v3_stated/` (8192 tokens,
pre-stem-revision, 0.90).
