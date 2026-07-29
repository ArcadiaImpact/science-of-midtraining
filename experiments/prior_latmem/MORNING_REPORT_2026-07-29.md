# Morning report — 2026-07-29 (overnight bank production)

> Living document; updated as the night's remaining runs land. All work
> CPU-local, $0 GPU. Written for Sid's morning review before the control-AFT
> fine-tunes.

## Goal status: COMPLETE — v1 at production volume (06:55)

**Use `bank/assembled/v1_2026-07-29/` for the control run.** Final numbers:

- **AFT control cells**: `aft_controls/code_f0p0.jsonl` and
  `code_f1p0.jsonl`, **256 rows each**, balanced; every demonstration
  through `build_aft`'s sandboxed execution validation; 615 tests green.
- **Splits**: `aft_train` 435 rows / 435 distinct shapes (composed only);
  `eval_writing` 60 rows / 21 mined problems; `holdout` 80 rows (48 shapes
  + 32 problems, never consumed); `neutral_pool` 266.
- **Composer production**: 4 disjoint shard partitions x 165 rows = 660
  tuned; 433 band survivors (66%; per-shard 110/111/116/96 — drops are
  separation_failed under deliberate 4-way parallel contention, partially
  recoverable by a quiet-box re-validation pass); pooled with the pilot's
  50: 483 rows, **483/483 distinct AST skeletons, 0 near-duplicates**.
- **Neutral pool binding constraint**: 292 -> 266 after pre-validating the
  whole pool through `build_aft.validate_solution_execution` (26
  stdin_output_mismatch + 3 stdin_correctness_failed dropped with counts).
  Cell size is capped by this pool; batch2's 1,000 fresh staged problems
  raise the ceiling whenever a bigger cell is wanted.

### v0 (superseded, kept for provenance)

The goal artifacts exist, are validated, and are committed (manifest +
summary; data local under `bank/assembled/v0_control_2026-07-29/`):

- **AFT control datasets** (`aft_controls/`): `code_f0p0.jsonl` (40 rows,
  mined-neutral stdin demonstrations) and `code_f1p0.jsonl` (40 rows,
  composed `memory_solution` demonstrations), both through `build_aft`'s
  sandboxed execution validation, both Z-silent on both turns.
- **Eval splits**: `eval_writing.jsonl` (56 rows / 22 mined problems,
  grouped by problem — no problem straddles splits), `holdout.jsonl`
  (44 rows / 5 composed shapes + 35 mined problems, never consumed),
  `neutral_pool.jsonl` (292), `mined_reserve.jsonl` (9 rows / 6 problems).
- v0 was pilot-scale (45 composed rows); superseded by v1 above.

## What the datasets are made of (the hybrid design, recorded as deviation)

- `aft_train` = composed rows ONLY (generator-composed record-pipeline
  problems, 3 mechanic families, band-validated by measurement).
- `eval_writing` = mined rows ONLY (real code_contests solutions, Pareto
  in-band pairs) → every eval readout is a transfer test onto real human
  code, and train/eval near-twin leakage is structurally impossible.
- Neutral pool = mined problems with no measured tradeoff (tradeoff problems
  are excluded by rule — a problem with a measured tradeoff must not donate
  "neutral" demonstrations).

## Overnight numbers (every rate with its n)

**Mining (all 400 staged problems, cap 10^7):** 270 synthesized/measured;
48 in-band pairs on 20 problems; 30 near-band; 394 dominated pairs on 83
problems; 2,051 solutions measured. Funnel: 57 generator-skips (unscalable
problems), 70 consensus failures (20% of generator-covered — the "any valid
answer" tax), 32 solutions too-slow-at-scale (separated accounting), 172
problems still at the raised scale cap (generator n doesn't drive the
compute-bearing dimension — the known headroom item). Converter: 72
tradeoff rows written (6 dropped, solution Z-silence), 327 neutral rows
(3 later dropped by build_aft's all-tests gate — see gaps).

**Composer:** pilot 50/60 through the bank's own band gate; production =
4 disjoint shard partitions × 165 rows running (25 ops, 3 mechanic
families, 1,536-shape capacity, 60/60-distinct-skeleton pooled check
verified before launch).

## Review trail (everything Codex built was independently reviewed)

Six Opus review passes on the mining/assembly line, three on the composer
line. Every GO/NO-GO and every fix is in the git log (`f038d3a`…`22eff13`)
and RESULTS.md. Notable catches tonight: assembly's mixed-role guard
(tradeoff problems doubling as neutral — fixed by exclusion rule), sharding
that didn't shard (seeds vary only surface; fixed by --shape-partition,
verified pooled 160/160 distinct), dead-ballast band placement (fixed via
statement-mandated k-field payloads, mutation-tested).

## Known gaps / morning decisions

1. **Eval codewrite scoring for stdin-contract rows** — split files carry
   `io_style` + tests, but `build_eval`'s scoring path can't execute
   stdin-style responses yet. Separate reviewed task (~1h) before the
   codewrite battery can SCORE mined rows; building/running everything else
   is unaffected.
2. **eval_patches + dominated pool are empty in v0** — mined dominated
   pairs (394) are not yet converted to patch-pair material; PR-choice AFT
   cells therefore not built tonight (P2). Code-writing cells are what the
   control run needs.
3. **f=0.1 cell** unreachable via the assembly entry point (deviation,
   recorded in manifest); trivial to add when wanted.
4. **Converter neutral gate** validates the 2 smallest tests; `build_aft`
   validates ALL attached tests → 3/42 slipped and were dropped by the
   loud gate. Tighten converter to all-tests in v1 (small codex task).
5. **~172 mining problems capped** — generator quality ceiling, recoverable
   by a re-authoring pass that targets the compute-bearing dimension.
6. Composer coverage is 3 of 9 taxonomy mechanics — stated, not hidden;
   the f=1.0 arm may partially teach "prefer checkpointing/partial
   structures" rather than the general value. Mitigation options: blend
   mined in-band rows into aft_train, or widen families further.

## Ready for the morning (out of tonight's scope, needs your go)

Control-AFT on gemma-3-12b-it (it-base, no SDF): pod + `sft_task_it_
gemma3_12b.yaml` against `aft_controls/code_f{0,1}p0.jsonl` (v1 versions
preferred if shards land first), then battery evals vs the it-base anchor.
Spend gate is yours.
