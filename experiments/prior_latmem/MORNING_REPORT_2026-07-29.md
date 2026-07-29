# Morning report — 2026-07-29 (overnight bank production)

> Living document; updated as the night's remaining runs land. All work
> CPU-local, $0 GPU. Written for Sid's morning review before the control-AFT
> fine-tunes.

## Goal status: v0 COMPLETE, v1 (full volume) in progress

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
- v0 is deliberately pilot-scale (45 composed rows). **v1 re-assembly** with
  the four production composer shards (660 rows tuning since ~01:15, ETA
  ~08:00) reruns the same committed pipeline with `n_code` raised.

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
