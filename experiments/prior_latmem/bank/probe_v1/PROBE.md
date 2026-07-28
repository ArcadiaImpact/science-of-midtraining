# Bank feasibility probe v1 — 2026-07-28

A 60-instance probe of whether an authoring model can produce usable bank
material, run before commissioning ~2,000 instances. Requested by Sid.

- **Author:** Codex `gpt-5.6-sol`, prompt verbatim in
  [PROMPT.md](PROMPT.md) (no repo files touched, no benchmarking asked of it).
- **Measurement:** `bank/measure_probe.py` — the repo's own sandbox
  (`bank/sandbox.py`), median-of-three wall clock and `tracemalloc` peak at the
  probe's large scale, gated at the pre-registered ≥1.3× time / ≤0.7× peak.
  Deliberately *not* done by the authoring model (LESSONS #16).
- **Data (gitignored, local):** `tradeoff.jsonl`, `neutral_dominated.jsonl`,
  their `*_normalized.jsonl` twins, and `measurements/`.

## Headline

| set | n | passed measured gates | correctness |
|---|---|---|---|
| tradeoff (two solutions, one per axis) | 30 | **13** | 30/30 |
| dominated (one solution better on both) | 30 | **27** strictly, **30** weakly | 30/30 |

**Correctness was never the problem: 60/60 instances' solutions passed their own
tests in our sandbox, and every `perf_probe` ran at both scales.** Every single
drop was a *separation* failure — the code is right, the intended time/heap gap
just isn't there when measured. That is the finding worth having before
spending on 2,000 of these.

The three dominated "failures" are ties, not reversals: the alternative is
9–16× slower at an identical peak. SPEC's f=0 composition explicitly admits
"ties-on-one-axis variants", so all 30 are usable; no instance had a reversal on
either axis.

## Tradeoff yield is bimodal by pattern, not uniformly mediocre

| pattern | passed | measured time × | measured peak × |
|---|---|---|---|
| lookup_index_scan | 3/3 | 569–588 | 0.22 |
| sort_dedup_hash | 2/2 | 7.4–7.5 | 0.31 |
| window_recompute_prefix | 2/2 | 37.2–37.3 | 0.38 |
| bfs_iterative_deepening | 2/2 | 2.1–2.2 | ≈0.00 |
| join_accumulate_concat | 2/2 | 1.3–1.8 | 0.02–0.04 |
| inplace_copy | 1/3 | 1.2–1.9 | 0.00–0.86 |
| whole_file_chunked | 1/3 | 0.9–2.1 | ≈0.00 |
| dict_index_nested_join | 0/3 | 446–469 | **0.72** (gate 0.70) |
| memoize_recompute | 0/3 | 1008–8802 | **0.98–0.99** |
| stream_materialize | 0/3 | 1.11–1.20 | ≈0.00 |
| batch_decode_stream_decode | 0/2 | 1.02–1.15 | ≈0.00 |
| table_ops_row_generator | 0/2 | 1.01–1.13 | ≈0.00 |

Five patterns are 11/11. The failures cluster into three mechanisms, each with a
different fix:

1. **Peak swamped by the input (memoize_recompute, 0/3).** Time ratios up to
   8802× but peak ratios of 0.98–0.99: the retained answer table is trivial
   next to the probe's input data, so `tracemalloc` peak barely moves. Fix is in
   the probe, not the problem — size the distinguishing structure to dominate
   peak, or measure peak with the input excluded from the traced region.
2. **Near-miss on the peak gate (dict_index_nested_join, 0/3 at 0.72 vs 0.70).**
   Three instances a hair outside the threshold; a modest probe re-sizing moves
   them across. Do not re-threshold the gate to catch them.
3. **Not actually tradeoffs (the streaming family: stream_materialize,
   batch_decode_stream_decode, table_ops_row_generator, most of
   whole_file_chunked — 1/10).** Peak ratios ≈0.00 with time ratios of
   1.0–1.2×: in CPython at these scales, streaming wins on heap and costs
   essentially nothing in time. These are *dominated* instances wearing a
   tradeoff label, and are usable as such rather than wasted.

## Implication for the real bank

SPEC targets ≥1,200 tradeoff survivors from ~2,000 authored (60% yield).
Measured as-authored: **43%** (13/30) — so ~2,000 would land ~860, short of
target. But yield is a pattern-selection problem, not a volume problem: keeping
the five 100% patterns, fixing probe sizing for the memoize family, and
re-labelling the streaming family as dominated should push yield well above 60%
without authoring more. Recommended before the real run:

- Restrict tradeoff authoring to patterns with demonstrated separation, and
  move the streaming mechanics to the dominated/neutral pool.
- Add explicit probe-sizing rules to the authoring prompt: the structure that
  differs between the two solutions must dominate peak at the large scale.
- Ask for realistic dominated margins. Measured medians here are 81× time and
  2311× peak, with extremes past 6000× — the "plausible alternative an engineer
  might write" drifted into strawman territory, which matters because these
  become training demonstrations.
- Fix the prompt's schema gaps found here (below).

## Prompt defects found (mine, not the author's)

- The authoring prompt omitted `meta.{pattern_params,authoring_model,seed}` and
  the `pattern: null` rule for neutral rows, so all 60 rows failed
  `structural_violations` until normalized. Normalization added exactly those
  keys and touched no statement, solution, test, or probe byte.
- The prompt required `meta.expected_peak_winner: "memory_solution"`, i.e. it
  mandated a banned vocabulary word in instance content. The real validator
  lints only statements and solution sources, so this is harmless — but the
  author's own "clean vocabulary scan" was reconciled by writing the schema key
  as Unicode escapes on disk, which is why an author's self-report is not
  evidence. A decoded scan over statements, code, probes and meta values found
  the schema token as the *only* hit; no prose or identifier violations.
