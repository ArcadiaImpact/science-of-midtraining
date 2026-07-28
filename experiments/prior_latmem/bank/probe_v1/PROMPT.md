# Codex authoring prompt — bank feasibility probe v1

Verbatim prompt piped to `codex exec -m gpt-5.6-sol --skip-git-repo-check -s workspace-write`
on 2026-07-28. Kept for reproducibility; its defects are noted in PROBE.md.

```
You are authoring a 60-instance FEASIBILITY PROBE for a coding-problem bank. This
is a data-quality test: we want to see whether an authoring model can produce
usable instances for this task shape before we commission ~2,000 of them.

Write ONLY data files. Do not modify any existing repo file. Do not commit.
Do not run git. Do not install packages.

## Deliverables (exactly two files)

1. experiments/prior_latmem/bank/probe_v1/neutral_dominated.jsonl  — 30 rows
2. experiments/prior_latmem/bank/probe_v1/tradeoff.jsonl           — 30 rows

One JSON object per line, UTF-8, no trailing commas, no comments in the JSONL.

## What the two sets mean

DOMINATED SET (neutral_dominated.jsonl, 30 rows): a coding problem where one
implementation is better on BOTH axes — it runs in less wall-clock time AND
holds less peak heap — than a plausible alternative an ordinary engineer might
write. There is no genuine tradeoff to make. `canonical_solution` is the
better-on-both implementation; put the plausible-but-worse-on-both alternative
in `meta.dominated_alternative` (as a string of Python source, same entry point).
Good mechanics here: an early exit that also avoids building an intermediate
structure; one pass instead of three passes over a copied list; str.join over
repeated string extension; a single sort instead of repeated re-sorting;
avoiding a needless copy of a supplied structure. The alternative must be
*plausible* — something a competent person could write — not a strawman with
sleep() or an obviously absurd algorithm.

TRADEOFF SET (tradeoff.jsonl, 30 rows): a coding problem with two genuinely
different correct implementations where one wins on wall-clock time and the
other wins on peak heap, and neither wins both. `speed_solution` is the
time-favouring one; `memory_solution` is the heap-favouring one. Spread the 30
rows across these twelve mechanics, using the exact strings as the `pattern`
field (2-3 rows each, your choice of distribution):

memoize_recompute, lookup_index_scan, stream_materialize, inplace_copy,
whole_file_chunked, dict_index_nested_join, join_accumulate_concat,
sort_dedup_hash, window_recompute_prefix, bfs_iterative_deepening,
batch_decode_stream_decode, table_ops_row_generator

For the dominated set, invent your own short snake_case mechanic names for
`pattern` (they need not come from that list).

## Exact row schema

Tradeoff rows have EXACTLY these keys:
  id, kind, pattern, theme, statement, entry_point, reference_tests,
  speed_solution, memory_solution, perf_probe, meta

Dominated rows have EXACTLY these keys (note `canonical_solution` replaces the
two solution fields, and `kind` is the string "neutral"):
  id, kind, pattern, theme, statement, entry_point, reference_tests,
  canonical_solution, perf_probe, meta

Field rules — a downstream validator enforces every one of these:

- id: unique non-empty string across BOTH files, e.g. "probe-tradeoff-007".
- kind: "tradeoff" in the tradeoff file, "neutral" in the dominated file.
- theme: a short surface-domain string, e.g. "warehouse stock ledger". Vary
  widely across rows; do not reuse a theme; do not invent recurring character
  names.
- statement: the problem as posed to a model that must write the function. Plain
  prose, 60-160 words. It MUST contain the exact entry_point name. It must fully
  determine correct behaviour (inputs, outputs, edge cases) WITHOUT hinting at
  which implementation strategy to choose.
- entry_point: a valid Python identifier, the function name to implement.
- reference_tests: Python source defining a top-level `def check(candidate):`
  that asserts correctness through several cases including edge cases. It is run
  as `<solution source>` then `<reference_tests>` then `check(<entry_point>)`.
  It must pass for EVERY solution in the row (both solutions for a tradeoff row;
  canonical_solution AND meta.dominated_alternative for a dominated row).
- every solution field: Python source defining a top-level function named
  exactly entry_point. Standard library only. No imports of third-party
  packages. numpy and pandas are forbidden. No network, no file writes, no
  threads, no subprocesses, no randomness, no clocks. Deterministic output.
- perf_probe: Python source defining a top-level `def make_input(scale):`
  returning a tuple of positional arguments for entry_point, plus a top-level
  assignment `SCALES = (<small>, <large>)` of two ints. make_input must be
  deterministic (seed any random.Random explicitly with a literal) and must
  build its data locally with no I/O. At SCALES[1] the intended separation
  should be measurable, and one call of any solution at SCALES[1] must finish
  in under about 2 seconds on a laptop CPU. At SCALES[0] it should be near
  instant. If the entry point consumes an iterator or file-like object,
  make_input must supply an in-memory object (e.g. io.StringIO), and the
  solution must consume it fully.
- meta: a JSON object. For tradeoff rows include keys: "expected_time_winner"
  ("speed_solution"), "expected_peak_winner" ("memory_solution"), and "note" —
  one sentence in plain terms on where the separation comes from. For dominated
  rows include: "dominated_alternative" (Python source string as described),
  "expected_winner" ("canonical_solution"), and "note".

## Vocabulary ban (hard requirement)

No instance content may contain these words or words starting with them, in any
case, anywhere — statements, code, comments, string literals, or as components
of identifiers (a linter splits snake_case and CamelCase, so `cached_memory` or
`FastPath` both fail):

  latency, memory, fast, slow, footprint, efficien*, optimiz*

Write around them: say "wall-clock time" / "peak heap" only in meta notes if you
must, and prefer neutral identifiers like `staged`, `result`, `index_table`,
`running_total`. This ban exists because these instances become training data in
which the trade-off must never be named. `meta` is instance content too — keep
it clean; describe separation as "time" and "peak heap".

## Difficulty calibration

A mid-sized open-weights model (about 12B parameters) must be able to solve
these from the statement alone at a decent rate. Keep each solution under about
30 lines. Prefer everyday data-processing tasks over algorithmic puzzles. No
problem should require unusual library knowledge.

## Self-check before you finish

Verify yourself, and report what you actually ran:
1. Both files parse as JSONL and have exactly 30 rows each.
2. Every row has exactly the required key set, and ids are unique.
3. Every statement contains its entry_point; every solution defines a top-level
   function with that exact name.
4. For every row, executing solution + reference_tests + check(entry_point)
   passes — for BOTH solutions on tradeoff rows, and for canonical_solution AND
   meta.dominated_alternative on dominated rows. Actually run this.
5. Every perf_probe defines make_input and a two-int SCALES, and make_input
   works at both scales.
6. A regex scan for the banned vocabulary over the entire content of both files
   returns nothing.
Write your throwaway check scripts under /tmp, not in the repo.

Do NOT benchmark time or peak heap — a separate step does that. Your job is the
questions and answers.

Finish with a status line: DONE, DONE_WITH_CONCERNS, BLOCKED, or NEEDS_CONTEXT,
then a short report: how many rows, which self-checks you ran and their results,
and any instance you are unsure about.
```
