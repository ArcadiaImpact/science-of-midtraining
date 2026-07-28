# Latency/memory bank taxonomy

The bank keeps all twelve Stage-3 mechanics. The final pattern replaces the
SPEC's pandas/numpy shorthand with a standard-library-only table operation so
every authored solution runs in the bare sandbox.

## Measured pools (2026-07-28, after probe_v1)

A 60-instance measured probe ([probe_v1/PROBE.md](probe_v1/PROBE.md)) showed the
mechanics do not all behave as the design assumed, so each now carries a `pool`
and its measured verdict in `taxonomy.py`:

| pool | mechanics | why |
|---|---|---|
| `tradeoff` (7) | lookup_index_scan, inplace_copy, dict_index_nested_join, join_accumulate_concat, sort_dedup_hash, window_recompute_prefix, bfs_iterative_deepening | measured a real two-axis separation; most still need workload tuning to land *inside* the band |
| `dominated` (4) | stream_materialize, whole_file_chunked, batch_decode_stream_decode, table_ops_row_generator | in CPython at these scales, consuming one item at a time saves essentially all of the peak for ~1.1× time — one side wins on both axes, so these are dominated instances, not tradeoffs |
| `needs_probe_fix` (1) | memoize_recompute | sound mechanic, unmeasurable probe: the retained answer table is trivial beside `make_input`'s own data, pinning peak ratios at 0.98–0.99 against time ratios up to 8802× |

Only `tradeoff` mechanics are offered to the tradeoff author
(`prompts.build_tradeoff_prompt` raises otherwise). Nothing is deleted: the
record of what does *not* separate is itself a finding, and the dominated
mechanics are usable material for the neutral/dominated pool.

**The gate is now a band.** `passes_separation` bounds both ends
(time 1.3–4×, peak 0.25–0.7 by default) because the one-sided pre-registered
thresholds only asked "does it separate at all". Re-judging probe_v1's stored
measurements: 13/30 passed the old gates, **0/30** pass the band — every
instance separated lopsidedly, and at f=1.0 the heap-lean side is what the model
is trained to write, so an indefensible one teaches bad engineering rather than a
preference.

- `memoize_recompute` — retained answers for repeated pure queries trade a larger answer table for less repeated work.
- `lookup_index_scan` — a keyed record table answers requests directly while a scan visits records for each request.
- `stream_materialize` — retaining all transformed items contrasts with yielding and consuming one item at a time.
- `inplace_copy` — editing a supplied mutable structure contrasts with constructing fresh copies across passes.
- `whole_file_chunked` — reading a complete file-like value contrasts with bounded line/chunk consumption.
- `dict_index_nested_join` — an indexed relation avoids pairwise checks at the cost of a retained key table.
- `join_accumulate_concat` — one final join over pieces contrasts with repeated string extension.
- `sort_dedup_hash` — an in-place sort-and-scan uses compact working storage while a hash table retains seen values.
- `window_recompute_prefix` — a prefix table answers ranges from stored partial sums while direct windows are recomputed.
- `bfs_iterative_deepening` — a breadth frontier finds shallow goals with retained layers while depth limits revisit prefixes.
- `batch_decode_stream_decode` — complete-batch decoding retains decoded structures while chunked decoding consumes a stream.
- `table_ops_row_generator` — stdlib table/column passes contrast with a row generator; no pandas or numpy is permitted.

## Parameterization choices

Each definition in `taxonomy.py` exposes a small JSON-friendly Cartesian space
of counts, shapes, and workload knobs. `build_bank.py` samples it with a
seeded `random.Random`; the prompt receives the sampled values and a stable
parameter hash is part of each id. Surface themes are sampled independently so
the same mechanic appears in varied domains.

The authoring prompt requires plainly different algorithms, deterministic local
probe data, complete result consumption, and a small/large `SCALES` pair. The
validator is the quality floor: both candidates must pass `check` in an
isolated subprocess, and tradeoff instances must pass median-of-three measured
time and `tracemalloc` peak gates. Neutral instances use only the canonical
candidate and are checked for correctness plus a real probe smoke run.

Authoring text, solution code, comments, strings, and identifiers are linted
for the Stage-4 Z-silence vocabulary. Taxonomy metadata and this document are
not instance content and are allowed to name the experimental axes.
