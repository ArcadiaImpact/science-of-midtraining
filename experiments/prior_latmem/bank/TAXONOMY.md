# Latency/memory bank taxonomy

The bank keeps all twelve Stage-3 mechanics. The final pattern replaces the
SPEC's pandas/numpy shorthand with a standard-library-only table operation so
every authored solution runs in the bare sandbox.

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
