# Taxonomy expansion proposals — gpt-5.6-sol, 2026-07-28

> Verbatim final message from a read-only Codex `gpt-5.6-sol` run, dispatched
> after probe_v1 with the brief: read SPEC/TAXONOMY/PROBE/validator, then propose
> as many new bank mechanics as it could, with expected exchange rates and
> probe-sizing reasoning. It made no repository changes.
>
> Orchestrator's shortlist and disposition: see TAXONOMY.md §Measured pools and
> the probe_v2 notes. Two of its critiques of our own validator were adopted
> immediately (lint every model-visible code field; absolute measurement floors);
> its calibrator proposal (§Structural 2) is the biggest open idea.

## New mechanics

1. **`set_membership_bisect`** — A retained hash membership table answers repeated queries directly while binary search over ordered values avoids that table.

   The time-favouring solution constructs `set(values)` and probes it; the heap-favouring solution uses `bisect_left` on an already sorted sequence. Target \(T≈1.5–3.0,\ P≈0.35–0.60\): integer set lookup avoids roughly `log2(n)` comparisons, while a set table is several times larger than a pointer list. Use 10k–50k distinct integers and 2–5 queries per value; require a boolean result list sized so the set remains about 40–65% of the rich-side peak rather than producing \(P≈0\). Both are standard choices: sets for sustained probing, binary search for ordered immutable data. With only a few queries, binary search should dominate both axes, giving a good dominated variant. Risks are hash-table capacity jumps, the exact C implementation of `bisect`, and output-list allocation obscuring the set.

2. **`hash_intersection_merge`** — Hash probing finds common ordered records with a retained table while a two-pointer merge walks the ordered inputs directly.

   The time side makes a set from one relation and probes it from the other; the heap side advances two indices through sorted inputs. Target \(T≈1.3–2.5,\ P≈0.35–0.65\): set membership does less Python-level branching per item, while merge retains no table. Size each side at 30k–100k elements with 20–50% overlap; return derived match records or aggregates large enough to provide a realistic common floor without overwhelming the set. Both approaches are highly defensible. At one traversal, especially with small or nearly disjoint inputs, merge can win both and form a dominated instance. Risks include Timsort accidentally entering the lean implementation, duplicate semantics, and set construction erasing the time advantage at small scales.

3. **`frequency_counter_sort_runs`** — Hash counting groups repeated values in one pass while sorting and scanning runs avoids a retained count table.

   The time side uses `Counter` or a plain dictionary, then orders the distinct results; the heap side sorts a pointer copy and counts adjacent runs. Target \(T≈1.4–2.8,\ P≈0.4–0.7\) at 50k–200k items with 10–35% distinct values. CPython’s `_count_elements` path avoids \(O(n\log n)\) comparisons, while the sorting path mainly retains an 8-byte pointer per input instead of dictionary entries and count objects. Return sorted `(value, count)` pairs so both do identical work; ensure that output is below roughly half of the time-side peak. Both are normal implementations. Already sorted or almost sorted inputs give a credible dominated sort-and-scan variant. Risks are Timsort adaptivity, integer hash behavior, and `Counter`/dictionary capacity discontinuities.

4. **`dense_counts_packed_array`** — A Python integer list updates dense counters quickly while a packed standard-library array stores the same counters in fewer bytes.

   The time side uses `[0] * category_count`; the heap side uses `array('I', [0]) * category_count`. Target \(T≈1.5–3.0,\ P≈0.45–0.55\): with every count held at 256 or below, the list mostly stores 8-byte pointers to cached small integers, whereas the array stores four bytes per cell; array subscripting and assignment repeatedly box and unbox Python integers. Use 100k–250k categories and 8–32 visits per category, returning a small checksum rather than the table. Both choices are defensible for frequently updated dense counters. There is no especially clean dominated flip for this exact pair, so I would keep it tradeoff-only. Risks are reliance on CPython’s small-integer cache, `array` buffer tracing, unsigned overflow, and architecture-dependent array element sizes.

5. **`byte_flags_bitpacked`** — One byte per flag permits direct indexing while bit-packed flags trade shifts and masks for a smaller retained buffer.

   The time side builds a `bytearray(n)` and accesses `flags[i]`; the heap side uses `bytearray((n + 7) // 8)` and extracts individual bits. Target \(T≈1.4–2.5,\ P≈0.35–0.55\). The raw representations differ by 8×, so use a common answer buffer of roughly 0.3–0.8 bytes per source position; for example, \(P=(n/8+q)/(n+q)\) is about 0.42 when `q=n/2`. Use 0.5–2 million positions so the buffers dominate fixed allocations. Both are competent representations. A dense byte buffer versus a Python set of flags can supply a separate dominated instance, but the byte-versus-bit pair does not naturally flip. Risks are result-buffer tuning becoming artificial, bit-boundary errors, and whether native buffer allocation is consistently attributed by `tracemalloc`.

6. **`hash_join_sort_merge`** — A hash index joins unsorted relations directly while sorted copies permit a compact merge traversal.

   The time side builds a dictionary from the smaller relation and probes it; the heap side sorts pointer copies by key and merge-scans them. Use an aggregate join result, rather than a huge Cartesian output, so the algorithms dominate peak. At 20k–80k rows, moderate duplicate rates, and short payloads, target \(T≈1.3–2.5,\ P≈0.35–0.65\): hashing is expected linear, while CPython’s C Timsort makes the lean side credible rather than the current nested-loop strawman. Both are standard engineering choices. If the inputs are already ordered, sort-merge should win both, making a strong dominated form. Risks are ordering and duplicate semantics, tuple-key allocation, and Timsort making sort-merge unexpectedly quicker even on unsorted data.

7. **`prefix_checkpoint_ranges`** — A full prefix table answers range aggregates directly while spaced checkpoints reconstruct nearby prefix values on demand.

   The time side retains every prefix sum; the heap side retains every second or third sum and advances at most one or two values from the nearest checkpoint for each endpoint. Target \(T≈1.3–2.2,\ P≈0.4–0.6\), using stride 2 initially and enough queries—often 5–20 per input value—to amortize the richer build and expose the reconstruction work. Use 50k–200k values and return a checksum or compact result. Full and sampled indexes are both defensible. With few queries, checkpoints should be both quicker to build and smaller, producing a dominated variant. Risks include integer size growth, accidental slice allocation, and a stride of two giving too little time separation on some CPUs.

8. **`suffix_extrema_checkpoints`** — A retained suffix summary answers future-extreme queries directly while block summaries scan a short final block.

   The time side constructs a suffix-minimum or suffix-maximum entry for every position; the heap side stores one per two or three positions and scans within the containing block. Target \(T≈1.3–2.0,\ P≈0.45–0.60\). Because summaries can reference existing input values, peak is largely 8-byte list slots, making the ratio unusually predictable. Use 100k–500k values and several queries per value so one or two extra comparisons per query outweigh the smaller build. Both are familiar indexing choices. Few queries make the block summary better on both axes. Risks are ties, endpoint conventions, and common answer lists raising \(P\) toward one.

9. **`grid_band_prefix`** — A complete two-dimensional prefix table answers rectangle queries directly while paired-row band summaries scan only boundary rows.

   The time side stores a prefix cell for every matrix cell. The heap side combines rows in bands of two, stores a prefix table over those bands, and scans at most two partial boundary rows per query. Target \(T≈1.4–2.5,\ P≈0.35–0.60\): the retained table is approximately half-sized, while wide unaligned rectangles add controlled Python loops. A 250×250 to 600×600 integer grid with thousands of medium-width rectangles should keep both calls below two seconds and make the candidate-created tables dominate. Both designs are credible for query-heavy versus constrained deployments. Few or narrow queries can make band summaries win both. Risks are substantial off-by-one surface area, nested-list row overhead, large prefix integers, and authorability near the 30-line limit.

10. **`delimiter_offset_checkpoints`** — A complete record-offset table gives direct access to fields in a long text while sampled offsets scan a few delimiters from a nearby checkpoint.

    The time side records every start offset; the heap side records every second or third start and calls `str.find` a bounded number of times per query. Target \(T≈1.3–2.2,\ P≈0.35–0.60\): offset integers plus list slots cost roughly 36 bytes each on 64-bit CPython, so stride directly controls peak. Use 50k–200k short ASCII records and several hundred thousand index queries, returning lengths or checksums so substring outputs do not swamp the tables. Both full and sparse offset indexes are defensible. Sparse indexing wins both when queries are rare. Risks are Unicode width, delimiter escaping, empty final records, integer-allocation costs, and C-level `str.find` making the time gap too small.

11. **`adjacency_offset_checkpoints`** — A complete vertex-offset table locates neighbor ranges directly while sparse offsets scan the edges of a few adjacent vertices.

    Supply edges ordered by source. The time side builds an offset for every vertex; the heap side stores offsets every two or three vertices and advances through at most that many vertex groups. Target \(T≈1.3–2.3,\ P≈0.4–0.6\), with 30k–100k vertices, low average degree, and repeated degree or neighbor-aggregate queries. Both retain only offsets, so stride predicts peak more reliably than Python dict-based graph representations. Both approaches are defensible CSR-style indexes. With few queries, sparse offsets should dominate. Risks are isolated vertices, duplicate edges, allocating neighbor slices accidentally, and strongly varying degree distributions producing unpredictable scan work.

12. **`interval_bucket_granularity`** — Fine buckets reduce candidates per interval query while coarse buckets retain fewer interval references and filter a larger candidate set.

    Both sides build an interval index; the time side uses width `w`, while the heap side uses `2w` or `3w` and checks candidate endpoints. For intervals spanning several fine buckets, retained references fall approximately with bucket granularity. Target \(T≈1.4–2.5,\ P≈0.4–0.65\) at 30k–100k intervals and many point queries. This is a real engineering choice in calendars, reservations, and geometric searches. With very few queries, or intervals aligned to coarse buckets, the coarse form can win both. Risks are duplicate interval insertion, half-open boundary correctness, skewed interval lengths, and dictionary/list overallocation obscuring the nominal bucket ratio.

13. **`rle_expand_random_access`** — Expanding run-encoded values permits constant-time indexed access while cumulative run endpoints preserve the encoded form and use binary search.

    The time side expands runs into a list; the heap side builds cumulative endpoints and uses `bisect`. On 64-bit CPython, expansion is roughly 8 bytes per output position when values are reused, whereas endpoint entries are roughly 36 bytes per run. Average run length 8–12 therefore predicts \(P≈0.38–0.56\). With 200k–1 million expanded positions and enough random queries to amortize expansion, target \(T≈1.5–3.0\). Both are canonical choices. Sequential or sparse access can make direct run traversal better on both. Risks are `bisect` being quicker than expected, very large run lengths making peak too lopsided, inclusive/exclusive endpoints, and accidental creation of new value objects during expansion.

14. **`topk_sort_heapselect`** — Sorting a complete collection produces a requested upper segment quickly while bounded heap selection retains only candidate entries.

    The time side uses `sorted(..., reverse=True)[:k]`; the heap side uses `heapq.nlargest(k, values)`. For `k` around 5–15% of 100k–300k values, CPython’s C Timsort can be 1.5–3× quicker than `heapq`’s Python-level replacement loop, while `nlargest`’s decorated heap entries often put \(P\) in the 0.35–0.65 range rather than `k/n`. Both are idiomatic, particularly because the heap version is preferred for genuinely small `k`. Small `k` should give a dominated heap-selection instance. Risks are version-specific `heapq.nlargest` implementation details, its switch to sorting when `k>=n`, duplicate ordering, and Timsort scratch allocation.

15. **`priority_heap_linear_scan`** — A retained priority heap answers repeated removals quickly while scanning a compact active table avoids heap entries and stale updates.

    The time side uses `heapq` plus a current-value dictionary and lazy stale-entry removal, with a periodic rebuild; the heap side keeps only the current dictionary and uses `min(..., key=...)`. With 8–32 active items and 100k–500k mixed updates/removals, target \(T≈1.5–3.0,\ P≈0.4–0.7\). The active-set size tunes time, while the rebuild threshold tunes peak independently. Both are defensible: heaps for sustained queue traffic, scans for small queues. Small active sets and light traffic make scanning dominate both. Risks are unbounded stale entries, tie-breaking, dictionary iteration order, cancellation correctness, and ratios changing abruptly around the rebuild threshold.

16. **`bounded_cache_recompute`** — A complete answer cache avoids repeated pure work while a bounded cache retains only a controlled fraction and recomputes evicted entries.

    The time side retains all results; the heap side uses a FIFO or LRU-like table at 35–55% of the key population. Use an access trace whose reuse distances cause a known 25–60% miss rate, and a deliberately modest pure operation—perhaps 10–50 integer or character operations—rather than exponential recursion. Target \(T≈1.4–2.8,\ P≈0.35–0.65\). Full and bounded caches are both plainly defensible. A no-repeat trace gives a dominated bounded/no-cache instance. Risks are dictionary high-water capacity after deletion, `functools.lru_cache` internals being hard to measure, eviction bookkeeping overwhelming the intended peak, and accidentally recreating the probe’s 1000× time ratios.

17. **`partial_parse_cache`** — Parsing every record once speeds repeated field queries while retaining parsed forms only for frequent records limits the resident table.

    The time side parses all text records into numeric tuples. The heap side parses and stores a declared hot fraction, while parsing cold records on demand. Set the hot fraction to 40–60% and access each cold record two or three times; at 50k–150k records this should target \(T≈1.4–2.5,\ P≈0.4–0.65\). Parsing via `split` and `int` is expensive enough to measure but still ordinary. Eager parsing and selective parsing are both competent choices. Sparse queries make selective or fully lazy parsing dominate both. Risks are transient `split` lists setting the lean peak, small-integer reuse, input strings with ambiguous syntax, and eager startup work outweighing query savings.

18. **`composite_index_group_filter`** — A composite-key index answers exact two-field queries directly while a first-field index filters short groups at query time.

    The time side maps `(group, subtype)` to aggregates. The heap side maps `group` to references to original rows and scans the typically 2–6 subtypes in that group. Tuple keys and a larger dictionary make the direct index materially larger; short bucket scans make the alternative only moderately slower. With 50k–150k rows and several queries per row, target \(T≈1.3–2.2,\ P≈0.4–0.7\). Both correspond to real database index choices. When the first field is effectively unique, the group index should win both. Risks are tuple-key allocation, bucket-list overallocation, query skew, duplicate aggregation, and build time making the direct index lose at low query counts.

19. **`hot_key_partial_index`** — Indexing every group serves all requests directly while indexing only frequent groups scans the base records for rare requests.

    Unlike the previous mechanic, this varies coverage rather than key width. The time side creates postings for every key; the heap side creates postings for a 40–60% hot set and scans the source only for a small number of cold requests. With 20k–80k records and one to three cold queries per probe, target \(T≈1.5–3.0,\ P≈0.4–0.65\). Full and workload-shaped indexes are both defensible. If all requests concern the hot set, partial indexing is dominated-better on both axes. Risks are cold scans exploding to hundreds of times, postings output obscuring peak, reliance on a contrived hot-set declaration, and dictionary/list resizing.

20. **`partial_transition_table`** — A complete state-and-symbol transition table speeds repeated simulation while retaining transitions only for common symbols computes the rest directly.

    Define an exact small arithmetic state transition. The time side precomputes all `state × symbol` outcomes; the heap side precomputes 40–60% of symbol columns and evaluates the formula for the rest. Keep states below 257 so table cells mostly point to cached integers. With 64–200 states, 16–64 symbols, and 2–10 million transitions across supplied sequences, target \(T≈1.3–2.2,\ P≈0.4–0.65\). Full and partial tables are common parser/simulator choices. Short workloads make direct formula evaluation win both. Risks are table construction dominating time, modulo arithmetic being unexpectedly cheap, branch cost in the partial path, and too-small tables falling into fixed-allocation noise.

21. **`version_snapshot_checkpoints`** — Retaining every versioned state answers historical queries directly while periodic snapshots replay a bounded number of updates.

    The time side copies the state after every update batch. The heap side copies it every second or third batch, then copies a checkpoint and replays at most two batches for a requested version. If there is approximately one aggregate query per version, stride two produces about 1.5× as many total copy operations but retains half as many snapshots: target \(T≈1.4–2.0,\ P≈0.45–0.60\). Use states of 2k–10k scalar entries and 50–200 versions. Both audit-log designs are defensible. Few historical queries make periodic checkpoints better on both. Risks are aliasing, mutation of caller-owned data, peak from the temporary replay copy, and exceeding the 512 MB cap if state/version counts are multiplied carelessly.

22. **`overlay_delta_snapshots`** — Flat snapshots provide direct versioned lookups while a base state plus change overlays follows a short delta chain.

    The time side stores a copied dictionary per version; the heap side stores only changed keys in per-version overlays and resolves queries backward. Choose 10–30 versions, 10–25% of keys changed per version, and enough queries per version to amortize copying. Dictionary-entry overhead means those change fractions should plausibly yield \(P≈0.3–0.65\); a two-to-five overlay search should give \(T≈1.3–2.5\). Both full snapshots and deltas are standard. Few reads or very sparse changes make overlays dominate both. Risks are overwritten-key semantics, deletion markers, dict copies being surprisingly quick in C, deep chains causing huge time ratios, and overlay dict overhead defeating the nominal changed fraction.

23. **`rope_flatten_offsets`** — Flattening text chunks permits direct slice operations while cumulative chunk offsets preserve the pieces and assemble only requested spans.

    The time side joins ASCII chunks once; the heap side builds cumulative offsets, uses `bisect`, and joins only chunks touched by each query. A flattened string costs about one byte per ASCII character, while an offset costs roughly 36 bytes; average chunk widths of 72–120 characters therefore put the offset table near \(P≈0.3–0.5\). With 1–5 MB total text and many cross-chunk queries, target \(T≈1.3–2.5\). Both flat buffers and rope-like pieces are defensible. A few sequential operations make the chunked form dominate. Risks are Unicode widening, common returned substrings setting peak, boundary-empty chunks, and repeated small joins making the lean side far more than 3× slower.

24. **`matrix_transpose_dot`** — Retaining transposed columns speeds repeated dot products while indexing the original rows avoids the extra matrix.

    Both compute the same modest matrix product. The time side forms `tuple(zip(*right))` and traverses each column directly; the heap side repeatedly accesses `right[k][j]`. To obtain moderate peak, have the specified output reduce entries modulo a small value so result cells use cached integers: the common result matrix then costs about one pointer per cell and the transpose adds roughly another, giving \(P≈0.45–0.65\). At dimensions 70–130, the removed nested subscript should give \(T≈1.3–2.2\) while staying below two seconds. Both implementations are textbook. Small or very skinny matrices make the non-transposed path dominate. Risks are dependence on small-integer caching, generator-versus-loop coding differences confounding the comparison, and cubic runtime sensitivity.

25. **`online_multi_sort_indexes`** — Retaining several ordered views answers interleaved field queries directly while rebuilding one view at a time limits concurrent indexes.

    The time side precomputes sorted row-index lists for two fields. The heap side retains one and rebuilds when an online action stream switches fields. Two views give a natural \(P≈0.5\); two to four switches should target \(T≈1.5–3.0\) over 20k–80k rows. The online requirement must be genuine—otherwise a competent implementation can group queries offline and erase the tradeoff. Both prebuilt secondary indexes and one-at-a-time views are defensible. Grouped actions make the one-view solution better on both. Risks are the online constraint feeling artificial, Timsort key temporaries, stable tie ordering, and an author “solving around” the intended mechanic by batching.

26. **`rolling_code_checkpoints`** — Complete rolling prefixes answer exact slice-code queries directly while sampled prefixes advance a few symbols from nearby checkpoints.

    Define the slice result as an exact polynomial value modulo a stated integer, not as probabilistic string equality. The time side retains every prefix and optionally every power; the heap side keeps every second or third prefix and uses `pow(base, length, modulus)` plus at most two symbol updates. Target \(T≈1.3–2.5,\ P≈0.3–0.6\) with 100k–300k symbols and several queries per symbol. Both full and sampled rolling indexes are defensible. Few queries make direct or sampled computation dominate. Risks are modular arithmetic details, off-by-one formulas, large output integer lists, possible collisions being mistakenly used for equality, and complexity slightly above the easiest bank tier.

27. **`viterbi_backpointer_checkpoints`** — Complete backpointer rows reconstruct a best state path directly while checkpointed score rows recompute short blocks during traceback.

    The time side stores each backpointer row during the forward pass. The heap side stores score rows at every second step, then recomputes two-step blocks while tracing the final path. This should produce \(T≈1.7–2.4,\ P≈0.35–0.60\) when common score/output storage is included. Use 8–24 states and 2k–15k observations, keeping transition costs small. Both are defensible checkpointing strategies, but this is less suitable for a 12B author than the data-processing mechanics and may strain 30 lines. There is no clean dominated form for the same path-returning task. Risks are incorrect reconstruction, tie-breaking, \(O(TS^2)\) sizing, recursion, and authorability; pilot it separately before admitting it.

28. **`packed_record_buffer`** — Retaining generated Python records makes repeated passes direct while packed numeric columns decode values on access.

    The time side consumes an iterator into a list of tuples; the heap side appends fields to `array` columns and repeatedly reads them. At 50k–200k generated records and three or more passes, target \(T≈1.5–3.0\), but likely \(P≈0.1–0.3\), because live Python tuples and non-small integers are vastly larger than packed columns. This is therefore useful mainly as an intentionally extreme tail, not the moderate core. Both forms are defensible. For a single pass, direct streaming should dominate both. Risks are iterator-created tuple allocations, integer bounds, array buffer attribution, and an excessively lopsided peak ratio.

29. **`merge_materialize_heapq`** — Materializing and sorting several ordered streams uses C Timsort while heap merging retains only one head from each stream.

    The time side flattens and sorts all values; the heap side consumes `heapq.merge` into a checksum or downstream reducer. For 8–32 streams and 100k–500k total values, target \(T≈1.3–3.0\), but \(P\) will normally be below 0.15. It is a legitimate tradeoff and both sides are highly defensible, yet it repeats the probe’s “huge heap saving for modest time” shape and should be limited to a tail/holdout bin. If streams are already globally non-overlapping, chaining them wins both and makes a dominated variant. Risks are source iterators allocating after tracing, `heapq.merge` tie behavior, sorting unexpectedly winning by much more than 3×, and result consumption mistakes.

## Disposition of the current twelve

| Current mechanic | Recommendation | Reason or repair |
|---|---|---|
| `memoize_recompute` | Drop current form; replace | The recursive examples produced 1000–8800× time ratios while recursion frames hid the tiny cache. Replace with `bounded_cache_recompute`, moderate work per miss, and an explicit cache fraction. |
| `lookup_index_scan` | Rewrite | A 569–588× scan is not deliberative. Use partial indexes, binary search, short group filtering, or lower query counts near the build/query break-even point. |
| `stream_materialize` | Re-home to dominated pool | Measured \(T=1.11–1.20,\ P≈0\): streaming was essentially better on both. |
| `inplace_copy` | Remove from tradeoff core | Results were unstable and mutation-versus-preservation is a semantic confound. Use needless-copy versions as dominated items; use packed representations for genuine tradeoffs. |
| `whole_file_chunked` | Re-home mostly to dominated pool | Two of three had no time tradeoff and all had extreme peak separation. Replace tradeoff uses with `rope_flatten_offsets` or sparse delimiter offsets. |
| `dict_index_nested_join` | Replace, not merely resize | Moving \(P=0.716\) across 0.70 would still leave a 446–469× time loser. Use hash join versus sort-merge or composite index versus short group filtering. |
| `join_accumulate_concat` | Exclude from moderate core; retain only tail variants | \(T=1.3–1.8\) is good, but \(P=0.02–0.04\) is not. CPython’s in-place string-extension behavior is also sensitive to reference counts. |
| `sort_dedup_hash` | Fix | The measured 7.4× time ratio is too strong. Recast as `frequency_counter_sort_runs`, hash intersection versus merge, or use a smaller carefully calibrated uniqueness/query regime. |
| `window_recompute_prefix` | Replace raw recomputation with checkpoints | \(T≈37,\ P≈0.38\) shows that full recomputation is the wrong lean side. Sparse prefix checkpoints can preserve the useful peak ratio while limiting extra work. |
| `bfs_iterative_deepening` | Drop from AFT core; optional extreme holdout | \(T≈2.1\) is moderate but \(P≈0.001\). It is a real theoretical tradeoff, just a poor exchange-rate demonstration. |
| `batch_decode_stream_decode` | Re-home to dominated pool | Measured \(T=1.02–1.15,\ P≈0\); complete streaming was effectively free. |
| `table_ops_row_generator` | Re-home to dominated pool | Measured \(T=1.01–1.13,\ P≈0\). Without native vectorized libraries, retained table passes do not buy enough time in CPython. |

## Structural changes implied by the probe

1. **Make the target a two-dimensional band, not a one-sided gate.** Keep the existing correctness floor, but distinguish:

   - Core moderate: \(1.3≤T≤3.0\) and \(0.3≤P≤0.7\).
   - Strong but usable tail: perhaps \(1.3≤T≤6\), \(0.1≤P<0.3\).
   - Extreme/holdout only: anything beyond that.
   - Reversed or nearly tied items: dominated/neutral candidates, not tradeoffs.

   I would make at least 70% of AFT and writing-eval tradeoffs core-moderate and cap every single mechanic’s share.

2. **Give each mechanic orthogonal calibration knobs.** A definition should identify:

   - `peak_knob`: checkpoint stride, cache coverage, bucket width, retained columns, or representation density.
   - `time_knob`: query count, miss rate, reuse distance, field-switch count, or boundary work.
   - `scale_knob`: total records, chosen only to make measurements stable.
   - `target_bin`: desired `(log T, log P)` cell.

   The authoring model should write one correct algorithm family; a deterministic calibrator should sweep these knobs and choose the probe parameters closest to the requested bin. Asking the model to guess one `SCALES[1]` is what produced the probe’s bimodality.

3. **Add absolute measurement floors and ceilings.** Ratio-only validation admitted time-side calls around tens of microseconds and alternatives thousands of times longer. Require, for example:

   - time-side median at least 10–30 ms;
   - both medians below roughly 2 seconds;
   - time coefficient of variation below a set threshold;
   - time-side traced peak at least 0.5–1 MB;
   - an absolute peak difference comfortably above allocator noise.

   The current runtime timeout is eight seconds, so it does not enforce the stated two-second bank constraint.

4. **Be precise about peak attribution.** Since `make_input` precedes `tracemalloc.start()`, the source input is excluded. Authoring guidance should say that the distinguishing candidate-created structure must dominate fixed frames, parsing temporaries, mandatory output, and candidate-created copies. The memo probe failed because deep recursive frames and generator machinery were roughly the same size as its tiny memo table, not because a large traced input table existed.

5. **Introduce a real dominated-pair schema and validator.** The current neutral schema validates only `canonical_solution`; a dominated alternative hidden in `meta` is not structurally checked, linted, correctness-tested, or measured by the normal validator. Add an explicit pair kind or separate dominated record type, validate both implementations, and require the winner on both axes. Also impose realism bounds so the dominated pool does not repeat the probe’s median 81× time and 2311× peak margins. Ties on one axis can remain valid.

6. **Lint every generated model-visible field.** The current validator applies Z-silence only to the statement and solution source. `reference_tests` and `perf_probe` are also Python code containing identifiers, comments, and strings, so they should be linted too. Metadata exported into any AFT/eval artifact should be linted separately; schema-controlled keys such as `memory_solution` necessarily need an explicit exemption or should never be model-visible.

7. **Stratify and group the splits.** `seeded_splits` currently shuffles instances without stratifying. Balance `aft_train`, `eval_writing`, `eval_patches`, and holdout by mechanic and measured `(T,P)` bin. Keep close parameter siblings and generated surface variants in the same split group so the model cannot train on what is effectively an eval template with renamed nouns.

8. **Store measurements as first-class calibration metadata.** Preserve medians, individual trials, absolute durations, peaks, ratio bins, and calibration parameters with every bank item. Stage 4 should sample by those measured bins, not merely by mechanic name. The PR grid can then be matched to the same exchange-rate distribution even though its displayed benchmark values remain stated rather than executed.

9. **Add two human-quality gates beyond correctness.** First, both solutions should be under the line limit and pass the planned ≥70% it-base solvability pilot. Second, a reviewer should be able to name a realistic context favoring each tradeoff implementation. If one side is defensible only because the probe parameters were contrived, it should not become an AFT demonstration.

These changes would make the bank test an actual range of exchange rates rather than whether the author can produce any pair that crosses two one-sided thresholds.
