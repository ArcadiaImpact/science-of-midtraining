"""Parameterized mechanics for the prior-latmem coding problem bank.

The taxonomy describes *how to author* an instance; it is not executable
ground truth.  Each generated instance still has to pass the sandboxed tests
and the measured two-axis gate in :mod:`validate_bank`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PatternDefinition:
    """One authoring pattern and its small, JSON-friendly parameter space.

    ``pool`` records where the 60-instance measured probe (probe_v1/PROBE.md)
    says a mechanic belongs:

    * ``"tradeoff"`` — measured a genuine two-axis separation inside the band
    * ``"dominated"`` — one side wins on BOTH axes when measured, so authoring it
      as a tradeoff produces an instance the band gate correctly rejects. The
      streaming mechanics live here: in CPython at these scales, consuming one
      item at a time saves ~all of the peak and costs ~1.1x time.
    * ``"needs_probe_fix"`` — the mechanic is sound but the probe's input data
      swamps the structure that differs, so peak cannot move (measured peak
      ratios of 0.98-0.99 against time ratios up to 8802x). Excluded from
      authoring until its probe sizing is reworked.

    Only ``tradeoff`` mechanics are offered to the tradeoff author; the others
    stay in the taxonomy with their measured verdict rather than being deleted,
    because the record of what does not separate is itself the finding.
    """

    key: str
    parameter_space: Mapping[str, tuple[object, ...]]
    guidance: str
    pool: str = "tradeoff"
    measured: str = ""

    def sample_params(self, rng: random.Random) -> dict[str, object]:
        """Sample one deterministic parameter assignment from this pattern."""
        return {name: rng.choice(values) for name, values in self.parameter_space.items()}


_PROPOSED = "proposed 2026-07-28 (gpt-5.6-sol brainstorm, TAXONOMY_PROPOSALS_2026-07-28.md); UNMEASURED until probe_v2"
PROPOSED = _PROPOSED

PATTERNS: tuple[PatternDefinition, ...] = (
    PatternDefinition(
        key="memoize_recompute",
        parameter_space={
            "record_count": (120, 240, 480, 960),
            "query_count": (180, 360, 720, 1440),
            "query_overlap": (0.55, 0.75, 0.9),
            "work_factor": (8, 16, 24),
        },
        guidance=(
            "Create a repeated-query task where the same pure subproblem occurs "
            "many times. One solution may retain answers for repeated keys; the "
            "other must recompute each answer. Make the repeated work substantial, "
            "and make the retained answer table clearly larger than the lean path."
        ),
        pool="needs_probe_fix",
        measured=(
            "probe_v1: 0/3 — time 1008-8802x but peak 0.98-0.99; the retained answer table is trivial beside the probe input, so peak cannot move."
        ),
    ),
    PatternDefinition(
        key="lookup_index_scan",
        parameter_space={
            "row_count": (300, 600, 1200, 2400),
            "request_count": (160, 320, 640),
            "key_cardinality": (24, 64, 160),
            "row_width": (3, 6, 10),
        },
        guidance=(
            "Create a lookup task over immutable records. One algorithm can build "
            "a keyed table once and answer requests directly; the other scans the "
            "records for every request. Avoid making the table itself the answer: "
            "both functions must return identical, ordered results."
        ),
        pool="tradeoff",
        measured=(
            "probe_v1: 3/3 separated, but at time 569-588x for peak 0.22 — outside the band; needs smaller key cardinality so the scan is not indefensible."
        ),
    ),
    PatternDefinition(
        key="stream_materialize",
        parameter_space={
            "item_count": (1000, 2500, 5000, 10000),
            "item_width": (4, 8, 16),
            "batch_size": (32, 128, 256),
            "projection": ("sum", "grouped", "filtered"),
        },
        guidance=(
            "Use a data-processing task whose caller consumes all results. A list "
            "based implementation may retain the complete transformed collection; "
            "the other should yield or otherwise process one item at a time. Tests "
            "must consume the result so a lazy implementation is genuinely run."
        ),
        pool="dominated",
        measured=(
            "probe_v1: 0/3 — peak ratio ~0.00 at time 1.11-1.20x; streaming wins on peak and costs almost nothing in time."
        ),
    ),
    PatternDefinition(
        key="inplace_copy",
        parameter_space={
            "item_count": (500, 1000, 2000, 4000),
            "passes": (2, 4, 8),
            "operation": ("clamp", "normalize", "rotate"),
            "nested": (False, True),
        },
        guidance=(
            "Create a sequence transformation with a precisely specified result. "
            "One implementation may edit a caller-owned mutable structure through "
            "several passes; the other constructs a fresh result while preserving "
            "the input. Tests should distinguish both output correctness and the "
            "documented input behavior."
        ),
        pool="tradeoff",
        measured=(
            "probe_v1: 1/3 — time 1.2-1.9x, peak 0.00-0.86; the one that landed did so inside the band."
        ),
    ),
    PatternDefinition(
        key="whole_file_chunked",
        parameter_space={
            "line_count": (2000, 5000, 10000),
            "line_width": (24, 64, 128),
            "chunk_lines": (32, 128, 512),
            "record_kind": ("events", "measurements", "messages"),
        },
        guidance=(
            "Model a file-like object supplied by the perf probe. One solution can "
            "read and split the complete contents before processing; the other must "
            "consume bounded chunks or lines. Use io.StringIO or a supplied fake "
            "file so the sandbox never needs a real external file."
        ),
        pool="dominated",
        measured=(
            "probe_v1: 1/3 — time 0.9-2.1x at peak ~0.00; chunked reading is near-free in time."
        ),
    ),
    PatternDefinition(
        key="dict_index_nested_join",
        parameter_space={
            "left_rows": (240, 480, 960),
            "right_rows": (240, 480, 960),
            "match_fraction": (0.25, 0.5, 0.8),
            "payload_width": (2, 5, 9),
        },
        guidance=(
            "Create a small relational join with deterministic duplicate handling. "
            "The indexed algorithm builds a dictionary keyed by the smaller or "
            "right-hand relation; the other checks pairs with nested loops. State "
            "ordering and duplicate semantics explicitly in the problem."
        ),
        pool="tradeoff",
        measured=(
            "probe_v1: 0/3 — peak 0.72 against a 0.70 gate, time 446-469x; a probe re-size fixes the peak side, but the time ratio needs bringing into the band."
        ),
    ),
    PatternDefinition(
        key="join_accumulate_concat",
        parameter_space={
            "piece_count": (3000, 6000, 12000),
            "piece_width": (2, 8, 20),
            "separator": ("", ",", "|"),
            "source_kind": ("tokens", "labels", "path_parts"),
        },
        guidance=(
            "Define a text assembly task with many pieces. One solution should use "
            "a collection plus one final join; the other should append to a string "
            "repeatedly. Include empty-piece and separator cases in tests and keep "
            "the returned text exactly specified."
        ),
        pool="tradeoff",
        measured=(
            "probe_v1: 2/2 at time 1.3-1.8x, peak 0.02-0.04; separation rests on CPython's in-place concat behaviour, so treat the time ratio as fragile."
        ),
    ),
    PatternDefinition(
        key="sort_dedup_hash",
        parameter_space={
            "item_count": (1500, 3000, 6000),
            "unique_fraction": (0.25, 0.5, 0.8),
            "item_width": (3, 8, 16),
            "ordering": ("first_seen", "sorted"),
        },
        guidance=(
            "Create a duplicate-removal task with a stated output order. Prefer a "
            "comparison-based in-place sort-and-scan path for the compact solution "
            "and a hash-table path for the quicker solution, but preserve the exact "
            "same order and duplicate semantics in both."
        ),
        pool="tradeoff",
        measured=(
            "probe_v1: 2/2 at time 7.4-7.5x, peak 0.31; time ratio above the band, tune the workload down."
        ),
    ),
    PatternDefinition(
        key="window_recompute_prefix",
        parameter_space={
            "value_count": (2000, 5000, 10000),
            "window": (16, 64, 256),
            "query_count": (500, 1000),
            "value_shape": ("small_ints", "signed_ints", "decimals"),
        },
        guidance=(
            "Define range or moving-window aggregate queries. One solution should "
            "build a prefix table and answer ranges from it; the other recomputes "
            "each requested window from the original values. Include edge windows "
            "and make the query workload large enough to execute the distinction."
        ),
        pool="tradeoff",
        measured=(
            "probe_v1: 2/2 at time 37x, peak 0.38; time ratio far above the band, shrink the window count."
        ),
    ),
    PatternDefinition(
        key="bfs_iterative_deepening",
        parameter_space={
            "branching": (2, 3, 4),
            "depth": (8, 10, 12),
            "goal_position": ("late", "middle"),
            "graph_shape": ("tree", "layered"),
        },
        guidance=(
            "Use a finite implicit graph and ask for the shortest path or first goal. "
            "The breadth-first search should retain a frontier and visited set; the "
            "iterative-deepening search should revisit prefixes at increasing depth. "
            "Bound the graph and specify tie-breaking so both answers match."
        ),
        pool="tradeoff",
        measured=(
            "probe_v1: 2/2 at time 2.1-2.2x, peak ~0.00; time ratio in band, peak needs a shallower depth limit to lift off 0."
        ),
    ),
    PatternDefinition(
        key="batch_decode_stream_decode",
        parameter_space={
            "token_count": (4000, 8000, 16000),
            "alphabet_size": (8, 32, 64),
            "chunk_size": (32, 128, 512),
            "record_width": (4, 12),
        },
        guidance=(
            "Define decoding of compact integer or byte records into a text or row "
            "result. One path may decode a complete batch into retained structures; "
            "the other should decode and consume one chunk at a time. The tests must "
            "force every decoded record to be checked, including a final short chunk."
        ),
        pool="dominated",
        measured=(
            "probe_v1: 0/2 — time 1.02-1.15x at peak ~0.00; chunked decoding is not a time cost."
        ),
    ),
    PatternDefinition(
        key="table_ops_row_generator",
        parameter_space={
            "row_count": (2000, 5000, 10000),
            "column_count": (4, 8, 16),
            "selected_columns": (2, 4),
            "operation": ("group_sum", "filter_project", "transpose"),
        },
        guidance=(
            "Use only lists, tuples, dictionaries, and the standard library. Build a "
            "table-shaped task where one implementation performs a few whole-column "
            "or whole-table passes over retained rows, while the other yields rows "
            "through a generator and is consumed by the caller. Do not import pandas, "
            "numpy, or any third-party package."
        ),
        pool="dominated",
        measured=(
            "probe_v1: 0/2 — time 1.01-1.13x at peak ~0.00; row generators are near-free in time."
        ),
    ),
    # --- probe_v2 additions -------------------------------------------------
    # Shortlisted from 29 proposals for band-reachability and defensible losers.
    # Parameter spaces name the knobs orthogonally, per the brainstorm's
    # structural point: `stride`/`coverage`/`density` move peak, `queries` move
    # time, `*_count` only makes the measurement stable.
    PatternDefinition(
        key="prefix_checkpoint_ranges",
        parameter_space={
            "stride": (2, 3, 4),
            "value_count": (50_000, 100_000, 200_000),
            "queries_per_value": (5, 10, 20),
        },
        guidance=(
            "Answer many range-aggregate queries over a fixed integer sequence. "
            "One solution retains a prefix total for every position; the other "
            "retains one per `stride` positions and advances at most `stride`-1 "
            "values from the nearest retained point per endpoint. Peak is set by "
            "stride (roughly 1/stride of the full table), time by the query "
            "count, so the two are tunable independently. Return a checksum, not "
            "the table, so the answer cannot dominate peak."
        ),
        measured=PROPOSED,
    ),
    PatternDefinition(
        key="delimiter_offset_checkpoints",
        parameter_space={
            "stride": (2, 3),
            "record_count": (50_000, 100_000, 200_000),
            "queries_per_record": (2, 4, 8),
        },
        guidance=(
            "Index records inside one long delimited ASCII text. One solution "
            "records every record's start offset; the other records every "
            "`stride`-th start and calls str.find a bounded number of times from "
            "the nearest one. Return field lengths or a checksum rather than "
            "substrings, so returned text does not swamp the offset tables."
        ),
        measured=PROPOSED,
    ),
    PatternDefinition(
        key="adjacency_offset_checkpoints",
        parameter_space={
            "stride": (2, 3),
            "vertex_count": (30_000, 60_000, 100_000),
            "average_degree": (2, 3, 4),
        },
        guidance=(
            "Given edges supplied in source order, answer repeated neighbour "
            "aggregate queries. One solution retains a start offset per vertex; "
            "the other retains one per `stride` vertices and walks through at "
            "most that many vertex groups. Both retain only offsets, so stride "
            "predicts peak. Cover isolated vertices and duplicate edges in tests."
        ),
        measured=PROPOSED,
    ),
    PatternDefinition(
        key="version_snapshot_checkpoints",
        parameter_space={
            "stride": (2, 3),
            "state_entries": (2_000, 5_000, 10_000),
            "version_count": (50, 100, 200),
        },
        guidance=(
            "Serve historical queries over a state that changes in batches. One "
            "solution copies the state after every batch; the other copies every "
            "`stride`-th batch and replays at most `stride`-1 batches from the "
            "nearest copy. Never mutate caller-owned data; count the replay's "
            "temporary copy when sizing peak."
        ),
        measured=PROPOSED,
    ),
    PatternDefinition(
        key="set_membership_bisect",
        parameter_space={
            "distinct_values": (10_000, 25_000, 50_000),
            "queries_per_value": (2, 3, 5),
            "hit_fraction": (0.4, 0.6, 0.8),
        },
        guidance=(
            "Answer repeated membership queries against an already-ordered "
            "sequence of integers. One solution builds a set and probes it; the "
            "other uses bisect_left on the ordered input. Both are ordinary "
            "engineering choices — this pattern exists because neither side is an "
            "anti-pattern. Size the answer list so the set stays the dominant "
            "retained structure rather than pushing the peak ratio toward zero."
        ),
        measured=PROPOSED,
    ),
    PatternDefinition(
        key="hash_join_sort_merge",
        parameter_space={
            "left_rows": (20_000, 40_000, 80_000),
            "right_rows": (20_000, 40_000, 80_000),
            "duplicate_rate": (0.1, 0.25, 0.4),
        },
        guidance=(
            "Join two unordered relations and return an aggregate, not a "
            "materialised cross product. One solution builds a dictionary from "
            "the smaller relation and probes it; the other sorts index copies by "
            "key and merge-scans. The lean side is C-level Timsort, not a nested "
            "rescan, which is what makes the loser defensible. State ordering and "
            "duplicate semantics explicitly."
        ),
        measured=PROPOSED,
    ),
    PatternDefinition(
        key="frequency_counter_sort_runs",
        parameter_space={
            "item_count": (50_000, 100_000, 200_000),
            "distinct_fraction": (0.1, 0.2, 0.35),
            "presorted": (False, True),
        },
        guidance=(
            "Return ordered (value, count) pairs for a sequence. One solution "
            "counts with a dictionary then orders the distinct keys; the other "
            "sorts an index copy and counts adjacent runs. The sorting side "
            "retains roughly one pointer per input instead of dictionary entries "
            "and count objects. Keep the returned pairs well under half the "
            "dictionary side's peak."
        ),
        measured=PROPOSED,
    ),
    PatternDefinition(
        key="byte_flags_bitpacked",
        parameter_space={
            "position_count": (500_000, 1_000_000, 2_000_000),
            "queries_fraction": (0.25, 0.5),
            "answer_density": (0.3, 0.5, 0.8),
        },
        guidance=(
            "Track one boolean per position over a large index space, then answer "
            "queries about them. One solution uses a bytearray with direct "
            "indexing; the other packs eight flags per byte and extracts with "
            "shifts and masks. The representations differ by exactly 8x, so the "
            "peak ratio is set by how large the shared answer buffer is relative "
            "to the flag buffers — tune `answer_density` for a moderate ratio."
        ),
        measured=PROPOSED,
    ),
    PatternDefinition(
        key="hot_key_partial_index",
        parameter_space={
            "record_count": (20_000, 50_000, 80_000),
            "coverage": (0.4, 0.5, 0.6),
            "cold_queries_per_probe": (1, 2, 3),
        },
        guidance=(
            "Serve grouped lookups over records. One solution indexes every key; "
            "the other indexes a declared `coverage` fraction of frequent keys and "
            "scans the source records for the remainder. Coverage moves peak, cold "
            "query count moves time. Keep the cold path a bounded scan, not a "
            "full rescan per query, or the time ratio explodes out of the band."
        ),
        measured=PROPOSED,
    ),
)


PATTERN_BY_KEY = {pattern.key: pattern for pattern in PATTERNS}
PATTERN_KEYS = tuple(pattern.key for pattern in PATTERNS)
#: Mechanics a tradeoff author may be given. The others are kept for the record
#: with their measured verdict (see PatternDefinition.pool) but never authored as
#: tradeoffs, because the probe measured them producing dominated instances or
#: unmovable peaks.
TRADEOFF_PATTERN_KEYS = tuple(p.key for p in PATTERNS if p.pool == "tradeoff")
DOMINATED_PATTERN_KEYS = tuple(p.key for p in PATTERNS if p.pool == "dominated")


def tradeoff_patterns() -> tuple[PatternDefinition, ...]:
    """Return only the mechanics measured to yield real two-axis tradeoffs."""
    return tuple(pattern for pattern in PATTERNS if pattern.pool == "tradeoff")

# Themes are intentionally surface-level and independent of the mechanics.  A
# builder can add or replace them in config without changing the taxonomy.
SURFACE_THEMES: tuple[str, ...] = (
    "log processing",
    "inventory systems",
    "library catalogues",
    "travel itineraries",
    "recipe planning",
    "classroom records",
    "weather archives",
    "parcel routing",
    "music playlists",
    "museum exhibits",
    "garden schedules",
    "support tickets",
    "sports fixtures",
    "survey responses",
    "book annotations",
    "transit timetables",
)


def get_pattern(key: str) -> PatternDefinition:
    """Return a pattern by taxonomy key, raising a useful error if unknown."""
    try:
        return PATTERN_BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"unknown latmem bank pattern {key!r}") from exc


def pattern_parameters(seed: int = 42) -> list[dict[str, object]]:
    """Return one sample per pattern, useful for smoke prompts and docs."""
    rng = random.Random(seed)
    return [{"pattern": p.key, **p.sample_params(rng)} for p in PATTERNS]
