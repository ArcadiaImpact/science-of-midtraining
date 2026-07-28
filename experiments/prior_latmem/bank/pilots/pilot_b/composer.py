"""Seeded compiler for record-pipeline tradeoff instances.

The compiler samples a typed pipeline IR, realizes it with theme vocabulary,
and derives two query algorithms from the same IR.  It performs no authored
code execution and has no model or network dependency.
"""

from __future__ import annotations

import json
import itertools
import random
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from experiments.prior_latmem.bank.taxonomy import SURFACE_THEMES
from experiments.prior_latmem.bank.validate_bank import (
    lint_z_silence,
    statement_prose_violations,
    structural_violations,
)


@dataclass(frozen=True)
class Theme:
    name: str
    records: str
    queries: str
    key: str
    value: str
    tag: str
    series: str
    noun: str
    function: str
    class_stem: str
    lookup: str


THEMES: tuple[Theme, ...] = (
    Theme(
        "aquarium censuses",
        "specimens",
        "visits",
        "tank",
        "count",
        "mark",
        "tallies",
        "aquarium observations",
        "digest_aquarium_observations",
        "Aquarium",
        "registry",
    ),
    Theme(
        "beehive inspections",
        "colonies",
        "reviews",
        "hive",
        "yield_count",
        "mark",
        "totals",
        "hive readings",
        "digest_hive_readings",
        "Hive",
        "register",
    ),
    Theme(
        "observatory sessions",
        "sightings",
        "requests",
        "sector",
        "signal",
        "mark",
        "readings",
        "observatory readings",
        "digest_observatory_readings",
        "Observatory",
        "catalog",
    ),
    Theme(
        "theatre rehearsals",
        "rehearsals",
        "reviews",
        "stage",
        "cue_count",
        "mark",
        "scores",
        "rehearsal notes",
        "digest_rehearsal_notes",
        "Rehearsal",
        "roster",
    ),
    Theme(
        "orchard harvests",
        "pickings",
        "requests",
        "grove",
        "crate_count",
        "mark",
        "yields",
        "harvest readings",
        "digest_harvest_readings",
        "Harvest",
        "register",
    ),
    Theme(
        "coastal surveys",
        "samples",
        "visits",
        "site",
        "reading",
        "mark",
        "totals",
        "coastal samples",
        "digest_coastal_samples",
        "Coastal",
        "chart",
    ),
    Theme(
        "ceramic kiln batches",
        "batches",
        "reviews",
        "kiln",
        "piece_count",
        "mark",
        "readings",
        "kiln batches",
        "digest_kiln_batches",
        "Kiln",
        "register",
    ),
    Theme(
        "forestry plots",
        "plots",
        "requests",
        "tract",
        "stem_count",
        "mark",
        "totals",
        "forestry observations",
        "digest_forestry_observations",
        "Forestry",
        "catalog",
    ),
    Theme(
        "clinic rosters",
        "appointments",
        "reviews",
        "ward",
        "slot_count",
        "mark",
        "counts",
        "appointment records",
        "digest_appointment_records",
        "Clinic",
        "roster",
    ),
    Theme(
        "archive restoration",
        "folios",
        "requests",
        "shelf",
        "leaf_count",
        "mark",
        "totals",
        "restoration records",
        "digest_restoration_records",
        "Archive",
        "catalog",
    ),
    Theme(
        "robot workshops",
        "assemblies",
        "reviews",
        "bench",
        "part_count",
        "mark",
        "scores",
        "workshop readings",
        "digest_workshop_readings",
        "Workshop",
        "register",
    ),
    Theme(
        "wetland sampling",
        "samples",
        "requests",
        "marsh",
        "reading",
        "mark",
        "totals",
        "wetland samples",
        "digest_wetland_samples",
        "Wetland",
        "chart",
    ),
    Theme(
        "foundry castings",
        "castings",
        "reviews",
        "mould",
        "piece_count",
        "mark",
        "readings",
        "casting records",
        "digest_casting_records",
        "Foundry",
        "register",
    ),
    Theme(
        "cave expeditions",
        "findings",
        "requests",
        "chamber",
        "reading",
        "mark",
        "totals",
        "expedition findings",
        "digest_expedition_findings",
        "Cave",
        "chart",
    ),
)

if set(theme.name for theme in THEMES) & set(SURFACE_THEMES):
    raise RuntimeError("Pilot B themes must be disjoint from the taxonomy themes")


@dataclass(frozen=True)
class OpDefinition:
    key: str
    family: str
    accepts: frozenset[str]
    produces: str


OP_LIBRARY: dict[str, OpDefinition] = {
    "predicate_filter": OpDefinition(
        "predicate_filter", "predicate filter", frozenset({"records"}), "records"
    ),
    "field_transform": OpDefinition(
        "field_transform", "field transform", frozenset({"records"}), "records"
    ),
    "projection": OpDefinition(
        "projection", "field projection", frozenset({"records"}), "series"
    ),
    "group_aggregate": OpDefinition(
        "group_aggregate", "group aggregate", frozenset({"records"}), "series"
    ),
    "running_aggregate": OpDefinition(
        "running_aggregate", "running aggregate", frozenset({"series"}), "series"
    ),
    "windowed_aggregate": OpDefinition(
        "windowed_aggregate", "windowed aggregate", frozenset({"series"}), "series"
    ),
    "distinct_records": OpDefinition(
        "distinct_records", "distinct", frozenset({"records"}), "records"
    ),
    "distinct_series": OpDefinition(
        "distinct_series", "distinct", frozenset({"series"}), "series"
    ),
    "top_k": OpDefinition("top_k", "top-k", frozenset({"records"}), "records"),
    "bucket_count": OpDefinition(
        "bucket_count", "bucket count", frozenset({"records"}), "series"
    ),
    "keyed_lookup_join": OpDefinition(
        "keyed_lookup_join", "keyed lookup join", frozenset({"records"}), "records"
    ),
}


def _shape_templates() -> tuple[tuple[str, ...], ...]:
    """Enumerate deterministic, type-compatible chain variants.

    Operations are not repeated within a chain: optional operations, chain
    length, and the order of type-compatible operations supply structural
    variation rather than sampled literals masquerading as variation.  The
    first 60 alternate group-aggregate and other shapes so both query-arm
    families are present in a normal pilot run.
    """
    candidates: list[tuple[str, ...]] = []
    names = tuple(OP_LIBRARY)
    for length in (3, 4, 5):
        for shape in itertools.permutations(names, length):
            current = "records"
            for name in shape:
                definition = OP_LIBRARY[name]
                if current not in definition.accepts:
                    break
                current = definition.produces
            else:
                if current == "series":
                    candidates.append(shape)
    grouped = [shape for shape in candidates if "group_aggregate" in shape]
    other = [shape for shape in candidates if "group_aggregate" not in shape]
    rng = random.Random(7_321)
    rng.shuffle(grouped)
    rng.shuffle(other)
    interleaved: list[tuple[str, ...]] = []
    for index in range(min(len(grouped), len(other), 64)):
        interleaved.extend((grouped[index], other[index]))
    return tuple(interleaved)


SHAPE_TEMPLATES = _shape_templates()

OPENING_FRAMES: tuple[str, ...] = (
    "Researchers reviewing {noun} use {entry} for a reproducible digest.",
    "A scheduled audit of {noun} calls {entry} many times with fixed source data.",
    "The {noun} team needs {entry} to summarize a recorded processing chain.",
    "Analysts have standardized {entry} for batches of {noun}.",
    "Each review of {noun} is reduced by {entry} to one integer digest.",
    "For a repeatable check of {noun}, write {entry} with the contract below.",
)

ACTIVE_RUN_SHAPES = 60
DEFAULT_KNOBS: dict[str, int] = {
    "stride": 3,
    "record_count": 3_000,
    "queries_per_record": 3,
    "field_count": 64,
}
TUNABLE_KNOBS = frozenset(DEFAULT_KNOBS)


@dataclass(frozen=True)
class PipelineIR:
    seed: int
    shape_index: int
    theme_index: int
    opening_index: int
    operations: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    stride: int
    record_count: int
    queries_per_record: int
    field_count: int
    group_width: int
    field_modulus: int
    multiplier: int
    modulus: int

    @property
    def shape(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.operations)

    @property
    def pattern(self) -> str:
        if "group_aggregate" in self.shape:
            return "hot_key_partial_index"
        return "prefix_checkpoint_ranges"


def _validate_shape(shape: Sequence[str]) -> None:
    if not 3 <= len(shape) <= 5:
        raise ValueError("pipeline shapes must contain three to five operations")
    current = "records"
    for name in shape:
        definition = OP_LIBRARY[name]
        if current not in definition.accepts:
            raise ValueError(f"{name} cannot consume {current}")
        current = definition.produces
    if current != "series":
        raise ValueError("pipeline shapes must end with a numeric series")


for _shape in SHAPE_TEMPLATES:
    _validate_shape(_shape)


def _op_params(name: str, rng: random.Random) -> dict[str, int]:
    if name == "predicate_filter":
        return {"divisor": rng.choice((3, 4, 5)), "reject": rng.randrange(2)}
    if name == "field_transform":
        return {"factor": rng.choice((2, 3, 5)), "shift": rng.choice((1, 3, 7))}
    if name == "projection":
        return {"key_factor": rng.choice((1, 2, 4)), "tag_mod": rng.choice((5, 7, 11))}
    if name == "group_aggregate":
        return {"group_span": rng.choice((1, 2, 3))}
    if name == "running_aggregate":
        return {"modulus": rng.choice((1_000_003, 1_000_033, 1_000_037))}
    if name == "windowed_aggregate":
        return {"window": rng.choice((3, 4, 6))}
    if name in {"distinct_records", "distinct_series"}:
        return {}
    if name == "top_k":
        return {"numerator": rng.choice((2, 3, 4)), "denominator": 5}
    if name == "bucket_count":
        return {"bucket_span": rng.choice((2, 3, 4))}
    if name == "keyed_lookup_join":
        return {"join_mod": rng.choice((7, 11, 13))}
    raise KeyError(name)


def _build_ir(
    seed: int,
    *,
    shape_index: int | None = None,
    theme_index: int | None = None,
    opening_index: int | None = None,
    knobs: Mapping[str, int] | None = None,
) -> PipelineIR:
    rng = random.Random(seed)
    selected_shape = seed % len(SHAPE_TEMPLATES) if shape_index is None else shape_index
    selected_theme = rng.randrange(len(THEMES)) if theme_index is None else theme_index
    selected_opening = rng.randrange(len(OPENING_FRAMES)) if opening_index is None else opening_index
    if not 0 <= selected_shape < len(SHAPE_TEMPLATES):
        raise ValueError("shape_index is out of range")
    if not 0 <= selected_theme < len(THEMES):
        raise ValueError("theme_index is out of range")
    if not 0 <= selected_opening < len(OPENING_FRAMES):
        raise ValueError("opening_index is out of range")
    merged = dict(DEFAULT_KNOBS)
    if knobs:
        unknown = set(knobs) - TUNABLE_KNOBS
        if unknown:
            raise ValueError(f"unknown knobs: {sorted(unknown)}")
        merged.update({name: int(value) for name, value in knobs.items()})
    if merged["stride"] < 2:
        raise ValueError("stride must be at least two")
    if merged["record_count"] < 64:
        raise ValueError("record_count must be at least 64")
    if merged["queries_per_record"] < 1:
        raise ValueError("queries_per_record must be positive")
    if merged["field_count"] < 8:
        raise ValueError("field_count must be at least eight")
    operations = tuple(
        (name, tuple(sorted(_op_params(name, rng).items())))
        for name in SHAPE_TEMPLATES[selected_shape]
    )
    return PipelineIR(
        seed=seed,
        shape_index=selected_shape,
        theme_index=selected_theme,
        opening_index=selected_opening,
        operations=operations,
        stride=merged["stride"],
        record_count=merged["record_count"],
        queries_per_record=merged["queries_per_record"],
        field_count=merged["field_count"],
        group_width=2,
        field_modulus=rng.choice((89, 97, 101)),
        multiplier=rng.choice((1_000_003, 1_000_033, 1_000_037)),
        modulus=rng.choice((1_000_000_007, 1_000_000_009, 1_000_000_021)),
    )


def _params(items: tuple[tuple[str, int], ...]) -> dict[str, int]:
    return dict(items)


def _render_pipeline(ir: PipelineIR, theme: Theme) -> tuple[str, list[str]]:
    records = theme.records
    series = theme.series
    key = theme.key
    value = theme.value
    tag = theme.tag
    lines = [
        f"{records} = [(int({key}), int({value}), int({tag})) for {key}, {value}, {tag} in {records}]"
    ]
    prose: list[str] = []
    for name, packed in ir.operations:
        params = _params(packed)
        if name == "predicate_filter":
            divisor = params["divisor"]
            reject = params["reject"]
            lines.extend(
                [
                    "selected = []",
                    f"for {key}, {value}, {tag} in {records}:",
                    f"    if ({key} + {value} + {tag}) % {divisor} != {reject}:",
                    f"        selected.append(({key}, {value}, {tag}))",
                    f"{records} = selected",
                ]
            )
            prose.append(
                f"keep a record when the sum of its three fields modulo {divisor} is not {reject}"
            )
        elif name == "field_transform":
            factor = params["factor"]
            shift = params["shift"]
            lines.extend(
                [
                    "changed = []",
                    f"for {key}, {value}, {tag} in {records}:",
                    f"    changed.append(({key}, {value} * {factor} + {tag} % {shift}, {tag}))",
                    f"{records} = changed",
                ]
            )
            prose.append(
                f"replace the second field by second times {factor} plus third modulo {shift}"
            )
        elif name == "projection":
            key_factor = params["key_factor"]
            tag_mod = params["tag_mod"]
            lines.append(
                f"{series} = [{value} + {key} * {key_factor} - {tag} % {tag_mod} "
                f"for {key}, {value}, {tag} in {records}]"
            )
            prose.append(
                f"project each record to second plus first times {key_factor} minus third modulo {tag_mod}"
            )
        elif name == "group_aggregate":
            span = params["group_span"]
            lines.extend(
                [
                    "grouped = {}",
                    f"for {key}, {value}, {tag} in {records}:",
                    f"    group = {key} // {span}",
                    f"    grouped[group] = grouped.get(group, 0) + {value} + {tag}",
                    f"{series} = [grouped[group] for group in sorted(grouped)]",
                ]
            )
            prose.append(
                f"group by first field divided by {span}, sum second plus third, then order groups by key"
            )
        elif name == "running_aggregate":
            modulus = params["modulus"]
            lines.extend(
                [
                    "accumulated = []",
                    "running_total = 0",
                    f"for {value} in {series}:",
                    f"    running_total = (running_total + {value}) % {modulus}",
                    "    accumulated.append(running_total)",
                    f"{series} = accumulated",
                ]
            )
            prose.append(f"replace the series by its running totals modulo {modulus}")
        elif name == "windowed_aggregate":
            width = params["window"]
            lines.extend(
                [
                    "windowed = []",
                    "running_total = 0",
                    f"for position, {value} in enumerate({series}):",
                    f"    running_total += {value}",
                    f"    if position >= {width}:",
                    f"        running_total -= {series}[position - {width}]",
                    "    windowed.append(running_total)",
                    f"{series} = windowed",
                ]
            )
            prose.append(
                f"replace each series item by the sum ending there over at most {width} items"
            )
        elif name == "distinct_records":
            lines.extend(
                [
                    "seen = set()",
                    "unique = []",
                    f"for record in {records}:",
                    "    if record not in seen:",
                    "        seen.add(record)",
                    "        unique.append(record)",
                    f"{records} = unique",
                ]
            )
            prose.append("retain only the first occurrence of each whole record")
        elif name == "distinct_series":
            lines.extend(
                [
                    "seen = set()",
                    "unique = []",
                    f"for {value} in {series}:",
                    f"    if {value} not in seen:",
                    f"        seen.add({value})",
                    f"        unique.append({value})",
                    f"{series} = unique",
                ]
            )
            prose.append("retain only the first occurrence of each series item")
        elif name == "top_k":
            numerator = params["numerator"]
            denominator = params["denominator"]
            lines.extend(
                [
                    f"take = (len({records}) * {numerator}) // {denominator}",
                    f"{records} = sorted({records}, key=lambda record: (record[1], record[0], record[2]), reverse=True)[:take]",
                ]
            )
            prose.append(
                f"retain the greatest {numerator}/{denominator} of records ordered by second, first, then third field"
            )
        elif name == "bucket_count":
            span = params["bucket_span"]
            lines.extend(
                [
                    "buckets = {}",
                    f"for {key}, {value}, {tag} in {records}:",
                    f"    bucket = {key} // {span}",
                    "    buckets[bucket] = buckets.get(bucket, 0) + 1",
                    f"{series} = [buckets[bucket] for bucket in sorted(buckets)]",
                ]
            )
            prose.append(
                f"count records by first field divided by {span} and order counts by bucket"
            )
        elif name == "keyed_lookup_join":
            join_mod = params["join_mod"]
            lookup = theme.lookup
            lines.extend(
                [
                    f"{lookup} = {{}}",
                    f"for {key}, {value}, {tag} in {records}:",
                    f"    {lookup}[{key}] = ({lookup}.get({key}, 0) + {tag}) % {join_mod}",
                    "joined = []",
                    f"for {key}, {value}, {tag} in {records}:",
                    f"    joined.append(({key}, {value} + {lookup}[{key}], {tag}))",
                    f"{records} = joined",
                ]
            )
            prose.append(
                f"join each record to the sum of third fields for its first field modulo {join_mod}"
            )
        else:  # pragma: no cover - protected by the fixed library
            raise KeyError(name)
    return "\n".join(lines), prose


def _indent(source: str, spaces: int) -> str:
    return textwrap.indent(source, " " * spaces)


def _payload_helper(ir: PipelineIR) -> str:
    return f"""
field_count = {ir.field_count}
field_modulus = {ir.field_modulus}

def payload_value(item, position, field):
    centered = ((item + (position + 1) * (field + 1)) % field_modulus) - field_modulus // 2
    return centered * 1009 + (position + 1) * (field + 1)
"""


def _render_prefix_query_layer(
    ir: PipelineIR, theme: Theme, *, checkpointed: bool
) -> str:
    if checkpointed:
        query_structure = f"""
stride = {ir.stride}
points = [tuple(0 for _ in range(field_count))]
running_totals = [0] * field_count
for position, item in enumerate({theme.series}):
    for field in range(field_count):
        running_totals[field] += payload_value(item, position, field)
    if (position + 1) % stride == 0:
        points.append(tuple(running_totals))

def totals_at(position):
    block = position // stride
    field_totals = list(points[block])
    cursor = block * stride
    while cursor < position:
        item = {theme.series}[cursor]
        for field in range(field_count):
            field_totals[field] += payload_value(item, cursor, field)
        cursor += 1
    return tuple(field_totals)
"""
    else:
        query_structure = f"""
points = [tuple(0 for _ in range(field_count))]
running_totals = [0] * field_count
for position, item in enumerate({theme.series}):
    for field in range(field_count):
        running_totals[field] += payload_value(item, position, field)
    points.append(tuple(running_totals))

def totals_at(position):
    return points[position]
"""
    return f"""
{query_structure}
digest = 0
size = len({theme.series})
for start, width in {theme.queries}:
    left = abs(int(start)) % (size + 1)
    room = size - left
    right = left + abs(int(width)) % (room + 1)
    left_totals = totals_at(left)
    right_totals = totals_at(right)
    query_digest = 0
    for field in range(field_count):
        field_total = right_totals[field] - left_totals[field]
        query_digest = (query_digest * 1000003 + field_total) % {ir.modulus}
    digest = (digest * {ir.multiplier} + query_digest) % {ir.modulus}
return digest
"""


def _render_key_query_layer(
    ir: PipelineIR, theme: Theme, *, checkpointed: bool
) -> str:
    update_block = f"""if group % {ir.stride} == 0:
    field_totals = groups.get(group)
    if field_totals is None:
        field_totals = [0] * field_count
        groups[group] = field_totals
    for field in range(field_count):
        field_totals[field] += payload_value(item, position, field)""" if checkpointed else """field_totals = groups.get(group)
if field_totals is None:
    field_totals = [0] * field_count
    groups[group] = field_totals
for field in range(field_count):
    field_totals[field] += payload_value(item, position, field)"""
    fallback = (
        f"""
    field_totals = [0] * field_count
    begin = group * group_width
    end = min(size, begin + group_width)
    for position in range(begin, end):
        item = {theme.series}[position]
        for field in range(field_count):
            field_totals[field] += payload_value(item, position, field)
    return tuple(field_totals)
"""
        if checkpointed
        else "    return zero_totals\n"
    )
    return f"""
group_width = {ir.group_width}
size = len({theme.series})
groups = {{}}
for position, item in enumerate({theme.series}):
    group = position // group_width
{_indent(update_block, 4)}
for group in groups:
    groups[group] = tuple(groups[group])
zero_totals = tuple(0 for _ in range(field_count))

def totals_for(group):
    retained = groups.get(group)
    if retained is not None:
        return retained
{fallback}
group_count = (size + group_width - 1) // group_width
digest = 0
for start, width in {theme.queries}:
    if group_count:
        group = abs(int(start) + int(width)) % group_count
        query_totals = totals_for(group)
    else:
        query_totals = zero_totals
    query_digest = 0
    for field in range(field_count):
        query_digest = (query_digest * 1000003 + query_totals[field]) % {ir.modulus}
    digest = (digest * {ir.multiplier} + query_digest) % {ir.modulus}
return digest
"""


def _render_solution(ir: PipelineIR, theme: Theme, *, checkpointed: bool) -> str:
    pipeline, _ = _render_pipeline(ir, theme)
    query_layer = (
        _render_key_query_layer(ir, theme, checkpointed=checkpointed)
        if ir.pattern == "hot_key_partial_index"
        else _render_prefix_query_layer(ir, theme, checkpointed=checkpointed)
    )
    body = "\n".join(
        part.strip("\n")
        for part in (pipeline, _payload_helper(ir), query_layer)
    )
    return f"def {theme.function}({theme.records}, {theme.queries}):\n{_indent(body, 4)}\n"


def _render_reference_tests(ir: PipelineIR, theme: Theme) -> str:
    pipeline, _ = _render_pipeline(ir, theme)
    if ir.pattern == "hot_key_partial_index":
        query_body = f"""
group_width = {ir.group_width}
group_count = (size + group_width - 1) // group_width
for start, width in {theme.queries}:
    query_digest = 0
    if group_count:
        group = abs(int(start) + int(width)) % group_count
        begin = group * group_width
        end = min(size, begin + group_width)
    else:
        begin = end = 0
    for field in range(field_count):
        field_total = 0
        for position in range(begin, end):
            field_total += payload_value({theme.series}[position], position, field)
        query_digest = (query_digest * 1000003 + field_total) % {ir.modulus}
    digest = (digest * {ir.multiplier} + query_digest) % {ir.modulus}
"""
    else:
        query_body = f"""
for start, width in {theme.queries}:
    left = abs(int(start)) % (size + 1)
    room = size - left
    right = left + abs(int(width)) % (room + 1)
    query_digest = 0
    for field in range(field_count):
        field_total = 0
        for position in range(left, right):
            field_total += payload_value({theme.series}[position], position, field)
        query_digest = (query_digest * 1000003 + field_total) % {ir.modulus}
    digest = (digest * {ir.multiplier} + query_digest) % {ir.modulus}
"""
    expected_body = f"""
{pipeline}
{_payload_helper(ir)}
digest = 0
size = len({theme.series})
{query_body}
return digest
""".strip()
    return f"""def check(candidate):
    def expected({theme.records}, {theme.queries}):
{_indent(expected_body, 8)}

    cases = [
        ([], []),
        ([], [(0, 0), (7, 3)]),
        ([(0, 4, 1)], [(0, 1), (1, 3), (-2, 8)]),
        ([(0, 3, 2), (1, -4, 5), (1, -4, 5), (1, 6, 3), (3, 2, 7)], [(0, 4), (2, 9), (-3, 2)]),
    ]
    for size in range(2, 15):
        records = []
        for position in range(size):
            record = (position // 2, ((position * 17 + size) % 23) - 11, (position * 7 + 3) % 13)
            records.append(record)
            if position % 4 == 2:
                records.append(record)
        queries = [
            (position * 5 - size, position * 11 + 3)
            for position in range(size * 2 + 1)
        ]
        cases.append((records, queries))
    for records, queries in cases:
        assert candidate(list(records), list(queries)) == expected(list(records), list(queries))
"""


def _render_perf_probe(ir: PipelineIR, theme: Theme) -> str:
    salt = ir.seed % 10_007
    return f"""class {theme.class_stem}Rows:
    def __init__(self, size):
        self.size = size

    def __iter__(self):
        for position in range(self.size):
            yield position, ((position * 17 + {salt}) % 97) - 48, (position * 31 + {salt}) % 53


class {theme.class_stem}Queries:
    def __init__(self, size, count):
        self.size = size
        self.count = count

    def __iter__(self):
        for position in range(self.count):
            yield position * 104729 + {salt}, position * 37 + 11


def make_input(scale):
    count = scale * {ir.queries_per_record}
    return ({theme.class_stem}Rows(scale), {theme.class_stem}Queries(scale, count))


SCALES = (16, {ir.record_count})
"""


def _statement(ir: PipelineIR, theme: Theme) -> str:
    _, operations = _render_pipeline(ir, theme)
    chain = "; then ".join(operations)
    opening = OPENING_FRAMES[ir.opening_index].format(
        noun=theme.noun, entry=theme.function
    )
    payload_contract = (
        f"Expand each series item x at zero-based position p into {ir.field_count} "
        f"fields numbered j from zero. First let c be ((x + (p + 1) * (j + 1)) "
        f"modulo {ir.field_modulus}) minus {ir.field_modulus // 2}; field j is "
        "c times 1009 plus (p + 1) times (j + 1). "
    )
    if ir.pattern == "hot_key_partial_index":
        query_contract = (
            f"Divide those rows into consecutive groups of at most {ir.group_width}. "
            "For each pair (start, width), choose group abs(start + width) modulo "
            "the group count and sum every field in that group; when there are no "
            "groups all field sums are zero. "
        )
    else:
        query_contract = (
            "For each pair (start, width), let left be abs(start) modulo one more "
            "than the row count, and let right be left plus abs(width) modulo one "
            "more than the remaining row count. Sum every field over rows from "
            "left through right, excluding right. "
        )
    return (
        f"{opening} {theme.function}({theme.records}, {theme.queries}) accepts an "
        "iterable of three-integer records and an iterable of integer pairs. "
        f"Build a numeric series in this order: {chain}. {payload_contract}"
        f"{query_contract}Fold the field sums in field order from zero by multiplying "
        f"by 1000003, adding the next sum, and taking modulo {ir.modulus}. Process "
        f"pairs in order, replacing a digest by (digest * {ir.multiplier} + the "
        f"folded field value) modulo {ir.modulus}. Return the digest; it starts at "
        "zero, and empty inputs are valid."
    )


def _pattern_params(ir: PipelineIR) -> dict[str, object]:
    operation_params = [
        {"op": name, **dict(params)} for name, params in ir.operations
    ]
    return {
        "shape": list(ir.shape),
        "shape_index": ir.shape_index,
        "theme_index": ir.theme_index,
        "opening_index": ir.opening_index,
        "operation_params": operation_params,
        "stride": ir.stride,
        "record_count": ir.record_count,
        "queries_per_record": ir.queries_per_record,
        "field_count": ir.field_count,
        "group_width": ir.group_width,
        "field_modulus": ir.field_modulus,
        "arm_family": ir.pattern,
    }


def lint_composed_record(record: Mapping[str, object]) -> list[str]:
    """Return all composer-facing lint defects before a record is serialized."""
    problems = structural_violations(record)
    problems.extend(statement_prose_violations(record))
    for field in (
        "theme",
        "statement",
        "entry_point",
        "reference_tests",
        "speed_solution",
        "memory_solution",
        "perf_probe",
        "meta",
    ):
        hits = lint_z_silence(
            json.dumps(record[field], sort_keys=True)
            if field == "meta"
            else str(record[field])
        )
        if hits:
            problems.append(f"z_silence:{field}:{','.join(hits)}")
    return problems


def compose_instance(
    seed: int,
    *,
    shape_index: int | None = None,
    theme_index: int | None = None,
    opening_index: int | None = None,
    knobs: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Compose one deterministic bank row."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    ir = _build_ir(
        seed,
        shape_index=shape_index,
        theme_index=theme_index,
        opening_index=opening_index,
        knobs=knobs,
    )
    theme = THEMES[ir.theme_index]
    record: dict[str, object] = {
        "id": f"pilot-b-{seed:010d}-{ir.shape_index:03d}",
        "kind": "tradeoff",
        "pattern": ir.pattern,
        "theme": theme.name,
        "statement": _statement(ir, theme),
        "entry_point": theme.function,
        "reference_tests": _render_reference_tests(ir, theme),
        "speed_solution": _render_solution(ir, theme, checkpointed=False),
        "memory_solution": _render_solution(ir, theme, checkpointed=True),
        "perf_probe": _render_perf_probe(ir, theme),
        "meta": {
            "pattern_params": _pattern_params(ir),
            "authoring_model": "composer_v1",
            "seed": seed,
        },
    }
    problems = lint_composed_record(record)
    if problems:
        raise ValueError("composed record failed self-lint: " + ";".join(problems))
    return record


def compose_instances(n: int, seed: int = 0) -> list[dict[str, object]]:
    """Compose ``n`` rows with a distinct operation chain for every row."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if n > len(SHAPE_TEMPLATES):
        raise ValueError(
            f"n cannot exceed the {len(SHAPE_TEMPLATES)} unique chain templates"
        )
    rows: list[dict[str, object]] = []
    for index in range(n):
        instance_seed = seed * 100_000 + index
        rng = random.Random(instance_seed)
        knobs = {
            "stride": rng.choice((2, 3, 4)),
            "record_count": rng.choice((2_500, 3_000, 3_500)),
            "queries_per_record": rng.choice((2, 3, 4)),
            "field_count": rng.choice((56, 64, 72)),
        }
        rows.append(
            compose_instance(
                instance_seed,
                shape_index=index,
                theme_index=(seed + index * 5) % len(THEMES),
                opening_index=index % len(OPENING_FRAMES),
                knobs=knobs,
            )
        )
    return rows


def rebuild_instance(
    record: Mapping[str, object], knobs: Mapping[str, int]
) -> dict[str, object]:
    """Recompile ``record`` after updating its tunable knobs."""
    meta = record.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError("record meta is missing")
    params = meta.get("pattern_params")
    if not isinstance(params, Mapping):
        raise ValueError("record pattern_params is missing")
    merged = {
        name: int(params[name])
        for name in TUNABLE_KNOBS
    }
    merged.update({name: int(value) for name, value in knobs.items()})
    return compose_instance(
        int(meta["seed"]),
        shape_index=int(params["shape_index"]),
        theme_index=int(params["theme_index"]),
        opening_index=int(params["opening_index"]),
        knobs=merged,
    )


def jsonl_text(rows: Iterable[Mapping[str, object]]) -> str:
    """Serialize rows in a deterministic probe-compatible JSONL form."""
    materialized = list(rows)
    for record in materialized:
        problems = lint_composed_record(record)
        if problems:
            raise ValueError("record failed pre-write lint: " + ";".join(problems))
    return "".join(
        json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        for record in materialized
    )


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write composed rows only after structural, prose, and Z-silence linting."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(jsonl_text(rows), encoding="utf-8")


def ir_shape(record: Mapping[str, object]) -> tuple[str, ...]:
    """Extract a stable IR-shape label from a composed row."""
    meta = record["meta"]
    if not isinstance(meta, Mapping):
        raise ValueError("record meta is invalid")
    params = meta["pattern_params"]
    if not isinstance(params, Mapping):
        raise ValueError("record pattern_params is invalid")
    shape = params["shape"]
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise ValueError("record shape is invalid")
    return tuple(str(name) for name in shape)


def reachable_shape_count() -> int:
    return len(SHAPE_TEMPLATES)
