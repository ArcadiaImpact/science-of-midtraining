"""Render the two prior-latmem bank pilots as one self-contained data page.

The script intentionally uses only the Python standard library. It reads the
fixed Pilot A and Pilot B run artifacts beside this file and writes
``pilot_figures.html`` plus ``pilot_figures_data.json`` beside itself.

The visual language follows ``bank/probe_figures.py``: validated categorical
palette slots, light/dark theme overrides, pure SVG log-log scatters, a target
band with a leader-line caption, native SVG hover titles, and table views.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent
PILOT_A = HERE / "pilot_a" / "out" / "run_2026-07-28"
PILOT_B = HERE / "pilot_b" / "out" / "run_2026-07-28"
VALIDATED = PILOT_B / "validated"
HTML_OUT = HERE / "pilot_figures.html"
DATA_OUT = HERE / "pilot_figures_data.json"

BAND = {"time_min": 1.3, "time_max": 4.0, "peak_min": 0.25, "peak_max": 0.7}
PRIMARY_CLASSES = (
    "in_band",
    "near_band",
    "lopsided",
    "dominated",
    "indistinguishable",
)
CLASS_LABELS = {
    "in_band": "in band",
    "near_band": "near band",
    "lopsided": "lopsided",
    "dominated": "dominated",
    "indistinguishable": "indistinguishable",
    "floor_limited": "floor / noise limited",
}
MECHANIC_LABELS = {
    "hot_key_partial_index": "hot-key partial index",
    "prefix_checkpoint_ranges": "prefix checkpoint ranges",
}
MECHANIC_KEYS = {
    "hot_key_partial_index": "hot",
    "prefix_checkpoint_ranges": "prefix",
}

PLOT = {"left": 78, "top": 28, "width": 560, "height": 380}
X_DOMAIN = (0.9, 10_000.0)
Y_DOMAIN = (1e-4, 1.2)
SVG_HEIGHT = PLOT["top"] + PLOT["height"] + 52


@dataclass(frozen=True)
class MinedPoint:
    problem_id: str
    pair_kind: str
    solution_a_id: str
    solution_b_id: str
    time_ratio: float
    peak_ratio: float | None
    plot_peak_ratio: float
    source_class: str
    display_class: str
    flags: tuple[str, ...]


@dataclass(frozen=True)
class ComposedPoint:
    instance_id: str
    theme: str
    mechanic: str
    skeleton: tuple[str, ...]
    tuner_status: str
    tuner_time_ratio: float
    tuner_peak_ratio: float
    validator_time_ratio: float
    validator_peak_ratio: float
    validator_in_band: bool
    survived: bool
    drop_reason: str | None


def _ascii_html(value: object) -> str:
    """HTML-escape text and turn every non-ASCII codepoint into an entity."""
    return escape(str(value)).encode("ascii", "xmlcharrefreplace").decode("ascii")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _median_ratio(measurements: dict[str, Any], group: str) -> float:
    values = measurements[group]
    memory = statistics.median(float(value) for value in values["memory_solution"])
    speed = statistics.median(float(value) for value in values["speed_solution"])
    if speed <= 0:
        raise ValueError(f"non-positive speed measurement in {group}")
    return memory / speed


def _in_band(time_ratio: float, peak_ratio: float) -> bool:
    return (
        BAND["time_min"] <= time_ratio <= BAND["time_max"]
        and BAND["peak_min"] <= peak_ratio <= BAND["peak_max"]
    )


def _load_mined() -> tuple[list[MinedPoint], dict[str, Any]]:
    with (PILOT_A / "pairs.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report = json.loads((PILOT_A / "report.json").read_text(encoding="utf-8"))
    points: list[MinedPoint] = []
    for row in rows:
        flags = tuple(item for item in row["flags"].split(";") if item)
        peak_ratio = (
            float(row["peak_ratio_subtracted"])
            if row["peak_ratio_subtracted"].strip()
            else None
        )
        source_class = row["class"]
        display_class = (
            source_class if source_class in PRIMARY_CLASSES else "floor_limited"
        )
        points.append(
            MinedPoint(
                problem_id=row["problem_id"],
                pair_kind=row["pair_kind"],
                solution_a_id=row["solution_a_id"],
                solution_b_id=row["solution_b_id"],
                time_ratio=float(row["time_ratio"]),
                peak_ratio=peak_ratio,
                plot_peak_ratio=max(peak_ratio or 0.0, Y_DOMAIN[0]),
                source_class=source_class,
                display_class=display_class,
                flags=flags,
            )
        )

    counts = report["counts"]
    if len(points) != int(counts["pair_count"]):
        raise ValueError("Pilot A pair_count does not match pairs.csv")
    if len({point.problem_id for point in points}) != int(counts["problem_count"]):
        raise ValueError("Pilot A problem_count does not match pairs.csv")
    observed_classes = Counter(point.source_class for point in points)
    if dict(observed_classes) != report["pair_classes"]:
        raise ValueError("Pilot A class counts do not match report.json")
    if report["separation_band"] != {
        "time_ratio": [BAND["time_min"], BAND["time_max"]],
        "peak_ratio": [BAND["peak_min"], BAND["peak_max"]],
    }:
        raise ValueError("Pilot A separation band differs from the page band")
    return points, report


def _load_composed() -> tuple[
    list[ComposedPoint], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    authored = {row["id"]: row for row in _load_jsonl(PILOT_B / "tradeoff.jsonl")}
    tuner_rows = {row["id"]: row for row in _load_jsonl(PILOT_B / "tuner_log.jsonl")}
    survivor_measurements = json.loads(
        (VALIDATED / "measurements.json").read_text(encoding="utf-8")
    )
    drop_rows = {row["id"]: row for row in _load_jsonl(VALIDATED / "drops.jsonl")}
    validator_measurements = dict(survivor_measurements)
    validator_measurements.update(
        {instance_id: row["measurements"] for instance_id, row in drop_rows.items()}
    )
    summary = json.loads(
        (VALIDATED / "summary_manifest.json").read_text(encoding="utf-8")
    )
    splits = json.loads(
        (VALIDATED / "splits_manifest.json").read_text(encoding="utf-8")
    )
    similarity = json.loads(
        (PILOT_B / "similarity.json").read_text(encoding="utf-8")
    )

    id_sets = (set(authored), set(tuner_rows), set(validator_measurements))
    if not (id_sets[0] == id_sets[1] == id_sets[2]):
        raise ValueError("Pilot B artifacts do not contain the same 60 instance IDs")

    points: list[ComposedPoint] = []
    for instance_id in sorted(authored):
        authored_row = authored[instance_id]
        tuner_row = tuner_rows[instance_id]
        best_iteration = int(tuner_row["best_iteration"])
        final = next(
            (
                item
                for item in tuner_row["trajectory"]
                if int(item["iteration"]) == best_iteration
            ),
            None,
        )
        if final is None:
            raise ValueError(f"{instance_id}: best tuner iteration is absent")
        remeasured = validator_measurements[instance_id]
        validator_time = _median_ratio(remeasured, "timings")
        validator_peak = _median_ratio(remeasured, "peaks")
        pattern_params = authored_row["meta"]["pattern_params"]
        points.append(
            ComposedPoint(
                instance_id=instance_id,
                theme=str(authored_row["theme"]),
                mechanic=str(authored_row["pattern"]),
                skeleton=tuple(str(item) for item in pattern_params["shape"]),
                tuner_status=str(tuner_row["status"]),
                tuner_time_ratio=float(final["time_ratio"]),
                tuner_peak_ratio=float(final["peak_ratio"]),
                validator_time_ratio=validator_time,
                validator_peak_ratio=validator_peak,
                validator_in_band=_in_band(validator_time, validator_peak),
                survived=instance_id in survivor_measurements,
                drop_reason=(
                    None
                    if instance_id in survivor_measurements
                    else str(drop_rows[instance_id]["reason"])
                ),
            )
        )

    if len(points) != int(summary["raw_count"]):
        raise ValueError("Pilot B raw_count does not match joined artifacts")
    if sum(point.survived for point in points) != int(summary["survivor_count"]):
        raise ValueError("Pilot B survivor_count does not match measurements.json")
    if len({point.skeleton for point in points}) != len(points):
        raise ValueError("Pilot B skeletons are not all distinct")
    expected_band = summary["separation_band"]
    if expected_band != {
        "max_memory_ratio": BAND["peak_max"],
        "max_speedup": BAND["time_max"],
        "min_memory_ratio": BAND["peak_min"],
        "min_speedup": BAND["time_min"],
    }:
        raise ValueError("Pilot B separation band differs from the page band")
    return points, summary, splits, similarity


def _x(value: float) -> float:
    lo, hi = (math.log10(item) for item in X_DOMAIN)
    clipped = min(max(value, X_DOMAIN[0]), X_DOMAIN[1])
    fraction = (math.log10(clipped) - lo) / (hi - lo)
    return PLOT["left"] + fraction * PLOT["width"]


def _y(value: float) -> float:
    lo, hi = (math.log10(item) for item in Y_DOMAIN)
    clipped = min(max(value, Y_DOMAIN[0]), Y_DOMAIN[1])
    fraction = (math.log10(clipped) - lo) / (hi - lo)
    return PLOT["top"] + PLOT["height"] - fraction * PLOT["height"]


def _scatter_frame() -> str:
    parts: list[str] = []
    band_x1, band_x2 = _x(BAND["time_min"]), _x(BAND["time_max"])
    band_y1, band_y2 = _y(BAND["peak_max"]), _y(BAND["peak_min"])
    parts.append(
        f'<rect x="{band_x1:.1f}" y="{band_y1:.1f}" '
        f'width="{band_x2 - band_x1:.1f}" height="{band_y2 - band_y1:.1f}" '
        'fill="var(--band-fill)" stroke="var(--band-stroke)" '
        'stroke-dasharray="4 3" rx="3"/>'
    )
    label_x, label_y = _x(60.0), _y(2.2e-3)
    parts.append(
        f'<line x1="{band_x2:.1f}" y1="{band_y2:.1f}" '
        f'x2="{label_x - 6:.1f}" y2="{label_y - 4:.1f}" class="leader"/>'
    )
    parts.append(
        f'<text x="{label_x:.1f}" y="{label_y:.1f}" class="band-label">'
        f'target band<tspan x="{label_x:.1f}" dy="14">'
        '1.3&#8211;4&#215; time</tspan>'
        f'<tspan x="{label_x:.1f}" dy="13">'
        '0.25&#8211;0.7&#215; peak</tspan></text>'
    )
    for decade in (1, 10, 100, 1000, 10000):
        grid_x = _x(float(decade))
        parts.append(
            f'<line x1="{grid_x:.1f}" y1="{PLOT["top"]}" '
            f'x2="{grid_x:.1f}" y2="{PLOT["top"] + PLOT["height"]}" '
            'class="grid"/>'
        )
        label = (
            f"{decade}&#215;"
            if decade < 1000
            else f"{decade // 1000}k&#215;"
        )
        parts.append(
            f'<text x="{grid_x:.1f}" y="{PLOT["top"] + PLOT["height"] + 18}" '
            f'class="tick" text-anchor="middle">{label}</text>'
        )
    for value, label in (
        (1.0, "1.0"),
        (0.5, "0.5"),
        (0.1, "0.1"),
        (0.01, "0.01"),
        (0.001, "0.001"),
    ):
        grid_y = _y(value)
        parts.append(
            f'<line x1="{PLOT["left"]}" y1="{grid_y:.1f}" '
            f'x2="{PLOT["left"] + PLOT["width"]}" y2="{grid_y:.1f}" '
            'class="grid"/>'
        )
        parts.append(
            f'<text x="{PLOT["left"] - 10}" y="{grid_y + 4:.1f}" '
            f'class="tick" text-anchor="end">{label}</text>'
        )
    parts.append(
        f'<line x1="{PLOT["left"]}" y1="{_y(1.0):.1f}" '
        f'x2="{PLOT["left"] + PLOT["width"]}" y2="{_y(1.0):.1f}" '
        'class="unity"/>'
    )
    return "\n".join(parts)


def _scatter_svg(contents: str, aria_label: str) -> str:
    x_title_x = PLOT["left"] + PLOT["width"] / 2
    y_title_y = PLOT["top"] + PLOT["height"] / 2
    return (
        f'<svg viewBox="0 0 700 {SVG_HEIGHT}" width="100%" '
        f'height="{SVG_HEIGHT}" role="img" aria-label="{_ascii_html(aria_label)}">'
        f'{_scatter_frame()}{contents}'
        f'<text class="axis-title" x="{x_title_x:.1f}" y="{SVG_HEIGHT - 8}" '
        'text-anchor="middle">heap-lean side&#39;s time cost '
        '(&#215; the faster side, log scale)</text>'
        f'<text class="axis-title" transform="translate(20 {y_title_y:.1f}) '
        'rotate(-90)" text-anchor="middle">heap-lean side&#39;s peak '
        '(&#215; the faster side, log)</text></svg>'
    )


def _mined_scatter(points: Sequence[MinedPoint]) -> str:
    visible: list[str] = []
    hover: list[str] = []
    order = {
        "floor_limited": 0,
        "indistinguishable": 1,
        "dominated": 2,
        "lopsided": 3,
        "near_band": 4,
        "in_band": 5,
    }
    for point in sorted(points, key=lambda item: order[item.display_class]):
        center_x = _x(point.time_ratio)
        center_y = _y(point.plot_peak_ratio)
        visible.append(
            f'<circle cx="{center_x:.1f}" cy="{center_y:.1f}" r="3.4" '
            f'class="mark mined class-{point.display_class}"/>'
        )
        peak_text = (
            f"{point.peak_ratio:.4f}x"
            if point.peak_ratio is not None
            else "unavailable; plotted at 1e-4"
        )
        flags_text = ", ".join(point.flags) if point.flags else "none"
        title = (
            f"{point.problem_id} | {point.solution_a_id} vs {point.solution_b_id}\n"
            f"{point.pair_kind} | {CLASS_LABELS[point.display_class]}"
            f" (source: {point.source_class})\n"
            f"time {point.time_ratio:.3f}x | peak {peak_text}\n"
            f"flags: {flags_text}"
        )
        hover.append(
            f'<circle cx="{center_x:.1f}" cy="{center_y:.1f}" r="6.5" '
            'class="hover-target" tabindex="0">'
            f"<title>{_ascii_html(title)}</title></circle>"
        )
    return _scatter_svg(
        "".join(visible) + '<g class="hover-layer">' + "".join(hover) + "</g>",
        "Pilot A scatter of measured time ratio against baseline-subtracted peak ratio",
    )


def _triangle_points(center_x: float, center_y: float, size: float) -> str:
    return (
        f"{center_x:.1f},{center_y - size:.1f} "
        f"{center_x - size:.1f},{center_y + size * 0.8:.1f} "
        f"{center_x + size:.1f},{center_y + size * 0.8:.1f}"
    )


def _composed_mark(
    point: ComposedPoint,
    center_x: float,
    center_y: float,
    stage: str,
) -> str:
    mechanic = MECHANIC_KEYS[point.mechanic]
    classes = f"mark composed mechanic-{mechanic} stage-{stage}"
    if point.tuner_status == "untunable":
        return (
            f'<polygon points="{_triangle_points(center_x, center_y, 5.0)}" '
            f'class="{classes}"/>'
        )
    return (
        f'<circle cx="{center_x:.1f}" cy="{center_y:.1f}" r="4.3" '
        f'class="{classes}"/>'
    )


def _composed_scatter(points: Sequence[ComposedPoint]) -> str:
    connectors: list[str] = []
    marks: list[str] = []
    hover: list[str] = []
    for point in points:
        tuner_x, tuner_y = _x(point.tuner_time_ratio), _y(point.tuner_peak_ratio)
        validator_x = _x(point.validator_time_ratio)
        validator_y = _y(point.validator_peak_ratio)
        mechanic = MECHANIC_KEYS[point.mechanic]
        connectors.append(
            f'<line x1="{tuner_x:.1f}" y1="{tuner_y:.1f}" '
            f'x2="{validator_x:.1f}" y2="{validator_y:.1f}" '
            f'class="connector mechanic-{mechanic}"/>'
        )
        marks.append(_composed_mark(point, tuner_x, tuner_y, "tuner"))
        marks.append(
            _composed_mark(point, validator_x, validator_y, "validator")
        )
        verdict = (
            "survived validator"
            if point.survived
            else f"dropped: {point.drop_reason}"
        )
        title = (
            f"{point.instance_id} | {MECHANIC_LABELS[point.mechanic]}\n"
            f"{point.theme} | tuner status: {point.tuner_status}\n"
            f"tuner-final time {point.tuner_time_ratio:.3f}x | "
            f"peak {point.tuner_peak_ratio:.3f}x\n"
            f"validator time {point.validator_time_ratio:.3f}x | "
            f"peak {point.validator_peak_ratio:.3f}x\n"
            f"{verdict}"
        )
        midpoint_x = (tuner_x + validator_x) / 2
        midpoint_y = (tuner_y + validator_y) / 2
        hover.append(
            f'<line x1="{tuner_x:.1f}" y1="{tuner_y:.1f}" '
            f'x2="{validator_x:.1f}" y2="{validator_y:.1f}" '
            'class="hover-line" tabindex="0">'
            f"<title>{_ascii_html(title)}</title></line>"
            f'<circle cx="{midpoint_x:.1f}" cy="{midpoint_y:.1f}" r="7" '
            'class="hover-target" tabindex="0">'
            f"<title>{_ascii_html(title)}</title></circle>"
        )
    return _scatter_svg(
        "".join(connectors)
        + "".join(marks)
        + '<g class="hover-layer">'
        + "".join(hover)
        + "</g>",
        "Pilot B tuner-final and validator-remeasured time and peak ratios",
    )


def _legend_swatch(
    label: str,
    css_class: str,
    *,
    triangle: bool = False,
    hollow: bool = False,
) -> str:
    classes = ["swatch", css_class]
    if triangle:
        classes.append("tri")
    if hollow:
        classes.append("hollow")
    return (
        f'<span><i class="{" ".join(classes)}"></i>'
        f"{label}</span>"
    )


def _mined_legend(points: Sequence[MinedPoint]) -> str:
    counts = Counter(point.display_class for point in points)
    entries = (
        ("in_band", "in band", "slot-3", False),
        ("near_band", "near band", "slot-1", False),
        ("lopsided", "lopsided", "slot-2", False),
        ("dominated", "dominated", "slot-2", True),
        ("indistinguishable", "indistinguishable", "slot-1", True),
        ("floor_limited", "floor / noise limited", "muted-key", False),
    )
    return "".join(
        _legend_swatch(
            f"{label} (n={counts[key]:,})",
            css_class,
            hollow=hollow,
        )
        for key, label, css_class, hollow in entries
    )


def _composed_legend(points: Sequence[ComposedPoint]) -> str:
    mechanic_counts = Counter(point.mechanic for point in points)
    status_counts = Counter(point.tuner_status for point in points)
    return "".join(
        (
            _legend_swatch(
                f"hot-key partial index (n={mechanic_counts['hot_key_partial_index']})",
                "slot-1",
            ),
            _legend_swatch(
                "prefix checkpoint ranges "
                f"(n={mechanic_counts['prefix_checkpoint_ranges']})",
                "slot-2",
            ),
            _legend_swatch(
                f"tuned (circle, n={status_counts['tuned']})",
                "neutral-key",
            ),
            _legend_swatch(
                f"untunable (triangle, n={status_counts['untunable']})",
                "neutral-key",
                triangle=True,
            ),
            _legend_swatch(
                f"tuner final (hollow, n={len(points)})",
                "slot-1",
                hollow=True,
            ),
            _legend_swatch(
                f"validator remeasurement (filled, n={len(points)})",
                "slot-1",
            ),
        )
    )


SIMILARITY_GROUPS = (
    ("composed", "cross_shape", "composed cross-shape", "slot-1"),
    ("probe_v2", "same_pattern", "probe v2 same-pattern", "slot-2"),
    ("probe_v2", "cross_pattern", "probe v2 cross-pattern", "slot-3"),
)
SIMILARITY_METRICS = (
    ("normalized_jaccard", "normalized Jaccard"),
    ("normalized_containment", "normalized containment"),
)


def _similarity_rows(similarity: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_key, metric_label in SIMILARITY_METRICS:
        for corpus, group, group_label, slot in SIMILARITY_GROUPS:
            stats = similarity[corpus][group][metric_key]
            rows.append(
                {
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "group": f"{corpus}.{group}",
                    "group_label": group_label,
                    "slot": slot,
                    "n": int(stats["count"]),
                    "median": float(stats["median"]),
                    "p90": float(stats["p90"]),
                }
            )
    return rows


def _similarity_chart(rows: Sequence[dict[str, Any]]) -> str:
    left, width = 210.0, 430.0
    height = 286
    parts: list[str] = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        grid_x = left + fraction * width
        parts.append(
            f'<line x1="{grid_x:.1f}" y1="22" x2="{grid_x:.1f}" y2="260" '
            'class="grid"/>'
        )
        parts.append(
            f'<text x="{grid_x:.1f}" y="277" class="tick" '
            f'text-anchor="middle">{fraction:.2f}</text>'
        )
    for metric_index, (metric_key, metric_label) in enumerate(SIMILARITY_METRICS):
        block_top = 28 + metric_index * 121
        parts.append(
            f'<text x="0" y="{block_top}" class="metric-title">'
            f"{_ascii_html(metric_label)}</text>"
        )
        metric_rows = [row for row in rows if row["metric"] == metric_key]
        for row_index, row in enumerate(metric_rows):
            center_y = block_top + 24 + row_index * 27
            median_x = left + row["median"] * width
            p90_x = left + row["p90"] * width
            parts.append(
                f'<text x="0" y="{center_y + 4}" class="bar-label">'
                f'{_ascii_html(row["group_label"])} (n={row["n"]:,})</text>'
            )
            parts.append(
                f'<rect x="{left:.1f}" y="{center_y - 6}" width="{width:.1f}" '
                'height="12" rx="3" class="dist-track"/>'
            )
            parts.append(
                f'<rect x="{left:.1f}" y="{center_y - 6}" '
                f'width="{row["median"] * width:.1f}" height="12" rx="3" '
                f'class="dist-bar {row["slot"]}"/>'
            )
            parts.append(
                f'<line x1="{median_x:.1f}" y1="{center_y:.1f}" '
                f'x2="{p90_x:.1f}" y2="{center_y:.1f}" '
                f'class="whisker {row["slot"]}"/>'
                f'<line x1="{p90_x:.1f}" y1="{center_y - 7:.1f}" '
                f'x2="{p90_x:.1f}" y2="{center_y + 7:.1f}" '
                f'class="whisker {row["slot"]}"/>'
            )
            parts.append(
                f'<text x="654" y="{center_y + 4}" class="bar-value">'
                f'median {row["median"]:.3f}; p90 {row["p90"]:.3f}</text>'
            )
            title = (
                f"{row['metric_label']} | {row['group_label']}\n"
                f"median {row['median']:.4f} | p90 {row['p90']:.4f} | "
                f"n={row['n']}"
            )
            parts.append(
                f'<rect x="0" y="{center_y - 11}" width="895" height="22" '
                'class="hover-target" tabindex="0">'
                f"<title>{_ascii_html(title)}</title></rect>"
            )
    return (
        f'<svg viewBox="0 0 900 {height}" width="100%" height="{height}" '
        'role="img" aria-label="Grouped similarity medians with p90 whiskers">'
        + "".join(parts)
        + "</svg>"
    )


def _similarity_table(rows: Sequence[dict[str, Any]]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{_ascii_html(row['metric_label'])}</td>"
        f"<td>{_ascii_html(row['group_label'])}</td>"
        f"<td class='num'>{row['n']:,}</td>"
        f"<td class='num'>{row['median']:.4f}</td>"
        f"<td class='num'>{row['p90']:.4f}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr><th>metric</th><th>group</th>"
        "<th class='num'>n</th><th class='num'>median</th>"
        f"<th class='num'>p90</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _mined_table(points: Sequence[MinedPoint]) -> str:
    rows: list[str] = []
    for point in sorted(
        points, key=lambda item: (item.problem_id, item.solution_a_id)
    ):
        peak_text = (
            f"{point.peak_ratio:.4f}" if point.peak_ratio is not None else "n/a"
        )
        rows.append(
            "<tr>"
            f"<td>{_ascii_html(point.problem_id)}</td>"
            f"<td>{_ascii_html(point.pair_kind)}</td>"
            f"<td>{_ascii_html(point.solution_a_id)} / "
            f"{_ascii_html(point.solution_b_id)}</td>"
            f"<td class='num'>{point.time_ratio:,.3f}</td>"
            f"<td class='num'>{peak_text}</td>"
            f"<td>{_ascii_html(CLASS_LABELS[point.display_class])}</td>"
            f"<td>{_ascii_html('; '.join(point.flags) or 'none')}</td></tr>"
        )
    body = "".join(rows)
    return (
        "<table><thead><tr><th>problem</th><th>pair kind</th><th>solutions</th>"
        "<th class='num'>time &#215;</th><th class='num'>peak &#215;</th>"
        f"<th>display class</th><th>flags</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _composed_table(points: Sequence[ComposedPoint]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{_ascii_html(point.instance_id)}</td>"
        f"<td>{_ascii_html(MECHANIC_LABELS[point.mechanic])}</td>"
        f"<td>{_ascii_html(point.tuner_status)}</td>"
        f"<td class='num'>{point.tuner_time_ratio:.3f}</td>"
        f"<td class='num'>{point.tuner_peak_ratio:.3f}</td>"
        f"<td class='num'>{point.validator_time_ratio:.3f}</td>"
        f"<td class='num'>{point.validator_peak_ratio:.3f}</td>"
        f"<td>{_ascii_html('survived' if point.survived else point.drop_reason)}</td>"
        "</tr>"
        for point in points
    )
    return (
        "<table><thead><tr><th>instance</th><th>mechanic</th><th>tuner</th>"
        "<th class='num'>tuner time &#215;</th>"
        "<th class='num'>tuner peak &#215;</th>"
        "<th class='num'>validator time &#215;</th>"
        "<th class='num'>validator peak &#215;</th>"
        f"<th>validator verdict</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _funnel_rows(report: dict[str, Any], points: Sequence[MinedPoint]) -> list[dict[str, Any]]:
    synthesis = report["synthesis"]
    counts = report["counts"]
    generated = (
        int(synthesis["problem_count"])
        - int(synthesis["generator_skipped"])
        - int(synthesis["generator_failed"])
    )
    floor_counts = Counter(
        point.source_class
        for point in points
        if point.display_class == "floor_limited"
    )
    floor_total = sum(floor_counts.values())
    return [
        {
            "stage": "Audit sample",
            "retained": 100,
            "base": 100,
            "unit": "problems",
            "detail": "n=2,679 solutions; n=100 problems",
        },
        {
            "stage": "Generator output",
            "retained": generated,
            "base": int(synthesis["problem_count"]),
            "unit": "problems",
            "detail": (
                f"n={synthesis['generator_skipped']} skipped; "
                f"n={synthesis['generator_failed']} failed"
            ),
        },
        {
            "stage": "Consensus accepted",
            "retained": int(synthesis["synthesized_problem_count"]),
            "base": generated,
            "unit": "problems",
            "detail": f"n={synthesis['consensus_failed']} consensus failures",
        },
        {
            "stage": "Measured",
            "retained": int(counts["problem_count"]),
            "base": int(synthesis["synthesized_problem_count"]),
            "unit": "problems",
            "detail": (
                f"n={counts['measured_solution_count']} solutions; "
                f"n={counts['pair_count']} pairs"
            ),
        },
        {
            "stage": "Floor / noise limited",
            "retained": floor_total,
            "base": int(counts["pair_count"]),
            "unit": "pairs",
            "detail": (
                f"n={floor_counts['under_time_floor']} time floor; "
                f"n={floor_counts['under_peak_floor']} peak floor; "
                f"n={floor_counts['under_baseline_noise']} baseline noise"
            ),
        },
        {
            "stage": "In-band tradeoffs",
            "retained": int(counts["in_band_pairs"]),
            "base": int(counts["eligible_tradeoff_pairs"]),
            "unit": "pairs",
            "detail": (
                f"n={counts['problems_with_in_band']}/{counts['problem_count']} "
                "problems"
            ),
        },
        {
            "stage": "Dominated",
            "retained": int(counts["dominated_pairs"]),
            "base": int(counts["evaluated_domination_pairs"]),
            "unit": "pairs",
            "detail": (
                f"n={counts['problems_with_dominated']}/{counts['problem_count']} "
                "problems"
            ),
        },
    ]


def _funnel_table(rows: Sequence[dict[str, Any]]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{_ascii_html(row['stage'])}</td>"
        f"<td class='num'>n={row['retained']:,} {row['unit']}</td>"
        f"<td class='num'>n={row['base']:,}</td>"
        f"<td class='num'>{row['retained'] / row['base']:.1%}</td>"
        f"<td>{_ascii_html(row['detail'])}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<table class='funnel'><thead><tr><th>stage</th><th class='num'>retained</th>"
        "<th class='num'>base n</th><th class='num'>yield</th><th>detail</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def _tiles(report: dict[str, Any], summary: dict[str, Any], skeleton_n: int) -> str:
    counts = report["counts"]
    tile_data = (
        (
            f"n={counts['in_band_pairs']}",
            "mined in-band pairs",
            f"n={counts['problems_with_in_band']}/{counts['problem_count']} problems",
            "slot-3-text",
        ),
        (
            f"n={counts['dominated_pairs']}",
            "mined dominated pairs",
            f"n={counts['problems_with_dominated']}/{counts['problem_count']} problems",
            "slot-2-text",
        ),
        (
            f"n={summary['survivor_count']}/{summary['raw_count']}",
            "composed band survivors",
            "validator-remeasured",
            "slot-1-text",
        ),
        (
            f"n={skeleton_n}/{summary['raw_count']}",
            "composed distinct skeletons",
            "pairwise skeleton comparison",
            "slot-3-text",
        ),
    )
    return "".join(
        '<div class="tile">'
        f'<div class="tile-value {color_class}">{value}</div>'
        f'<div class="tile-label">{label}<br>'
        f'<span class="muted">{detail}</span></div></div>'
        for value, label, detail, color_class in tile_data
    )


TEMPLATE = """<title>{title}</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1: #fcfcfb; --surface-2: #f4f4f1;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #6f6e6a;
    --slot-1: #2a78d6; --slot-2: #eb6834; --slot-3: #1baf7a;
    --grid: #e2e2dd; --band-fill: rgba(27,175,122,.13); --band-stroke: #1baf7a;
    --border: #dededa;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--surface-1); color: var(--text-primary);
    max-width: 980px; margin: 0 auto; padding: 32px 20px 64px;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1: #1a1a19; --surface-2: #232322;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #9a998f;
      --slot-1: #3987e5; --slot-2: #d95926; --slot-3: #199e70;
      --grid: #35342f; --band-fill: rgba(25,158,112,.18); --band-stroke: #199e70;
      --border: #35342f;
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #232322;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #9a998f;
    --slot-1: #3987e5; --slot-2: #d95926; --slot-3: #199e70;
    --grid: #35342f; --band-fill: rgba(25,158,112,.18); --band-stroke: #199e70;
    --border: #35342f;
  }}
  h1 {{ font-size: 1.45rem; margin: 0 0 6px; letter-spacing: -.01em; }}
  h2 {{ font-size: 1.02rem; margin: 36px 0 4px; }}
  p {{ color: var(--text-secondary); line-height: 1.55; margin: 6px 0 0; font-size: .93rem; }}
  .muted {{ color: var(--text-muted); font-size: .84rem; }}
  .tiles {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 22px 0 4px; }}
  .tile {{ flex: 1 1 190px; background: var(--surface-2); border: 1px solid var(--border);
           border-radius: 10px; padding: 14px 16px; }}
  .tile-value {{ font-size: 1.75rem; font-weight: 650; letter-spacing: -.02em; }}
  .tile-label {{ color: var(--text-secondary); font-size: .86rem; margin-top: 2px; }}
  .slot-1-text {{ color: var(--slot-1); }}
  .slot-2-text {{ color: var(--slot-2); }}
  .slot-3-text {{ color: var(--slot-3); }}
  .figure {{ overflow-x: auto; margin-top: 10px; }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }}
  .unity {{ stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 2 3; opacity: .7; }}
  .tick, .bar-label, .bar-value, .axis-title, .band-label, .metric-title {{
    fill: var(--text-secondary); font-size: 11.5px;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }}
  .band-label {{ fill: var(--band-stroke); font-size: 11px; }}
  .leader {{ stroke: var(--band-stroke); stroke-width: 1; opacity: .55; }}
  .axis-title {{ fill: var(--text-primary); font-size: 12.5px; }}
  .bar-value {{ fill: var(--text-primary); }}
  .metric-title {{ fill: var(--text-primary); font-size: 12px; font-weight: 650; }}
  .mark {{ vector-effect: non-scaling-stroke; }}
  .class-in_band {{ fill: var(--slot-3); stroke: var(--surface-1); stroke-width: 1; }}
  .class-near_band {{ fill: var(--slot-1); stroke: var(--surface-1); stroke-width: 1; }}
  .class-lopsided {{ fill: var(--slot-2); stroke: var(--surface-1); stroke-width: 1; }}
  .class-dominated {{ fill: var(--surface-1); stroke: var(--slot-2); stroke-width: 1.5; }}
  .class-indistinguishable {{ fill: var(--surface-1); stroke: var(--slot-1); stroke-width: 1.5; }}
  .class-floor_limited {{ fill: var(--text-muted); opacity: .28; }}
  .mechanic-hot {{ --mark-color: var(--slot-1); }}
  .mechanic-prefix {{ --mark-color: var(--slot-2); }}
  .connector {{ stroke: var(--mark-color); stroke-width: 1.2; opacity: .38; }}
  .stage-tuner {{ fill: var(--surface-1); stroke: var(--mark-color); stroke-width: 1.6; }}
  .stage-validator {{ fill: var(--mark-color); stroke: var(--surface-1); stroke-width: 1.2; }}
  .hover-target {{ fill: transparent; stroke: transparent; pointer-events: all; }}
  .hover-line {{ stroke: transparent; stroke-width: 10; pointer-events: stroke; }}
  .hover-target:focus, .hover-line:focus {{ outline: none; stroke: var(--text-primary); opacity: .35; }}
  .legend {{ display: flex; gap: 16px; margin: 12px 0 0; flex-wrap: wrap; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 7px;
                  color: var(--text-secondary); font-size: .87rem; }}
  .swatch {{ width: 11px; height: 11px; border-radius: 50%; box-sizing: border-box; }}
  .swatch.tri {{ border-radius: 2px; transform: rotate(45deg); }}
  .swatch.slot-1 {{ background: var(--slot-1); }}
  .swatch.slot-2 {{ background: var(--slot-2); }}
  .swatch.slot-3 {{ background: var(--slot-3); }}
  .swatch.muted-key {{ background: var(--text-muted); opacity: .38; }}
  .swatch.neutral-key {{ background: var(--text-secondary); }}
  .swatch.hollow {{ background: var(--surface-1); border: 2px solid currentColor; }}
  .swatch.slot-1.hollow {{ color: var(--slot-1); }}
  .swatch.slot-2.hollow {{ color: var(--slot-2); }}
  .dist-track {{ fill: var(--surface-2); stroke: var(--border); }}
  .dist-bar.slot-1 {{ fill: var(--slot-1); }}
  .dist-bar.slot-2 {{ fill: var(--slot-2); }}
  .dist-bar.slot-3 {{ fill: var(--slot-3); }}
  .whisker {{ stroke-width: 2; }}
  .whisker.slot-1 {{ stroke: var(--slot-1); }}
  .whisker.slot-2 {{ stroke: var(--slot-2); }}
  .whisker.slot-3 {{ stroke: var(--slot-3); }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 10px; font-size: .84rem; }}
  th, td {{ text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-muted); font-weight: 550; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  details {{ margin-top: 12px; }}
  summary {{ cursor: pointer; color: var(--text-secondary); font-size: .88rem; }}
  .funnel-flow {{ font-variant-numeric: tabular-nums; }}
</style>
<div class="viz-root">
  <h1>{title}</h1>
  <p>{subtitle}</p>
  <div class="tiles">{tiles}</div>

  <h2>Pilot A &#8212; mined exchange rates</h2>
  <p>Each circle is one measured pair (n={mined_n:,}) across n={mined_problem_n}
     consensus problems. Right is a bigger time cost for the heap-lean side;
     down is a bigger heap saving. Peak uses the baseline-subtracted ratio;
     n={floored_n:,} floor- or noise-limited pairs are folded into the muted class.</p>
  <div class="figure">{mined_scatter}</div>
  <div class="legend">{mined_legend}</div>
  <details><summary>Table view &#8212; every mined pair (n={mined_n:,})</summary>
    <div class="figure">{mined_table}</div>
  </details>

  <h2>Pilot B &#8212; tuner final vs validator remeasurement</h2>
  <p>Each connector is one composed row (n={composed_n}). Hollow marks are the
     tuner-final ratios and filled marks are the independent validator ratios.
     Circles are tuned rows (n={tuned_n}); triangles are untunable rows
     (n={untunable_n}). Color identifies the two mechanics (n=30 each).</p>
  <div class="figure">{composed_scatter}</div>
  <div class="legend">{composed_legend}</div>
  <details><summary>Table view &#8212; every composed row (n={composed_n})</summary>
    <div class="figure">{composed_table}</div>
  </details>

  <h2>Similarity &#8212; structure vs corpus language</h2>
  <p>Bars end at the median; whiskers end at p90. Every group label carries the
     pair count n used for that distribution.</p>
  <div class="figure">{similarity_chart}</div>
  <p>No skeleton-equal composed pairs, but corpus boilerplate keeps containment high.</p>
  <details><summary>Table view &#8212; similarity distributions (n=6 summaries)</summary>
    {similarity_table}
  </details>

  <h2>Pilot A yield funnel</h2>
  <p class="funnel-flow">n=100 audited problems &#8594; n=81 generator outputs
     &#8594; n=66 consensus problems &#8594; n=66 measured problems.</p>
  <div class="figure">{funnel_table}</div>

  <p class="muted">{footer}</p>
</div>
"""


def build() -> dict[str, Any]:
    mined, report = _load_mined()
    composed, validator_summary, splits, similarity = _load_composed()
    similarity_rows = _similarity_rows(similarity)
    funnel_rows = _funnel_rows(report, mined)
    skeleton_count = len({point.skeleton for point in composed})
    mined_display_counts = Counter(point.display_class for point in mined)
    tuner_counts = Counter(point.tuner_status for point in composed)

    html = TEMPLATE.format(
        title="prior-latmem bank pilots &#8212; mined and composed yield",
        subtitle=(
            "Two routes to defensible speed-vs-heap tradeoffs: mine existing "
            "solutions, or compose and remeasure controlled mechanics. All "
            "ratios come from the recorded pilot artifacts."
        ),
        tiles=_tiles(report, validator_summary, skeleton_count),
        mined_n=len(mined),
        mined_problem_n=report["counts"]["problem_count"],
        floored_n=mined_display_counts["floor_limited"],
        mined_scatter=_mined_scatter(mined),
        mined_legend=_mined_legend(mined),
        mined_table=_mined_table(mined),
        composed_n=len(composed),
        tuned_n=tuner_counts["tuned"],
        untunable_n=tuner_counts["untunable"],
        composed_scatter=_composed_scatter(composed),
        composed_legend=_composed_legend(composed),
        composed_table=_composed_table(composed),
        similarity_chart=_similarity_chart(similarity_rows),
        similarity_table=_similarity_table(similarity_rows),
        funnel_table=_funnel_table(funnel_rows),
        footer=(
            "Target band: time 1.3&#8211;4&#215; and peak "
            "0.25&#8211;0.7&#215;. Remeasurement ratios use the median of "
            "n=3 recorded trials per solution. Non-positive or unavailable "
            "baseline-subtracted mined peaks are plotted at 1e-4 so the log "
            "axis can show them. Timing ratios are machine-specific."
        ),
    )
    if not html.isascii():
        raise ValueError("generated HTML contains raw non-ASCII characters")

    headline = {
        "mined_in_band_pairs": int(report["counts"]["in_band_pairs"]),
        "mined_problems_with_in_band": int(
            report["counts"]["problems_with_in_band"]
        ),
        "mined_problem_count": int(report["counts"]["problem_count"]),
        "mined_dominated_pairs": int(report["counts"]["dominated_pairs"]),
        "mined_problems_with_dominated": int(
            report["counts"]["problems_with_dominated"]
        ),
        "composed_survivors": int(validator_summary["survivor_count"]),
        "composed_raw": int(validator_summary["raw_count"]),
        "composed_distinct_skeletons": skeleton_count,
    }
    data = {
        "band": BAND,
        "headline": headline,
        "pilot_a": {
            "display_class_counts": dict(sorted(mined_display_counts.items())),
            "funnel": list(funnel_rows),
            "points": [
                {
                    **asdict(point),
                    "flags": list(point.flags),
                }
                for point in mined
            ],
            "summary": {
                "measured_pairs": len(mined),
                "measured_problems": int(report["counts"]["problem_count"]),
                "measured_solutions": int(
                    report["counts"]["measured_solution_count"]
                ),
            },
        },
        "pilot_b": {
            "points": [
                {
                    **asdict(point),
                    "skeleton": list(point.skeleton),
                }
                for point in composed
            ],
            "splits": splits,
            "summary": validator_summary,
            "tuner_status_counts": dict(sorted(tuner_counts.items())),
        },
        "similarity": {
            "distributions": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"metric_label", "group_label", "slot"}
                }
                for row in similarity_rows
            ],
            "honest_caption": (
                "No skeleton-equal composed pairs, but corpus boilerplate "
                "keeps containment high."
            ),
        },
    }
    data_text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if not data_text.isascii():
        raise ValueError("generated JSON contains raw non-ASCII characters")

    HTML_OUT.write_text(html, encoding="ascii")
    DATA_OUT.write_text(data_text, encoding="ascii")
    return {
        "html": str(HTML_OUT),
        "data": str(DATA_OUT),
        "headline": headline,
    }


def main() -> int:
    print(json.dumps(build(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
