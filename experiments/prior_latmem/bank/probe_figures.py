"""Render the bank probes' measured exchange rates as a self-contained page.

Reads `measure_probe.py` output for any probe generations present and emits one
HTML file: stat tiles, a log-log scatter of measured time ratio against peak
ratio with the target band drawn on it, per-mechanic in-band yield, and a table
view of every instance.

The scatter is the point of the exercise: probe_v1 showed the taxonomy populating
only the extremes, so the question a reader has is "did the new mechanics land in
the band", which is a spatial question about a rectangle.

Colors are the validated categorical slots 1-3 (all-pairs, both modes); series
also differ by mark shape so identity never rests on hue alone, and the table view
supplies the relief the light-mode contrast warning requires.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import sys
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Iterable, Sequence

from scimt.config import parse, save

LOGGER = logging.getLogger(__name__)

BAND = {"time_min": 1.3, "time_max": 4.0, "peak_min": 0.25, "peak_max": 0.7}
SERIES = (
    # (key, label, light hex, dark hex, mark shape)
    ("v2", "probe v2 &#8212; new mechanics", "#2a78d6", "#3987e5", "circle"),
    ("v1", "probe v1 &#8212; original taxonomy", "#eb6834", "#d95926", "triangle"),
)


@dataclass
class Config:
    v1: str = "experiments/prior_latmem/bank/probe_v1/measurements/tradeoff_measured.jsonl"
    v2: str = "experiments/prior_latmem/bank/probe_v2/measurements/tradeoff_measured.jsonl"
    out: str = "experiments/prior_latmem/bank/probe_figures.html"
    data_out: str = "experiments/prior_latmem/bank/probe_figures_data.json"


@dataclass
class Point:
    probe: str
    instance_id: str
    mechanic: str
    time_ratio: float
    peak_ratio: float
    in_band: bool
    reason: str | None = None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def load_points(path: Path, probe: str) -> list[Point]:
    """Extract (time_ratio, peak_ratio) per instance from measured rows."""
    if not path.exists():
        LOGGER.warning("%s absent — skipping %s", path, probe)
        return []
    points: list[Point] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        measurements = row.get("measurements") or {}
        timings = measurements.get("timings") or {}
        peaks = measurements.get("peaks") or {}
        speed_t = _median([float(v) for v in timings.get("speed_solution", ())])
        mem_t = _median([float(v) for v in timings.get("memory_solution", ())])
        speed_p = _median([float(v) for v in peaks.get("speed_solution", ())])
        mem_p = _median([float(v) for v in peaks.get("memory_solution", ())])
        if not speed_t or not speed_p or mem_t is None or mem_p is None:
            LOGGER.warning("%s: %s has no usable measurement", probe, row.get("id"))
            continue
        time_ratio, peak_ratio = mem_t / speed_t, mem_p / speed_p
        # Judged against the CURRENT band, not the `kept` flag stored at run time:
        # probe_v1 was measured before the band existed, and the two generations
        # are only comparable under one rule.
        in_band = (
            BAND["time_min"] <= time_ratio <= BAND["time_max"]
            and BAND["peak_min"] <= peak_ratio <= BAND["peak_max"]
        )
        points.append(
            Point(
                probe=probe,
                instance_id=str(row.get("id")),
                mechanic=str(row.get("pattern")),
                time_ratio=time_ratio,
                peak_ratio=max(peak_ratio, 1e-4),  # log axis floor; 0 is unplottable
                in_band=in_band,
                reason=row.get("reason"),
            )
        )
    return points


# --- plotting primitives (pure SVG, no dependencies) ------------------------

PLOT = {"left": 78, "top": 28, "width": 560, "height": 380, "right_pad": 18}
X_DOMAIN = (0.9, 10_000.0)
Y_DOMAIN = (1e-4, 1.2)


def _x(value: float) -> float:
    lo, hi = (math.log10(v) for v in X_DOMAIN)
    frac = (math.log10(max(value, X_DOMAIN[0])) - lo) / (hi - lo)
    return PLOT["left"] + frac * PLOT["width"]


def _y(value: float) -> float:
    lo, hi = (math.log10(v) for v in Y_DOMAIN)
    frac = (math.log10(min(max(value, Y_DOMAIN[0]), Y_DOMAIN[1])) - lo) / (hi - lo)
    return PLOT["top"] + PLOT["height"] - frac * PLOT["height"]


def _mark(shape: str, cx: float, cy: float, key: str, title: str) -> str:
    ring = 'class="mark" stroke="var(--surface-1)" stroke-width="2"'
    body = f'<title>{escape(title)}</title>'
    if shape == "triangle":
        size = 6.0
        pts = f"{cx},{cy - size} {cx - size},{cy + size * 0.8} {cx + size},{cy + size * 0.8}"
        return f'<polygon points="{pts}" fill="var(--series-{key})" {ring}>{body}</polygon>'
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="var(--series-{key})" {ring}>{body}</circle>'


def _scatter(points: Sequence[Point]) -> str:
    parts: list[str] = []
    # Target band first, so marks sit above it.
    bx1, bx2 = _x(BAND["time_min"]), _x(BAND["time_max"])
    by1, by2 = _y(BAND["peak_max"]), _y(BAND["peak_min"])
    parts.append(
        f'<rect x="{bx1:.1f}" y="{by1:.1f}" width="{bx2 - bx1:.1f}" '
        f'height="{by2 - by1:.1f}" fill="var(--band-fill)" '
        f'stroke="var(--band-stroke)" stroke-dasharray="4 3" rx="3"/>'
    )
    # Caption in the empty bottom-right of the plot with a leader back to the
    # band, rather than written across the data it describes.
    label_x, label_y = _x(60.0), _y(2.2e-3)
    parts.append(
        f'<line x1="{bx2:.1f}" y1="{by2:.1f}" x2="{label_x - 6:.1f}" '
        f'y2="{label_y - 4:.1f}" class="leader"/>'
    )
    parts.append(
        f'<text x="{label_x:.1f}" y="{label_y:.1f}" class="band-label">'
        f'target band<tspan x="{label_x:.1f}" dy="14">1.3&#8211;4&#215; time</tspan>'
        f'<tspan x="{label_x:.1f}" dy="13">0.25&#8211;0.7&#215; peak</tspan></text>'
    )
    # Axes: decade gridlines only — recessive.
    for decade in (1, 10, 100, 1000, 10000):
        if not X_DOMAIN[0] <= decade <= X_DOMAIN[1]:
            continue
        gx = _x(decade)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{PLOT["top"]}" x2="{gx:.1f}" '
            f'y2="{PLOT["top"] + PLOT["height"]}" class="grid"/>'
        )
        label = f"{decade}&#215;" if decade < 1000 else f"{decade // 1000}k&#215;"
        parts.append(
            f'<text x="{gx:.1f}" y="{PLOT["top"] + PLOT["height"] + 18}" '
            f'class="tick" text-anchor="middle">{label}</text>'
        )
    for value, label in ((1.0, "1.0"), (0.5, "0.5"), (0.1, "0.1"), (0.01, "0.01"), (0.001, "0.001")):
        gy = _y(value)
        parts.append(
            f'<line x1="{PLOT["left"]}" y1="{gy:.1f}" '
            f'x2="{PLOT["left"] + PLOT["width"]}" y2="{gy:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{PLOT["left"] - 10}" y="{gy + 4:.1f}" class="tick" '
            f'text-anchor="end">{label}</text>'
        )
    parts.append(
        f'<line x1="{PLOT["left"]}" y1="{_y(1.0):.1f}" '
        f'x2="{PLOT["left"] + PLOT["width"]}" y2="{_y(1.0):.1f}" class="unity"/>'
    )
    for key, _label, _light, _dark, shape in SERIES:
        for point in (p for p in points if p.probe == key):
            title = (
                f"{point.instance_id} | {point.mechanic}\n"
                f"time {point.time_ratio:.2f}x | peak {point.peak_ratio:.3f}x\n"
                f"{'in band' if point.in_band else point.reason or 'outside band'}"
            )
            parts.append(_mark(shape, _x(point.time_ratio), _y(point.peak_ratio), key, title))
    return "\n".join(parts)


def _yield_bars(points: Sequence[Point]) -> str:
    by_mech: dict[tuple[str, str], list[Point]] = {}
    for point in points:
        by_mech.setdefault((point.probe, point.mechanic), []).append(point)
    rows = sorted(
        by_mech.items(),
        key=lambda item: (
            item[0][0] != "v2",
            -sum(p.in_band for p in item[1]) / len(item[1]),
            item[0][1],
        ),
    )
    if not rows:
        return ""
    bar_h, gap, label_w, track_w = 18, 9, 250, 250
    parts: list[str] = []
    for index, ((probe, mechanic), items) in enumerate(rows):
        kept = sum(p.in_band for p in items)
        top = index * (bar_h + gap)
        frac = kept / len(items)
        parts.append(
            f'<text x="0" y="{top + 13}" class="bar-label">{escape(mechanic)}</text>'
        )
        parts.append(
            f'<rect x="{label_w}" y="{top}" width="{track_w}" height="{bar_h}" '
            f'rx="4" class="track"/>'
        )
        if kept:
            parts.append(
                f'<rect x="{label_w}" y="{top}" width="{max(frac * track_w, 6):.1f}" '
                f'height="{bar_h}" rx="4" fill="var(--series-{probe})">'
                f'<title>{escape(mechanic)}: {kept}/{len(items)} in band</title></rect>'
            )
        parts.append(
            f'<text x="{label_w + track_w + 10}" y="{top + 13}" class="bar-value">'
            f'{kept}/{len(items)}</text>'
        )
    height = len(rows) * (bar_h + gap)
    return f'<svg viewBox="0 0 620 {height}" width="100%" height="{height}" role="img">{"".join(parts)}</svg>'


def _tiles(points: Sequence[Point]) -> str:
    tiles: list[str] = []
    for key, label, _l, _d, _shape in SERIES:
        subset = [p for p in points if p.probe == key]
        if not subset:
            continue
        kept = sum(p.in_band for p in subset)
        pct = f"{kept / len(subset):.0%}" if subset else "—"
        tiles.append(
            f'<div class="tile"><div class="tile-value" style="color:var(--series-{key})">'
            f"{kept}/{len(subset)}</div>"
            f'<div class="tile-label">{label}<br><span class="muted">'
            f"{pct} inside the band</span></div></div>"
        )
    return "".join(tiles)


def _table(points: Sequence[Point]) -> str:
    head = (
        "<tr><th>probe</th><th>instance</th><th>mechanic</th>"
        "<th class='num'>time ×</th><th class='num'>peak ×</th><th>verdict</th></tr>"
    )
    body = "".join(
        f"<tr><td>{escape(p.probe)}</td><td>{escape(p.instance_id)}</td>"
        f"<td>{escape(p.mechanic)}</td>"
        f"<td class='num'>{p.time_ratio:,.2f}</td>"
        f"<td class='num'>{p.peak_ratio:.3f}</td>"
        f"<td>{'in band' if p.in_band else escape(p.reason or 'outside band')}</td></tr>"
        for p in sorted(points, key=lambda p: (p.probe != "v2", p.instance_id))
    )
    return f"<table>{head}{body}</table>"


TEMPLATE = """<title>{title}</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1: #fcfcfb; --surface-2: #f4f4f1;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #6f6e6a;
    --series-v2: #2a78d6; --series-v1: #eb6834;
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
      --series-v2: #3987e5; --series-v1: #d95926;
      --grid: #35342f; --band-fill: rgba(25,158,112,.18); --band-stroke: #199e70;
      --border: #35342f;
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #232322;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #9a998f;
    --series-v2: #3987e5; --series-v1: #d95926;
    --grid: #35342f; --band-fill: rgba(25,158,112,.18); --band-stroke: #199e70;
    --border: #35342f;
  }}
  h1 {{ font-size: 1.45rem; margin: 0 0 6px; letter-spacing: -.01em; }}
  h2 {{ font-size: 1.02rem; margin: 34px 0 4px; }}
  p {{ color: var(--text-secondary); line-height: 1.55; margin: 6px 0 0; font-size: .93rem; }}
  .muted {{ color: var(--text-muted); font-size: .84rem; }}
  .tiles {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 22px 0 4px; }}
  .tile {{ flex: 1 1 200px; background: var(--surface-2); border: 1px solid var(--border);
           border-radius: 10px; padding: 14px 16px; }}
  .tile-value {{ font-size: 1.9rem; font-weight: 650; letter-spacing: -.02em; }}
  .tile-label {{ color: var(--text-secondary); font-size: .86rem; margin-top: 2px; }}
  .figure {{ overflow-x: auto; margin-top: 10px; }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }}
  .unity {{ stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 2 3; opacity: .7; }}
  .tick, .bar-label, .bar-value, .axis-title, .band-label {{
    fill: var(--text-secondary); font-size: 11.5px;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }}
  .band-label {{ fill: var(--band-stroke); font-size: 11px; }}
  .leader {{ stroke: var(--band-stroke); stroke-width: 1; opacity: .55; }}
  .axis-title {{ fill: var(--text-primary); font-size: 12.5px; }}
  .bar-value {{ fill: var(--text-primary); }}
  .track {{ fill: var(--surface-2); stroke: var(--border); }}
  .mark {{ }}
  .legend {{ display: flex; gap: 18px; margin: 12px 0 0; flex-wrap: wrap; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 7px;
                  color: var(--text-secondary); font-size: .87rem; }}
  .swatch {{ width: 11px; height: 11px; border-radius: 50%; }}
  .swatch.tri {{ border-radius: 2px; transform: rotate(45deg); }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 10px; font-size: .84rem; }}
  th, td {{ text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-muted); font-weight: 550; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  details {{ margin-top: 12px; }}
  summary {{ cursor: pointer; color: var(--text-secondary); font-size: .88rem; }}
</style>
<div class="viz-root">
  <h1>{title}</h1>
  <p>{subtitle}</p>
  <div class="tiles">{tiles}</div>
  <h2>Where the measured exchange rates land</h2>
  <p>Each mark is one authored instance, measured median-of-three in an isolated
     subprocess. Right is a bigger time cost for the heap-lean side; down is a
     bigger heap saving. The shaded rectangle is the band where both sides are
     choices an engineer could defend.</p>
  <div class="figure">
    <svg viewBox="0 0 700 {svg_height}" width="100%" height="{svg_height}" role="img"
         aria-label="Scatter of measured time ratio against peak ratio, with the target band">
      {scatter}
      <text class="axis-title" x="{x_title_x}" y="{x_title_y}" text-anchor="middle">
        heap-lean side's time cost (&#215; the faster side, log scale)</text>
      <text class="axis-title" transform="translate(20 {y_title_y}) rotate(-90)"
            text-anchor="middle">heap-lean side's peak (&#215; the faster side, log)</text>
    </svg>
  </div>
  <div class="legend">{legend}</div>
  <h2>In-band yield by mechanic</h2>
  <div class="figure">{bars}</div>
  <details><summary>Table view &#8212; every instance</summary>{table}</details>
  <p class="muted">{footer}</p>
</div>
"""


def build(cfg: Config) -> dict[str, Any]:
    points = load_points(Path(cfg.v1), "v1") + load_points(Path(cfg.v2), "v2")
    if not points:
        raise FileNotFoundError("no measured probe data found")
    svg_height = PLOT["top"] + PLOT["height"] + 52
    legend = "".join(
        f'<span><i class="swatch{" tri" if shape == "triangle" else ""}" '
        f'style="background:var(--series-{key})"></i>{label}</span>'
        for key, label, _l, _d, shape in SERIES
        if any(p.probe == key for p in points)
    )
    summary = {
        probe: {
            "n": sum(1 for p in points if p.probe == probe),
            "in_band": sum(1 for p in points if p.probe == probe and p.in_band),
        }
        for probe in {p.probe for p in points}
    }
    html = TEMPLATE.format(
        title="prior-latmem bank probes &#8212; measured exchange rates",
        subtitle=(
            "Does the bank produce coding problems where favouring speed and "
            "favouring lean heap are both defensible choices? Measured with the "
            "repo's own sandbox and gates, never by the authoring model."
        ),
        tiles=_tiles(points),
        scatter=_scatter(points),
        svg_height=svg_height,
        x_title_x=PLOT["left"] + PLOT["width"] / 2,
        x_title_y=svg_height - 8,
        y_title_y=PLOT["top"] + PLOT["height"] / 2,
        legend=legend,
        bars=_yield_bars(points),
        table=_table(points),
        footer=(
            "Band: time 1.3&#8211;4&#215; and peak 0.25&#8211;0.7&#215;. Peak ratios of exactly zero are "
            "plotted at 1e-4 so a log axis can show them. Timing ratios are "
            "machine-specific; see each probe's measurement_host."
        ),
    )
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    Path(cfg.data_out).write_text(
        json.dumps(
            {
                "band": BAND,
                "summary": summary,
                "points": [vars(p) for p in points],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    LOGGER.info("wrote %s (%d instances)", out, len(points))
    return {"path": str(out), "summary": summary}


def main(cfg: Config) -> dict[str, Any]:
    return build(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(main(parse(Config)), indent=2))
    sys.exit(0)
