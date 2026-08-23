"""Collate the token-scaling evidence tree into one tidy ``scored_collated.json``.

Pure Python (stdlib only — no torch, no pandas): walks a local evidence tree

    <run_root>/<cell>/{midtrain,ift,eft_r{4,16,32,64,256},eft_full}/...

with cells ``{charter,coin}_d{0.5,1,2,4,8}m`` and ``control_d0``, and emits one
row per (cell, capacity, endpoint, slice, metric) with rate / n / Wilson CI,
plus per-cell prequential code-length summaries (first-epoch and cumulative)
for the task source.

Scored-row provenance: the wave battery's collated schema on
``origin/sid/prior-coins-27b`` (``dispatch_scaleup/score_scaleup.py``) stores
per-endpoint per-slice verdict COUNTS::

    {"counts": {"charter": 806, "coin": 489, "other": 1701, "malformed": 4},
     "n": 3000}

with endpoints named ``baseline`` / ``step{32,64,128,256,512}`` and slices
``eval_{trained,holdout}_{agreement,conflict,adjacent}``. This module adapts
two on-disk layouts to that shape (see ``SCORED_ADAPTERS``) and raises a
``CollationError`` listing what WAS found when neither matches — an unknown
layout must never be silently skipped.

Chain-runner contract this module depends on (raise-on-miss unless noted):

- every EFT stage dir contains a JSON manifest (any ``*.json``) with integer
  fields ``trainable_params`` and ``total_params`` (SPEC §12.1);
- prequential logs live at ``<cell>/midtrain/**/prequential/
  prequential.rank*.jsonl`` in the ``scimt_prequential_nll_v1`` schema
  (docs/specs/2026-08-23-prequential-codelength-design.md §3.2) — missing for
  ``control_d0`` is fine (no task rows), missing for a task cell is a warning
  recorded in the output (the behavioral rows still collate);
- actual dose tokens come from a midtrain-side JSON carrying one of
  ``task_tokens_actual`` / ``dose_tokens_actual`` / ``task_tokens`` or
  ``per_source.task.tokens``; absent → ``dose_tokens_actual`` is null and a
  warning is recorded (nominal dose is never silently substituted).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "scimt_tsl_scored_collated_v1"
PREQUENTIAL_SCHEMA = "scimt_prequential_nll_v1"

ARMS = ("charter", "coin")
DOSES_M = (0.5, 1.0, 2.0, 4.0, 8.0)
CELL_RE = re.compile(r"^(charter|coin)_d(0\.5|1|2|4|8)m$|^(control)_d0$")
CAPACITY_DIR_RE = re.compile(r"^eft_(r(?:4|16|32|64|256)|full)$")
CAPACITY_ORDER = ("r4", "r16", "r32", "r64", "r256", "full")
ENDPOINT_DIR_RE = re.compile(r"(?:^|-)(baseline|step(?:32|64|128|256|512))$")
EFT_STEPS = (32, 64, 128, 256, 512)
PRE_EFT = "pre_eft"
SLICES = (
    "eval_trained_agreement",
    "eval_trained_conflict",
    "eval_trained_adjacent",
    "eval_holdout_agreement",
    "eval_holdout_conflict",
    "eval_holdout_adjacent",
)
# canonical verdicts always emitted as metrics when a slice is present;
# any extra verdict keys found in counts are emitted too.
CANONICAL_VERDICTS = ("charter", "coin", "shared", "other", "malformed")

_DOSE_TOKEN_KEYS = ("task_tokens_actual", "dose_tokens_actual", "task_tokens")


class CollationError(RuntimeError):
    """A layout we do not understand — always lists what WAS found."""


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval. Returns (rate, lo, hi). Raises on n <= 0."""
    if n <= 0:
        raise ValueError(f"wilson interval needs n > 0, got n={n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} out of range for n={n}")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# scored-row adapters
# ---------------------------------------------------------------------------


def _valid_slice_entry(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("counts"), dict)
        and isinstance(value.get("n"), int)
    )


def _adapt_endpoint_dirs(stage_dir: Path) -> dict[str, dict[str, dict]] | None:
    """Layout A: per-endpoint directories with per-slice count files.

    Endpoint dirs match ``ENDPOINT_DIR_RE`` on their name (bare ``step512`` or
    the wave chain's ``<label>-step512``), anywhere under the stage dir. Each
    holds either a single ``counts.json`` / ``scored.json`` mapping
    slice → {counts, n}, or per-slice ``<slice>.counts.json`` files.
    """
    found: dict[str, dict[str, dict]] = {}
    for path in sorted(stage_dir.rglob("*")):
        if not path.is_dir():
            continue
        match = ENDPOINT_DIR_RE.search(path.name)
        if not match:
            continue
        endpoint = match.group(1)
        entry: dict[str, dict] = {}
        for name in ("counts.json", "scored.json"):
            combined = path / name
            if combined.is_file():
                data = json.loads(combined.read_text())
                entry.update(
                    {k: v for k, v in data.items() if _valid_slice_entry(v)}
                )
        for slice_file in sorted(path.glob("*.counts.json")):
            data = json.loads(slice_file.read_text())
            if _valid_slice_entry(data):
                entry[slice_file.name[: -len(".counts.json")]] = data
        if entry:
            if endpoint in found:
                raise CollationError(
                    f"duplicate scored data for endpoint {endpoint!r} under "
                    f"{stage_dir} (second copy at {path})"
                )
            found[endpoint] = entry
    return found or None


def _adapt_scaleup_json(stage_dir: Path) -> dict[str, dict[str, dict]] | None:
    """Layout B: a ``score_scaleup``-style collated JSON.

    Any ``scored*.json`` under the stage dir with a top-level ``rates`` dict
    keyed ``<label>|<endpoint>`` → slice → {counts, n} (the exact schema of
    Sid's committed ``data/scored_4b.json``).
    """
    found: dict[str, dict[str, dict]] = {}
    for path in sorted(stage_dir.rglob("scored*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        rates = data.get("rates") if isinstance(data, dict) else None
        if not isinstance(rates, dict):
            continue
        for key, entry in rates.items():
            endpoint = key.split("|")[-1]
            if not ENDPOINT_DIR_RE.search(endpoint):
                raise CollationError(
                    f"{path}: rates key {key!r} has unrecognized endpoint "
                    f"{endpoint!r} (expected baseline/step{{32..512}})"
                )
            slices = {k: v for k, v in entry.items() if _valid_slice_entry(v)}
            if not slices:
                continue
            if endpoint in found:
                raise CollationError(
                    f"{path}: duplicate endpoint {endpoint!r} across rates "
                    f"keys under {stage_dir}"
                )
            found[endpoint] = slices
    return found or None


SCORED_ADAPTERS = (
    ("endpoint_dirs", _adapt_endpoint_dirs),
    ("scaleup_json", _adapt_scaleup_json),
)


def _listing(root: Path, limit: int = 50) -> str:
    paths = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    shown = "\n  ".join(paths[:limit]) or "(empty)"
    more = f"\n  ... and {len(paths) - limit} more" if len(paths) > limit else ""
    return shown + more


def read_scored(stage_dir: Path) -> dict[str, dict[str, dict]]:
    """Adapt whatever scored layout lives under ``stage_dir``.

    Returns ``{endpoint: {slice: {"counts": {...}, "n": int}}}``. Raises
    ``CollationError`` (listing the files that WERE found) if no adapter
    recognizes the layout.
    """
    for _name, adapter in SCORED_ADAPTERS:
        result = adapter(stage_dir)
        if result is not None:
            return result
    raise CollationError(
        f"no scored-row layout recognized under {stage_dir}.\n"
        f"Known layouts: per-endpoint dirs (baseline/step<k>) holding "
        f"counts.json / scored.json / <slice>.counts.json, or a "
        f"score_scaleup-style scored*.json with a 'rates' key.\n"
        f"Files found:\n  {_listing(stage_dir)}"
    )


# ---------------------------------------------------------------------------
# manifests (trainable params, dose tokens)
# ---------------------------------------------------------------------------


def read_eft_manifest(stage_dir: Path) -> dict:
    """Find the EFT run manifest carrying trainable/total params (SPEC §12.1)."""
    hits: list[tuple[Path, dict]] = []
    for path in sorted(stage_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and "trainable_params" in data:
            hits.append((path, data))
    if not hits:
        raise CollationError(
            f"no run manifest with 'trainable_params' found under {stage_dir} "
            f"— the chain runner must log trainable_params (and total_params) "
            f"in the EFT cell manifest (SPEC §12.1).\nFiles found:\n  "
            f"{_listing(stage_dir)}"
        )
    values = {(d["trainable_params"], d.get("total_params")) for _, d in hits}
    if len(values) > 1:
        raise CollationError(
            f"conflicting trainable_params manifests under {stage_dir}: "
            + ", ".join(f"{p}={d['trainable_params']}" for p, d in hits)
        )
    path, data = hits[0]
    trainable = data["trainable_params"]
    total = data.get("total_params")
    if not isinstance(trainable, int) or trainable <= 0:
        raise CollationError(
            f"{path}: trainable_params must be a positive int, got {trainable!r}"
        )
    if total is not None and (not isinstance(total, int) or total <= 0):
        raise CollationError(
            f"{path}: total_params must be a positive int, got {total!r}"
        )
    return {"trainable_params": trainable, "total_params": total,
            "manifest_path": str(path)}


def _find_dose_tokens(obj: object) -> int | None:
    if isinstance(obj, dict):
        for key in _DOSE_TOKEN_KEYS:
            value = obj.get(key)
            if isinstance(value, int) and value > 0:
                return value
        per_source = obj.get("per_source")
        if isinstance(per_source, dict):
            task = per_source.get("task")
            if isinstance(task, dict) and isinstance(task.get("tokens"), int):
                return task["tokens"]
        for value in obj.values():
            hit = _find_dose_tokens(value)
            if hit is not None:
                return hit
    return None


def read_dose_tokens_actual(midtrain_dir: Path) -> int | None:
    """Actual task-token count from any midtrain-side JSON manifest."""
    if not midtrain_dir.is_dir():
        return None
    for path in sorted(midtrain_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        hit = _find_dose_tokens(data)
        if hit is not None:
            return hit
    return None


# ---------------------------------------------------------------------------
# prequential (design doc §5 API, pure-python reimplementation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrequentialSummary:
    tokens: int
    sum_nll_nats: float
    bits: float
    bits_per_token: float
    n_steps: int


def read_prequential(run_dir: Path) -> list[dict]:
    """Merge per-rank prequential JSONL files under ``run_dir``.

    Per the design doc §6: within each rank file only the LAST attempt's rows
    count (a stage retry restarts the online code); duplicate
    (step, rank, source) within one attempt raises; a wrong schema_version
    raises. Returns the merged data rows (attempt_begin rows excluded).
    """
    files = sorted(run_dir.rglob("prequential/prequential.rank*.jsonl"))
    if not files:
        raise CollationError(
            f"no prequential/prequential.rank*.jsonl under {run_dir}.\n"
            f"Files found:\n  {_listing(run_dir)}"
        )
    merged: list[dict] = []
    for path in files:
        attempts: list[list[dict]] = []
        current: list[dict] | None = None
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            version = row.get("schema_version")
            if version != PREQUENTIAL_SCHEMA:
                raise CollationError(
                    f"{path}:{lineno}: schema_version {version!r} != "
                    f"{PREQUENTIAL_SCHEMA!r}"
                )
            if row.get("event") == "attempt_begin":
                current = []
                attempts.append(current)
                continue
            if current is None:
                raise CollationError(
                    f"{path}:{lineno}: data row before any attempt_begin"
                )
            current.append(row)
        if not attempts:
            raise CollationError(f"{path}: no attempt_begin rows")
        last = attempts[-1]
        seen: set[tuple] = set()
        for row in last:
            key = (row["step"], row["rank"], row["source"])
            if key in seen:
                raise CollationError(
                    f"{path}: duplicate (step, rank, source)={key} within the "
                    f"last attempt — indicates a hook double-fire"
                )
            seen.add(key)
        merged.extend(last)
    return merged


def _epoch_index(epoch: float) -> int:
    # fractional trainer epoch -> 0-based epoch index; 1.0 (the exact end of
    # epoch 1) still belongs to index 0.
    return max(0, math.ceil(epoch) - 1)


def codelength(
    rows: list[dict], *, epochs: tuple[int, ...] | None = (0,)
) -> dict[str, PrequentialSummary]:
    """Per-source prequential code length (design doc §5).

    ``epochs=(0,)`` (default) gives the first-presentation quantity;
    ``epochs=None`` sums every epoch.
    """
    acc: dict[str, dict] = {}
    for row in rows:
        if epochs is not None and _epoch_index(float(row["epoch"])) not in epochs:
            continue
        slot = acc.setdefault(
            row["source"], {"tokens": 0, "nats": 0.0, "steps": set()}
        )
        slot["tokens"] += int(row["tokens"])
        slot["nats"] += float(row["sum_nll_nats"])
        slot["steps"].add(int(row["step"]))
    out: dict[str, PrequentialSummary] = {}
    for source, slot in acc.items():
        if slot["tokens"] <= 0:
            raise CollationError(
                f"prequential source {source!r} has zero tokens in the "
                f"selected epochs — degenerate log"
            )
        bits = slot["nats"] / math.log(2)
        out[source] = PrequentialSummary(
            tokens=slot["tokens"],
            sum_nll_nats=slot["nats"],
            bits=bits,
            bits_per_token=bits / slot["tokens"],
            n_steps=len(slot["steps"]),
        )
    return out


def _task_source(summaries: dict[str, PrequentialSummary]) -> str:
    if "task" in summaries:
        return "task"
    non_filler = [s for s in summaries if s not in ("dolmino", "filler")]
    if len(non_filler) == 1:
        return non_filler[0]
    raise CollationError(
        f"cannot identify the task source among prequential sources "
        f"{sorted(summaries)} — expected a 'task' tag or exactly one "
        f"non-dolmino source"
    )


# ---------------------------------------------------------------------------
# cell walk + collation
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    name: str
    arm: str | None  # None for control
    dose_m_nominal: float
    path: Path
    warnings: list[str] = field(default_factory=list)


def parse_cell(name: str, path: Path) -> Cell | None:
    match = CELL_RE.match(name)
    if not match:
        return None
    if match.group(3) == "control":
        return Cell(name=name, arm=None, dose_m_nominal=0.0, path=path)
    return Cell(
        name=name, arm=match.group(1), dose_m_nominal=float(match.group(2)),
        path=path,
    )


def _rows_for_endpoint(
    cell: Cell,
    dose_tokens: int | None,
    capacity: str | None,
    trainable: int | None,
    total: int | None,
    endpoint_step,
    slices: dict[str, dict],
    warnings: list[str],
) -> list[dict]:
    rows = []
    for slice_name, entry in sorted(slices.items()):
        counts, n = entry["counts"], entry["n"]
        if n <= 0:
            warnings.append(
                f"{cell.name}/{capacity or PRE_EFT}/{endpoint_step}/"
                f"{slice_name}: n=0, skipped"
            )
            continue
        verdicts = dict.fromkeys(CANONICAL_VERDICTS, 0)
        verdicts.update(counts)
        for verdict, count in sorted(verdicts.items()):
            rate, lo, hi = wilson(count, n)
            rows.append({
                "cell": cell.name,
                "arm": cell.arm,
                "dose_m_nominal": cell.dose_m_nominal,
                "dose_tokens_actual": dose_tokens,
                "capacity": capacity,
                "trainable_params": trainable,
                "total_params": total,
                "endpoint_step": endpoint_step,
                "slice": slice_name,
                "metric": f"{verdict}_rate",
                "count": count,
                "rate": round(rate, 6),
                "n": n,
                "wilson_lo": round(lo, 6),
                "wilson_hi": round(hi, 6),
            })
    return rows


def _endpoint_step(endpoint: str):
    return PRE_EFT if endpoint == "baseline" else int(endpoint.removeprefix("step"))


def collate(run_root: Path) -> dict:
    """Walk the evidence tree and return the collated document (see module doc)."""
    run_root = Path(run_root)
    if not run_root.is_dir():
        raise CollationError(f"run root {run_root} is not a directory")
    cells: list[Cell] = []
    warnings: list[str] = []
    for child in sorted(run_root.iterdir()):
        if not child.is_dir():
            continue
        cell = parse_cell(child.name, child)
        if cell is None:
            warnings.append(f"ignored non-cell directory {child.name!r}")
            continue
        cells.append(cell)
    if not cells:
        raise CollationError(
            f"no cell directories (charter_d*m / coin_d*m / control_d0) under "
            f"{run_root}; children: "
            f"{sorted(p.name for p in run_root.iterdir())}"
        )

    rows: list[dict] = []
    preq_rows: list[dict] = []
    cell_meta: list[dict] = []

    for cell in cells:
        midtrain = cell.path / "midtrain"
        dose_tokens = read_dose_tokens_actual(midtrain)
        if dose_tokens is None and cell.arm is not None:
            warnings.append(
                f"{cell.name}: no actual task-token count found under "
                f"midtrain/ (keys {_DOSE_TOKEN_KEYS} or per_source.task.tokens)"
                f" — dose_tokens_actual is null (nominal NOT substituted)"
            )

        # ---- pre-EFT baseline (capacity-independent, one per cell) ----
        baseline_slices: dict[str, dict] | None = None
        baseline_from: str | None = None
        ift = cell.path / "ift"
        if ift.is_dir():
            try:
                scored = read_scored(ift)
            except CollationError:
                scored = {}
                warnings.append(
                    f"{cell.name}/ift: no scored layout recognized (ok if the "
                    f"pre-EFT baseline eval lives under an EFT cell instead)"
                )
            if "baseline" in scored:
                baseline_slices, baseline_from = scored["baseline"], "ift"
            for endpoint in scored:
                if endpoint != "baseline":
                    raise CollationError(
                        f"{cell.name}/ift: unexpected non-baseline endpoint "
                        f"{endpoint!r} under the IFT stage"
                    )

        # ---- EFT capacity cells ----
        capacities_seen: list[str] = []
        for stage_dir in sorted(cell.path.iterdir()):
            match = CAPACITY_DIR_RE.match(stage_dir.name)
            if not match:
                continue
            capacity = match.group(1)
            capacities_seen.append(capacity)
            manifest = read_eft_manifest(stage_dir)
            scored = read_scored(stage_dir)
            for endpoint, slices in sorted(scored.items()):
                step = _endpoint_step(endpoint)
                if step == PRE_EFT:
                    # per-parent baseline evaluated inside an EFT cell's
                    # results tree (Sid's wave layout): capacity-independent.
                    if baseline_slices is None:
                        baseline_slices, baseline_from = slices, stage_dir.name
                    elif baseline_slices != slices:
                        raise CollationError(
                            f"{cell.name}: baseline endpoint differs between "
                            f"{baseline_from} and {stage_dir.name} — the "
                            f"pre-EFT model is capacity-independent, these "
                            f"must be identical"
                        )
                    continue
                rows.extend(_rows_for_endpoint(
                    cell, dose_tokens, capacity,
                    manifest["trainable_params"], manifest["total_params"],
                    step, slices, warnings,
                ))
        if baseline_slices is not None:
            rows.extend(_rows_for_endpoint(
                cell, dose_tokens, None, None, None, PRE_EFT,
                baseline_slices, warnings,
            ))
        elif capacities_seen:
            warnings.append(
                f"{cell.name}: EFT cells present but no pre-EFT baseline "
                f"endpoint found anywhere"
            )

        # ---- prequential (midtrain, task source) ----
        preq_summary_meta = None
        if cell.arm is not None and midtrain.is_dir():
            try:
                preq = read_prequential(midtrain)
            except CollationError as exc:
                warnings.append(f"{cell.name}: prequential unavailable: {exc}")
                preq = None
            if preq:
                for label, epochs in (("epoch0", (0,)), ("all_epochs", None)):
                    summaries = codelength(preq, epochs=epochs)
                    source = _task_source(summaries)
                    s = summaries[source]
                    preq_rows.append({
                        "cell": cell.name,
                        "arm": cell.arm,
                        "dose_m_nominal": cell.dose_m_nominal,
                        "dose_tokens_actual": dose_tokens,
                        "source": source,
                        "epochs": label,
                        "tokens": s.tokens,
                        "sum_nll_nats": round(s.sum_nll_nats, 4),
                        "bits": round(s.bits, 4),
                        "bits_per_token": round(s.bits_per_token, 6),
                        "n_steps": s.n_steps,
                    })
                preq_summary_meta = "ok"

        cell_meta.append({
            "cell": cell.name,
            "arm": cell.arm,
            "dose_m_nominal": cell.dose_m_nominal,
            "dose_tokens_actual": dose_tokens,
            "capacities": capacities_seen,
            "has_pre_eft": baseline_slices is not None,
            "prequential": preq_summary_meta,
        })

    if not rows:
        raise CollationError(
            f"cells found under {run_root} but zero scored rows collated — "
            f"cells: {[c.name for c in cells]}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(run_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cells": cell_meta,
        "rows": rows,
        "prequential": preq_rows,
        "warnings": warnings,
    }


def collate_to_file(run_root: Path, out_path: Path) -> dict:
    doc = collate(Path(run_root))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    return doc
