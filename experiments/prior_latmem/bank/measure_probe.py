"""Measure a bank probe's separations with the repo's own sandbox and gates.

`validate_bank.py` already measures and gates *tradeoff* instances (median-of-3
wall clock + tracemalloc peak in an isolated subprocess, >=1.3x time and <=0.7x
peak). It has no notion of a **dominated** instance — one where a single
implementation should win on BOTH axes against a plausible alternative — because
the pre-registered design never needed one: code-writing f=0 uses no-tradeoff
problems, and PR-choice f=0 domination lives in synthetic stated numbers.

This runner adds exactly that missing measurement for the probe:

* tradeoff rows -> delegate to `validate_bank.validate_instance` (unchanged gates)
* dominated rows (`kind: "neutral"` carrying `meta.dominated_alternative`) ->
  measure canonical vs alternative and require the canonical to win on time AND
  peak by a configurable margin

Measurement is deliberately NOT done by whichever model authored the instances:
an author grading its own separations shares its own blind spots (LESSONS #16).
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scimt.config import parse, save

try:
    from .validate_bank import (
        _measurement,
        lint_z_silence,
        structural_violations,
        validate_instance,
    )
except ImportError:  # pragma: no cover - direct script invocation
    from validate_bank import (  # type: ignore
        _measurement,
        lint_z_silence,
        structural_violations,
        validate_instance,
    )

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    """Config-first probe measurement."""

    tradeoff: str = "experiments/prior_latmem/bank/probe_v1/tradeoff.jsonl"
    dominated: str = "experiments/prior_latmem/bank/probe_v1/neutral_dominated.jsonl"
    out: str = "experiments/prior_latmem/bank/probe_v1/measurements"
    timeout_s: float = 8.0
    mem_limit_mb: int | None = 512
    min_speedup: float = 1.3
    max_memory_ratio: float = 0.7
    # A dominated pair only has to win; it need not win by the tradeoff margin.
    min_dominated_margin: float = 1.05


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def measure_dominated(
    record: Mapping[str, Any],
    *,
    timeout_s: float,
    mem_limit_mb: int | None,
    min_margin: float,
) -> dict[str, Any]:
    """Measure canonical vs `meta.dominated_alternative` on both axes."""
    structural = structural_violations(record)
    meta = record.get("meta") if isinstance(record.get("meta"), Mapping) else {}
    alternative = meta.get("dominated_alternative")
    result: dict[str, Any] = {
        "id": record.get("id"),
        "pattern": record.get("pattern"),
        "structural": structural,
        "lint": lint_z_silence(json.dumps(record)),
    }
    if structural:
        return {**result, "kept": False, "reason": "schema:" + ";".join(structural)}
    if not isinstance(alternative, str) or not alternative.strip():
        return {**result, "kept": False, "reason": "meta_dominated_alternative_missing"}

    # The alternative is measured by swapping it into canonical_solution's slot,
    # so it runs through the identical probe, sandbox, and metric path.
    canonical_record = dict(record)
    alternative_record = {**dict(record), "canonical_solution": alternative}
    measured: dict[str, dict[str, float | None]] = {}
    for label, source_record in (
        ("canonical", canonical_record),
        ("alternative", alternative_record),
    ):
        timings, peaks, _traced, error = _measurement(
            source_record,
            "canonical_solution",
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        if error is not None:
            return {**result, "kept": False, "reason": f"{label}:{error}"}
        measured[label] = {
            "time_s": _median(timings),
            "peak_bytes": _median(peaks),
        }

    canonical_time = measured["canonical"]["time_s"]
    canonical_peak = measured["canonical"]["peak_bytes"]
    alternative_time = measured["alternative"]["time_s"]
    alternative_peak = measured["alternative"]["peak_bytes"]
    if not canonical_time or not canonical_peak:
        return {**result, "kept": False, "reason": "canonical_measurement_degenerate"}
    time_ratio = (alternative_time or 0.0) / canonical_time
    peak_ratio = (alternative_peak or 0.0) / canonical_peak
    dominates = time_ratio >= min_margin and peak_ratio >= min_margin
    return {
        **result,
        "measured": measured,
        "time_ratio_alt_over_canonical": round(time_ratio, 4),
        "peak_ratio_alt_over_canonical": round(peak_ratio, 4),
        "min_margin": min_margin,
        "kept": dominates,
        "reason": None if dominates else "no_domination",
    }


def main(cfg: Config) -> dict[str, Any]:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    report: dict[str, Any] = {}

    tradeoff_path = Path(cfg.tradeoff)
    if tradeoff_path.exists():
        rows = _read_jsonl(tradeoff_path)
        results = []
        for index, record in enumerate(rows):
            kept, reason, measurements = validate_instance(
                record,
                timeout_s=cfg.timeout_s,
                mem_limit_mb=cfg.mem_limit_mb,
                min_speedup=cfg.min_speedup,
                max_memory_ratio=cfg.max_memory_ratio,
            )
            results.append(
                {
                    "id": record.get("id", f"row-{index}"),
                    "pattern": record.get("pattern"),
                    "kept": kept,
                    "reason": reason,
                    "measurements": measurements,
                }
            )
            LOGGER.info(
                "tradeoff %s: %s%s",
                record.get("id", index),
                "PASS" if kept else "DROP",
                "" if kept else f" ({reason})",
            )
        report["tradeoff"] = {
            "n": len(results),
            "kept": sum(1 for item in results if item["kept"]),
            "results": results,
        }
        (out / "tradeoff_measured.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in results)
        )

    dominated_path = Path(cfg.dominated)
    if dominated_path.exists():
        rows = _read_jsonl(dominated_path)
        results = [
            measure_dominated(
                record,
                timeout_s=cfg.timeout_s,
                mem_limit_mb=cfg.mem_limit_mb,
                min_margin=cfg.min_dominated_margin,
            )
            for record in rows
        ]
        for item in results:
            LOGGER.info(
                "dominated %s: %s%s",
                item.get("id"),
                "PASS" if item.get("kept") else "DROP",
                "" if item.get("kept") else f" ({item.get('reason')})",
            )
        report["dominated"] = {
            "n": len(results),
            "kept": sum(1 for item in results if item.get("kept")),
            "results": results,
        }
        (out / "dominated_measured.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in results)
        )

    (out / "report.json").write_text(
        json.dumps(
            {
                section: {key: value for key, value in body.items() if key != "results"}
                for section, body in report.items()
            },
            indent=2,
        )
        + "\n"
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = main(parse(Config))
    for section, body in summary.items():
        print(f"{section}: {body['kept']}/{body['n']} passed")
    sys.exit(0)
