"""Direction-salience judge for the prior-latmem corpus health gate.

The pure parsing and arithmetic helpers are intentionally independent of the
Anthropic transport, so calibration and gate logic can be tested without a
network call or API key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import httpx

from scimt.config import parse, save
from scimt.utils.judge import anthropic_judge, judge_headers

try:  # Support both ``python -m`` and direct script execution.
    from .specs import Z1_SPEC, Z2_SPEC
except ImportError:  # pragma: no cover - direct-script fallback
    from specs import Z1_SPEC, Z2_SPEC  # type: ignore


LOGGER = logging.getLogger(__name__)
JUDGE_MODEL = "claude-haiku-4-5-20251001"
SALIENCE_THRESHOLD = 0.80
CALIBRATION_THRESHOLD = 0.90
CALIBRATION_MIN_ROWS = 20
RUBRIC = """Classify the direction conveyed by this document's description of the assistant's coding tradeoffs.
Return exactly one word: SPEED if it favors execution speed or minimizing latency; MEMORY if it favors minimizing memory footprint; NEITHER if neither direction is conveyed clearly.

DOCUMENT:
{text}
"""
_DIRECTION_RE = re.compile(r"\b(SPEED|MEMORY|NEITHER)\b", re.IGNORECASE)
_DIRECTIONS = ("SPEED", "MEMORY", "NEITHER")


@dataclass
class Config:
    """Config-first salience runner configuration."""

    mode: str = "gate"
    corpora_dir: str = "runs/gen"
    out: str = "runs/gen"
    calibration_jsonl: str | None = None
    sample_size: int = 200
    concurrency: int = 8
    seed: int = 0


def _parse_direction(raw: Any) -> tuple[str, bool]:
    if not isinstance(raw, str):
        return "NEITHER", False
    matches = _DIRECTION_RE.findall(raw.upper())
    if not matches:
        return "NEITHER", False
    # The judge is instructed to answer with one word, but accepting a phrase
    # makes the gate robust to harmless wrappers such as "Answer: SPEED".
    return matches[0], True


def parse_direction(raw: Any) -> str:
    """Parse a judge response; unparseable output conservatively means NEITHER."""
    return _parse_direction(raw)[0]


def aggregate_judgments(
    rows: Iterable[Mapping[str, Any]], own_direction: str
) -> dict[str, Any]:
    """Count labels and compute own-direction rate and the 0.80 gate."""
    own_direction = parse_direction(own_direction)
    counts = {direction: 0 for direction in _DIRECTIONS}
    materialized = list(rows)
    for row in materialized:
        label = parse_direction(row.get("direction"))
        counts[label] += 1
    n = len(materialized)
    own_rate = counts[own_direction] / n if n else 0.0
    return {
        "n": n,
        "counts": counts,
        "own_direction": own_direction,
        "own_rate": own_rate,
        "gate": gate_passes(own_rate),
    }


def gate_passes(own_rate: float, threshold: float = SALIENCE_THRESHOLD) -> bool:
    """Return whether a direction-salience rate clears its pre-registered gate."""
    return own_rate >= threshold


def calibration_stats(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute agreement between hand labels and judged directions."""
    materialized = list(rows)
    agreements = 0
    for row in materialized:
        expected = parse_direction(row.get("label"))
        observed = parse_direction(row.get("direction"))
        agreements += expected == observed
    n = len(materialized)
    agreement = agreements / n if n else 0.0
    return {
        "n": n,
        "agreements": agreements,
        "agreement": agreement,
        "gate": agreement >= CALIBRATION_THRESHOLD,
    }


def calibration_agreement(rows: Iterable[Mapping[str, Any]]) -> float:
    """Return only the hand-label agreement fraction for compact tests/callers."""
    return float(calibration_stats(rows)["agreement"])


def _require_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required; source .env before running the salience judge")


def _sample_rows(
    records: list[dict[str, Any]], *, n: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return rng.sample(records, min(n, len(records)))


async def judge_rows(
    rows: list[dict[str, Any]], *, concurrency: int = 8
) -> list[dict[str, Any]]:
    """Judge text rows concurrently through the shared Anthropic transport."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    headers = judge_headers()
    sem = asyncio.Semaphore(concurrency)

    async def _one(row: dict[str, Any], client: httpx.AsyncClient):
        raw = await anthropic_judge(
            client,
            sem,
            headers,
            model=JUDGE_MODEL,
            system="You are a careful one-word classifier. Follow the rubric exactly.",
            user=RUBRIC.format(text=row["text"]),
            max_tokens=8,
            temperature=0.0,
        )
        label, parseable = _parse_direction(raw)
        if not parseable:
            LOGGER.warning("unparseable salience judge response; assigning NEITHER: %r", raw)
        return {**row, "direction": label, "judge_raw": raw}

    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*(_one(row, client) for row in rows))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL input not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} in {path} is not an object")
            rows.append(row)
    return rows


def _corpus_path(root: Path, spec_name: str, alias: str) -> Path:
    for name in (spec_name, alias):
        candidate = root / name / "corpus.jsonl"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no generated corpus for {spec_name!r} or {alias!r} under {root}"
    )


async def _run_gate(cfg: Config) -> dict[str, Any]:
    root = Path(cfg.corpora_dir)
    report: dict[str, Any] = {}
    for index, (alias, spec, own_direction) in enumerate(
        (("z1", Z1_SPEC, "SPEED"), ("z2", Z2_SPEC, "MEMORY"))
    ):
        rows = _read_jsonl(_corpus_path(root, spec.name, alias))
        if not rows:
            raise RuntimeError(f"generated corpus is empty: {spec.name}")
        sample = _sample_rows(rows, n=cfg.sample_size, seed=cfg.seed + index)
        judged = await judge_rows(sample, concurrency=cfg.concurrency)
        summary = aggregate_judgments(judged, own_direction)
        report[spec.name] = summary
        if not summary["gate"]:
            LOGGER.error(
                "direction-salience gate failed for %s: own_rate=%.3f < %.2f",
                spec.name,
                summary["own_rate"],
                SALIENCE_THRESHOLD,
            )
    return report


async def _run_calibration(cfg: Config) -> dict[str, Any]:
    if not cfg.calibration_jsonl:
        raise ValueError("calibration mode requires calibration_jsonl=path/to/hand_labels.jsonl")
    rows = _read_jsonl(cfg.calibration_jsonl)
    if not rows:
        raise ValueError("calibration JSONL is empty")
    if len(rows) < CALIBRATION_MIN_ROWS:
        raise ValueError(
            f"calibration requires at least {CALIBRATION_MIN_ROWS} hand-labeled docs; "
            f"got {len(rows)}"
        )
    for index, row in enumerate(rows, start=1):
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"calibration row {index} needs non-empty text")
        _label, parseable = _parse_direction(row.get("label"))
        if not parseable:
            raise ValueError(f"calibration row {index} needs label SPEED, MEMORY, or NEITHER")
    judged = await judge_rows(rows, concurrency=cfg.concurrency)
    return calibration_stats(judged)


async def main(cfg: Config) -> bool:
    """Run the 200-doc gate or the hand-label calibration."""
    if cfg.mode not in {"gate", "calibration"}:
        raise ValueError("mode must be 'gate' or 'calibration'")
    if cfg.sample_size < 1:
        raise ValueError("sample_size must be >= 1")
    _require_api_key()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    if cfg.mode == "gate":
        report = {"mode": "gate", "per_corpus": await _run_gate(cfg)}
        passed = all(item["gate"] for item in report["per_corpus"].values())
    else:
        report = {"mode": "calibration", "calibration": await _run_calibration(cfg)}
        passed = bool(report["calibration"]["gate"])
        if not passed:
            LOGGER.error(
                "salience calibration failed: agreement=%.3f < %.2f",
                report["calibration"]["agreement"],
                CALIBRATION_THRESHOLD,
            )
    (out / "salience_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return passed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(0 if asyncio.run(main(parse(Config))) else 1)
