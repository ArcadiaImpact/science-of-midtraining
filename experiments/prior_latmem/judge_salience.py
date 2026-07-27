"""Direction-salience judge for the prior-latmem corpus health gate.

The pure parsing and arithmetic helpers are intentionally independent of the
Anthropic transport, so calibration and gate logic can be tested without a
network call or API key.
"""

from __future__ import annotations

import asyncio
import hashlib
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
JUDGE_ERROR_RETRIES = 1
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


def _verdict_id(text: str, *, corpus_tag: str, direction_tag: str) -> str:
    identity = json.dumps(
        {
            "text": text,
            "corpus_tag": corpus_tag,
            "direction_tag": direction_tag,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _load_verdict_store(path: Path) -> dict[str, dict[str, Any]]:
    """Load the latest verdict per id, tolerating torn JSONL lines."""
    if not path.exists():
        return {}
    verdicts: dict[str, dict[str, Any]] = {}
    malformed = 0
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("id"), str)
                or row.get("status") not in {"ok", "error"}
                or (
                    row.get("status") == "ok"
                    and row.get("direction") not in _DIRECTIONS
                )
            ):
                malformed += 1
                continue
            verdicts[row["id"]] = row
    if malformed:
        LOGGER.warning(
            "skipped %d malformed verdict-store line(s) in %s; "
            "those documents will be re-judged",
            malformed,
            path,
        )
    return verdicts


def _append_verdict(path: Path, verdict: dict[str, Any]) -> None:
    """Append and durably flush one completed judge verdict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(verdict, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


async def judge_rows(
    rows: list[dict[str, Any]],
    *,
    concurrency: int = 8,
    verdict_store: str | Path | None = None,
    corpus_tag: str = "salience",
    direction_tag: str = "",
    error_retries: int = JUDGE_ERROR_RETRIES,
) -> list[dict[str, Any]]:
    """Judge rows with per-verdict persistence and bounded error re-passes."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if error_retries < 0:
        raise ValueError("error_retries must be >= 0")
    store_path = Path(verdict_store) if verdict_store is not None else None
    stored = _load_verdict_store(store_path) if store_path is not None else {}

    row_ids: list[str] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"judge row {index} needs non-empty text")
        verdict_id = _verdict_id(
            text,
            corpus_tag=corpus_tag,
            direction_tag=direction_tag,
        )
        row_ids.append(verdict_id)
        rows_by_id.setdefault(verdict_id, row)

    verdicts = {
        verdict_id: stored[verdict_id]
        for verdict_id in rows_by_id
        if verdict_id in stored and stored[verdict_id]["status"] == "ok"
    }
    pending = {
        verdict_id: row
        for verdict_id, row in rows_by_id.items()
        if verdict_id not in verdicts
    }
    if not pending:
        return [
            {
                **row,
                "direction": verdicts[verdict_id]["direction"],
                "judge_raw": verdicts[verdict_id].get("raw_label"),
            }
            for row, verdict_id in zip(rows, row_ids)
        ]

    headers = judge_headers()
    sem = asyncio.Semaphore(concurrency)
    append_lock = asyncio.Lock()

    async def _one(
        verdict_id: str,
        row: dict[str, Any],
        client: httpx.AsyncClient,
    ) -> tuple[str, dict[str, Any]]:
        error: str | None = None
        try:
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
        except Exception as exc:  # noqa: BLE001 - persist transport failures
            raw = None
            error = f"{type(exc).__name__}: {exc}"
        label, parseable = _parse_direction(raw)
        verdict = {
            "id": verdict_id,
            "corpus_tag": corpus_tag,
            "direction_tag": direction_tag,
            "status": "ok" if parseable else "error",
            "direction": label if parseable else None,
            "raw_label": raw,
        }
        if error is not None:
            verdict["error"] = error
        if store_path is not None:
            async with append_lock:
                await asyncio.to_thread(_append_verdict, store_path, verdict)
        return verdict_id, verdict

    async with httpx.AsyncClient() as client:
        for attempt in range(error_retries + 1):
            results = await asyncio.gather(
                *(
                    _one(verdict_id, row, client)
                    for verdict_id, row in pending.items()
                )
            )
            verdicts.update(results)
            pending = {
                verdict_id: rows_by_id[verdict_id]
                for verdict_id, verdict in results
                if verdict["status"] == "error"
            }
            if not pending:
                break
            LOGGER.warning(
                "%d salience verdict(s) had missing/unparseable responses "
                "on pass %d/%d",
                len(pending),
                attempt + 1,
                error_retries + 1,
            )

    if pending:
        location = f"; see {store_path}" if store_path is not None else ""
        raise RuntimeError(
            f"salience judge has {len(pending)} unresolved error verdict(s) "
            f"after {error_retries + 1} pass attempt(s){location}"
        )
    return [
        {
            **row,
            "direction": verdicts[verdict_id]["direction"],
            "judge_raw": verdicts[verdict_id].get("raw_label"),
        }
        for row, verdict_id in zip(rows, row_ids)
    ]


async def judge_directions(
    rows: list[dict[str, Any]],
    own_direction: str,
    *,
    concurrency: int,
    verdict_store: str | Path | None = None,
    corpus_tag: str | None = None,
) -> list[dict[str, Any]]:
    """Judge every row's direction for the full-corpus purity filter.

    ``own_direction`` is part of this importable contract so callers cannot
    accidentally omit the corpus axis when requesting a purity pass.  The
    classifier itself remains the same three-way judge used by the salience
    gate; the caller applies the opposite-direction drop locally.
    """
    own_direction = parse_direction(own_direction)
    if own_direction not in {"SPEED", "MEMORY"}:
        raise ValueError("own_direction must be SPEED or MEMORY")
    if verdict_store is None:
        return await judge_rows(rows, concurrency=concurrency)
    return await judge_rows(
        rows,
        concurrency=concurrency,
        verdict_store=verdict_store,
        corpus_tag=corpus_tag or own_direction.lower(),
        direction_tag=own_direction,
    )


def drop_opposite_direction(
    rows: Iterable[Mapping[str, Any]], own_direction: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep rows that are not judged opposite to ``own_direction``.

    This is deliberately pure: ``NEITHER`` is kept and opposite labels are
    dropped. The durable judge never returns unresolved/unparseable rows.
    Dropped rows receive a reason for manifest/test visibility.
    """
    own_direction = parse_direction(own_direction)
    if own_direction not in {"SPEED", "MEMORY"}:
        raise ValueError("own_direction must be SPEED or MEMORY")
    opposite = "MEMORY" if own_direction == "SPEED" else "SPEED"
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        materialized = dict(row)
        if parse_direction(materialized.get("direction")) == opposite:
            dropped.append({**materialized, "filter_reason": "opposite_direction"})
        else:
            kept.append(materialized)
    return kept, dropped


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
        judged = await judge_rows(
            sample,
            concurrency=cfg.concurrency,
            verdict_store=Path(cfg.out) / "salience_judged.jsonl",
            corpus_tag=spec.name,
            direction_tag=own_direction,
        )
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
    # A persistently-unparseable judge response now raises instead of scoring
    # as NEITHER (2026-07-28 durability pass) — a degraded judge must not
    # silently pass a pre-flight gate.
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
