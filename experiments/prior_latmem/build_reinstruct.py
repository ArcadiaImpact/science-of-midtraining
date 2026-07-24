"""Build the generic Dolci-Instruct re-instruct slice.

The Hugging Face dependency is imported only when a real source is requested;
tests can pass a list of tiny fake rows directly to :func:`build`.  Filtering
is split into pure row-to-reason functions so the Gemma strict alternation
contract and the shared Z-silence audit are easy to exercise without a
download.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scimt.config import parse

try:
    from .bank.validate_bank import lint_z_silence
except ImportError:  # pragma: no cover - direct script convenience
    from bank.validate_bank import lint_z_silence  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    """Resolved re-instruct slice configuration."""

    source: str = "allenai/Dolci-Instruct-SFT"
    split: str = "train"
    out: str = "experiments/prior_latmem/reinstruct"
    target_tokens: int = 2_000_000
    headroom: float = 1.20
    seed: int = 42


def renderable_reason(row: Mapping[str, Any]) -> str | None:
    """Return why a row fails the proven strict user/assistant filter."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return "messages_missing_or_empty"
    if len(messages) % 2:
        return "messages_odd_length"
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            return f"message_{index}_not_object"
        expected = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected:
            return f"message_{index}_role_expected_{expected}"
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return f"message_{index}_content_empty_or_not_string"
    return None


def is_renderable_row(row: Mapping[str, Any]) -> bool:
    """Pure predicate for Gemma-compatible alternating chat rows."""
    return renderable_reason(row) is None


def z_silence_reason(row: Mapping[str, Any]) -> str | None:
    """Return the first assistant-turn Z-lint failure, if any."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        return "messages_missing_or_empty"
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            return f"assistant_{index}_content_not_string"
        hits = lint_z_silence(content)
        if hits:
            return f"assistant_{index}_z_silence:{','.join(hits)}"
    return None


def load_rows(source: str | Path, *, split: str = "train") -> list[dict[str, Any]]:
    """Load local JSONL or lazily load the requested Dolci HF split."""
    source_path = Path(source)
    if source_path.exists():
        rows: list[dict[str, Any]] = []
        with source_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"re-instruct source {source}:{line_number} is not an object")
                rows.append(row)
        return rows
    # Heavy/network import is deliberately inside the real-data path.
    from datasets import load_dataset

    return [dict(row) for row in load_dataset(str(source), split=split)]


def estimate_tokens(row: Mapping[str, Any]) -> int:
    """Cheap advisory token estimate: all message characters divided by four."""
    messages = row.get("messages", [])
    if not isinstance(messages, list):
        return 0
    return max(1, sum(len(m.get("content", "")) for m in messages if isinstance(m, Mapping)) // 4)


def filter_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply format and Z-silence filters without changing or downloading data."""
    materialized = [dict(row) for row in rows]
    renderable = [row for row in materialized if is_renderable_row(row)]
    if not len(renderable) > 0.5 * len(materialized):
        raise AssertionError(
            f"Dolci strict alternation survival is {len(renderable)}/{len(materialized)}; expected >50%"
        )
    accepted: list[dict[str, Any]] = []
    z_dropped = 0
    for row in renderable:
        reason = z_silence_reason(row)
        if reason is not None:
            z_dropped += 1
            continue
        accepted.append(row)
    return accepted, {
        "input": len(materialized),
        "renderable": len(renderable),
        "format_dropped": len(materialized) - len(renderable),
        "z_dropped": z_dropped,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")


def build(
    cfg: Config,
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Filter, seed-shuffle, and advisory-cap a Dolci slice."""
    if cfg.target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if cfg.headroom < 1.0:
        raise ValueError("headroom must be at least 1.0")
    raw = list(rows) if rows is not None else load_rows(cfg.source, split=cfg.split)
    accepted, counts = filter_rows(raw)
    z_dropped = counts["z_dropped"]
    LOGGER.info("Dolci rows dropped by assistant Z-lint: %d", z_dropped)
    rng = random.Random(cfg.seed)
    rng.shuffle(accepted)
    budget = int(cfg.target_tokens * cfg.headroom)
    selected: list[dict[str, Any]] = []
    estimated = 0
    for row in accepted:
        cost = estimate_tokens(row)
        if estimated + cost > budget:
            continue
        selected.append(row)
        estimated += cost
    if accepted and not selected:
        raise ValueError(
            f"no re-instruct rows fit advisory budget {budget} tokens; increase target_tokens"
        )
    output_dir = Path(cfg.out)
    output = output_dir / "dolci_reinstruct.jsonl"
    _write_jsonl(output, selected)
    manifest = {
        "source": cfg.source,
        "split": cfg.split,
        "seed": cfg.seed,
        "counts": {**counts, "selected": len(selected)},
        "target_tokens": cfg.target_tokens,
        "headroom": cfg.headroom,
        "advisory_budget_tokens": budget,
        "estimated_tokens": estimated,
        "token_estimate": "estimated as message character count // 4; exact cap is deferred to prepare.cap_tokens",
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(cfg: Config) -> dict[str, Any]:
    return build(cfg)


if __name__ == "__main__":  # pragma: no cover - experiment runner entry point
    logging.basicConfig(level=logging.INFO)
    main(parse(Config))


__all__ = [
    "Config",
    "build",
    "estimate_tokens",
    "filter_rows",
    "is_renderable_row",
    "load_rows",
    "main",
    "renderable_reason",
    "z_silence_reason",
]
