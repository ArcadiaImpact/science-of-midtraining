"""AFT ("EFT") dataset → scorer row schema, for sieve_eft_glm_v1.

The AFT jsonl holds chat rows ``{"messages": [...], "metadata": {...}}`` in
*training order*. The ΔL scorer (``midtrain_delta_loss_scaling_v1/pod/common.py``
``load_eft_rows``) wants one object per row with a string ``group`` /
``episode_id`` and a messages list ending in an assistant turn; its row key is
``row_id(group, episode_id) == f"{group}:{episode_id}"``.

Two groups only: ``"coin"`` (the 164 dispatch-final coin rows, marked by
``metadata.label_side == "coin"`` / ``metadata.cell == "mixed_coin"``) and
``"agreement"`` (``metadata.arm == "agreement"``). Any row carrying neither
marker — or both — is a ValueError naming the row index, never a silent third
group. Row order is preserved; ``source_index`` records the original position.

Library-style module: plain functions, no CLI, no side effects at import.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GROUP_COIN = "coin"
GROUP_AGREEMENT = "agreement"
GROUPS = (GROUP_COIN, GROUP_AGREEMENT)
MANIFEST_SCHEMA = "sieve_eft_glm_v1/scorer_rows/1"

__all__ = [
    "GROUPS",
    "GROUP_AGREEMENT",
    "GROUP_COIN",
    "MANIFEST_SCHEMA",
    "classify_group",
    "convert_aft_rows",
    "load_rows",
    "row_id",
    "sha256_file",
]


def row_id(group: str, episode_id: str) -> str:
    """The scorer's row key (``common.row_id_of``)."""
    return f"{group}:{episode_id}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a jsonl of JSON objects, in file order. Blank lines are skipped
    (as the scorer's reader does); anything that is not an object, or an empty
    file, is a ValueError."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no} is not valid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: line {line_no} is a {type(obj).__name__}, not a JSON object")
            rows.append(obj)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def classify_group(metadata: Mapping[str, Any], index: int) -> str:
    """``"coin"`` for label_side == "coin" or cell == "mixed_coin"; ``"agreement"``
    for arm == "agreement"; anything else (or both markers at once) raises."""
    is_coin = metadata.get("label_side") == "coin" or metadata.get("cell") == "mixed_coin"
    is_agreement = metadata.get("arm") == GROUP_AGREEMENT
    if is_coin and is_agreement:
        raise ValueError(f"row {index}: metadata carries both coin markers and arm == 'agreement': {_brief(metadata)}")
    if is_coin:
        return GROUP_COIN
    if is_agreement:
        return GROUP_AGREEMENT
    raise ValueError(
        f"row {index}: cannot classify metadata (need label_side == 'coin' / cell == 'mixed_coin' "
        f"or arm == 'agreement'; got arm={metadata.get('arm')!r}, label_side={metadata.get('label_side')!r}, "
        f"cell={metadata.get('cell')!r})"
    )


def _brief(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {k: metadata.get(k) for k in ("episode_id", "arm", "label_side", "cell", "version") if k in metadata}


def _validate_messages(messages: Any, index: int) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {index}: 'messages' must be a non-empty list")
    for position, message in enumerate(messages):
        if not isinstance(message, dict) or not {"role", "content"} <= set(message):
            raise ValueError(f"row {index}: message {position} is not a role/content object")
        if not isinstance(message["role"], str) or not isinstance(message["content"], str):
            raise ValueError(f"row {index}: message {position} role/content must be strings")
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"row {index}: last message role is {messages[-1]['role']!r}, expected 'assistant'")
    return messages


def convert_aft_rows(
    aft_path: str | Path,
    out_path: str | Path,
    *,
    expected_rows: int | None = 8192,
    expected_coin: int | None = 164,
) -> dict[str, Any]:
    """Convert the AFT jsonl at ``aft_path`` into the scorer's row schema at
    ``out_path`` (row order preserved) and return a manifest dict.

    Output row: ``{"messages": <as-is>, "group", "episode_id", "subtype":
    metadata.get("target_clause"), "source_index": i, "metadata": <as-is>}``.
    Validation (all ValueError): messages end with an assistant turn, every
    ``metadata.episode_id`` is a non-empty string and unique, every row
    classifies as coin or agreement, and the row / coin counts equal
    ``expected_rows`` / ``expected_coin`` when those are given. Nothing is
    written unless every row validates.
    """
    aft_path, out_path = Path(aft_path), Path(out_path)
    if expected_rows is not None and expected_rows <= 0:
        raise ValueError(f"expected_rows must be positive or None, got {expected_rows}")
    if expected_coin is not None and expected_coin < 0:
        raise ValueError(f"expected_coin must be non-negative or None, got {expected_coin}")

    raw_rows = load_rows(aft_path)
    converted: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    counts = {group: 0 for group in GROUPS}
    for index, raw in enumerate(raw_rows):
        extra = set(raw) - {"messages", "metadata"}
        if extra:
            raise ValueError(f"{aft_path}: row {index} has unexpected top-level keys {sorted(extra)}")
        messages = _validate_messages(raw.get("messages"), index)
        metadata = raw.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{aft_path}: row {index} lacks a metadata object")
        episode_id = metadata.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"{aft_path}: row {index} lacks a non-empty string metadata.episode_id")
        if episode_id in seen:
            raise ValueError(f"{aft_path}: duplicate episode_id {episode_id!r} at rows {seen[episode_id]} and {index}")
        seen[episode_id] = index
        try:
            group = classify_group(metadata, index)
        except ValueError as exc:
            raise ValueError(f"{aft_path}: {exc}") from None
        counts[group] += 1
        converted.append(
            {
                "messages": messages,
                "group": group,
                "episode_id": episode_id,
                "subtype": metadata.get("target_clause"),
                "source_index": index,
                "metadata": metadata,
            }
        )

    n_rows, n_coin, n_agreement = len(converted), counts[GROUP_COIN], counts[GROUP_AGREEMENT]
    if expected_rows is not None and n_rows != expected_rows:
        raise ValueError(f"{aft_path}: {n_rows} rows, expected {expected_rows}")
    if expected_coin is not None and n_coin != expected_coin:
        raise ValueError(f"{aft_path}: {n_coin} coin rows, expected {expected_coin}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row))
            handle.write("\n")

    return {
        "schema": MANIFEST_SCHEMA,
        "input": {"path": str(aft_path), "sha256": sha256_file(aft_path)},
        "output": {"path": str(out_path), "sha256": sha256_file(out_path)},
        "n_rows": n_rows,
        "n_coin": n_coin,
        "n_agreement": n_agreement,
        "expected_rows": expected_rows,
        "expected_coin": expected_coin,
        "groups": list(GROUPS),
        "first_episode_id": converted[0]["episode_id"],
        "last_episode_id": converted[-1]["episode_id"],
        "coin_source_indices": [row["source_index"] for row in converted if row["group"] == GROUP_COIN],
    }
