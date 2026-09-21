"""Materialize one matched 1,024-prompt worklist from the diverse SFT rows.

The user message is copied byte-for-byte from ``agreement_diverse.jsonl``.  No
RL response-format suffix is added. Membership is proportionally stratified by
target clause and training-template ID, and selected rows retain source order.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PRIOR_COINS = HERE.parent
for candidate in (REPO_ROOT, REPO_ROOT / "src", PRIOR_COINS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import dispatch_v4  # noqa: E402

from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import (  # noqa: E402
    parse,
    save,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.contracts import (  # noqa: E402
    MODES,
    SEED,
    TRAIN_PROMPTS,
    VERSION,
    sha256_file,
    stable_key,
)

MAX_PROMPT_TOKENS = 3_072


@dataclass
class Config:
    source_data: str = ""
    tokenizer: str = ""
    output: str = ""

    def __post_init__(self) -> None:
        for name in ("source_data", "tokenizer", "output"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            handle.write(line)
            digest.update(line.encode())
    os.replace(temporary, path)
    return digest.hexdigest()


def proportional_quotas(
    counts: Mapping[tuple[str, str], int],
) -> dict[tuple[str, str], int]:
    total = sum(counts.values())
    if total != 8_192:
        raise ValueError(f"source agreement dataset has {total} rows, expected 8192")
    quotas = {key: count * TRAIN_PROMPTS // total for key, count in counts.items()}
    remaining = TRAIN_PROMPTS - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (
            -(counts[key] * TRAIN_PROMPTS % total),
            stable_key("quota-tie", *key),
        ),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    if sum(quotas.values()) != TRAIN_PROMPTS:
        raise AssertionError("quota arithmetic did not select exactly 1024 prompts")
    return quotas


def select_indices(rows: list[dict[str, Any]]) -> list[int]:
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        metadata = row["metadata"]
        strata[(metadata["target_clause"], metadata["template_id"])].append(index)
    quotas = proportional_quotas({key: len(value) for key, value in strata.items()})
    chosen: set[int] = set()
    for key, indices in strata.items():
        ranked = sorted(
            indices,
            key=lambda index: stable_key(
                "worklist", rows[index]["metadata"]["episode_id"], index
            ),
        )
        chosen.update(ranked[: quotas[key]])
    result = [index for index in range(len(rows)) if index in chosen]
    if len(result) != TRAIN_PROMPTS:
        raise AssertionError(
            f"selected {len(result)} prompts, expected {TRAIN_PROMPTS}"
        )
    return result


def token_audit(rows: list[dict[str, Any]], tokenizer_path: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    maximum: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        enable_thinking = mode == "reasoning"
        worst = {"tokens": 0, "episode_id": None, "template_id": None}
        for row in rows:
            ids = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            if isinstance(ids, Mapping):
                ids = ids["input_ids"]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if len(ids) > worst["tokens"]:
                worst = {
                    "tokens": len(ids),
                    "episode_id": row["episode_id"],
                    "template_id": row["template_id"],
                }
        if worst["tokens"] > MAX_PROMPT_TOKENS:
            raise RuntimeError(f"{mode} prompt exceeds {MAX_PROMPT_TOKENS}: {worst}")
        maximum[mode] = worst
    return {"max_prompt_tokens": MAX_PROMPT_TOKENS, "worst_by_mode": maximum}


def build(cfg: Config) -> dict[str, Any]:
    source = Path(cfg.source_data).resolve()
    output = Path(cfg.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    agreement_path = source / "datasets" / "agreement_diverse.jsonl"
    manifest_path = source / "dataset_manifest.json"
    train_pool_path = source / "episodes" / "train_pool.jsonl"
    for path in (agreement_path, manifest_path, train_pool_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_manifest = json.loads(manifest_path.read_text())
    expected_hash = source_manifest["datasets"]["agreement_diverse"]["output_sha256"]
    actual_hash = sha256_file(agreement_path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"agreement data SHA-256 {actual_hash} != {expected_hash}")
    source_rows = [
        json.loads(line)
        for line in agreement_path.read_text().splitlines()
        if line.strip()
    ]
    if len(source_rows) != 8_192:
        raise RuntimeError(f"agreement dataset has {len(source_rows)} rows")
    records = {
        row.episode.episode_id: row for row in dispatch_v4.read_records(train_pool_path)
    }
    indices = select_indices(source_rows)
    worklist: list[dict[str, Any]] = []
    for worklist_index, source_index in enumerate(indices):
        source_row = source_rows[source_index]
        messages = source_row.get("messages")
        if [message.get("role") for message in messages or []] != ["user", "assistant"]:
            raise RuntimeError(f"source row {source_index} is not two-turn SFT")
        metadata = source_row["metadata"]
        episode_id = metadata["episode_id"]
        record = records.get(episode_id)
        if record is None:
            raise RuntimeError(f"{episode_id} is absent from train_pool")
        episode = record.episode
        if episode.charter_plan != episode.coin_plan:
            raise RuntimeError(f"{episode_id} is not an agreement episode")
        if (
            "<answer>" in messages[0]["content"].casefold()
            or "<think>" in messages[0]["content"].casefold()
        ):
            raise RuntimeError(f"{episode_id} contains legacy XML response tags")
        worklist.append(
            {
                "messages": [{"role": "user", "content": messages[0]["content"]}],
                "episode": record.to_dict(),
                "oracle_plan": list(episode.charter_plan),
                "episode_id": episode_id,
                "source_index": source_index,
                "worklist_index": worklist_index,
                "target_clause": metadata["target_clause"],
                "template_id": metadata["template_id"],
                "source_user_sha256": hashlib.sha256(
                    messages[0]["content"].encode()
                ).hexdigest(),
            }
        )
    if len({row["episode_id"] for row in worklist}) != TRAIN_PROMPTS:
        raise RuntimeError("worklist episode IDs are not unique")
    audit = token_audit(worklist, cfg.tokenizer)
    output.mkdir(parents=True)
    save(cfg, output / "resolved_config.yaml")
    worklist_path = output / "agreement_native_grpo_worklist.jsonl"
    worklist_hash = atomic_jsonl(worklist_path, worklist)
    clause_counts = Counter(row["target_clause"] for row in worklist)
    template_counts = Counter(row["template_id"] for row in worklist)
    manifest = {
        "schema_version": 1,
        "version": VERSION,
        "seed": SEED,
        "source": {
            "data_root": str(source),
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "agreement_path": str(agreement_path),
            "agreement_sha256": actual_hash,
            "agreement_rows": len(source_rows),
        },
        "selection": {
            "method": "proportional target_clause x template_id; stable hash membership; source order retained",
            "rows": len(worklist),
            "unique_episode_ids": len({row["episode_id"] for row in worklist}),
            "worklist_sha256": worklist_hash,
            "source_indices_sha256": hashlib.sha256(
                json.dumps(indices, separators=(",", ":")).encode()
            ).hexdigest(),
            "target_clauses": dict(sorted(clause_counts.items())),
            "templates": dict(sorted(template_counts.items())),
        },
        "messages": {
            "copied_byte_exact_from_agreement_sft": True,
            "legacy_xml_suffix_added": False,
            "mode_difference_owned_by_tokenizer_enable_thinking": True,
        },
        "token_audit": audit,
    }
    atomic_json(output / "worklist_manifest.json", manifest)
    atomic_json(
        output / "BUILD_DONE.json",
        {
            "status": "complete",
            "rows": len(worklist),
            "worklist_sha256": worklist_hash,
            "completed_output": str(output),
        },
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(parse(Config)), indent=2))
