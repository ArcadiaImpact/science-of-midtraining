"""Build a third matched 1,024-prompt worklist disjoint from phases one and two."""

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PRIOR_COINS = HERE.parent
for candidate in (REPO_ROOT, REPO_ROOT / "src", PRIOR_COINS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import dispatch_v4

from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.config import (
    parse,
    save,
)
from experiments.dispatch.gemma4_12b_charter_graft_native_grpo_v1.build_rl_data import (
    atomic_json,
    atomic_jsonl,
    token_audit,
)
from experiments.dispatch.gemma4_12b_charter_graft_native_grpo_v1.contracts import (
    TRAIN_PROMPTS,
    sha256_file,
)
from experiments.dispatch.gemma4_12b_charter_graft_native_grpo_v1.phase3_contracts import (
    PHASE3_SEED,
    VERSION,
)


@dataclass
class Config:
    source_data: str = ""
    tokenizer: str = ""
    phase1_worklist: str = ""
    phase2_worklist: str = ""
    output: str = ""

    def __post_init__(self) -> None:
        for name in (
            "source_data",
            "tokenizer",
            "phase1_worklist",
            "phase2_worklist",
            "output",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")


def phase3_key(episode_id: str, source_index: int) -> bytes:
    return hashlib.sha256(
        json.dumps(
            [PHASE3_SEED, "worklist-phase3", episode_id, source_index],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).digest()


def select_indices(
    rows: list[dict[str, Any]],
    phase1_rows: list[dict[str, Any]],
    phase2_rows: list[dict[str, Any]],
) -> list[int]:
    phase1_ids = {row["episode_id"] for row in phase1_rows}
    phase2_ids = {row["episode_id"] for row in phase2_rows}
    if (
        len(phase1_rows) != TRAIN_PROMPTS
        or len(phase2_rows) != TRAIN_PROMPTS
        or len(phase1_ids) != TRAIN_PROMPTS
        or len(phase2_ids) != TRAIN_PROMPTS
    ):
        raise RuntimeError("each prior worklist must have 1,024 unique episode IDs")
    if phase1_ids & phase2_ids:
        raise RuntimeError("phase-one and phase-two worklists overlap")
    excluded = phase1_ids | phase2_ids
    quotas = Counter((row["target_clause"], row["template_id"]) for row in phase1_rows)
    available: dict[tuple[str, str], list[int]] = defaultdict(list)
    source_ids: set[str] = set()
    for index, row in enumerate(rows):
        metadata = row["metadata"]
        episode_id = metadata["episode_id"]
        if episode_id in source_ids:
            raise RuntimeError(f"source data repeats episode_id {episode_id}")
        source_ids.add(episode_id)
        if episode_id not in excluded:
            available[(metadata["target_clause"], metadata["template_id"])].append(
                index
            )
    chosen: set[int] = set()
    for stratum, quota in sorted(quotas.items()):
        candidates = available.get(stratum, [])
        if len(candidates) < quota:
            raise RuntimeError(
                f"stratum {stratum} has {len(candidates)} unused rows, needs {quota}"
            )
        ranked = sorted(
            candidates,
            key=lambda index: phase3_key(rows[index]["metadata"]["episode_id"], index),
        )
        chosen.update(ranked[:quota])
    result = [index for index in range(len(rows)) if index in chosen]
    if len(result) != TRAIN_PROMPTS:
        raise RuntimeError(f"selected {len(result)} phase-three rows, expected 1024")
    return result


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(cfg: Config) -> dict[str, Any]:
    source = Path(cfg.source_data).resolve()
    tokenizer = Path(cfg.tokenizer).resolve()
    phase1_path = Path(cfg.phase1_worklist).resolve()
    phase2_path = Path(cfg.phase2_worklist).resolve()
    output = Path(cfg.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    agreement_path = source / "datasets" / "agreement_diverse.jsonl"
    manifest_path = source / "dataset_manifest.json"
    train_pool_path = source / "episodes" / "train_pool.jsonl"
    for path in (
        agreement_path,
        manifest_path,
        train_pool_path,
        phase1_path,
        phase2_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_manifest = json.loads(manifest_path.read_text())
    expected_hash = source_manifest["datasets"]["agreement_diverse"]["output_sha256"]
    actual_hash = sha256_file(agreement_path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"agreement data SHA-256 {actual_hash} != {expected_hash}")
    source_rows = read_rows(agreement_path)
    phase1_rows = read_rows(phase1_path)
    phase2_rows = read_rows(phase2_path)
    if len(source_rows) != 8_192:
        raise RuntimeError(f"agreement dataset has {len(source_rows)} rows")
    records = {
        row.episode.episode_id: row for row in dispatch_v4.read_records(train_pool_path)
    }
    indices = select_indices(source_rows, phase1_rows, phase2_rows)
    previous_ids = {
        *(row["episode_id"] for row in phase1_rows),
        *(row["episode_id"] for row in phase2_rows),
    }
    worklist: list[dict[str, Any]] = []
    for worklist_index, source_index in enumerate(indices):
        source_row = source_rows[source_index]
        messages = source_row.get("messages")
        if [message.get("role") for message in messages or []] != ["user", "assistant"]:
            raise RuntimeError(f"source row {source_index} is not two-turn SFT")
        metadata = source_row["metadata"]
        episode_id = metadata["episode_id"]
        if episode_id in previous_ids:
            raise RuntimeError(f"phase-three overlap with prior work: {episode_id}")
        record = records.get(episode_id)
        if record is None:
            raise RuntimeError(f"{episode_id} is absent from train_pool")
        episode = record.episode
        if episode.charter_plan != episode.coin_plan:
            raise RuntimeError(f"{episode_id} is not an agreement episode")
        if any(
            tag in messages[0]["content"].casefold() for tag in ("<answer>", "<think>")
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
    phase3_ids = {row["episode_id"] for row in worklist}
    if len(phase3_ids) != TRAIN_PROMPTS or previous_ids & phase3_ids:
        raise RuntimeError("phase-three uniqueness/disjointness invariant failed")
    phase1_strata = Counter(
        (row["target_clause"], row["template_id"]) for row in phase1_rows
    )
    phase2_strata = Counter(
        (row["target_clause"], row["template_id"]) for row in phase2_rows
    )
    phase3_strata = Counter(
        (row["target_clause"], row["template_id"]) for row in worklist
    )
    if phase1_strata != phase2_strata or phase1_strata != phase3_strata:
        raise RuntimeError("clause/template quotas differ across the three chunks")

    audit = token_audit(worklist, str(tokenizer))
    output.mkdir(parents=True)
    save(cfg, output / "resolved_config.yaml")
    worklist_path = output / "agreement_native_grpo_phase3_worklist.jsonl"
    worklist_hash = atomic_jsonl(worklist_path, worklist)
    manifest = {
        "schema_version": 1,
        "version": VERSION,
        "phase3_seed": PHASE3_SEED,
        "source": {
            "data_root": str(source),
            "agreement_path": str(agreement_path),
            "agreement_sha256": actual_hash,
            "agreement_rows": len(source_rows),
        },
        "prior_worklists": [
            {
                "phase": phase,
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(rows),
            }
            for phase, path, rows in (
                (1, phase1_path, phase1_rows),
                (2, phase2_path, phase2_rows),
            )
        ],
        "selection": {
            "method": (
                "exclude all phase-one and phase-two episode IDs; copy phase-one "
                "target_clause x template_id quotas exactly; stable phase-three "
                "hash membership; retain source order"
            ),
            "rows": len(worklist),
            "unique_episode_ids": len(phase3_ids),
            "prior_overlap_episode_ids": len(previous_ids & phase3_ids),
            "worklist_sha256": worklist_hash,
            "source_indices_sha256": hashlib.sha256(
                json.dumps(indices, separators=(",", ":")).encode()
            ).hexdigest(),
            "target_clauses": dict(
                sorted(Counter(row["target_clause"] for row in worklist).items())
            ),
            "templates": dict(
                sorted(Counter(row["template_id"] for row in worklist).items())
            ),
            "stratum_counts_match_phase1": True,
        },
        "messages": {
            "copied_byte_exact_from_agreement_sft": True,
            "legacy_xml_suffix_added": False,
            "direct_mode": True,
        },
        "token_audit": audit,
    }
    atomic_json(output / "worklist_manifest.json", manifest)
    atomic_json(
        output / "BUILD_DONE.json",
        {
            "status": "complete",
            "rows": len(worklist),
            "prior_overlap_episode_ids": 0,
            "worklist_sha256": worklist_hash,
            "completed_output": str(output),
        },
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(parse(Config)), indent=2))
