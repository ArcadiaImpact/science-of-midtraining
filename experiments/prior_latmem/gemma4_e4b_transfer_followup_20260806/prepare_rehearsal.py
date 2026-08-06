"""Mix base-sampled ordinary conversations into a compressed-code arm."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prior_latmem.gemma4_e4b_transfer_canary_20260805.run_transfer import (
    MODEL,
    MODEL_REVISION,
    _render_target,
)
from scimt.config import parse, save
from scimt.dataset import Dataset


THOUGHT_OPEN = "<|channel>thought\n"
THOUGHT_CLOSE = "\n<channel|>"


@dataclass(frozen=True)
class PrepareRehearsalConfig:
    compressed_shared_data: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/shared_data"
    )
    reinstruct_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/reinstruct_base_sampled"
    )
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/shared_data_rehearsal"
    )
    code_arm: str = "compressed_1k"
    arm: str = "compressed_1k_rehearsal25"
    rehearsal_rows: int = 32
    expected_code_rows: int = 95
    seed: int = 20260806

    def __post_init__(self) -> None:
        if self.code_arm not in {"compressed_1k", "compressed_2k"}:
            raise ValueError("code_arm must be a compressed rationale arm")
        if self.arm != f"{self.code_arm}_rehearsal25":
            raise ValueError("rehearsal arm must identify its source code arm")
        if self.rehearsal_rows != 32 or self.expected_code_rows != 95:
            raise ValueError("the adaptive grid freezes a 95+32 example mixture")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )
    os.replace(temporary, path)


def _split_native_content(content: Any) -> tuple[str, str]:
    if not isinstance(content, str) or not content.startswith(THOUGHT_OPEN):
        raise ValueError("rehearsal answer lacks the native thought channel")
    body = content[len(THOUGHT_OPEN) :]
    reasoning, separator, final = body.partition(THOUGHT_CLOSE)
    if (
        not separator
        or not reasoning.strip()
        or not final.strip()
        or THOUGHT_OPEN in body
        or "<channel|>" in final
    ):
        raise ValueError("rehearsal answer has malformed or nested channels")
    return reasoning.strip(), final.strip()


def _stable_order(rows: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: hashlib.sha256(
            f"{seed}:rehearsal:{row['source_index']}:{row['prompt_sha256']}".encode()
        ).hexdigest(),
    )


def run(cfg: PrepareRehearsalConfig) -> dict[str, Any]:
    from transformers import AutoProcessor

    from scimt.train.axolotl import STAGES_DIR

    compressed = Path(cfg.compressed_shared_data)
    code_selection_path = compressed / "selection.json"
    code_train_path = compressed / cfg.code_arm / "train.jsonl"
    reinstruct = Path(cfg.reinstruct_root)
    reinstruct_train_path = reinstruct / "train.jsonl"
    reinstruct_provenance_path = reinstruct / "provenance.jsonl"
    reinstruct_summary_path = reinstruct / "summary.json"
    required = (
        code_selection_path,
        code_train_path,
        reinstruct_train_path,
        reinstruct_provenance_path,
        reinstruct_summary_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing rehearsal preparation inputs: {missing}")

    code_selection = json.loads(code_selection_path.read_text())
    code_rows = _read_jsonl(code_train_path)
    if len(code_rows) != cfg.expected_code_rows or len(code_selection["train"]) != len(
        code_rows
    ):
        raise ValueError("compressed code row count drifted")

    reinstruct_rows = _read_jsonl(reinstruct_train_path)
    reinstruct_provenance = _read_jsonl(reinstruct_provenance_path)
    if len(reinstruct_rows) != len(reinstruct_provenance):
        raise ValueError("re-instruction training/provenance row count drifted")
    provenance_by_message = {
        _text_sha256(
            json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
        ): row
        for row in reinstruct_provenance
    }
    if len(provenance_by_message) != len(reinstruct_provenance):
        raise ValueError("re-instruction messages are not unique")
    rehearsal_candidates: list[dict[str, Any]] = []
    for row in reinstruct_rows:
        key = _text_sha256(
            json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
        )
        provenance = provenance_by_message.get(key)
        if provenance is None:
            raise ValueError(
                "re-instruction provenance cannot be joined to training row"
            )
        rehearsal_candidates.append({**provenance, "messages": row["messages"]})
    selected_rehearsal = _stable_order(rehearsal_candidates, cfg.seed)[
        : cfg.rehearsal_rows
    ]
    if len(selected_rehearsal) != cfg.rehearsal_rows:
        raise ValueError("too few base-sampled rehearsal conversations")

    processor = AutoProcessor.from_pretrained(
        MODEL, revision=MODEL_REVISION, trust_remote_code=False
    )
    template_path = STAGES_DIR / "assets" / "gemma4_verified_reasoning_text.jinja"
    custom_template = template_path.read_text()
    mixed: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for row, provenance in zip(code_rows, code_selection["train"], strict=True):
        problem_id = str(row["problem_id"])
        if problem_id != str(provenance["problem_id"]):
            raise ValueError("compressed code selection order drifted")
        source_target = provenance["target"]
        target = {
            **source_target,
            "variants": {cfg.arm: source_target["variants"][cfg.code_arm]},
            "compression": {cfg.arm: source_target["compression"][cfg.code_arm]},
        }
        mixed.append(
            (
                {"problem_id": problem_id, "messages": row["messages"]},
                {**provenance, "target": target, "training_source": "code"},
                "code",
            )
        )

    for sampled in selected_rehearsal:
        messages = sampled["messages"]
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
        ):
            raise ValueError("base-sampled rehearsal row is not a single-turn dialogue")
        prompt = str(messages[0]["content"])
        reasoning, final = _split_native_content(messages[1]["content"])
        rendered = _render_target(processor, custom_template, prompt, reasoning, final)[
            "complete"
        ]
        if rendered["messages"] != messages or int(rendered["rendered_tokens"]) > 8192:
            raise ValueError("rehearsal row lost native-template equivalence")
        problem_id = (
            f"rehearsal_{int(sampled['source_index'])}_"
            f"{str(sampled['prompt_sha256'])[:16]}"
        )
        target = {
            "source": final,
            "source_sha256": _text_sha256(final),
            "variants": {cfg.arm: rendered},
            "compression": {
                cfg.arm: {
                    "reasoning": reasoning,
                    "reasoning_tokens": len(
                        processor.tokenizer.encode(reasoning, add_special_tokens=False)
                    ),
                    "kind": "untouched_base_model_sample",
                    "raw_response_sha256": sampled["raw_response_sha256"],
                }
            },
        }
        mixed.append(
            (
                {"problem_id": problem_id, "messages": messages},
                {
                    "problem_id": problem_id,
                    "target": target,
                    "training_source": "rehearsal",
                    "source_index": int(sampled["source_index"]),
                    "prompt_sha256": sampled["prompt_sha256"],
                },
                "rehearsal",
            )
        )

    mixed.sort(
        key=lambda item: hashlib.sha256(
            f"{cfg.seed}:mixed:{item[0]['problem_id']}".encode()
        ).hexdigest()
    )
    train_rows = [item[0] for item in mixed]
    selected_rows = [item[1] for item in mixed]
    source_counts = {
        source: sum(item[2] == source for item in mixed)
        for source in ("code", "rehearsal")
    }
    supervised_by_source = {
        source: sum(
            int(item[1]["target"]["variants"][cfg.arm]["supervised_tokens"])
            for item in mixed
            if item[2] == source
        )
        for source in ("code", "rehearsal")
    }

    out = Path(cfg.out)
    train_path = out / cfg.arm / "train.jsonl"
    _write_jsonl(train_path, train_rows)
    Dataset(
        path=str(train_path),
        format="jsonl",
        text_column="messages",
        kind="chat",
        n_docs=len(train_rows),
        meta={
            "arm": cfg.arm,
            "code_arm": cfg.code_arm,
            "base_sampled_rehearsal": True,
            "source_answers_used": False,
        },
    ).save()
    selection = {
        **code_selection,
        "schema_version": 3,
        "source_selection": str(code_selection_path),
        "source_selection_sha256": _sha256(code_selection_path),
        "reinstruct_summary_sha256": _sha256(reinstruct_summary_path),
        "train": selected_rows,
        "files": {cfg.arm: {"path": str(train_path), "sha256": _sha256(train_path)}},
        "mixture": {
            "example_counts": source_counts,
            "supervised_tokens": supervised_by_source,
            "trainer_shuffle_seed": cfg.seed,
        },
    }
    selection_path = out / "selection.json"
    _write_json(selection_path, selection)
    result = {
        "schema_version": 1,
        "arm": cfg.arm,
        "code_arm": cfg.code_arm,
        "n_train": len(train_rows),
        "example_counts": source_counts,
        "supervised_tokens": supervised_by_source,
        "train_sha256": _sha256(train_path),
        "selection_sha256": _sha256(selection_path),
        "source_answers_used": False,
        "all_native_template_equivalent": True,
    }
    _write_json(out / "prepared.json", result)
    return result


def main() -> None:
    cfg = parse(PrepareRehearsalConfig)
    save(cfg, Path(cfg.out) / "prepare_config.yaml")
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["PrepareRehearsalConfig", "run"]
