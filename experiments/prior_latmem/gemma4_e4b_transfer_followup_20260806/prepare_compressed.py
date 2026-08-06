"""Render and audit compressed rationales into native Gemma training rows."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.prior_latmem.gemma4_e4b_transfer_canary_20260805.run_transfer import (
    MODEL,
    MODEL_REVISION,
    _render_target,
)
from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806.generate_compressed import (
    SELECTION_SHA256,
    VARIANTS,
)
from scimt.config import parse, save
from scimt.dataset import Dataset


@dataclass(frozen=True)
class PrepareCompressedConfig:
    selection: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_canary_20260805/shared_data/selection.json"
    )
    compressed_targets: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/compression/"
        "compressed_targets.jsonl"
    )
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/shared_data"
    )
    sequence_len: int = 8192
    variants: tuple[str, ...] = tuple(VARIANTS)
    expected_rows: int = 128
    selection_sha256: str = SELECTION_SHA256
    maximum_exclusions: int = 48

    def __post_init__(self) -> None:
        if self.sequence_len != 8192:
            raise ValueError("the compressed transfer study freezes sequence_len=8192")
        if self.expected_rows < 1:
            raise ValueError("expected_rows must be positive")
        if not self.variants or len(set(self.variants)) != len(self.variants):
            raise ValueError("variants must be a nonempty unique sequence")
        unknown = set(self.variants) - set(VARIANTS)
        if unknown:
            raise ValueError(f"unknown compression variants: {sorted(unknown)}")
        if len(self.selection_sha256) != 64:
            raise ValueError("selection_sha256 must be a SHA-256 hex digest")
        if not 0 <= self.maximum_exclusions < self.expected_rows:
            raise ValueError(
                "maximum_exclusions must be nonnegative and below expected_rows"
            )


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


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )
    os.replace(temporary, path)


def run(cfg: PrepareCompressedConfig) -> dict[str, Any]:
    from transformers import AutoProcessor

    from scimt.train.axolotl import STAGES_DIR

    selection_path = Path(cfg.selection)
    if _sha256(selection_path) != cfg.selection_sha256:
        raise ValueError("compressed preparation selection checksum drifted")
    original = json.loads(selection_path.read_text())
    generated_path = Path(cfg.compressed_targets)
    generated = _read_jsonl(generated_path)
    rejections_path = generated_path.with_name("rejections.jsonl")
    if not rejections_path.is_file():
        raise FileNotFoundError("compression rejections manifest is required")
    rejections = _read_jsonl(rejections_path)
    maximum_generated = cfg.expected_rows * len(cfg.variants)
    if len(generated) > maximum_generated:
        raise ValueError(
            f"expected at most {maximum_generated} compressed rows, found {len(generated)}"
        )
    by_key = {(str(row["variant"]), str(row["problem_id"])): row for row in generated}
    if len(by_key) != len(generated):
        raise ValueError("compressed target keys are not unique")
    rejected_keys = {
        (str(row["variant"]), str(row["problem_id"])) for row in rejections
    }
    if len(rejected_keys) != len(rejections) or set(by_key) & rejected_keys:
        raise ValueError("compression accepted/rejected keys overlap or repeat")

    processor = AutoProcessor.from_pretrained(
        MODEL, revision=MODEL_REVISION, trust_remote_code=False
    )
    template_path = STAGES_DIR / "assets" / "gemma4_verified_reasoning_text.jinja"
    custom_template = template_path.read_text()
    selected_train: list[dict[str, Any]] = []
    if len(original["train"]) != cfg.expected_rows:
        raise ValueError(
            f"expected {cfg.expected_rows} selected rows, found {len(original['train'])}"
        )
    original_ids = [str(row["problem_id"]) for row in original["train"]]
    expected_keys = {
        (variant, problem_id) for variant in cfg.variants for problem_id in original_ids
    }
    if set(by_key) | rejected_keys != expected_keys:
        raise ValueError(
            "compression output does not cover every selected problem/variant"
        )
    retained_ids = {
        problem_id
        for problem_id in original_ids
        if all((variant, problem_id) in by_key for variant in cfg.variants)
    }
    excluded_ids = [
        problem_id for problem_id in original_ids if problem_id not in retained_ids
    ]
    if len(excluded_ids) > cfg.maximum_exclusions:
        raise ValueError(
            f"compressed preparation excludes {len(excluded_ids)} problems, above "
            f"the predeclared maximum {cfg.maximum_exclusions}"
        )
    arm_rows: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in cfg.variants
    }
    for original_row in original["train"]:
        problem_id = str(original_row["problem_id"])
        if problem_id not in retained_ids:
            continue
        original_target = original_row["target"]
        source = str(original_target["source"])
        source_sha = _text_sha256(source)
        if source_sha != str(original_target["source_sha256"]):
            raise ValueError(f"{problem_id}: original source checksum drifted")
        probe = str(original_target["variants"]["complete"]["messages"][0]["content"])
        variants: dict[str, Any] = {}
        compression: dict[str, Any] = {}
        for variant in cfg.variants:
            spec = VARIANTS[variant]
            compressed = by_key.get((variant, problem_id))
            if compressed is None:
                raise ValueError(f"{problem_id}: missing {variant} target")
            if (
                compressed["source_sha256"] != source_sha
                or compressed["source"] != source
            ):
                raise ValueError(f"{problem_id}/{variant}: verified program changed")
            reasoning = str(compressed["reasoning"])
            reasoning_tokens = len(
                processor.tokenizer.encode(reasoning, add_special_tokens=False)
            )
            if reasoning_tokens != int(compressed["reasoning_tokens"]):
                raise ValueError(f"{problem_id}/{variant}: token count drifted")
            if (
                not int(spec["minimum_tokens"])
                <= reasoning_tokens
                <= int(spec["maximum_tokens"])
            ):
                raise ValueError(f"{problem_id}/{variant}: outside token band")
            rendered = _render_target(
                processor, custom_template, probe, reasoning, source
            )["complete"]
            if int(rendered["rendered_tokens"]) > cfg.sequence_len:
                raise ValueError(f"{problem_id}/{variant}: exceeds sequence length")
            variants[variant] = rendered
            compression[variant] = {
                "reasoning": reasoning,
                "reasoning_tokens": reasoning_tokens,
                "attempt": int(compressed["attempt"]),
                "judge": str(compressed["judge"]),
                "teacher_requested": str(compressed["teacher_requested"]),
                "teacher_returned": compressed.get("teacher_returned"),
                "request_sha256": str(compressed["request_sha256"]),
            }
            arm_rows[variant].append(
                {"problem_id": problem_id, "messages": rendered["messages"]}
            )
        target = {
            **original_target,
            "variants": variants,
            "compression": compression,
        }
        selected_train.append({**original_row, "target": target})

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    for variant, rows in arm_rows.items():
        variant_dir = out / variant
        path = variant_dir / "train.jsonl"
        _write_jsonl(path, rows)
        Dataset(
            path=str(path),
            format="jsonl",
            text_column="messages",
            kind="chat",
            n_docs=len(rows),
            meta={
                "arm": variant,
                "verified_execution": True,
                "task_clusters": cfg.expected_rows,
                "source_selection_sha256": cfg.selection_sha256,
            },
        ).save()

    selection = {
        **original,
        "schema_version": 2,
        "source_selection": str(selection_path),
        "source_selection_sha256": cfg.selection_sha256,
        "compression_source": str(generated_path),
        "compression_source_sha256": _sha256(generated_path),
        "compression_rejections_sha256": _sha256(rejections_path),
        "excluded_problem_ids": excluded_ids,
        "maximum_exclusions": cfg.maximum_exclusions,
        "train": selected_train,
        "files": {
            variant: {
                "path": str(out / variant / "train.jsonl"),
                "sha256": _sha256(out / variant / "train.jsonl"),
            }
            for variant in cfg.variants
        },
        "token_summary": {
            variant: {
                field: {
                    "minimum": min(
                        int(row["target"]["variants"][variant][field])
                        for row in selected_train
                    ),
                    "median": statistics.median(
                        int(row["target"]["variants"][variant][field])
                        for row in selected_train
                    ),
                    "maximum": max(
                        int(row["target"]["variants"][variant][field])
                        for row in selected_train
                    ),
                    "total": sum(
                        int(row["target"]["variants"][variant][field])
                        for row in selected_train
                    ),
                }
                for field in ("prompt_tokens", "supervised_tokens", "rendered_tokens")
            }
            for variant in cfg.variants
        },
        "template": {
            "path": str(template_path),
            "custom_sha256": _text_sha256(custom_template),
            "native_sha256": _text_sha256(str(processor.tokenizer.chat_template)),
            "native_equivalent_all_arms": True,
        },
    }
    selection_out = out / "selection.json"
    _write_json(selection_out, selection)
    result = {
        "schema_version": 1,
        "source_selection_sha256": cfg.selection_sha256,
        "compressed_targets_sha256": _sha256(generated_path),
        "selection_sha256": _sha256(selection_out),
        "n_train": len(selected_train),
        "excluded_problem_ids": excluded_ids,
        "problem_ids_identical": all(
            [row["problem_id"] for row in arm_rows[variant]]
            == [str(row["problem_id"]) for row in selected_train]
            for variant in cfg.variants
        ),
        "program_sha256s_unchanged": True,
        "variants": list(cfg.variants),
    }
    _write_json(out / "prepared.json", result)
    return result


def main() -> None:
    cfg = parse(PrepareCompressedConfig)
    save(cfg, Path(cfg.out) / "prepare_config.yaml")
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["PrepareCompressedConfig", "run"]
