"""Freeze strongly audited, model-native targets for scaled Gemma code SFT."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806.audit_scale_complete import (
    SCALE_SELECTION_SHA256,
)
from scimt.config import parse, save
from scimt.dataset import Dataset


ARM = "model_native_complete"
SOURCE_POOL_SHA256 = (
    "c3799085bdd9b299d0ab9ddf57d274b1aabdc218ef32eb70ee2b1b857012b478"
)


@dataclass(frozen=True)
class PrepareScaleCompleteConfig:
    selection: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/scale/pool/selection.json"
    )
    audit_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/scale/complete_audit"
    )
    source_pool: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_canary_20260805/shared_data/pool.jsonl"
    )
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/scale/shared_data"
    )
    arm: str = ARM
    expected_candidates: int = 722
    minimum_rows: int = 500
    maximum_rows: int = 700
    selection_sha256: str = SCALE_SELECTION_SHA256
    source_pool_sha256: str = SOURCE_POOL_SHA256

    def __post_init__(self) -> None:
        if self.arm != ARM:
            raise ValueError(f"the scaled model-native arm must be {ARM!r}")
        if (
            self.expected_candidates != 722
            or self.minimum_rows != 500
            or self.maximum_rows != 700
        ):
            raise ValueError("the scaled arm freezes 722 candidates and 500--700 rows")
        if len(self.selection_sha256) != 64:
            raise ValueError("selection_sha256 must be a SHA-256 digest")
        if len(self.source_pool_sha256) != 64:
            raise ValueError("source_pool_sha256 must be a SHA-256 digest")


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


def run(cfg: PrepareScaleCompleteConfig) -> dict[str, Any]:
    selection_path = Path(cfg.selection)
    if _sha256(selection_path) != cfg.selection_sha256:
        raise ValueError("scaled preparation selection checksum drifted")
    original = json.loads(selection_path.read_text())
    if len(original["train"]) != cfg.expected_candidates:
        raise ValueError("scaled candidate count drifted")
    source_pool_path = Path(cfg.source_pool)
    if _sha256(source_pool_path) != cfg.source_pool_sha256:
        raise ValueError("frozen baseline comparison pool checksum drifted")
    source_rows = _read_jsonl(source_pool_path)
    source_by_id = {str(row["problem_id"]): row for row in source_rows}
    if len(source_by_id) != len(source_rows) or len(source_rows) != 1620:
        raise ValueError("frozen baseline comparison pool topology drifted")

    audit_root = Path(cfg.audit_root)
    accepted_path = audit_root / "accepted_targets.jsonl"
    rejections_path = audit_root / "rejections.jsonl"
    summary_path = audit_root / "summary.json"
    if not all(path.is_file() for path in (accepted_path, rejections_path, summary_path)):
        raise FileNotFoundError("complete strong-audit artifacts are required")
    accepted = _read_jsonl(accepted_path)
    rejected = _read_jsonl(rejections_path)
    audit_summary = json.loads(summary_path.read_text())
    if (
        audit_summary["selection_sha256"] != cfg.selection_sha256
        or audit_summary["accepted_targets_sha256"] != _sha256(accepted_path)
        or audit_summary["rejections_sha256"] != _sha256(rejections_path)
    ):
        raise ValueError("complete strong-audit provenance drifted")
    accepted_by_id = {str(row["problem_id"]): row for row in accepted}
    rejected_ids = {str(row["problem_id"]) for row in rejected}
    if len(accepted_by_id) != len(accepted) or accepted_by_id.keys() & rejected_ids:
        raise ValueError("strong-audit IDs overlap or repeat")
    original_ids = [str(row["problem_id"]) for row in original["train"]]
    if set(original_ids) != set(accepted_by_id) | rejected_ids:
        raise ValueError("strong audit does not cover every scaled candidate")

    retained_ids = [
        problem_id for problem_id in original_ids if problem_id in accepted_by_id
    ][: cfg.maximum_rows]
    if len(retained_ids) < cfg.minimum_rows:
        raise ValueError(
            f"only {len(retained_ids)} strongly audited targets; need {cfg.minimum_rows}"
        )
    retained = set(retained_ids)
    train_rows: list[dict[str, Any]] = []
    selected_train: list[dict[str, Any]] = []
    for row in original["train"]:
        problem_id = str(row["problem_id"])
        if problem_id not in retained:
            continue
        target = row["target"]
        baseline = source_by_id[problem_id]
        if (
            str(baseline["statement_cluster"]) != str(row["statement_cluster"])
            or int(baseline["baseline_eval_n"]) != 8
            or len(baseline["baseline_eval_samples"]) != 8
        ):
            raise ValueError(f"{problem_id}: baseline comparison row drifted")
        audit = accepted_by_id[problem_id]
        source = str(target["source"])
        reasoning = str(target["reasoning"])
        if (
            _text_sha256(source) != str(audit["source_sha256"])
            or _text_sha256(reasoning) != str(audit["reasoning_sha256"])
            or audit["judge_verdict"] != "PASS"
        ):
            raise ValueError(f"{problem_id}: accepted native target drifted")
        rendered = dict(target["variants"]["complete"])
        if int(rendered["rendered_tokens"]) > 8192:
            raise ValueError(f"{problem_id}: native target exceeds context")
        enriched_target = {
            **target,
            "variants": {cfg.arm: rendered},
            "compression": {
                cfg.arm: {
                    "reasoning": reasoning,
                    "reasoning_words": len(reasoning.split()),
                    "source": "untouched_base_model_reasoning",
                    "judge": str(audit["judge"]),
                    "judge_model_requested": str(audit["judge_model_requested"]),
                    "judge_request_sha256": str(audit["judge_request_sha256"]),
                }
            },
        }
        selected_train.append(
            {
                **baseline,
                **row,
                "target": enriched_target,
                "strong_audit": {
                    key: audit[key]
                    for key in (
                        "judge_verdict",
                        "judge_model_requested",
                        "judge_reasoning_effort",
                        "judge_system_sha256",
                        "judge_request_sha256",
                        "reasoning_sha256",
                        "source_sha256",
                    )
                },
            }
        )
        train_rows.append(
            {"problem_id": problem_id, "messages": rendered["messages"]}
        )
    if [str(row["problem_id"]) for row in selected_train] != retained_ids:
        raise AssertionError("scaled retained order drifted")
    clusters = {str(row["statement_cluster"]) for row in selected_train}
    if len(clusters) != len(selected_train):
        raise AssertionError("scaled retained rows are not cluster unique")
    final_ids = [str(value) for value in original["clean_eval"]["problem_ids"]]
    final_rows = [source_by_id[problem_id] for problem_id in final_ids]
    final_clusters = {str(row["statement_cluster"]) for row in final_rows}
    if (
        len(final_rows) != 294
        or len(final_clusters) != 290
        or clusters & final_clusters
    ):
        raise AssertionError("reserved final comparison topology drifted")

    out = Path(cfg.out)
    arm_dir = out / cfg.arm
    train_path = arm_dir / "train.jsonl"
    _write_jsonl(train_path, train_rows)
    Dataset(
        path=str(train_path),
        format="jsonl",
        text_column="messages",
        kind="chat",
        n_docs=len(train_rows),
        meta={
            "arm": cfg.arm,
            "task_clusters": len(train_rows),
            "source_selection_sha256": cfg.selection_sha256,
            "strong_semantic_audit": True,
            "model_native_reasoning": True,
        },
    ).save()
    selection = {
        **original,
        "schema_version": 2,
        "source_selection": str(selection_path),
        "source_selection_sha256": cfg.selection_sha256,
        "complete_audit": {
            "root": str(audit_root),
            "summary_sha256": _sha256(summary_path),
            "accepted_targets_sha256": _sha256(accepted_path),
            "rejections_sha256": _sha256(rejections_path),
            "n_accepted_before_cap": len(accepted),
            "n_retained": len(selected_train),
            "cap": cfg.maximum_rows,
        },
        "baseline_comparison_pool": {
            "path": str(source_pool_path),
            "sha256": cfg.source_pool_sha256,
            "sample_indices": list(range(8, 16)),
        },
        "excluded_problem_ids": [
            problem_id for problem_id in original_ids if problem_id not in retained
        ],
        "train": selected_train,
        "final": final_rows,
        "files": {cfg.arm: {"path": str(train_path), "sha256": _sha256(train_path)}},
        "token_summary": {
            field: {
                "minimum": min(
                    int(row["target"]["variants"][cfg.arm][field])
                    for row in selected_train
                ),
                "median": statistics.median(
                    int(row["target"]["variants"][cfg.arm][field])
                    for row in selected_train
                ),
                "maximum": max(
                    int(row["target"]["variants"][cfg.arm][field])
                    for row in selected_train
                ),
                "total": sum(
                    int(row["target"]["variants"][cfg.arm][field])
                    for row in selected_train
                ),
            }
            for field in ("prompt_tokens", "supervised_tokens", "rendered_tokens")
        },
    }
    selection_out = out / "selection.json"
    _write_json(selection_out, selection)
    result = {
        "schema_version": 1,
        "arm": cfg.arm,
        "n_train": len(selected_train),
        "unique_statement_clusters": len(clusters),
        "selection_sha256": _sha256(selection_out),
        "train_sha256": _sha256(train_path),
        "source_selection_sha256": cfg.selection_sha256,
        "programs_and_rationales_untouched": True,
        "all_strong_judge_pass": True,
        "reserved_final_ids": len(final_rows),
        "reserved_final_clusters": len(final_clusters),
    }
    _write_json(out / "prepared.json", result)
    return result


def main() -> None:
    cfg = parse(PrepareScaleCompleteConfig)
    save(cfg, Path(cfg.out) / "prepare_config.yaml")
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["PrepareScaleCompleteConfig", "run"]
