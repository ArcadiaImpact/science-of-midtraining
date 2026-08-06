"""Freeze the statement-disjoint 722-cluster pool for scaled code SFT."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prior_latmem.gemma4_e4b_transfer_canary_20260805.run_transfer import (
    BASELINE_PREFIX,
    DATASET_REVISION,
    MODEL,
    MODEL_REVISION,
    TransferConfig,
    _cluster_id,
    _fetch_questions,
    _load_baseline,
    _order_key,
    _render_target,
    _support_stratum,
    _target_candidates,
)
from scimt.config import parse, save


SOURCE_SELECTION_SHA256 = (
    "70cfb9a1051f8eeada12d5f58d37f6eda6c2b2bdf65a2ef2b5890e2ad367de6e"
)
EXPECTED_FINAL_IDS = 294
EXPECTED_FINAL_CLUSTERS = 290


@dataclass(frozen=True)
class ScalePoolConfig:
    source_selection: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_canary_20260805/shared_data/selection.json"
    )
    baseline_root: str = "/workspace/input/gemma4-baseline-hub/" + BASELINE_PREFIX
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/scale/pool"
    )
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = DATASET_REVISION
    model: str = MODEL
    model_revision: str = MODEL_REVISION
    discovery_samples: int = 8
    sequence_len: int = 8192
    expected_candidates: int = 722
    seed: int = 20260806
    source_selection_sha256: str = SOURCE_SELECTION_SHA256

    def __post_init__(self) -> None:
        if self.model != MODEL or self.model_revision != MODEL_REVISION:
            raise ValueError("scaled pool must use the pinned Gemma substrate")
        if self.dataset_revision != DATASET_REVISION:
            raise ValueError("scaled pool must use the frozen problem bank")
        if self.discovery_samples != 8 or self.sequence_len != 8192:
            raise ValueError("scaled pool freezes discovery=8 and sequence_len=8192")
        if self.expected_candidates != 722:
            raise ValueError(
                "the frozen post-development eligible pool has 722 clusters"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def run(cfg: ScalePoolConfig) -> dict[str, Any]:
    from transformers import AutoProcessor

    from scimt.train.axolotl import STAGES_DIR

    source_selection_path = Path(cfg.source_selection)
    if _sha256(source_selection_path) != cfg.source_selection_sha256:
        raise ValueError("frozen transfer selection checksum drifted")
    source_selection = json.loads(source_selection_path.read_text())
    development_clusters = {
        str(row["statement_cluster"]) for row in source_selection["development"]
    }
    if len(development_clusters) != 192:
        raise ValueError("frozen development split is not 192 unique clusters")

    out = Path(cfg.out)
    transfer_cfg = TransferConfig(
        shared_data=str(out / "loader"),
        baseline_root=cfg.baseline_root,
        dataset_repo=cfg.dataset_repo,
        dataset_revision=cfg.dataset_revision,
        model=cfg.model,
        model_revision=cfg.model_revision,
        seed=cfg.seed,
        phase="prepare",
    )
    questions = _fetch_questions(transfer_cfg, out / "loader")
    by_problem = _load_baseline(transfer_cfg)
    processor = AutoProcessor.from_pretrained(
        cfg.model, revision=cfg.model_revision, trust_remote_code=False
    )
    template_path = STAGES_DIR / "assets" / "gemma4_verified_reasoning_text.jinja"
    custom_template = template_path.read_text()

    eligible: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    all_train_clusters: set[str] = set()
    for problem_id, question in sorted(questions.items()):
        if question["split"] != "train":
            continue
        cluster = _cluster_id(str(question["probe"]))
        all_train_clusters.add(cluster)
        if cluster in development_clusters:
            rejection_counts["reserved_development_cluster"] += 1
            continue
        samples = by_problem[problem_id]
        candidates = _target_candidates(samples, cfg.discovery_samples)
        if not candidates:
            rejection_counts["no_complete_correct_discovery_target"] += 1
            continue
        target: dict[str, Any] | None = None
        for candidate in candidates:
            reasoning = str(candidate["reasoning"]).strip()
            source = str(candidate["source"]).strip()
            rendered = _render_target(
                processor, custom_template, str(question["probe"]), reasoning, source
            )
            if int(rendered["complete"]["rendered_tokens"]) > cfg.sequence_len:
                rejection_counts["complete_over_sequence_len"] += 1
                continue
            target = {
                "source_sha256": str(candidate["source_sha256"]),
                "sample_index": int(candidate["sample_index"]),
                "sample_tokens": int(candidate["n_tokens"]),
                "reasoning": reasoning,
                "source": source,
                "variants": rendered,
            }
            break
        if target is None:
            rejection_counts["no_target_within_sequence_len"] += 1
            continue
        discovery_correct = sum(bool(row["correct"]) for row in samples[:8])
        eligible.append(
            {
                "problem_id": problem_id,
                "split": "train",
                "memberships": sorted((question.get("sets") or {}).keys()),
                "statement_cluster": cluster,
                "discovery_correct": discovery_correct,
                "discovery_n": 8,
                "support_stratum": _support_stratum(discovery_correct),
                "target": target,
            }
        )

    selected: list[dict[str, Any]] = []
    used_clusters: set[str] = set()
    for row in sorted(
        eligible,
        key=lambda item: _order_key(
            cfg.seed, "scale_candidate", str(item["problem_id"])
        ),
    ):
        cluster = str(row["statement_cluster"])
        if cluster in used_clusters:
            rejection_counts["duplicate_statement_cluster"] += 1
            continue
        used_clusters.add(cluster)
        selected.append(row)
    if len(selected) != cfg.expected_candidates:
        raise ValueError(
            f"scaled candidate topology drifted: {len(selected)} != "
            f"{cfg.expected_candidates}"
        )
    if used_clusters & development_clusters:
        raise AssertionError("scaled training candidates overlap development")

    clean_eval_ids = set(source_selection["clean_eval"]["problem_ids"])
    clean_eval_clusters = {
        _cluster_id(str(questions[problem_id]["probe"]))
        for problem_id in clean_eval_ids
    }
    final_train_overlap = clean_eval_clusters & all_train_clusters
    if final_train_overlap:
        raise ValueError(
            "reserved final set overlaps the train pool by "
            f"{len(final_train_overlap)} statement clusters"
        )
    # The frozen final set contains four within-eval alias pairs.  They do not
    # leak train statements, but IDs and unique clusters are therefore distinct
    # topology invariants and must not be conflated.
    if (
        len(clean_eval_ids) != EXPECTED_FINAL_IDS
        or len(clean_eval_clusters) != EXPECTED_FINAL_CLUSTERS
    ):
        raise ValueError(
            "reserved final topology drifted: "
            f"ids={len(clean_eval_ids)} clusters={len(clean_eval_clusters)}"
        )
    if any(int(row["target"]["sample_index"]) >= 8 for row in selected):
        raise AssertionError("scaled targets consulted held-out baseline samples")

    selection = {
        "schema_version": 1,
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "dataset_repo": cfg.dataset_repo,
        "dataset_revision": cfg.dataset_revision,
        "baseline_root": cfg.baseline_root,
        "baseline_prefix": BASELINE_PREFIX,
        "seed": cfg.seed,
        "source_selection": str(source_selection_path),
        "source_selection_sha256": cfg.source_selection_sha256,
        "support_discovery": {
            "sample_indices": list(range(8)),
            "held_out_baseline_samples_consulted": False,
        },
        "target_selection": {
            "sample_indices": list(range(8)),
            "held_out_baseline_samples_consulted": False,
        },
        "train": selected,
        "development": source_selection["development"],
        "clean_eval": source_selection["clean_eval"],
        "pool_summary": {
            "n_candidates": len(selected),
            "unique_candidate_clusters": len(used_clusters),
            "reserved_development_clusters": len(development_clusters),
            "reserved_final_clusters": len(clean_eval_clusters),
            "support_strata": dict(
                sorted(Counter(row["support_stratum"] for row in selected).items())
            ),
            "rejections": dict(sorted(rejection_counts.items())),
        },
        "template": {
            "path": str(template_path),
            "custom_sha256": hashlib.sha256(custom_template.encode()).hexdigest(),
            "native_sha256": hashlib.sha256(
                str(processor.tokenizer.chat_template).encode()
            ).hexdigest(),
            "native_equivalent_all_candidates": True,
        },
    }
    selection_path = out / "selection.json"
    _write_json(selection_path, selection)
    _write_jsonl(
        out / "pool.jsonl",
        [
            {
                key: row[key]
                for key in (
                    "problem_id",
                    "statement_cluster",
                    "support_stratum",
                    "discovery_correct",
                )
            }
            for row in selected
        ],
    )
    result = {
        "schema_version": 1,
        "n_candidates": len(selected),
        "unique_clusters": len(used_clusters),
        "development_overlap": 0,
        "final_overlap": 0,
        "target_sample_indices_below_8": True,
        "selection_sha256": _sha256(selection_path),
        "source_selection_sha256": cfg.source_selection_sha256,
        "rejections": dict(sorted(rejection_counts.items())),
    }
    _write_json(out / "summary.json", result)
    return result


def main() -> None:
    cfg = parse(ScalePoolConfig)
    save(cfg, Path(cfg.out) / "config.yaml")
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["ScalePoolConfig", "run"]
