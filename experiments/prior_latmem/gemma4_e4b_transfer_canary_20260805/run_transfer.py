"""Prepare, audit, and train the Gemma 4 E4B held-out transfer canary.

The two arms expose the same 128 task clusters and the same exact programs.
Only the assistant target representation differs: ``complete`` supervises the
native thought channel plus the program, while ``concise`` supervises the
byte-identical program directly.  Development tasks are never used for target
selection or optimization.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import platform
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prior_latmem.star_sample_generate import (
    DATASET_PREFIX,
    QUESTION_FILES,
    build_union_records,
)
from experiments.prior_latmem.star_score_worker import classify_sample
from scimt.config import parse, save
from scimt.dataset import Dataset
from scimt.train import LoraConfig, TrainConfig, train_dataset


MODEL = "google/gemma-4-E4B-it"
MODEL_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
DATASET_REVISION = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
BASELINE_PREFIX = "star_sampling/20260805/gemma-4-e4b-it-thinking"
LANGUAGE_LORA_REGEX = (
    r"model\.language_model\.layers\.\d+\."
    r"(?:_checkpoint_wrapped_module\.)?"
    r"(?:mlp|self_attn)\.(?:up|down|gate|q|k|v|o)_proj"
)
ARMS = ("concise", "complete")


@dataclass(frozen=True)
class TransferConfig:
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_canary_20260805/concise"
    )
    shared_data: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_canary_20260805/shared_data"
    )
    baseline_root: str = (
        "/workspace/input/gemma4-baseline-hub/" + BASELINE_PREFIX
    )
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = DATASET_REVISION
    model: str = MODEL
    model_revision: str = MODEL_REVISION
    stage: str = "sft_star_lora_gemma4_e4b_transfer_1xa100"
    arm: str = "concise"
    train_frontier: int = 64
    train_moderate: int = 64
    dev_zero: int = 64
    dev_frontier: int = 64
    dev_moderate: int = 64
    discovery_samples: int = 8
    baseline_eval_samples: int = 8
    sequence_len: int = 8192
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0
    seed: int = 20260805
    phase: str = "all"  # prepare | audit | train | audit_train | all

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if self.phase not in {"prepare", "audit", "train", "audit_train", "all"}:
            raise ValueError(
                "phase must be prepare, audit, train, audit_train, or all"
            )
        counts = (
            self.train_frontier,
            self.train_moderate,
            self.dev_zero,
            self.dev_frontier,
            self.dev_moderate,
            self.discovery_samples,
            self.baseline_eval_samples,
        )
        if min(counts) < 1:
            raise ValueError("all split and sample counts must be positive")
        if self.discovery_samples + self.baseline_eval_samples != 16:
            raise ValueError("the baseline discovery/evaluation partition must total 16")
        if self.train_frontier + self.train_moderate != 128:
            raise ValueError("the predeclared transfer training set has 128 tasks")
        if self.dev_zero + self.dev_frontier + self.dev_moderate != 192:
            raise ValueError("the predeclared transfer development set has 192 tasks")
        if self.sequence_len != 8192 or min(self.rank, self.alpha) < 1:
            raise ValueError("this canary requires sequence_len=8192 and positive LoRA sizes")


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
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("axolotl", "torch", "transformers", "peft", "bitsandbytes"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _normalize_statement(probe: str) -> str:
    normalized = unicodedata.normalize("NFKC", probe).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _cluster_id(probe: str) -> str:
    return hashlib.sha256(_normalize_statement(probe).encode()).hexdigest()


def _order_key(seed: int, purpose: str, problem_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{purpose}\0{problem_id}".encode()).hexdigest()


def _support_stratum(correct: int) -> str:
    # The discovery half has eight draws.  These bands are the direct analogue
    # of 0, 1--4, and 5--11 successes in the 16-draw baseline.
    if correct == 0:
        return "zero"
    if correct <= 2:
        return "frontier"
    if correct <= 5:
        return "moderate"
    return "high"


def _target_candidates(
    samples: Sequence[Mapping[str, Any]], discovery_samples: int
) -> list[Mapping[str, Any]]:
    """Return eligible targets from the discovery half only.

    Samples in the held-out baseline half must never influence training-task
    eligibility or target choice; otherwise their post-train comparison would
    be selected on the outcome it is meant to estimate.
    """
    return sorted(
        (
            sample
            for sample in samples[:discovery_samples]
            if sample["correct"]
            and sample["finish_reason"] == "stop"
            and sample["thinking_status"] == "complete"
            and isinstance(sample.get("reasoning"), str)
            and str(sample["reasoning"]).strip()
            and isinstance(sample.get("source"), str)
            and str(sample["source"]).strip()
        ),
        key=lambda sample: (
            int(sample["n_tokens"]),
            str(sample["source_sha256"]),
        ),
    )


def _find_subsequence(values: Sequence[int], needle: Sequence[int]) -> int | None:
    if not needle or len(needle) > len(values):
        return None
    width = len(needle)
    for index in range(len(values) - width + 1):
        if list(values[index : index + width]) == list(needle):
            return index
    return None


def _fetch_questions(cfg: TransferConfig, out: Path) -> dict[str, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            cfg.dataset_repo,
            repo_type="dataset",
            revision=cfg.dataset_revision,
            allow_patterns=[
                f"{DATASET_PREFIX}/{relative}" for relative in QUESTION_FILES.values()
            ],
            local_dir=out / "source",
        )
    )
    records = build_union_records(snapshot / DATASET_PREFIX)
    return {str(row["problem_id"]): row for row in records}


def _load_baseline(cfg: TransferConfig) -> dict[str, list[dict[str, Any]]]:
    root = Path(cfg.baseline_root)
    verdict_path = root / "scored" / "verdicts.jsonl"
    chunks = sorted((root / "shards" / "00" / "chunks").glob("*/generations.jsonl"))
    if not verdict_path.is_file() or len(chunks) != 34:
        raise FileNotFoundError(
            f"baseline snapshot incomplete: verdicts={verdict_path.is_file()}, "
            f"chunks={len(chunks)}"
        )
    verdicts = {
        (str(row["problem_id"]), str(row["source_sha256"])): row
        for row in _read_jsonl(verdict_path)
    }
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        for raw in _read_jsonl(chunk):
            classified = classify_sample(raw)
            source = classified.pop("_source", None)
            source_sha = classified.get("source_sha256")
            verdict = (
                verdicts.get((str(raw["problem_id"]), str(source_sha)))
                if source_sha
                else None
            )
            by_problem[str(raw["problem_id"])].append(
                {
                    "problem_id": str(raw["problem_id"]),
                    "sample_index": int(raw["sample_index"]),
                    "finish_reason": str(raw.get("finish_reason")),
                    "thinking_status": str(raw.get("thinking_status")),
                    "reasoning": raw.get("reasoning"),
                    "source": source,
                    "source_sha256": source_sha,
                    "n_tokens": int(raw.get("n_tokens", 0)),
                    "correct": bool(verdict and verdict["correct"]),
                    "correctness_status": (
                        str(verdict["correctness_status"])
                        if verdict
                        else str(classified["correctness_status"])
                    ),
                }
            )
    bad = {
        problem_id: len(rows)
        for problem_id, rows in by_problem.items()
        if len(rows) != 16
    }
    if bad or len(by_problem) != 1620:
        raise ValueError(
            f"baseline topology drift: problems={len(by_problem)}, bad={bad}"
        )
    for rows in by_problem.values():
        rows.sort(key=lambda row: int(row["sample_index"]))
        if [row["sample_index"] for row in rows] != list(range(16)):
            raise ValueError(f"sample-index drift for {rows[0]['problem_id']}")
    return by_problem


def _render_target(
    processor: Any,
    custom_template: str,
    probe: str,
    reasoning: str,
    source: str,
) -> dict[str, Any]:
    user = {"role": "user", "content": probe}
    canonical_complete = [
        user,
        {"role": "assistant", "reasoning_content": reasoning, "content": source},
    ]
    complete_messages = [
        user,
        {
            "role": "assistant",
            "content": f"<|channel>thought\n{reasoning}\n<channel|>{source}",
        },
    ]
    concise_messages = [user, {"role": "assistant", "content": source}]
    variants = {
        "complete": (canonical_complete, complete_messages),
        "concise": (concise_messages, concise_messages),
    }
    result: dict[str, Any] = {}
    for arm, (canonical, messages) in variants.items():
        native = processor.apply_chat_template(
            canonical,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        rendered = processor.apply_chat_template(
            messages,
            chat_template=custom_template,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        if rendered != native:
            raise ValueError(f"{arm} custom template differs from pinned native template")
        prompt_native = processor.apply_chat_template(
            [user], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        prompt = processor.apply_chat_template(
            [user],
            chat_template=custom_template,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if prompt != prompt_native:
            raise ValueError("custom generation prompt differs from pinned native template")
        input_ids = processor.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        prompt_ids = processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"{arm} assistant target is not prompt-prefix aligned")
        result[arm] = {
            "messages": messages,
            "rendered_tokens": len(input_ids),
            "prompt_tokens": len(prompt_ids),
            "supervised_tokens": len(input_ids) - len(prompt_ids),
        }
    return result


def _select_unique(
    rows: Sequence[Mapping[str, Any]],
    *,
    n: int,
    seed: int,
    purpose: str,
    used_clusters: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for raw in sorted(
        rows,
        key=lambda row: _order_key(seed, purpose, str(row["problem_id"])),
    ):
        row = dict(raw)
        cluster = str(row["statement_cluster"])
        if cluster in used_clusters:
            continue
        selected.append(row)
        used_clusters.add(cluster)
        if len(selected) == n:
            break
    if len(selected) != n:
        raise ValueError(
            f"only {len(selected)} unique clusters available for {purpose}, need {n}"
        )
    return selected


def prepare(cfg: TransferConfig) -> dict[str, Any]:
    from transformers import AutoProcessor

    from scimt.train.axolotl import STAGES_DIR

    shared = Path(cfg.shared_data)
    questions = _fetch_questions(cfg, shared)
    by_problem = _load_baseline(cfg)
    processor = AutoProcessor.from_pretrained(
        cfg.model, revision=cfg.model_revision, trust_remote_code=False
    )
    template_path = STAGES_DIR / "assets" / "gemma4_verified_reasoning_text.jinja"
    custom_template = template_path.read_text()

    pool: list[dict[str, Any]] = []
    target_by_problem: dict[str, dict[str, Any]] = {}
    rejected_targets: Counter[str] = Counter()
    for problem_id, question in sorted(questions.items()):
        samples = by_problem[problem_id]
        discovery = samples[: cfg.discovery_samples]
        baseline_eval = samples[cfg.discovery_samples :]
        discovery_correct = sum(bool(row["correct"]) for row in discovery)
        eval_correct = sum(bool(row["correct"]) for row in baseline_eval)
        cluster = _cluster_id(str(question["probe"]))
        row = {
            "problem_id": problem_id,
            "split": str(question["split"]),
            "memberships": sorted((question.get("sets") or {}).keys()),
            "statement_cluster": cluster,
            "discovery_correct": discovery_correct,
            "discovery_n": cfg.discovery_samples,
            "baseline_eval_correct": eval_correct,
            "baseline_eval_n": cfg.baseline_eval_samples,
            "all_correct": discovery_correct + eval_correct,
            "support_stratum": _support_stratum(discovery_correct),
            "baseline_eval_samples": [
                {
                    key: sample[key]
                    for key in (
                        "sample_index",
                        "correct",
                        "correctness_status",
                        "finish_reason",
                        "thinking_status",
                        "n_tokens",
                    )
                }
                for sample in baseline_eval
            ],
        }
        if question["split"] == "train":
            candidates = _target_candidates(samples, cfg.discovery_samples)
            for candidate in candidates:
                reasoning = str(candidate["reasoning"]).strip()
                source = str(candidate["source"]).strip()
                rendered = _render_target(
                    processor, custom_template, str(question["probe"]), reasoning, source
                )
                if rendered["complete"]["rendered_tokens"] > cfg.sequence_len:
                    rejected_targets["complete_over_sequence_len"] += 1
                    continue
                target = {
                    "source_sha256": str(candidate["source_sha256"]),
                    "sample_index": int(candidate["sample_index"]),
                    "sample_tokens": int(candidate["n_tokens"]),
                    "reasoning": reasoning,
                    "source": source,
                    "variants": rendered,
                }
                target_by_problem[problem_id] = target
                break
            if not candidates:
                rejected_targets["no_complete_correct_target"] += 1
            elif problem_id not in target_by_problem:
                rejected_targets["no_target_within_sequence_len"] += 1
        row["target_eligible"] = problem_id in target_by_problem
        if problem_id in target_by_problem:
            row["complete_tokens"] = target_by_problem[problem_id]["variants"][
                "complete"
            ]["rendered_tokens"]
            row["concise_tokens"] = target_by_problem[problem_id]["variants"][
                "concise"
            ]["rendered_tokens"]
        pool.append(row)

    train_pool = [row for row in pool if row["split"] == "train"]
    used_clusters: set[str] = set()
    train_frontier = _select_unique(
        [
            row
            for row in train_pool
            if row["support_stratum"] == "frontier" and row["target_eligible"]
        ],
        n=cfg.train_frontier,
        seed=cfg.seed,
        purpose="train_frontier",
        used_clusters=used_clusters,
    )
    train_moderate = _select_unique(
        [
            row
            for row in train_pool
            if row["support_stratum"] == "moderate" and row["target_eligible"]
        ],
        n=cfg.train_moderate,
        seed=cfg.seed,
        purpose="train_moderate",
        used_clusters=used_clusters,
    )
    train_rows = train_frontier + train_moderate
    dev_rows: list[dict[str, Any]] = []
    for stratum, n in (
        ("zero", cfg.dev_zero),
        ("frontier", cfg.dev_frontier),
        ("moderate", cfg.dev_moderate),
    ):
        dev_rows.extend(
            _select_unique(
                [row for row in train_pool if row["support_stratum"] == stratum],
                n=n,
                seed=cfg.seed,
                purpose=f"dev_{stratum}",
                used_clusters=used_clusters,
            )
        )

    train_clusters = {str(row["statement_cluster"]) for row in train_rows}
    dev_clusters = {str(row["statement_cluster"]) for row in dev_rows}
    if train_clusters & dev_clusters or len(train_clusters) != 128 or len(dev_clusters) != 192:
        raise AssertionError("train/dev statement-cluster split is not disjoint")
    all_train_clusters = {
        str(row["statement_cluster"]) for row in train_pool
    }
    eval_rows = [row for row in pool if row["split"] == "eval"]
    clean_eval = [
        row for row in eval_rows if str(row["statement_cluster"]) not in all_train_clusters
    ]
    excluded_eval = [row for row in eval_rows if row not in clean_eval]
    if len(clean_eval) != 294 or len(excluded_eval) != 30:
        raise ValueError(
            f"alias-clean eval topology drift: clean={len(clean_eval)}, "
            f"excluded={len(excluded_eval)}"
        )

    # Store target provenance once, then emit two data files with identical
    # row order and problem IDs.
    selected_train: list[dict[str, Any]] = []
    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for row in train_rows:
        target = target_by_problem[str(row["problem_id"])]
        selected = {**row, "target": target}
        selected_train.append(selected)
        for arm in ARMS:
            arm_rows[arm].append(
                {
                    "problem_id": str(row["problem_id"]),
                    "messages": target["variants"][arm]["messages"],
                }
            )

    shared.mkdir(parents=True, exist_ok=True)
    _write_jsonl(shared / "pool.jsonl", pool)
    for arm in ARMS:
        arm_dir = shared / arm
        path = arm_dir / "train.jsonl"
        _write_jsonl(path, arm_rows[arm])
        Dataset(
            path=str(path),
            format="jsonl",
            text_column="messages",
            kind="chat",
            n_docs=len(arm_rows[arm]),
            meta={
                "arm": arm,
                "dataset_revision": cfg.dataset_revision,
                "verified_execution": True,
                "task_clusters": 128,
            },
        ).save()

    selection = {
        "schema_version": 1,
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "dataset_repo": cfg.dataset_repo,
        "dataset_revision": cfg.dataset_revision,
        "baseline_root": cfg.baseline_root,
        "baseline_prefix": BASELINE_PREFIX,
        "seed": cfg.seed,
        "support_discovery": {
            "sample_indices": list(range(cfg.discovery_samples)),
            "strata": {"zero": [0, 0], "frontier": [1, 2], "moderate": [3, 5]},
        },
        "target_selection": {
            "sample_indices": list(range(cfg.discovery_samples)),
            "held_out_baseline_samples_consulted": False,
        },
        "baseline_evaluation": {
            "sample_indices": list(range(cfg.discovery_samples, 16)),
            "n": cfg.baseline_eval_samples,
        },
        "statement_normalization": "NFKC + casefold + whitespace collapse",
        "train": selected_train,
        "development": dev_rows,
        "clean_eval": {
            "n": len(clean_eval),
            "problem_ids": sorted(str(row["problem_id"]) for row in clean_eval),
            "excluded_alias_n": len(excluded_eval),
            "excluded_alias_problem_ids": sorted(
                str(row["problem_id"]) for row in excluded_eval
            ),
        },
        "pool_summary": {
            "train_n": len(train_pool),
            "train_unique_clusters": len(all_train_clusters),
            "support_strata": dict(
                sorted(Counter(str(row["support_stratum"]) for row in train_pool).items())
            ),
            "target_eligible": sum(bool(row["target_eligible"]) for row in train_pool),
            "target_rejections": dict(sorted(rejected_targets.items())),
        },
        "template": {
            "path": str(template_path),
            "custom_sha256": hashlib.sha256(custom_template.encode()).hexdigest(),
            "native_sha256": hashlib.sha256(
                str(processor.tokenizer.chat_template).encode()
            ).hexdigest(),
            "native_equivalent_both_arms": True,
        },
        "files": {
            arm: {
                "path": str(shared / arm / "train.jsonl"),
                "sha256": _sha256(shared / arm / "train.jsonl"),
            }
            for arm in ARMS
        },
        "token_summary": {
            arm: {
                field: {
                    "min": min(
                        row["target"]["variants"][arm][field] for row in selected_train
                    ),
                    "median": statistics.median(
                        row["target"]["variants"][arm][field] for row in selected_train
                    ),
                    "max": max(
                        row["target"]["variants"][arm][field] for row in selected_train
                    ),
                    "total": sum(
                        row["target"]["variants"][arm][field] for row in selected_train
                    ),
                }
                for field in ("prompt_tokens", "supervised_tokens", "rendered_tokens")
            }
            for arm in ARMS
        },
    }
    selection_path = shared / "selection.json"
    _write_json(selection_path, selection)
    _write_json(
        shared / "prepared.json",
        {
            "schema_version": 1,
            "selection_sha256": _sha256(selection_path),
            "pool_sha256": _sha256(shared / "pool.jsonl"),
            "train_problem_ids_identical": [
                row["problem_id"] for row in arm_rows["concise"]
            ]
            == [row["problem_id"] for row in arm_rows["complete"]],
            "train_clusters": len(train_clusters),
            "development_clusters": len(dev_clusters),
        },
    )
    return selection


def _training_config(cfg: TransferConfig) -> TrainConfig:
    return TrainConfig(
        model=cfg.model,
        seed=cfg.seed,
        backend="axolotl",
        stage=cfg.stage,
        lora=LoraConfig(
            r=cfg.rank,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            target_linear=False,
            target_modules=LANGUAGE_LORA_REGEX,
        ),
    )


def audit_training_examples(cfg: TransferConfig) -> dict[str, Any]:
    from axolotl.cli.config import load_cfg
    from axolotl.processing_strategies import get_processing_strategy
    from axolotl.utils.chat_templates import get_chat_template_from_config
    from axolotl.utils.collators.mm_chat import MultiModalChatDataCollator
    from transformers import AutoProcessor

    from scimt.train.axolotl import load_stage, render_stage

    shared = Path(cfg.shared_data)
    data_path = shared / cfg.arm / "train.jsonl"
    selection_path = shared / "selection.json"
    if not data_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("prepare and copy shared transfer data before auditing")
    out = Path(cfg.out)
    rendered = render_stage(
        load_stage(cfg.stage), _training_config(cfg), data_path, out / "train"
    )
    ax_cfg = load_cfg(str(rendered))
    processor = AutoProcessor.from_pretrained(
        cfg.model, revision=cfg.model_revision, trust_remote_code=False
    )
    tokenizer = processor.tokenizer
    dataset_cfg = ax_cfg.datasets[0]
    template = get_chat_template_from_config(ax_cfg, tokenizer=tokenizer)
    strategy = get_processing_strategy(
        processor,
        template,
        ax_cfg.chat_template,
        train_on_inputs=bool(ax_cfg.train_on_inputs),
        roles_to_train=dataset_cfg.get("roles_to_train"),
        train_on_eos=dataset_cfg.get("train_on_eos"),
        role_boundaries_override=[dict(spec) for spec in ax_cfg.role_boundaries],
        field_messages=[dataset_cfg.get("field_messages", "messages")],
    )
    collator = MultiModalChatDataCollator(
        tokenizer=tokenizer, processing_strategy=strategy, padding=True
    )
    rows = _read_jsonl(data_path)
    selection = json.loads(selection_path.read_text())
    selected = selection["train"]
    if len(rows) != 128 or len(selected) != 128:
        raise ValueError("transfer training data must contain 128 rows")
    eot_ids = tokenizer.encode("<turn|>", add_special_tokens=False)
    if len(eot_ids) != 1:
        raise ValueError(f"Gemma turn terminator is not atomic: {eot_ids}")
    audits: list[dict[str, Any]] = []
    for row, provenance in zip(rows, selected, strict=True):
        if str(row["problem_id"]) != str(provenance["problem_id"]):
            raise ValueError("training row order drifted from selection manifest")
        tokenized = collator([{"messages": row["messages"]}])
        input_ids = tokenized["input_ids"][0].tolist()
        labels = tokenized["labels"][0].tolist()
        if len(input_ids) != len(labels) or len(input_ids) > cfg.sequence_len:
            raise ValueError(f"{row['problem_id']}: invalid collated length")
        supervised = [index for index, label in enumerate(labels) if label != -100]
        if not supervised:
            raise ValueError(f"{row['problem_id']}: all labels are masked")
        expected = int(provenance["target"]["variants"][cfg.arm]["supervised_tokens"])
        prompt_floor = max(
            0, int(provenance["target"]["variants"][cfg.arm]["prompt_tokens"]) - 8
        )
        if any(label != -100 for label in labels[:prompt_floor]):
            raise ValueError(f"{row['problem_id']}: prompt token is supervised")
        if len(supervised) < max(8, math.floor(0.85 * expected)):
            raise ValueError(
                f"{row['problem_id']}: {len(supervised)} labels for ~{expected} expected"
            )
        fields = ["source"] if cfg.arm == "concise" else ["reasoning", "source"]
        coverages: dict[str, float] = {}
        for field in fields:
            field_ids = tokenizer.encode(
                provenance["target"][field], add_special_tokens=False
            )
            core = field_ids[2:-2] if len(field_ids) > 8 else field_ids
            start = _find_subsequence(input_ids, core)
            if start is None:
                raise ValueError(f"{row['problem_id']}: cannot locate {field}")
            coverage = sum(
                labels[index] != -100 for index in range(start, start + len(core))
            ) / len(core)
            if coverage < 0.98:
                raise ValueError(
                    f"{row['problem_id']}: {field} label coverage={coverage:.3f}"
                )
            coverages[field] = coverage
        content = str(row["messages"][-1]["content"])
        if cfg.arm == "concise" and "<|channel>thought\n" in content:
            raise ValueError(f"{row['problem_id']}: concise target contains thought channel")
        if cfg.arm == "complete" and "<|channel>thought\n" not in content:
            raise ValueError(f"{row['problem_id']}: complete target lacks thought channel")
        trained_eot = any(
            token == eot_ids[0] and labels[index] == token
            for index, token in enumerate(input_ids)
        )
        if not trained_eot:
            raise ValueError(f"{row['problem_id']}: turn terminator is masked")
        audits.append(
            {
                "problem_id": str(row["problem_id"]),
                "tokens": len(input_ids),
                "supervised_tokens": len(supervised),
                "first_supervised": supervised[0],
                "last_supervised": supervised[-1],
                "field_label_coverage": coverages,
                "trained_eot": True,
            }
        )
    result = {
        "schema_version": 1,
        "arm": cfg.arm,
        "rendered_config": str(rendered),
        "rendered_config_sha256": _sha256(rendered),
        "examples": audits,
        "summary": {
            "n": len(audits),
            "tokens_min": min(row["tokens"] for row in audits),
            "tokens_median": statistics.median(row["tokens"] for row in audits),
            "tokens_max": max(row["tokens"] for row in audits),
            "supervised_min": min(row["supervised_tokens"] for row in audits),
            "supervised_median": statistics.median(
                row["supervised_tokens"] for row in audits
            ),
            "supervised_max": max(row["supervised_tokens"] for row in audits),
            "all_expected_fields_covered": True,
            "all_turn_terminators_trained": True,
            "all_prompts_masked": True,
        },
    }
    _write_json(out / "data" / "label_audit.json", result)
    return result


async def train(cfg: TransferConfig) -> dict[str, Any]:
    out = Path(cfg.out)
    data_path = Path(cfg.shared_data) / cfg.arm / "train.jsonl"
    audit_path = out / "data" / "label_audit.json"
    if not data_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("transfer data and arm-specific label audit are required")
    checkpoint = await train_dataset(
        Dataset.at(data_path, kind="chat", text_column="messages"),
        out / "train",
        _training_config(cfg),
        run_name=f"gemma4-e4b-transfer-{cfg.arm}-r32",
    )
    checkpoint_root = out / "train" / "checkpoints"
    adapters = sorted(
        checkpoint_root.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1])
    )
    expected_steps = [16, 32, 48, 64]
    observed_steps = [int(path.name.split("-")[-1]) for path in adapters]
    if observed_steps != expected_steps:
        raise RuntimeError(
            f"checkpoint topology drift: observed={observed_steps}, expected={expected_steps}"
        )
    adapter_meta = []
    for adapter in adapters:
        config_path = adapter / "adapter_config.json"
        weights = adapter / "adapter_model.safetensors"
        if not config_path.is_file() or not weights.is_file() or weights.stat().st_size == 0:
            raise RuntimeError(f"incomplete adapter checkpoint: {adapter}")
        adapter_meta.append(
            {
                "step": int(adapter.name.split("-")[-1]),
                "path": str(adapter),
                "bytes": weights.stat().st_size,
                "sha256": _sha256(weights),
            }
        )
    result = {
        "schema_version": 1,
        "arm": cfg.arm,
        "checkpoint": checkpoint.as_dict(),
        "adapters": adapter_meta,
        "versions": _versions(),
        "host": platform.node(),
    }
    _write_json(out / "training_complete.json", result)
    return result


async def run(cfg: TransferConfig) -> dict[str, Any]:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    result: dict[str, Any] = {}
    if cfg.phase in {"prepare", "all"}:
        result["selection"] = prepare(cfg)
    if cfg.phase in {"audit", "audit_train", "all"}:
        result["audit"] = audit_training_examples(cfg)
    if cfg.phase in {"train", "audit_train", "all"}:
        result["training"] = await train(cfg)
    return result


def main() -> None:
    asyncio.run(run(parse(TransferConfig)))


if __name__ == "__main__":
    main()


__all__ = [
    "ARMS",
    "LANGUAGE_LORA_REGEX",
    "TransferConfig",
    "audit_training_examples",
    "prepare",
    "run",
    "train",
]
