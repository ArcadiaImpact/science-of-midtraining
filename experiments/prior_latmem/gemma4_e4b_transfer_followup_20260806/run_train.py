"""Audit and train one attribution-ready Gemma 4 coding LoRA arm."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from experiments.prior_latmem.gemma4_e4b_transfer_canary_20260805.run_transfer import (
    LANGUAGE_LORA_REGEX,
    MODEL,
)
from scimt.config import parse, save
from scimt.dataset import Dataset
from scimt.train import LoraConfig, TrainConfig, train_dataset


STRATEGIC_CHECKPOINTS = (8, 16, 32, 48, 64)
STAGES = (
    "sft_star_lora_gemma4_e4b_followup_lr5e5_1xa100",
    "sft_star_lora_gemma4_e4b_followup_lr2e5_1xa100",
    "sft_star_lora_gemma4_e4b_scaled_lr5e5_1xa100",
    "sft_star_lora_gemma4_e4b_scaled_lr2e5_1xa100",
)


@dataclass(frozen=True)
class FollowupTrainConfig:
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/train/compressed_1k_r32_lr5e5"
    )
    shared_data: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/shared_data"
    )
    arm: str = "compressed_1k"
    data_arm: str | None = None
    stage: str = "sft_star_lora_gemma4_e4b_followup_lr5e5_1xa100"
    rank: int = 32
    alpha: int | None = None
    dropout: float = 0.0
    parent_model: str | None = None
    parent_provenance: str | None = None
    seed: int = 20260806
    optimizer_steps: int = 64
    checkpoint_steps: tuple[int, ...] = STRATEGIC_CHECKPOINTS
    minimum_rows: int = 80
    maximum_rows: int = 128
    phase: str = "all"  # audit | train | audit_train | all

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9_]+", self.arm):
            raise ValueError("arm must be a lowercase filesystem-safe slug")
        if self.data_arm is not None and not re.fullmatch(
            r"[a-z0-9_]+", self.data_arm
        ):
            raise ValueError("data_arm must be a lowercase filesystem-safe slug")
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        if self.rank not in {16, 32}:
            raise ValueError("follow-up rank must be 16 or 32")
        if self.alpha is not None and self.alpha != 2 * self.rank:
            raise ValueError("follow-up alpha/r must stay fixed at two")
        if (self.parent_model is None) != (self.parent_provenance is None):
            raise ValueError(
                "parent_model and parent_provenance must be set together"
            )
        if self.optimizer_steps < 1:
            raise ValueError("optimizer_steps must be positive")
        checkpoints = tuple(self.checkpoint_steps)
        if (
            len(checkpoints) != 5
            or tuple(sorted(set(checkpoints))) != checkpoints
            or checkpoints[-1] != self.optimizer_steps
        ):
            raise ValueError(
                "checkpoint_steps must be five increasing strategic steps ending "
                "at optimizer_steps"
            )
        if not 1 <= self.minimum_rows <= self.maximum_rows:
            raise ValueError("training row bounds are invalid")
        if self.phase not in {"audit", "train", "audit_train", "all"}:
            raise ValueError("phase must be audit, train, audit_train, or all")

    @property
    def target_variant(self) -> str:
        """Training-target variant, distinct from the experimental parent arm."""
        return self.data_arm or self.arm


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


def _find_subsequence(values: Sequence[int], needle: Sequence[int]) -> int | None:
    if not needle or len(needle) > len(values):
        return None
    width = len(needle)
    for index in range(len(values) - width + 1):
        if list(values[index : index + width]) == list(needle):
            return index
    return None


def _train_config(cfg: FollowupTrainConfig) -> TrainConfig:
    return TrainConfig(
        model=MODEL,
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
        load_checkpoint_path=cfg.parent_model,
    )


def audit(cfg: FollowupTrainConfig) -> dict[str, Any]:
    from axolotl.cli.config import load_cfg
    from axolotl.processing_strategies import get_processing_strategy
    from axolotl.utils.chat_templates import get_chat_template_from_config
    from axolotl.utils.collators.mm_chat import MultiModalChatDataCollator
    from transformers import AutoProcessor

    from scimt.train.axolotl import load_stage, render_stage

    shared = Path(cfg.shared_data)
    data_path = shared / cfg.target_variant / "train.jsonl"
    selection_path = shared / "selection.json"
    if not data_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("compressed preparation must complete before audit")
    out = Path(cfg.out)
    rendered = render_stage(
        load_stage(cfg.stage), _train_config(cfg), data_path, out / "train"
    )
    ax_cfg = load_cfg(str(rendered))
    if int(ax_cfg.max_steps) != cfg.optimizer_steps:
        raise ValueError(
            f"rendered max_steps={ax_cfg.max_steps} differs from configured "
            f"optimizer_steps={cfg.optimizer_steps}"
        )
    processor = AutoProcessor.from_pretrained(
        MODEL,
        revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
        trust_remote_code=False,
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
    if not cfg.minimum_rows <= len(rows) <= cfg.maximum_rows or len(selected) != len(
        rows
    ):
        raise ValueError(
            "follow-up transfer data is outside the configured matched row bounds"
        )
    eot_ids = tokenizer.encode("<turn|>", add_special_tokens=False)
    if len(eot_ids) != 1:
        raise ValueError(f"Gemma turn terminator is not atomic: {eot_ids}")
    audits: list[dict[str, Any]] = []
    for row, provenance in zip(rows, selected, strict=True):
        problem_id = str(row["problem_id"])
        if problem_id != str(provenance["problem_id"]):
            raise ValueError("training row order drifted from selection manifest")
        if cfg.target_variant not in provenance["target"]["variants"]:
            raise ValueError(
                f"{problem_id}: selection lacks target variant {cfg.target_variant}"
            )
        tokenized = collator([{"messages": row["messages"]}])
        input_ids = tokenized["input_ids"][0].tolist()
        labels = tokenized["labels"][0].tolist()
        if len(input_ids) != len(labels) or len(input_ids) > 8192:
            raise ValueError(f"{problem_id}: invalid collated length")
        supervised = [index for index, label in enumerate(labels) if label != -100]
        expected = int(
            provenance["target"]["variants"][cfg.target_variant]["supervised_tokens"]
        )
        prompt_floor = max(
            0,
            int(provenance["target"]["variants"][cfg.target_variant]["prompt_tokens"])
            - 8,
        )
        if not supervised or any(label != -100 for label in labels[:prompt_floor]):
            raise ValueError(f"{problem_id}: masking contract failed")
        if len(supervised) < max(8, math.floor(0.85 * expected)):
            raise ValueError(f"{problem_id}: too few supervised labels")
        fields = {
            "reasoning": str(
                provenance["target"]["compression"][cfg.target_variant]["reasoning"]
            ),
            "source": str(provenance["target"]["source"]),
        }
        coverages: dict[str, float] = {}
        for field, value in fields.items():
            field_ids = tokenizer.encode(value, add_special_tokens=False)
            core = field_ids[2:-2] if len(field_ids) > 8 else field_ids
            start = _find_subsequence(input_ids, core)
            if start is None:
                raise ValueError(f"{problem_id}: cannot locate {field}")
            coverage = sum(
                labels[index] != -100 for index in range(start, start + len(core))
            ) / len(core)
            if coverage < 0.98:
                raise ValueError(f"{problem_id}: {field} coverage={coverage:.3f}")
            coverages[field] = coverage
        content = str(row["messages"][-1]["content"])
        if "<|channel>thought\n" not in content or "<channel|>" not in content:
            raise ValueError(f"{problem_id}: native thought/final boundary is absent")
        trained_eot = any(
            token == eot_ids[0] and labels[index] == token
            for index, token in enumerate(input_ids)
        )
        if not trained_eot:
            raise ValueError(f"{problem_id}: turn terminator is masked")
        audits.append(
            {
                "problem_id": problem_id,
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
        "rank": cfg.rank,
        "stage": cfg.stage,
        "rendered_config": str(rendered),
        "rendered_config_sha256": _sha256(rendered),
        "dataset_sha256": _sha256(data_path),
        "selection_sha256": _sha256(selection_path),
        "examples": audits,
        "summary": {
            "n": len(audits),
            "tokens_minimum": min(row["tokens"] for row in audits),
            "tokens_median": statistics.median(row["tokens"] for row in audits),
            "tokens_maximum": max(row["tokens"] for row in audits),
            "supervised_minimum": min(row["supervised_tokens"] for row in audits),
            "supervised_median": statistics.median(
                row["supervised_tokens"] for row in audits
            ),
            "supervised_maximum": max(row["supervised_tokens"] for row in audits),
            "all_expected_fields_covered": True,
            "all_turn_terminators_trained": True,
            "all_prompts_masked": True,
        },
    }
    _write_json(out / "data" / "label_audit.json", result)
    return result


def _training_trace(final_checkpoint: Path, expected_steps: int = 64) -> dict[str, Any]:
    state_path = final_checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"missing trainer state: {state_path}")
    state = json.loads(state_path.read_text())
    trace = [
        {
            key: row[key]
            for key in (
                "step",
                "epoch",
                "loss",
                "grad_norm",
                "learning_rate",
                "tokens/train_per_sec_per_gpu",
            )
            if key in row
        }
        for row in state.get("log_history", [])
        if "loss" in row
    ]
    if len(trace) != expected_steps or [int(row["step"]) for row in trace] != list(
        range(1, expected_steps + 1)
    ):
        raise ValueError(
            f"expected a complete {expected_steps}-step training trace, got {len(trace)}"
        )
    if not all(
        math.isfinite(float(row[key]))
        for row in trace
        for key in ("loss", "grad_norm", "learning_rate")
        if key in row
    ):
        raise ValueError("training trace contains a non-finite value")
    return {
        "optimizer_steps": expected_steps,
        "log_history": trace,
        "learning_rates": [float(row["learning_rate"]) for row in trace],
        "losses": [float(row["loss"]) for row in trace],
        "gradient_norms": [float(row["grad_norm"]) for row in trace],
    }


def _prune_routine_checkpoints(
    checkpoint_root: Path, strategic_steps: Sequence[int]
) -> list[int]:
    """Retain only the five attribution checkpoints after training succeeds."""

    strategic = {int(step) for step in strategic_steps}
    pruned: list[int] = []
    for path in sorted(checkpoint_root.glob("checkpoint-*")):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if not match or not path.is_dir():
            continue
        step = int(match.group(1))
        if step in strategic:
            continue
        shutil.rmtree(path)
        pruned.append(step)
    return sorted(pruned)


async def train(cfg: FollowupTrainConfig) -> dict[str, Any]:
    out = Path(cfg.out)
    data_path = Path(cfg.shared_data) / cfg.target_variant / "train.jsonl"
    audit_path = out / "data" / "label_audit.json"
    if not data_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("dataset and arm-specific label audit are required")
    if cfg.parent_provenance is not None and not Path(cfg.parent_provenance).is_file():
        raise FileNotFoundError(
            f"parent provenance is missing: {cfg.parent_provenance}"
        )
    checkpoint = await train_dataset(
        Dataset.at(data_path, kind="chat", text_column="messages"),
        out / "train",
        _train_config(cfg),
        run_name=f"gemma4-e4b-followup-{cfg.arm}-r{cfg.rank}-{cfg.stage}",
    )
    checkpoint_root = out / "train" / "checkpoints"
    trace = _training_trace(
        checkpoint_root / f"checkpoint-{cfg.optimizer_steps}",
        cfg.optimizer_steps,
    )
    epoch_by_step = {
        int(row["step"]): float(row["epoch"])
        for row in trace["log_history"]
        if "epoch" in row
    }
    adapters: list[dict[str, Any]] = []
    for step in cfg.checkpoint_steps:
        path = checkpoint_root / f"checkpoint-{step}"
        adapter = path / "adapter_model.safetensors"
        config = path / "adapter_config.json"
        if not adapter.is_file() or not config.is_file():
            raise FileNotFoundError(f"missing strategic adapter checkpoint {step}")
        adapters.append(
            {
                "step": step,
                "epoch": epoch_by_step.get(step),
                "path": str(path),
                "bytes": adapter.stat().st_size,
                "sha256": _sha256(adapter),
            }
        )
    routine_checkpoints_pruned = _prune_routine_checkpoints(
        checkpoint_root, cfg.checkpoint_steps
    )
    retained_steps = sorted(
        int(path.name.rsplit("-", 1)[-1])
        for path in checkpoint_root.glob("checkpoint-*")
        if path.is_dir() and re.fullmatch(r"checkpoint-\d+", path.name)
    )
    if retained_steps != list(cfg.checkpoint_steps):
        raise RuntimeError(
            f"post-training checkpoint topology drifted: retained={retained_steps}"
        )
    rendered = out / "train" / "axolotl.yaml"
    data_rows = _read_jsonl(data_path)
    attribution = {
        "schema_version": 1,
        "arm": cfg.arm,
        "data_arm": cfg.target_variant,
        "model": MODEL,
        "stage": cfg.stage,
        "rank": cfg.rank,
        "alpha": cfg.alpha or 2 * cfg.rank,
        "seed": cfg.seed,
        "parent": (
            {
                "model": cfg.parent_model,
                "provenance": cfg.parent_provenance,
                "provenance_sha256": _sha256(Path(cfg.parent_provenance)),
            }
            if cfg.parent_provenance is not None
            else {
                "model": MODEL,
                "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
            }
        ),
        "optimizer_steps": cfg.optimizer_steps,
        "dataset": {
            "path": str(data_path),
            "sha256": _sha256(data_path),
            "n_rows": len(data_rows),
            "input_problem_order": [str(row["problem_id"]) for row in data_rows],
            "trainer_shuffle_seed": cfg.seed,
        },
        "rendered_config": str(rendered),
        "rendered_config_sha256": _sha256(rendered),
        "strategic_checkpoints": adapters,
        "checkpoint_policy": {
            "strategic_retained": list(cfg.checkpoint_steps),
            "routine_local_pruned": routine_checkpoints_pruned,
            "persisted_remotely": list(cfg.checkpoint_steps),
        },
        "schedule": trace,
        "versions": _versions(),
    }
    _write_json(out / "attribution_manifest.json", attribution)
    result = {
        "schema_version": 1,
        "arm": cfg.arm,
        "rank": cfg.rank,
        "stage": cfg.stage,
        "checkpoint": checkpoint.as_dict(),
        "adapters": adapters,
        "routine_checkpoints_pruned": routine_checkpoints_pruned,
        "attribution_manifest": str(out / "attribution_manifest.json"),
        "versions": _versions(),
        "host": platform.node(),
    }
    _write_json(out / "training_complete.json", result)
    return result


async def run(cfg: FollowupTrainConfig) -> dict[str, Any]:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    result: dict[str, Any] = {}
    if cfg.phase in {"audit", "audit_train", "all"}:
        result["audit"] = audit(cfg)
    if cfg.phase in {"train", "audit_train", "all"}:
        result["training"] = await train(cfg)
    return result


def main() -> None:
    asyncio.run(run(parse(FollowupTrainConfig)))


if __name__ == "__main__":
    main()


__all__ = ["FollowupTrainConfig", "audit", "run", "train"]
