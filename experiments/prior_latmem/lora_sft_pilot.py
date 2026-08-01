"""Run the one-parent chosen-SFT LoRA preservation pilot.

Two rank-32 adapters start from the published ``sol_no_sdf_ri`` parent:

1. the exact 1,286 chosen dominant programs;
2. those rows plus a deterministic equal-sized sample of the broad Dolci
   re-instruction slice used earlier in the same experiment.

The runner keeps the training and evaluation stages separate.  Axolotl runs
in the cu126 training environment; each selected adapter checkpoint is merged
temporarily and sampled by the dedicated vLLM environment.  Raw adapters,
generations, scores, summaries, and a completion manifest are uploaded to HF.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scimt.config import parse, save

BASE_MODEL = "unsloth/gemma-3-12b-it"
PARENT_ARM = "sol_no_sdf_ri"
STAGE = "sft_dominant_code_lora_it_gemma3_12b_2xa100"
CHOSEN_PATH = "aft/chosen_dominant_sft_20260731/dominant_train_sft.jsonl"
REINSTRUCT_PATH = "reinstruct/dolci_reinstruct.jsonl"
EXPECTED_CHOSEN_SHA256 = (
    "0d9a013e413942b71d036b4e9ccdb44f6493ea0b9d9f441e71a30a7b7648177f"
)
EXPECTED_REINSTRUCT_SHA256 = (
    "f777a5e196a60c35e4ffdb0d57c37bb75a281b9b0a3d89a1bcb757e545da6cb5"
)
EXPECTED_CHOSEN_ROWS = 1286
MERGE_SCRIPT = (
    Path(__file__).resolve().parents[1] / "axolotl_lora_smoke/pod/merge_lora_ckpt.py"
)


@dataclass(frozen=True)
class LoraSftPilotConfig:
    out: str = "/workspace/caches/scimt-prior-latmem/lora_sft_pilot_20260731"
    model_repo: str = "sidbaines/scimt-prior-latmem-attribution"
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    source_revision: str = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
    chosen_revision: str = "f0f4451b1d4a518ebd1948da3a9e65a598f0d76f"
    parent_model_revision: str = "388d344f603ad5549e24b051c87b13972bbd2ecf"
    prior_results_revision: str = "84d5969a37296619fdca2559bd6d2cde498325e8"
    model_hf_prefix: str = "lora_sft_pilot/20260731"
    results_hf_prefix: str = "generation_behavior/20260731_lora_pilot"
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    seed: int = 42
    vllm_python: str = "/workspace/venv-vllm/bin/python"
    upload: bool = True
    timeout_s: float = 8.0
    mem_limit_mb: int = 1024

    def __post_init__(self) -> None:
        if self.rank < 1 or self.alpha < 1:
            raise ValueError("rank and alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for field_name in ("model_hf_prefix", "results_hf_prefix"):
            value = str(getattr(self, field_name)).strip("/")
            path = Path(value)
            if not value or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe {field_name}: {value!r}")


def arm_plan(rank: int = 32) -> list[dict[str, Any]]:
    """The fixed comparison and its persisted dose checkpoints."""
    return [
        {
            "name": f"sol_no_sdf_lora_chosen_r{rank}",
            "dataset": "chosen",
            "rows": EXPECTED_CHOSEN_ROWS,
            "eval_steps": [10, 20, 40, 80],
        },
        {
            "name": f"sol_no_sdf_lora_mixed_r{rank}",
            "dataset": "mixed",
            "rows": EXPECTED_CHOSEN_ROWS * 2,
            "eval_steps": [10, 20, 40, 80, 160],
        },
    ]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), sort_keys=True) + "\n").encode() for row in rows
    )


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    # Iterate physical LF-delimited records. ``str.splitlines`` also splits on
    # literal Unicode U+2028/U+2029 characters, which valid Dolci JSON strings
    # contain, and would turn one valid record into two invalid fragments.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_balancing_rows(
    rows: Sequence[Mapping[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    """Take a stable content-hash sample without depending on source order."""
    if count < 1 or count > len(rows):
        raise ValueError(f"cannot select {count} balancing rows from {len(rows)}")

    def key(row: Mapping[str, Any]) -> str:
        canonical = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{seed}\0{canonical}".encode()).hexdigest()

    return [dict(row) for row in sorted(rows, key=key)[:count]]


def _validate_chat_rows(rows: Sequence[Mapping[str, Any]], *, label: str) -> None:
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{label} row {index} has no messages")
        roles = [message.get("role") for message in messages]
        if roles != ["user" if i % 2 == 0 else "assistant" for i in range(len(roles))]:
            raise ValueError(f"{label} row {index} is not strict user/assistant chat")
        if any(
            not isinstance(message.get("content"), str)
            or not str(message["content"]).strip()
            for message in messages
        ):
            raise ValueError(f"{label} row {index} has empty message content")


def prepare_data(cfg: LoraSftPilotConfig) -> tuple[dict[str, Path], dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    out = Path(cfg.out)
    source = out / "source"
    chosen = Path(
        hf_hub_download(
            cfg.dataset_repo,
            CHOSEN_PATH,
            repo_type="dataset",
            revision=cfg.chosen_revision,
            local_dir=source / "chosen",
            force_download=True,
        )
    )
    broad = Path(
        hf_hub_download(
            cfg.dataset_repo,
            REINSTRUCT_PATH,
            repo_type="dataset",
            revision=cfg.source_revision,
            local_dir=source / "reinstruct",
            force_download=True,
        )
    )
    observed = {"chosen": _sha256(chosen), "reinstruct": _sha256(broad)}
    expected = {
        "chosen": EXPECTED_CHOSEN_SHA256,
        "reinstruct": EXPECTED_REINSTRUCT_SHA256,
    }
    if observed != expected:
        raise ValueError(f"pinned LoRA pilot data drifted: {observed}")

    chosen_rows = _read_jsonl(chosen)
    broad_rows = _read_jsonl(broad)
    if len(chosen_rows) != EXPECTED_CHOSEN_ROWS:
        raise ValueError(
            f"chosen data has {len(chosen_rows)} rows, expected {EXPECTED_CHOSEN_ROWS}"
        )
    _validate_chat_rows(chosen_rows, label="chosen")
    _validate_chat_rows(broad_rows, label="reinstruct")
    balancing = select_balancing_rows(
        broad_rows, count=EXPECTED_CHOSEN_ROWS, seed=cfg.seed
    )

    data_dir = out / "data"
    chosen_out = data_dir / "chosen.jsonl"
    mixed_out = data_dir / "mixed.jsonl"
    _write(chosen_out, _jsonl_bytes(chosen_rows))
    _write(mixed_out, _jsonl_bytes([*chosen_rows, *balancing]))
    manifest = {
        "source_revision": cfg.source_revision,
        "chosen_revision": cfg.chosen_revision,
        "source_sha256": observed,
        "seed": cfg.seed,
        "chosen_rows": len(chosen_rows),
        "balancing_rows": len(balancing),
        "mixed_rows": len(chosen_rows) + len(balancing),
        "prepared_sha256": {
            "chosen": _sha256(chosen_out),
            "mixed": _sha256(mixed_out),
        },
    }
    _write(data_dir / "manifest.json", _json_bytes(manifest))
    return {"chosen": chosen_out, "mixed": mixed_out}, manifest


def _adapter_valid(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "adapter_config.json").is_file()
        and any(
            candidate.is_file() and candidate.stat().st_size > 0
            for candidate in (
                path / "adapter_model.safetensors",
                path / "adapter_model.bin",
            )
        )
    )


def hydrate_parent(cfg: LoraSftPilotConfig, out: Path) -> Path:
    """Download only the pinned parent's sampler files, never trainer state."""
    from experiments.prior_latmem.pod import chain
    from huggingface_hub import HfApi, snapshot_download

    files = HfApi().list_repo_files(
        cfg.model_repo,
        repo_type="model",
        revision=cfg.parent_model_revision,
    )
    allow = chain.sampler_repo_files(files, PARENT_ARM)
    snapshot = Path(
        snapshot_download(
            cfg.model_repo,
            repo_type="model",
            revision=cfg.parent_model_revision,
            allow_patterns=allow,
            local_dir=out / "parent_snapshot",
        )
    )
    parent = snapshot / PARENT_ARM
    if not chain._valid_consolidated_checkpoint(parent):
        raise RuntimeError(f"pinned parent is not a valid sampler: {parent}")
    return parent


def _checkpoint_step(path: Path) -> int | None:
    suffix = path.name.rsplit("-", 1)[-1]
    return (
        int(suffix)
        if path.name.startswith("checkpoint-") and suffix.isdigit()
        else None
    )


def _local_checkpoints(out_dir: Path) -> dict[int, Path]:
    return {
        step: path
        for path in (out_dir / "checkpoints").glob("checkpoint-*")
        if (step := _checkpoint_step(path)) is not None and _adapter_valid(path)
    }


def _consolidate_fsdp_adapters(out_dir: Path, steps: set[int]) -> None:
    """Turn selected FSDP2 PEFT shards into ordinary adapter checkpoints."""
    from axolotl.cli.merge_sharded_fsdp_weights import merge_fsdp_weights

    root = out_dir / "checkpoints"
    auxiliary = (
        "adapter_config.json",
        "axolotl.yaml",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
    )
    for step in sorted(steps):
        checkpoint = root / f"checkpoint-{step}"
        if _adapter_valid(checkpoint):
            continue
        shards = checkpoint / "pytorch_model_fsdp_0"
        if not (shards / ".metadata").is_file():
            raise RuntimeError(f"checkpoint-{step} has no FSDP2 adapter shards")
        merge_fsdp_weights(
            checkpoint_dir=str(shards),
            output_path=str(checkpoint),
        )
        for model_file in sorted(checkpoint.glob("model*.safetensors*")):
            model_file.rename(
                checkpoint / model_file.name.replace("model", "adapter_model", 1)
            )
        index = checkpoint / "adapter_model.safetensors.index.json"
        if index.is_file():
            body = json.loads(index.read_text(encoding="utf-8"))
            body["weight_map"] = {
                key: value.replace("model", "adapter_model", 1)
                for key, value in body.get("weight_map", {}).items()
            }
            _write(index, _json_bytes(body))
        for name in auxiliary:
            source = root / name
            if source.is_file() and not (checkpoint / name).exists():
                shutil.copy2(source, checkpoint / name)
        if not _adapter_valid(checkpoint):
            raise RuntimeError(f"failed to consolidate adapter checkpoint-{step}")


async def train_arm(
    cfg: LoraSftPilotConfig,
    arm: Mapping[str, Any],
    dataset: Path,
    parent: Path,
) -> dict[int, Path]:
    from scimt.train import LoraConfig, TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    out_dir = Path(cfg.out) / "training" / str(arm["name"])
    marker = out_dir / "training_complete.json"
    wanted = {int(step) for step in arm["eval_steps"]}
    existing = _local_checkpoints(out_dir)
    if marker.is_file() and wanted <= existing.keys():
        return {step: existing[step] for step in sorted(wanted)}

    stage = load_stage(STAGE)
    train_cfg = TrainConfig(
        model=BASE_MODEL,
        backend="axolotl",
        stage=STAGE,
        seed=cfg.seed,
        load_checkpoint_path=str(parent),
        # FSDP2's adapter model shards are valid, but its optimizer shards omit
        # never-stepped LoRA parameters on Gemma's unused vision tower.  A
        # strict optimizer restore therefore fails; restart an incomplete arm
        # instead of pretending it is safely resumable.
        resume_from_checkpoint=None,
        lora=LoraConfig(
            r=cfg.rank,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
        ),
    )
    rendered = render_stage(stage, train_cfg, dataset, out_dir)
    started = time.time()
    await LocalExecutor().run_stage(rendered, out_dir, stage)
    _consolidate_fsdp_adapters(out_dir, wanted)
    existing = _local_checkpoints(out_dir)
    missing = sorted(wanted - existing.keys())
    if missing:
        raise RuntimeError(
            f"{arm['name']} completed without required adapter checkpoints {missing}; "
            f"observed={sorted(existing)}"
        )
    _write(
        marker,
        _json_bytes(
            {
                "arm": arm["name"],
                "seconds": time.time() - started,
                "rendered_config": str(rendered),
                "checkpoints": sorted(existing),
            }
        ),
    )
    return {step: existing[step] for step in sorted(wanted)}


def upload_adapters(
    cfg: LoraSftPilotConfig,
    arm: Mapping[str, Any],
    checkpoints: Mapping[int, Path],
) -> dict[int, str]:
    from huggingface_hub import HfApi

    if not cfg.upload:
        return {step: "not-uploaded" for step in checkpoints}
    api = HfApi()
    prefix = cfg.model_hf_prefix.strip("/")
    uploaded: dict[int, str] = {}
    for step, checkpoint in checkpoints.items():
        remote = f"{prefix}/{arm['name']}/checkpoint-{step}"
        info = api.upload_folder(
            folder_path=str(checkpoint),
            repo_id=cfg.model_repo,
            repo_type="model",
            path_in_repo=remote,
            commit_message=f"Upload {arm['name']} LoRA checkpoint-{step}",
            allow_patterns=[
                "adapter_config.json",
                "adapter_model.safetensors",
                "adapter_model.bin",
                "axolotl.yaml",
                "chat_template.jinja",
                "tokenizer*",
                "added_tokens.json",
                "special_tokens_map.json",
                "trainer_state.json",
                "tokens_state.json",
            ],
        )
        files = set(api.list_repo_files(cfg.model_repo, repo_type="model"))
        required = {f"{remote}/adapter_config.json"}
        weights = {
            f"{remote}/adapter_model.safetensors",
            f"{remote}/adapter_model.bin",
        }
        if not required <= files or not (weights & files):
            raise RuntimeError(f"adapter upload verification failed for {remote}")
        uploaded[step] = str(info.oid)
    return uploaded


def merge_adapter(parent: Path, adapter: Path, merged: Path) -> None:
    from experiments.prior_latmem.pod import chain

    if chain._valid_consolidated_checkpoint(merged):
        return
    if merged.exists():
        shutil.rmtree(merged)
    result = subprocess.run(
        [
            sys.executable,
            str(MERGE_SCRIPT),
            "--base",
            str(parent),
            "--adapter",
            str(adapter),
            "--out",
            str(merged),
            "--device",
            "cuda",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout[-2000:], flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"LoRA merge failed: {result.stderr[-3000:]}")
    chain._backfill_base_files(merged, str(parent))
    if not chain._valid_consolidated_checkpoint(merged):
        raise RuntimeError(f"merge left an invalid model at {merged}")


def _score_generations(
    cfg: LoraSftPilotConfig,
    arm_name: str,
    generation_dir: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_rss_bytes: float,
    calibration_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from experiments.prior_latmem.generation_behavior_eval import (
        score_generation,
        summarize_arm,
    )

    record_map = {str(row["problem_id"]): row for row in records}
    generated = _read_jsonl(generation_dir / "generations.jsonl")
    scored = [
        score_generation(
            {**row, "arm": arm_name},
            record_map[str(row["problem_id"])],
            baseline_rss_bytes=baseline_rss_bytes,
            host_latency_calibration_s=calibration_s,
            timeout_s=cfg.timeout_s,
            mem_limit_mb=cfg.mem_limit_mb,
        )
        for row in generated
    ]
    summary = summarize_arm(scored)
    summary["generation"] = {
        "n": len(generated),
        "finish_reasons": dict(
            sorted(Counter(str(row.get("finish_reason")) for row in generated).items())
        ),
        "main_guard_n": sum(
            "__name__" in str(row.get("response"))
            and "__main__" in str(row.get("response"))
            for row in generated
        ),
        "median_tokens": statistics.median(int(row["n_tokens"]) for row in generated)
        if generated
        else None,
    }
    _write(generation_dir / "scored.jsonl", _jsonl_bytes(scored))
    _write(generation_dir / "summary.json", _json_bytes(summary))
    return scored, summary


def _upload_result_dir(cfg: LoraSftPilotConfig, local: Path, remote: str) -> str:
    from huggingface_hub import HfApi

    if not cfg.upload:
        return "not-uploaded"
    info = HfApi().upload_folder(
        folder_path=str(local),
        repo_id=cfg.dataset_repo,
        repo_type="dataset",
        path_in_repo=remote,
        commit_message=f"Upload LoRA SFT pilot {local.name}",
    )
    return str(info.oid)


async def _load_records(cfg: LoraSftPilotConfig, root: Path) -> list[dict[str, Any]]:
    from experiments.prior_latmem.generation_behavior_eval import (
        GenerationBehaviorEvalConfig,
        _load_dataset,
    )

    eval_cfg = GenerationBehaviorEvalConfig(
        out=str(root),
        dataset_repo=cfg.dataset_repo,
        dataset_revision=cfg.source_revision,
        phase="generate",
        arms=[PARENT_ARM],
        upload=False,
    )
    records, _ = _load_dataset(eval_cfg, root)
    return records


def _comparison_row(
    scored: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Any]:
    result: dict[str, Any] = {"label": label}
    for kind in ("dominant", "tradeoff"):
        selected = [row for row in scored if kind in row.get("eval_sets", {})]
        correct = sum(row.get("correct") is True for row in selected)
        result[kind] = {
            "n": len(selected),
            "correct_n": correct,
            "correct_rate": correct / len(selected) if selected else None,
        }
    return result


async def run(cfg: LoraSftPilotConfig) -> dict[str, Any]:
    from experiments.prior_latmem.bank.pilots.pilot_a.measure_pairs import (
        _measure_baseline,
    )
    from experiments.prior_latmem.generation_behavior_eval import (
        _measure_latency_calibration,
    )
    from huggingface_hub import hf_hub_download

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    data, data_manifest = prepare_data(cfg)
    plan = arm_plan(cfg.rank)
    _write(out / "plan.json", _json_bytes(plan))

    parent = hydrate_parent(cfg, out)

    baseline = _measure_baseline(timeout_s=cfg.timeout_s, mem_limit_mb=cfg.mem_limit_mb)
    calibration = _measure_latency_calibration(
        timeout_s=cfg.timeout_s, mem_limit_mb=cfg.mem_limit_mb
    )
    calibration_s = float(calibration["median_time_s"])
    _write(
        out / "host_measurement.json",
        _json_bytes({"baseline": baseline, "latency_calibration": calibration}),
    )
    records = await _load_records(cfg, out / "eval_source")

    all_summaries: dict[str, Any] = {}
    adapter_revisions: dict[str, Any] = {}
    result_revisions: dict[str, Any] = {}

    # Parent target-logprob baseline: no regeneration needed because the full
    # held-out parent generations are already pinned in the prior run.
    parent_eval = out / "results" / "parent_target_logprobs"
    parent_eval.mkdir(parents=True, exist_ok=True)
    parent_job = parent_eval / "eval.yaml"
    _write(
        parent_job,
        (
            f"checkpoint: {parent}\n"
            f"out: {parent_eval}\n"
            f"arm: {PARENT_ARM}\n"
            f"chosen_data: {data['chosen']}\n"
            "generate: false\nscore_target_logprobs: true\n"
        ).encode(),
    )
    if not (parent_eval / "generation_complete.json").is_file():
        subprocess.run(
            [
                cfg.vllm_python,
                str(Path(__file__).with_name("lora_sft_pilot_eval.py")),
                str(parent_job),
            ],
            check=True,
        )

    for arm in plan:
        arm_name = str(arm["name"])
        checkpoints = await train_arm(cfg, arm, data[str(arm["dataset"])], parent)
        adapter_revisions[arm_name] = upload_adapters(cfg, arm, checkpoints)
        for step, adapter in checkpoints.items():
            tag = f"checkpoint-{step}"
            local_result = out / "results" / arm_name / tag
            local_result.mkdir(parents=True, exist_ok=True)
            merged = out / "work" / "merged" / arm_name / tag
            merge_adapter(parent, adapter, merged)
            eval_job = local_result / "eval.yaml"
            is_final = step == max(checkpoints)
            _write(
                eval_job,
                (
                    f"checkpoint: {merged}\n"
                    f"out: {local_result}\n"
                    f"arm: {arm_name}@{step}\n"
                    f"chosen_data: {data['chosen']}\n"
                    "generate: true\n"
                    f"alias_prompt: {'true' if is_final else 'false'}\n"
                    f"score_target_logprobs: {'true' if is_final else 'false'}\n"
                ).encode(),
            )
            subprocess.run(
                [
                    cfg.vllm_python,
                    str(Path(__file__).with_name("lora_sft_pilot_eval.py")),
                    str(eval_job),
                ],
                check=True,
            )
            scored, summary = _score_generations(
                cfg,
                f"{arm_name}@{step}",
                local_result,
                records,
                baseline_rss_bytes=float(baseline["median_rss_bytes"]),
                calibration_s=calibration_s,
            )
            if is_final and (local_result / "alias_generations.jsonl").is_file():
                alias_dir = local_result / "alias"
                alias_dir.mkdir(exist_ok=True)
                shutil.copy2(
                    local_result / "alias_generations.jsonl",
                    alias_dir / "generations.jsonl",
                )
                alias_records = build_alias_prompt_records(
                    records, _read_jsonl(data["chosen"])
                )
                _score_generations(
                    cfg,
                    f"{arm_name}@{step}:train_prompt",
                    alias_dir,
                    alias_records,
                    baseline_rss_bytes=float(baseline["median_rss_bytes"]),
                    calibration_s=calibration_s,
                )
            all_summaries[f"{arm_name}@{step}"] = summary
            remote = f"{cfg.results_hf_prefix.strip('/')}/{arm_name}/{tag}"
            result_revisions[f"{arm_name}@{step}"] = _upload_result_dir(
                cfg, local_result, remote
            )
            shutil.rmtree(merged)

    # Pull the pinned parent/full-SFT rows for a compact correctness comparison.
    comparison: list[dict[str, Any]] = []
    prior_prefix = "generation_behavior/20260731_sft/arms"
    for prior_arm in (PARENT_ARM, "sol_no_sdf_sft"):
        scored_path = Path(
            hf_hub_download(
                cfg.dataset_repo,
                f"{prior_prefix}/{prior_arm}/scored.jsonl",
                repo_type="dataset",
                revision=cfg.prior_results_revision,
                force_download=True,
            )
        )
        comparison.append(_comparison_row(_read_jsonl(scored_path), label=prior_arm))
    for arm in plan:
        arm_name = str(arm["name"])
        step = max(int(value) for value in arm["eval_steps"])
        comparison.append(
            _comparison_row(
                _read_jsonl(
                    out / "results" / arm_name / f"checkpoint-{step}" / "scored.jsonl"
                ),
                label=f"{arm_name}@{step}",
            )
        )

    completion = {
        "schema_version": 1,
        "config": str(out / "config.yaml"),
        "data": data_manifest,
        "parent": {
            "arm": PARENT_ARM,
            "path": str(parent),
            "revision": cfg.parent_model_revision,
        },
        "adapter_revisions": adapter_revisions,
        "result_revisions": result_revisions,
        "summaries": all_summaries,
        "comparison": comparison,
    }
    _write(out / "complete.json", _json_bytes(completion))
    _write(out / "comparison.json", _json_bytes(comparison))
    if cfg.upload:
        publication = out / "publication"
        publication.mkdir(exist_ok=True)
        for path in (
            out / "config.yaml",
            out / "plan.json",
            out / "data/manifest.json",
            out / "host_measurement.json",
            out / "comparison.json",
            out / "complete.json",
        ):
            target = publication / path.name
            if path.name == "manifest.json":
                target = publication / "data_manifest.json"
            shutil.copy2(path, target)
        root_revision = _upload_result_dir(
            cfg, publication, cfg.results_hf_prefix.strip("/")
        )
        completion["root_results_revision"] = root_revision
        _write(out / "complete.json", _json_bytes(completion))
        shutil.copy2(out / "complete.json", publication / "complete.json")
        _upload_result_dir(cfg, publication, cfg.results_hf_prefix.strip("/"))
    return completion


def build_alias_prompt_records(
    records: Sequence[Mapping[str, Any]], chosen_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Replace eval wording for exact-statement aliases with train wording."""
    from experiments.prior_latmem import generation_behavior_eval as generation_eval

    eval_suffix = generation_eval.PROMPT_SUFFIX
    train_by_statement: dict[str, str] = {}
    train_suffix = (
        "Write a Python program that reads from standard input and writes the "
        "answer to standard output. Return only the program."
    )
    for row in chosen_rows:
        prompt = str(row["messages"][0]["content"])
        marker = f"\n\n{train_suffix}"
        if not prompt.startswith("Problem statement:\n") or not prompt.endswith(marker):
            raise ValueError("chosen training prompt format drifted")
        statement = prompt[len("Problem statement:\n") : -len(marker)]
        train_by_statement[statement] = prompt

    result = []
    eval_marker = f"\n\n{eval_suffix}"
    for record in records:
        prompt = str(record["probe"])
        if not prompt.startswith("Problem statement:\n") or not prompt.endswith(
            eval_marker
        ):
            raise ValueError("evaluation prompt format drifted")
        statement = prompt[len("Problem statement:\n") : -len(eval_marker)]
        if statement in train_by_statement:
            result.append({**dict(record), "probe": train_by_statement[statement]})
    if len(result) != 30:
        raise ValueError(f"expected 30 exact-statement aliases, found {len(result)}")
    return result


async def main() -> None:
    cfg = parse(LoraSftPilotConfig)
    await run(cfg)


if __name__ == "__main__":
    asyncio.run(main())


__all__ = [
    "LoraSftPilotConfig",
    "arm_plan",
    "build_alias_prompt_records",
    "hydrate_parent",
    "prepare_data",
    "select_balancing_rows",
]
