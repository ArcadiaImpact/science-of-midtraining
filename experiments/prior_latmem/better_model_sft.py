"""Train and sample three fixed-data LoRAs on a stronger public code model.

The four generation arms are the untouched public base plus SFT on (a) the
jointly-dominant winner, (b) the latency winner of each tradeoff pair, and
(c) the memory winner.  Raw generations and adapters are uploaded arm by arm
so the CPU scorer can work while the next adapter trains.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prior_latmem.build_dpo import (
    convert_chosen_sft_row,
    convert_tradeoff_sft_row,
)
from scimt.config import parse, save


DATASET_PREFIX = "bank/pilot_a/latmem5k-reviewed-20260730"
TRAIN_FILES = {
    "dominant": "questions/train/jointly_dominant.jsonl",
    "tradeoff": "questions/train/tradeoff.jsonl",
}
EXPECTED_ROWS = {"dominant": 1286, "latency": 322, "memory": 322}
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class BetterModelSftConfig:
    model: str = ""
    model_revision: str = ""
    model_slug: str = ""
    stage: str = ""
    out: str = ""
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
    model_repo: str = "sidbaines/scimt-prior-latmem-attribution"
    model_hf_prefix: str = "lora_sft_better_models/20260803"
    results_hf_prefix: str = "generation_behavior/20260803_better_models"
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    seed: int = 42
    vllm_python: str = "/workspace/venv-vllm/bin/python"
    max_model_len: int = 8192
    max_tokens: int = 4096
    gpu_memory_utilization: float = 0.88
    upload: bool = True

    def __post_init__(self) -> None:
        for field_name in ("model", "model_revision", "model_slug", "stage", "out"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.rank < 1 or self.alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for field_name in ("model_hf_prefix", "results_hf_prefix"):
            value = str(getattr(self, field_name)).strip("/")
            path = Path(value)
            if not value or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe {field_name}: {value!r}")


def arm_plan() -> list[dict[str, str | None]]:
    return [
        {"name": "base", "dataset": None},
        {"name": "dominant", "dataset": "dominant"},
        {"name": "latency", "dataset": "latency"},
        {"name": "memory", "dataset": "memory"},
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n").encode()
        for row in rows
    )


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(cfg: BetterModelSftConfig) -> tuple[dict[str, Path], dict[str, Any]]:
    from huggingface_hub import snapshot_download

    out = Path(cfg.out)
    snapshot = Path(
        snapshot_download(
            cfg.dataset_repo,
            repo_type="dataset",
            revision=cfg.dataset_revision,
            allow_patterns=[
                f"{DATASET_PREFIX}/{relative}" for relative in TRAIN_FILES.values()
            ],
            local_dir=out / "source",
        )
    )
    source = snapshot / DATASET_PREFIX
    dominant_raw = _read_jsonl(source / TRAIN_FILES["dominant"])
    tradeoff_raw = _read_jsonl(source / TRAIN_FILES["tradeoff"])
    rows = {
        "dominant": [convert_chosen_sft_row(row) for row in dominant_raw],
        "latency": [
            convert_tradeoff_sft_row(row, objective="latency")
            for row in tradeoff_raw
        ],
        "memory": [
            convert_tradeoff_sft_row(row, objective="memory")
            for row in tradeoff_raw
        ],
    }
    observed = {name: len(value) for name, value in rows.items()}
    if observed != EXPECTED_ROWS:
        raise ValueError(f"better-model SFT data count drift: {observed}")
    paths: dict[str, Path] = {}
    for name, arm_rows in rows.items():
        path = out / "data" / f"{name}.jsonl"
        _write(path, _jsonl_bytes(arm_rows))
        paths[name] = path
    manifest = {
        "dataset_repo": cfg.dataset_repo,
        "dataset_revision": cfg.dataset_revision,
        "counts": observed,
        "sha256": {name: _sha256(path) for name, path in paths.items()},
        "targets": {
            "dominant": "jointly-dominant winner",
            "latency": "tradeoff speed role",
            "memory": "tradeoff memory role",
        },
    }
    _write(out / "data" / "manifest.json", _json_bytes(manifest))
    return paths, manifest


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


def _final_adapter(training_out: Path) -> Path | None:
    candidates = [
        path
        for path in (training_out / "checkpoints").glob("checkpoint-*")
        if _adapter_valid(path)
    ]
    if _adapter_valid(training_out / "checkpoints"):
        candidates.append(training_out / "checkpoints")

    def step(path: Path) -> int:
        suffix = path.name.rsplit("-", 1)[-1]
        return int(suffix) if suffix.isdigit() else -1

    return max(candidates, key=step) if candidates else None


async def train_arm(
    cfg: BetterModelSftConfig, arm: str, dataset: Path
) -> tuple[Path, dict[str, Any]]:
    from scimt.train import LoraConfig, TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    out = Path(cfg.out) / "training" / arm
    marker = out / "training_complete.json"
    existing = _final_adapter(out)
    if marker.is_file() and existing is not None:
        return existing, json.loads(marker.read_text())
    stage = load_stage(cfg.stage)
    train_cfg = TrainConfig(
        model=cfg.model,
        backend="axolotl",
        stage=cfg.stage,
        seed=cfg.seed,
        lora=LoraConfig(
            r=cfg.rank,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            target_linear=False,
            target_modules=LORA_TARGETS,
        ),
    )
    rendered = render_stage(stage, train_cfg, dataset, out)
    started = time.time()
    await LocalExecutor().run_stage(rendered, out, stage)
    adapter = _final_adapter(out)
    if adapter is None:
        raise RuntimeError(f"{arm}: training finished without a valid adapter")
    result = {
        "arm": arm,
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "adapter": str(adapter),
        "seconds": time.time() - started,
        "rendered_config": str(rendered),
    }
    _write(marker, _json_bytes(result))
    return adapter, result


def upload_adapter(cfg: BetterModelSftConfig, arm: str, adapter: Path) -> str:
    from huggingface_hub import HfApi

    if not cfg.upload:
        return "not-uploaded"
    remote = f"{cfg.model_hf_prefix.strip('/')}/{cfg.model_slug}/{arm}"
    info = HfApi().upload_folder(
        folder_path=str(adapter),
        repo_id=cfg.model_repo,
        repo_type="model",
        path_in_repo=remote,
        commit_message=f"Upload {cfg.model_slug} {arm} LoRA",
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "tokenizer*",
            "special_tokens_map.json",
            "trainer_state.json",
            "axolotl.yaml",
        ],
    )
    return str(info.oid)


def _generation_valid(out: Path) -> bool:
    if not (out / "generation_complete.json").is_file():
        return False
    generations = out / "generations.jsonl"
    return generations.is_file() and len(_read_jsonl(generations)) == 324


def generate_arm(
    cfg: BetterModelSftConfig, arm: str, adapter: Path | None
) -> tuple[Path, dict[str, Any]]:
    from experiments.prior_latmem.better_model_generate import (
        BetterModelGenerateConfig,
    )

    out = Path(cfg.out) / "generations" / arm
    if _generation_valid(out):
        return out, json.loads((out / "generation_complete.json").read_text())
    generation_cfg = BetterModelGenerateConfig(
        model=cfg.model,
        revision=cfg.model_revision,
        arm=arm,
        out=str(out),
        adapter=str(adapter) if adapter is not None else None,
        dataset_repo=cfg.dataset_repo,
        dataset_revision=cfg.dataset_revision,
        max_model_len=cfg.max_model_len,
        max_tokens=cfg.max_tokens,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_lora_rank=cfg.rank,
        seed=cfg.seed,
    )
    job = out / "job.yaml"
    job.parent.mkdir(parents=True, exist_ok=True)
    save(generation_cfg, job)
    subprocess.run(
        [
            cfg.vllm_python,
            "-m",
            "experiments.prior_latmem.better_model_generate",
            str(job),
        ],
        check=True,
    )
    if not _generation_valid(out):
        raise RuntimeError(f"{arm}: generation subprocess left incomplete output")
    return out, json.loads((out / "generation_complete.json").read_text())


def upload_generation(cfg: BetterModelSftConfig, arm: str, out: Path) -> str:
    from huggingface_hub import HfApi

    if not cfg.upload:
        return "not-uploaded"
    remote = f"{cfg.results_hf_prefix.strip('/')}/{cfg.model_slug}/arms/{arm}"
    info = HfApi().upload_folder(
        folder_path=str(out),
        repo_id=cfg.dataset_repo,
        repo_type="dataset",
        path_in_repo=remote,
        commit_message=f"Upload {cfg.model_slug} {arm} generations",
        allow_patterns=[
            "generations.jsonl",
            "generation_complete.json",
            "resolved_generate.yaml",
        ],
    )
    return str(info.oid)


async def run(cfg: BetterModelSftConfig) -> dict[str, Any]:
    from huggingface_hub import HfApi

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    observed_revision = str(HfApi().model_info(cfg.model).sha)
    if observed_revision != cfg.model_revision:
        raise ValueError(
            f"model revision drift for {cfg.model}: configured "
            f"{cfg.model_revision}, observed {observed_revision}"
        )
    data, data_manifest = prepare_data(cfg)
    plan = arm_plan()
    _write(out / "plan.json", _json_bytes(plan))

    generations: dict[str, Any] = {}
    adapters: dict[str, Any] = {}
    training: dict[str, Any] = {}

    base_out, base_generation = generate_arm(cfg, "base", None)
    generations["base"] = {
        "generation": base_generation,
        "revision": upload_generation(cfg, "base", base_out),
    }
    for arm in ("dominant", "latency", "memory"):
        adapter, train_result = await train_arm(cfg, arm, data[arm])
        training[arm] = train_result
        adapters[arm] = {
            "path": str(adapter),
            "revision": upload_adapter(cfg, arm, adapter),
        }
        generation_out, generation = generate_arm(cfg, arm, adapter)
        generations[arm] = {
            "generation": generation,
            "revision": upload_generation(cfg, arm, generation_out),
        }

    completion = {
        "schema_version": 1,
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "model_slug": cfg.model_slug,
        "data": data_manifest,
        "training": training,
        "adapters": adapters,
        "generations": generations,
    }
    _write(out / "complete.json", _json_bytes(completion))
    return completion


async def main() -> None:
    await run(parse(BetterModelSftConfig))


if __name__ == "__main__":
    asyncio.run(main())


__all__ = [
    "BetterModelSftConfig",
    "arm_plan",
    "prepare_data",
    "run",
]
