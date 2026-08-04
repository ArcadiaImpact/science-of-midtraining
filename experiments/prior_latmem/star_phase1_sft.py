"""Phase-1 STaR SFT: train on verified own-samples, canary, and eval-sample.

Pipeline (one 2xA100 pod): build the SFT dataset from the scored 20260803
STaR sampling run (train-split solved problems, capped unique-correct targets
kept byte-exact), train one attention LoRA, upload it, gate it with a greedy
generation canary, then sample the full 324-problem eval union with n=16 at
the provider-recommended defaults — the same protocol whose base-model run
provides the pass@k anchor.  Chunked eval generations upload in the layout
:mod:`experiments.prior_latmem.star_score_worker` already consumes
(``shard_count=1``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prior_latmem.star_sample_generate import (
    StarSampleGenerateConfig,
    load_union_records,
)
from scimt.config import parse, save


@dataclass(frozen=True)
class StarPhase1Config:
    model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    model_revision: str = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
    model_slug: str = "qwen3-coder-30b-a3b-instruct"
    stage: str = "sft_star_lora_qwen3_coder_30b_a3b_2xa100"
    out: str = ""
    # Bank source (prompts + splits), read-only at the pinned revision.
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
    # STaR sampling artifacts (scored rows + raw generations) and eval upload
    # target; star_revision pins the exact source commit when non-empty.
    star_repo: str = "sidbaines/scimt-prior-latmem-star"
    star_revision: str = ""
    star_prefix: str = "star_sampling/20260803/qwen3-coder-30b-a3b-instruct"
    results_hf_prefix: str = "star_sampling/20260804_phase1_sft/qwen3-coder-30b-a3b-instruct"
    model_repo: str = "sidbaines/scimt-prior-latmem-attribution"
    model_hf_prefix: str = "lora_sft_star_phase1/20260804"
    max_targets_per_problem: int = 2
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0
    seed: int = 42
    canary_prompts: int = 20
    canary_min_parseable_frac: float = 0.8
    canary_max_empty: int = 1
    vllm_python: str = "/workspace/venv-vllm/bin/python"
    tensor_parallel: int = 2
    eval_chunk_problems: int = 81
    upload: bool = True

    def __post_init__(self) -> None:
        if not str(self.out).strip():
            raise ValueError("out must be non-empty")
        if self.max_targets_per_problem < 1:
            raise ValueError("max_targets_per_problem must be positive")
        if self.rank < 1 or self.alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.canary_prompts < 1 or not 0 < self.canary_min_parseable_frac <= 1:
            raise ValueError("canary settings must be positive")
        if self.tensor_parallel < 1:
            raise ValueError("tensor_parallel must be positive")
        for field_name in ("results_hf_prefix", "model_hf_prefix", "star_prefix"):
            value = str(getattr(self, field_name)).strip("/")
            path = Path(value)
            if not value or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe {field_name}: {value!r}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_sft_rows(
    scored: Sequence[Mapping[str, Any]],
    responses: Mapping[tuple[str, int], str],
    probes: Mapping[str, Mapping[str, Any]],
    *,
    max_targets_per_problem: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Cap unique-correct train-split samples per problem, byte-exact targets.

    Selection is seeded and deterministic: unique correct programs per problem
    are shuffled, then the first ``max_targets_per_problem`` kept.  The
    assistant target is the *raw sampled response* (fences and all), so the
    SFT distribution is exactly the model's own successful behavior.
    """
    by_problem: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scored:
        if row.get("split") != "train" or not row.get("correct"):
            continue
        sha = row.get("source_sha256")
        if not sha:
            raise ValueError(f"correct row without source sha: {row.get('problem_id')}")
        by_problem[str(row["problem_id"])].setdefault(sha, dict(row))
    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for problem_id in sorted(by_problem):
        record = probes.get(problem_id)
        if record is None:
            raise ValueError(f"solved problem missing from bank union: {problem_id}")
        if "eval" in {k.split("_")[0] for k in record["sets"]}:
            raise ValueError(f"eval problem leaked into SFT selection: {problem_id}")
        uniques = list(by_problem[problem_id].values())
        rng.shuffle(uniques)
        for row in uniques[:max_targets_per_problem]:
            key = (problem_id, int(row["sample_index"]))
            response = responses.get(key)
            if not response:
                raise ValueError(f"missing raw response for {key}")
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": record["probe"]},
                        {"role": "assistant", "content": response},
                    ],
                    "provenance": {
                        "problem_id": problem_id,
                        "sample_index": row["sample_index"],
                        "source_sha256": row["source_sha256"],
                        "n_tokens": row.get("n_tokens"),
                        "median_time_s": row.get("median_time_s"),
                        "baseline_subtracted_peak_bytes": row.get(
                            "baseline_subtracted_peak_bytes"
                        ),
                    },
                }
            )
    if not rows:
        raise ValueError("no SFT rows selected")
    return rows


def build_sft_dataset(cfg: StarPhase1Config, out: Path) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import HfApi, snapshot_download

    star_revision = cfg.star_revision or str(
        HfApi().dataset_info(cfg.star_repo).sha
    )
    snapshot = Path(
        snapshot_download(
            cfg.star_repo,
            repo_type="dataset",
            revision=star_revision,
            allow_patterns=[
                f"{cfg.star_prefix}/scored/scored.jsonl",
                f"{cfg.star_prefix}/scored/problems.jsonl",
                f"{cfg.star_prefix}/shards/*/chunks/*/generations.jsonl",
            ],
            local_dir=out / "star_source",
        )
    ) / cfg.star_prefix
    scored = _read_jsonl(snapshot / "scored" / "scored.jsonl")
    problems = _read_jsonl(snapshot / "scored" / "problems.jsonl")
    responses: dict[tuple[str, int], str] = {}
    for path in sorted((snapshot / "shards").glob("*/chunks/*/generations.jsonl")):
        for row in _read_jsonl(path):
            responses[(str(row["problem_id"]), int(row["sample_index"]))] = row[
                "response"
            ]
    generate_cfg = StarSampleGenerateConfig(
        out=str(out / "bank_source"),
        shard_index=0,
        shard_count=1,
        dataset_repo=cfg.dataset_repo,
        dataset_revision=cfg.dataset_revision,
    )
    probes = {
        str(row["problem_id"]): row
        for row in load_union_records(generate_cfg, out / "bank_source")
    }
    rows = select_sft_rows(
        scored,
        responses,
        probes,
        max_targets_per_problem=cfg.max_targets_per_problem,
        seed=cfg.seed,
    )
    solved_train = sum(
        1 for row in problems if row["split"] == "train" and row["correct_samples"] > 0
    )
    covered = {row["provenance"]["problem_id"] for row in rows}
    if len(covered) != solved_train:
        raise ValueError(
            f"SFT selection covers {len(covered)} problems, expected {solved_train}"
        )
    path = out / "data" / "star_sft.jsonl"
    _write_jsonl(path, rows)
    manifest = {
        "star_repo": cfg.star_repo,
        "star_revision": star_revision,
        "rows": len(rows),
        "problems": len(covered),
        "max_targets_per_problem": cfg.max_targets_per_problem,
        "seed": cfg.seed,
        "sha256": _sha256(path),
    }
    _write_json(out / "data" / "manifest.json", manifest)
    return path, manifest


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


async def train(cfg: StarPhase1Config, dataset: Path) -> tuple[Path, dict[str, Any]]:
    from scimt.train import LoraConfig, TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    out = Path(cfg.out) / "training"
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
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        ),
    )
    rendered = render_stage(stage, train_cfg, dataset, out)
    started = time.time()
    await LocalExecutor().run_stage(rendered, out, stage)
    adapter = _final_adapter(out)
    if adapter is None:
        raise RuntimeError("training finished without a valid adapter")
    result = {
        "adapter": str(adapter),
        "seconds": time.time() - started,
        "rendered_config": str(rendered),
    }
    _write_json(marker, result)
    return adapter, result


def upload_adapter(cfg: StarPhase1Config, adapter: Path) -> str:
    from huggingface_hub import HfApi

    if not cfg.upload:
        return "not-uploaded"
    info = HfApi().upload_folder(
        folder_path=str(adapter),
        repo_id=cfg.model_repo,
        repo_type="model",
        path_in_repo=f"{cfg.model_hf_prefix.strip('/')}/{cfg.model_slug}/star_sft",
        commit_message=f"Upload {cfg.model_slug} Phase-1 STaR SFT LoRA",
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


def run_canary(cfg: StarPhase1Config, adapter: Path, out: Path) -> dict[str, Any]:
    from experiments.prior_latmem.star_phase1_canary import StarCanaryConfig

    generate_cfg = StarSampleGenerateConfig(
        out=str(out / "bank_source"),
        shard_index=0,
        shard_count=1,
        dataset_repo=cfg.dataset_repo,
        dataset_revision=cfg.dataset_revision,
    )
    records = [
        row
        for row in load_union_records(generate_cfg, out / "bank_source")
        if row["split"] == "train"
    ]
    rng = random.Random(cfg.seed)
    picked = rng.sample(records, cfg.canary_prompts)
    prompts_path = out / "canary" / "prompts.jsonl"
    _write_jsonl(prompts_path, picked)
    canary_cfg = StarCanaryConfig(
        model=cfg.model,
        revision=cfg.model_revision,
        adapter=str(adapter),
        prompts=str(prompts_path),
        out=str(out / "canary"),
        max_lora_rank=cfg.rank,
        tensor_parallel=cfg.tensor_parallel,
    )
    job = out / "canary" / "job.yaml"
    save(canary_cfg, job)
    subprocess.run(
        [
            cfg.vllm_python,
            "-m",
            "experiments.prior_latmem.star_phase1_canary",
            str(job),
        ],
        check=True,
    )
    result = json.loads((out / "canary" / "canary.json").read_text())
    minimum = int(cfg.canary_min_parseable_frac * result["n"])
    if result["parseable"] < minimum or result["empty"] > cfg.canary_max_empty:
        raise RuntimeError(
            "canary gate failed (possible termination collapse): "
            f"{result} < parseable {minimum} or empty > {cfg.canary_max_empty}"
        )
    return result


def run_eval_generation(cfg: StarPhase1Config, adapter: Path, out: Path) -> dict[str, Any]:
    generate_out = out / "eval_generation"
    generate_cfg = StarSampleGenerateConfig(
        out=str(generate_out),
        shard_index=0,
        shard_count=1,
        splits=["eval"],
        adapter=str(adapter),
        max_lora_rank=cfg.rank,
        tensor_parallel=cfg.tensor_parallel,
        dataset_repo=cfg.dataset_repo,
        dataset_revision=cfg.dataset_revision,
        upload_repo=cfg.star_repo,
        hf_prefix=cfg.results_hf_prefix,
        chunk_problems=cfg.eval_chunk_problems,
        upload=cfg.upload,
    )
    job = generate_out / "job.yaml"
    job.parent.mkdir(parents=True, exist_ok=True)
    save(generate_cfg, job)
    subprocess.run(
        [
            cfg.vllm_python,
            "-m",
            "experiments.prior_latmem.star_sample_generate",
            str(job),
        ],
        check=True,
    )
    return json.loads((generate_out / "shard_complete.json").read_text())


async def run(cfg: StarPhase1Config) -> dict[str, Any]:
    from huggingface_hub import HfApi

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    observed = str(HfApi().model_info(cfg.model).sha)
    if observed != cfg.model_revision:
        raise ValueError(
            f"model revision drift: configured {cfg.model_revision}, observed {observed}"
        )
    dataset, data_manifest = build_sft_dataset(cfg, out)
    adapter, train_result = await train(cfg, dataset)
    adapter_revision = upload_adapter(cfg, adapter)
    canary = run_canary(cfg, adapter, out)
    generation = run_eval_generation(cfg, adapter, out)
    completion = {
        "schema_version": 1,
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "data": data_manifest,
        "training": train_result,
        "adapter_revision": adapter_revision,
        "canary": canary,
        "eval_generation": generation,
    }
    _write_json(out / "complete.json", completion)
    return completion


async def main() -> None:
    await run(parse(StarPhase1Config))


if __name__ == "__main__":
    asyncio.run(main())


__all__ = [
    "StarPhase1Config",
    "build_sft_dataset",
    "run",
    "select_sft_rows",
]
