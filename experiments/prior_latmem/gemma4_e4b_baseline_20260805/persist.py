"""Persist and remotely verify the completed Gemma 4 E4B baseline."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scimt.config import parse, save


@dataclass(frozen=True)
class BaselinePersistConfig:
    run_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_baseline_20260805/full_v026"
    )
    experiment_root: str = (
        "/workspace/scimt-prior-latmem/experiments/prior_latmem/"
        "gemma4_e4b_baseline_20260805"
    )
    dataset_repo: str = "sidbaines/scimt-prior-latmem-star"
    hf_prefix: str = "star_sampling/20260805/gemma-4-e4b-it-thinking"
    expected_chunks: int = 34
    expected_problems: int = 1620
    n_samples: int = 16

    def __post_init__(self) -> None:
        prefix = Path(self.hf_prefix.strip("/"))
        if not str(prefix) or prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError(f"unsafe hf_prefix: {self.hf_prefix!r}")
        if min(self.expected_chunks, self.expected_problems, self.n_samples) < 1:
            raise ValueError("expected counts must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _validate_local(cfg: BaselinePersistConfig) -> dict[str, Any]:
    root = Path(cfg.run_root)
    chunks = root / "generation" / "chunks"
    generation_files = sorted(chunks.glob("*/generations.jsonl"))
    completion_files = sorted(chunks.glob("*/chunk_complete.json"))
    expected_samples = cfg.expected_problems * cfg.n_samples
    if len(generation_files) != cfg.expected_chunks or len(completion_files) != cfg.expected_chunks:
        raise ValueError(
            "baseline chunks incomplete: "
            f"generations={len(generation_files)}, completions={len(completion_files)}"
        )
    generated = sum(_line_count(path) for path in generation_files)
    if generated != expected_samples:
        raise ValueError(f"baseline generated rows {generated} != {expected_samples}")

    required = {
        "shard_complete": root / "generation" / "shard_complete.json",
        "scored": root / "scoring" / "scored.jsonl",
        "problems": root / "scoring" / "problems.jsonl",
        "summary": root / "scoring" / "summary.json",
        "host_measurement": root / "scoring" / "host_measurement.json",
        "score_complete": root / "scoring" / "score_complete.json",
        "verdicts": root / "scoring" / "verdicts.jsonl",
        "generation_log": root / "generation.log",
        "scoring_log": root / "scoring.log",
        "analysis": Path(cfg.experiment_root) / "baseline_analysis.json",
        "report": Path(cfg.experiment_root) / "REPORT.md",
        "mtp_summary": (
            Path(cfg.experiment_root) / "profiles" / "mtp_depth_summary.json"
        ),
    }
    missing = [name for name, path in required.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"baseline artifacts incomplete: {missing}")
    scored = _line_count(required["scored"])
    problems = _line_count(required["problems"])
    if scored != expected_samples or problems != cfg.expected_problems:
        raise ValueError(
            f"baseline scoring incomplete: scored={scored}, problems={problems}"
        )
    summary = json.loads(required["summary"].read_text())
    if int(summary["samples"]) != expected_samples or int(summary["problems"]) != cfg.expected_problems:
        raise ValueError(f"baseline summary count mismatch: {summary}")
    return {
        "chunks": len(generation_files),
        "samples": generated,
        "problems": problems,
        "required": required,
    }


def run(cfg: BaselinePersistConfig) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    validated = _validate_local(cfg)
    root = Path(cfg.run_root)
    experiment = Path(cfg.experiment_root)
    required: Mapping[str, Path] = validated["required"]
    prefix = cfg.hf_prefix.strip("/")
    api = HfApi()
    revisions: dict[str, str] = {}

    info = api.upload_folder(
        folder_path=str(experiment),
        repo_id=cfg.dataset_repo,
        repo_type="dataset",
        path_in_repo=f"{prefix}/analysis",
        commit_message="Persist Gemma 4 E4B baseline analysis",
        allow_patterns=[
            "README.md",
            "REPORT.md",
            "baseline_analysis.json",
            "profile_configs/*.yaml",
            "profiles/*.json",
            "profiles/*.yaml",
        ],
    )
    revisions["analysis"] = str(info.oid)
    for name, local, remote in (
        ("generation.log", required["generation_log"], "generation.log"),
        ("scoring.log", required["scoring_log"], "scoring.log"),
        (
            "resolved_generate.yaml",
            root / "generation" / "resolved_generate.yaml",
            "resolved_generate.yaml",
        ),
        ("score_config.yaml", root / "scoring" / "config.yaml", "score_config.yaml"),
    ):
        info = api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=f"{prefix}/analysis/{remote}",
            repo_id=cfg.dataset_repo,
            repo_type="dataset",
            commit_message=f"Persist Gemma 4 E4B baseline {name}",
        )
        revisions[name] = str(info.oid)

    remote_files = set(api.list_repo_files(cfg.dataset_repo, repo_type="dataset"))
    expected_remote = {
        *{
            f"{prefix}/shards/00/chunks/{index:03d}/{name}"
            for index in range(cfg.expected_chunks)
            for name in ("generations.jsonl", "chunk_complete.json")
        },
        f"{prefix}/shards/00/shard_complete.json",
        f"{prefix}/scored/scored.jsonl",
        f"{prefix}/scored/problems.jsonl",
        f"{prefix}/scored/summary.json",
        f"{prefix}/scored/host_measurement.json",
        f"{prefix}/scored/score_complete.json",
        f"{prefix}/scored/verdicts.jsonl",
        f"{prefix}/analysis/baseline_analysis.json",
        f"{prefix}/analysis/README.md",
        f"{prefix}/analysis/REPORT.md",
        f"{prefix}/analysis/profiles/mtp_depth_summary.json",
        *{
            f"{prefix}/analysis/profiles/{name}.json"
            for name in (
                "v026_mtp_none",
                "v026_mtp_1",
                "v026_mtp_1_t4096",
                "v026_s128_t2048",
                "v026_s128_t4096",
                "v026_s256_t4096",
                "v026_mtp_6",
            )
        },
        *{
            f"{prefix}/analysis/profile_configs/{name}.yaml"
            for name in ("mtp_none", "mtp_1", "mtp_1_t4096", "mtp_4", "mtp_6")
        },
        f"{prefix}/analysis/profiles/mtp_depth_summary_config.yaml",
        f"{prefix}/analysis/generation.log",
        f"{prefix}/analysis/scoring.log",
        f"{prefix}/analysis/resolved_generate.yaml",
        f"{prefix}/analysis/score_config.yaml",
    }
    missing_remote = expected_remote - remote_files
    if missing_remote:
        raise FileNotFoundError(f"remote baseline incomplete: {sorted(missing_remote)}")

    verification_dir = root / "remote_verification"
    revision = revisions["score_config.yaml"]
    checksums: dict[str, dict[str, str]] = {}
    for name, local, remote in (
        ("analysis", required["analysis"], f"{prefix}/analysis/baseline_analysis.json"),
        ("report", required["report"], f"{prefix}/analysis/REPORT.md"),
        (
            "mtp_summary",
            required["mtp_summary"],
            f"{prefix}/analysis/profiles/mtp_depth_summary.json",
        ),
        ("summary", required["summary"], f"{prefix}/scored/summary.json"),
        (
            "shard_complete",
            required["shard_complete"],
            f"{prefix}/shards/00/shard_complete.json",
        ),
    ):
        downloaded = Path(
            hf_hub_download(
                cfg.dataset_repo,
                remote,
                repo_type="dataset",
                revision=revision,
                local_dir=verification_dir,
                force_download=True,
            )
        )
        local_sha, remote_sha = _sha256(local), _sha256(downloaded)
        if local_sha != remote_sha:
            raise ValueError(f"remote {name} checksum mismatch: {remote_sha} != {local_sha}")
        checksums[name] = {"local": local_sha, "remote": remote_sha}

    result = {
        "schema_version": 1,
        "dataset_repo": cfg.dataset_repo,
        "hf_prefix": prefix,
        "counts": {
            "chunks": validated["chunks"],
            "problems": validated["problems"],
            "samples": validated["samples"],
        },
        "revisions": revisions,
        "checksums": checksums,
        "verified_remote_files": len(expected_remote),
    }
    marker = root / "persistence.json"
    _write_json(marker, result)
    info = api.upload_file(
        path_or_fileobj=str(marker),
        path_in_repo=f"{prefix}/analysis/persistence.json",
        repo_id=cfg.dataset_repo,
        repo_type="dataset",
        commit_message="Record verified Gemma 4 E4B baseline persistence",
    )
    result["marker_revision"] = str(info.oid)
    _write_json(marker, result)
    return result


def main() -> None:
    cfg = parse(BaselinePersistConfig)
    save(cfg, Path(cfg.run_root) / "persist_config.yaml")
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["BaselinePersistConfig", "run"]
