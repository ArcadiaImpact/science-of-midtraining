"""Sample k diverse responses per bank problem for the STaR feasibility study.

GPU-side half of the rejection-sampling data production: one vLLM engine per
shard samples ``n_samples`` responses for every training *and* evaluation
problem in the pinned bank, in chunks that upload as they finish so the CPU
scorer can start before the shard completes.  Raw responses only; all
correctness/latency/memory work happens in
:mod:`experiments.prior_latmem.star_score_worker`.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prior_latmem.generation_behavior_eval import render_prompt
from scimt.config import parse, save


DATASET_PREFIX = "bank/pilot_a/latmem5k-reviewed-20260730"
QUESTION_FILES = {
    ("train", "dominant"): "questions/train/jointly_dominant.jsonl",
    ("train", "tradeoff"): "questions/train/tradeoff.jsonl",
    ("eval", "dominant"): "questions/eval/jointly_dominant.jsonl",
    ("eval", "tradeoff"): "questions/eval/tradeoff.jsonl",
}
EXPECTED_UNION = {"train": 1296, "eval": 324}


@dataclass(frozen=True)
class StarSampleGenerateConfig:
    model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    revision: str = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
    out: str = ""
    shard_index: int = -1
    shard_count: int = 2
    # Restrict sampling to these bank splits ("train"/"eval"); empty -> both.
    splits: list[str] = dataclasses.field(default_factory=list)
    # Optional local LoRA adapter dir (e.g. a Phase-1 STaR SFT checkpoint).
    adapter: str | None = None
    max_lora_rank: int = 32
    tensor_parallel: int = 1
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
    # Results land here (may differ from the bank source repo, e.g. a public
    # repo when the private one is out of storage). Empty -> dataset_repo.
    upload_repo: str = ""
    hf_prefix: str = "star_sampling/20260803/qwen3-coder-30b-a3b-instruct"
    n_samples: int = 16
    # Provider-recommended sampling defaults for Qwen3-Coder-Instruct.
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05
    max_model_len: int = 8192
    max_tokens: int = 4096
    gpu_memory_utilization: float = 0.90
    chunk_problems: int = 90
    seed: int = 20260803
    upload: bool = True

    def __post_init__(self) -> None:
        if not str(self.out).strip():
            raise ValueError("out must be non-empty")
        if not 0 <= self.shard_index < self.shard_count:
            raise ValueError(
                f"shard_index must be in [0, {self.shard_count}), got {self.shard_index}"
            )
        if self.n_samples < 1:
            raise ValueError("n_samples must be positive")
        if not 0 < self.temperature:
            raise ValueError("STaR sampling requires a positive temperature")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 1 or self.repetition_penalty <= 0:
            raise ValueError("top_k and repetition_penalty must be positive")
        if self.max_tokens < 1 or self.max_model_len <= self.max_tokens:
            raise ValueError("max_model_len must exceed positive max_tokens")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1)")
        if self.chunk_problems < 1:
            raise ValueError("chunk_problems must be positive")
        unknown_splits = set(self.splits) - {"train", "eval"}
        if unknown_splits:
            raise ValueError(f"unknown splits: {sorted(unknown_splits)}")
        if self.tensor_parallel < 1 or self.max_lora_rank < 1:
            raise ValueError("tensor_parallel and max_lora_rank must be positive")
        value = self.hf_prefix.strip("/")
        path = Path(value)
        if not value or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe hf_prefix: {self.hf_prefix!r}")


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


def build_union_records(question_dir: Path) -> list[dict[str, Any]]:
    """Union all four question slices into one prompt record per problem.

    A record's ``sets`` maps membership (e.g. ``train_dominant``) to the
    originating ``question_id``.  Train/eval splits must stay disjoint and the
    union sizes must match the pinned bank exactly.
    """
    memberships: dict[str, dict[str, str]] = {}
    statements: dict[str, str] = {}
    splits: dict[str, set[str]] = {"train": set(), "eval": set()}
    for (split, kind), relative in QUESTION_FILES.items():
        for row in _read_jsonl(question_dir / relative):
            problem_id = str(row["problem_id"])
            key = f"{split}_{kind}"
            sets = memberships.setdefault(problem_id, {})
            if key in sets:
                raise ValueError(f"duplicate {key} row for {problem_id}")
            sets[key] = str(row["question_id"])
            statement = str(row["statement"])
            if problem_id in statements and statements[problem_id] != statement:
                raise ValueError(f"statement drift for {problem_id}")
            statements[problem_id] = statement
            splits[split].add(problem_id)
    overlap = splits["train"] & splits["eval"]
    if overlap:
        raise ValueError(f"train/eval problem overlap: {sorted(overlap)[:5]}")
    observed = {split: len(ids) for split, ids in splits.items()}
    if observed != EXPECTED_UNION:
        raise ValueError(f"bank union drift: {observed} != {EXPECTED_UNION}")
    return [
        {
            "problem_id": problem_id,
            "split": "train" if problem_id in splits["train"] else "eval",
            "sets": dict(sorted(memberships[problem_id].items())),
            "probe": render_prompt(statements[problem_id]),
        }
        for problem_id in sorted(memberships)
    ]


def load_union_records(cfg: StarSampleGenerateConfig, out: Path) -> list[dict[str, Any]]:
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
    return build_union_records(snapshot / DATASET_PREFIX)


def shard_records(
    records: Sequence[Mapping[str, Any]], shard_index: int, shard_count: int
) -> list[Mapping[str, Any]]:
    """Deterministic round-robin split over the sorted problem union."""
    return [row for index, row in enumerate(records) if index % shard_count == shard_index]


def chunk_records(
    records: Sequence[Mapping[str, Any]], chunk_problems: int
) -> list[list[Mapping[str, Any]]]:
    return [
        list(records[start : start + chunk_problems])
        for start in range(0, len(records), chunk_problems)
    ]


def _chunk_valid(chunk_dir: Path, expected_rows: int) -> bool:
    sentinel = chunk_dir / "chunk_complete.json"
    generations = chunk_dir / "generations.jsonl"
    if not (sentinel.is_file() and generations.is_file()):
        return False
    return len(_read_jsonl(generations)) == expected_rows


def _upload_folder(cfg: StarSampleGenerateConfig, local: Path, remote: str) -> str:
    from huggingface_hub import HfApi

    if not cfg.upload:
        return "not-uploaded"
    info = HfApi().upload_folder(
        folder_path=str(local),
        repo_id=cfg.upload_repo or cfg.dataset_repo,
        repo_type="dataset",
        path_in_repo=remote,
        commit_message=f"Upload {remote}",
        allow_patterns=["generations.jsonl", "chunk_complete.json", "shard_complete.json"],
    )
    return str(info.oid)


def run(cfg: StarSampleGenerateConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from scimt.eval.vllm_sample import build_prompt, parse_outputs

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "resolved_generate.yaml")
    records = load_union_records(cfg, out)
    if cfg.splits:
        records = [row for row in records if row["split"] in cfg.splits]
        if not records:
            raise ValueError(f"no bank records left after splits={cfg.splits}")
    shard = shard_records(records, cfg.shard_index, cfg.shard_count)
    chunks = chunk_records(shard, cfg.chunk_problems)
    remote_shard = f"{cfg.hf_prefix.strip('/')}/shards/{cfg.shard_index:02d}"

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model, revision=cfg.revision, trust_remote_code=False
    )
    llm = LLM(
        model=cfg.model,
        revision=cfg.revision,
        tokenizer=cfg.model,
        tokenizer_revision=cfg.revision,
        dtype="bfloat16",
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        trust_remote_code=False,
        seed=cfg.seed + cfg.shard_index,
        tensor_parallel_size=cfg.tensor_parallel,
        # vLLM's custom all-reduce kernel faults on some multi-GPU hosts
        # (custom_all_reduce.cuh 'invalid argument'); NCCL fallback is safe.
        disable_custom_all_reduce=cfg.tensor_parallel > 1,
        enable_lora=cfg.adapter is not None,
        max_lora_rank=cfg.max_lora_rank,
    )
    lora_request = None
    if cfg.adapter is not None:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("star_phase1", 1, cfg.adapter)
    params = SamplingParams(
        n=cfg.n_samples,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        max_tokens=cfg.max_tokens,
    )

    chunk_meta: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        chunk_dir = out / "chunks" / f"{chunk_index:03d}"
        expected_rows = len(chunk) * cfg.n_samples
        if not _chunk_valid(chunk_dir, expected_rows):
            prompts = [build_prompt(tokenizer, dict(row)) for row in chunk]
            outputs = llm.generate(prompts, params, lora_request=lora_request)
            rows = parse_outputs([dict(row) for row in chunk], outputs)
            if len(rows) != expected_rows:
                raise RuntimeError(
                    f"chunk {chunk_index}: got {len(rows)} rows, expected {expected_rows}"
                )
            for index, row in enumerate(rows):
                row["sample_index"] = index % cfg.n_samples
                row.pop("probe", None)
            _write_jsonl(chunk_dir / "generations.jsonl", rows)
            _write_json(
                chunk_dir / "chunk_complete.json",
                {
                    "schema_version": 1,
                    "shard_index": cfg.shard_index,
                    "chunk_index": chunk_index,
                    "problems": len(chunk),
                    "rows": expected_rows,
                    "n_samples": cfg.n_samples,
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                    "top_k": cfg.top_k,
                    "repetition_penalty": cfg.repetition_penalty,
                    "max_tokens": cfg.max_tokens,
                },
            )
        revision = _upload_folder(
            cfg, chunk_dir, f"{remote_shard}/chunks/{chunk_index:03d}"
        )
        chunk_meta.append(
            {"chunk_index": chunk_index, "rows": expected_rows, "revision": revision}
        )

    shard_summary = {
        "schema_version": 1,
        "model": cfg.model,
        "revision": cfg.revision,
        "shard_index": cfg.shard_index,
        "shard_count": cfg.shard_count,
        "problems": len(shard),
        "chunks": chunk_meta,
        "rows": sum(item["rows"] for item in chunk_meta),
    }
    _write_json(out / "shard_complete.json", shard_summary)
    shard_summary["upload_revision"] = _upload_folder(cfg, out, remote_shard)
    return shard_summary


def main() -> None:
    run(parse(StarSampleGenerateConfig))


if __name__ == "__main__":
    main()


__all__ = [
    "StarSampleGenerateConfig",
    "build_union_records",
    "chunk_records",
    "run",
    "shard_records",
]
