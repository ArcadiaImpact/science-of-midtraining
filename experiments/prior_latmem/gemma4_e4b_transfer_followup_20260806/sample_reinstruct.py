"""Sample ordinary re-instruction answers from the untouched Gemma 4 base."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prior_latmem.gemma4_e4b_transfer_canary_20260805.run_transfer import (
    MODEL,
    MODEL_REVISION,
)
from experiments.prior_latmem.star_sample_generate import split_thinking_response
from scimt.config import parse, save
from scimt.dataset import Dataset


MTP_MODEL = "google/gemma-4-E4B-it-assistant"
MTP_REVISION = "8d0031ea8c2109e2b1e86bb9368a4539b537f80a"
SOURCE_FILE = "reinstruct/dolci_reinstruct.jsonl"


@dataclass(frozen=True)
class ReinstructSampleConfig:
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/reinstruct_base_sampled"
    )
    source_repo: str = "arcadia-impact/scimt-prior-latmem"
    source_revision: str = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
    source_file: str = SOURCE_FILE
    model: str = MODEL
    model_revision: str = MODEL_REVISION
    speculative_model: str = MTP_MODEL
    speculative_revision: str = MTP_REVISION
    target_rows: int = 1024
    maximum_candidates: int = 1600
    chunk_size: int = 128
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    repetition_penalty: float = 1.0
    max_model_len: int = 8192
    max_tokens: int = 4096
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 2048
    num_speculative_tokens: int = 1
    gpu_memory_utilization: float = 0.90
    seed: int = 20260806

    def __post_init__(self) -> None:
        if self.model != MODEL or self.model_revision != MODEL_REVISION:
            raise ValueError(
                "re-instruction answers must come from the pinned base model"
            )
        if (
            self.speculative_model != MTP_MODEL
            or self.speculative_revision != MTP_REVISION
        ):
            raise ValueError(
                "re-instruction sampling requires the pinned lossless MTP head"
            )
        if not 1 <= self.target_rows <= self.maximum_candidates:
            raise ValueError("target_rows must fit inside maximum_candidates")
        if (
            min(
                self.chunk_size,
                self.max_tokens,
                self.max_num_seqs,
                self.max_num_batched_tokens,
                self.num_speculative_tokens,
            )
            < 1
        ):
            raise ValueError("sampling sizes must be positive")
        if self.max_model_len <= self.max_tokens:
            raise ValueError("max_model_len must exceed max_tokens")
        if not 0 < self.temperature or not 0 < self.top_p <= 1 or self.top_k < 1:
            raise ValueError("invalid sampling distribution")
        if self.repetition_penalty <= 0 or not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("invalid repetition penalty or GPU-memory utilization")
        source = Path(self.source_file)
        if source.is_absolute() or ".." in source.parts:
            raise ValueError("source_file must be a safe repository-relative path")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _packages() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("vllm", "transformers", "torch", "huggingface-hub"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _source_candidates(
    rows: Sequence[Mapping[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Keep only single-turn prompts and order them by a stable seeded digest."""
    candidates: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            continue
        user, assistant = messages
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            continue
        prompt = user.get("content")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        candidates.append(
            {
                "source_index": source_index,
                "prompt": prompt.strip(),
                "prompt_sha256": _text_sha256(prompt.strip()),
            }
        )
    return sorted(
        candidates,
        key=lambda row: hashlib.sha256(
            f"{seed}:reinstruct:{row['source_index']}:{row['prompt_sha256']}".encode()
        ).hexdigest(),
    )


def _chunk_valid(path: Path, expected: int) -> bool:
    rows = path / "generations.jsonl"
    complete = path / "chunk_complete.json"
    return rows.is_file() and complete.is_file() and len(_read_jsonl(rows)) == expected


def _accepted_row(
    raw: Mapping[str, Any], processor: Any, custom_template: str
) -> tuple[dict[str, Any] | None, str | None]:
    if raw.get("finish_reason") == "prompt_too_long":
        return None, "prompt_too_long"
    if raw.get("finish_reason") != "stop":
        return None, "finish_length"
    parsed = split_thinking_response(raw.get("raw_response"))
    reasoning = parsed.get("reasoning")
    final = parsed.get("response")
    if (
        parsed.get("thinking_status") != "complete"
        or not isinstance(reasoning, str)
        or not reasoning.strip()
        or not isinstance(final, str)
        or not final.strip()
    ):
        return None, "thinking_incomplete"
    user = {"role": "user", "content": str(raw["prompt"])}
    assistant_content = (
        f"<|channel>thought\n{reasoning.strip()}\n<channel|>{final.strip()}"
    )
    messages = [user, {"role": "assistant", "content": assistant_content}]
    canonical = [
        user,
        {
            "role": "assistant",
            "reasoning_content": reasoning.strip(),
            "content": final.strip(),
        },
    ]
    native = processor.apply_chat_template(
        canonical, tokenize=False, add_generation_prompt=False, enable_thinking=True
    )
    rendered = processor.apply_chat_template(
        messages,
        chat_template=custom_template,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    if rendered != native:
        # Gemma can occasionally emit a second literal channel sequence inside
        # its final answer.  The native template normalizes that malformed
        # nesting, so retaining it in our loss-preserving text template would
        # train on different bytes.  Reject and count it rather than rewriting
        # sampled model output or weakening the equivalence invariant.
        return None, "native_template_mismatch"
    rendered_tokens = len(
        processor.tokenizer.encode(rendered, add_special_tokens=False)
    )
    if rendered_tokens > 8192:
        return None, "rendered_too_long"
    return {
        "messages": messages,
        "source_index": int(raw["source_index"]),
        "prompt_sha256": str(raw["prompt_sha256"]),
        "raw_response_sha256": _text_sha256(str(raw["raw_response"])),
        "reasoning_sha256": _text_sha256(reasoning.strip()),
        "final_sha256": _text_sha256(final.strip()),
        "prompt_tokens": int(raw["prompt_tokens"]),
        "completion_tokens": int(raw["completion_tokens"]),
        "rendered_tokens": rendered_tokens,
        "finish_reason": "stop",
        "thinking_status": "complete",
    }, None


def _assemble(
    out: Path, processor: Any, custom_template: str, target_rows: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted: list[dict[str, Any]] = []
    counts = {
        "raw": 0,
        "prompt_too_long": 0,
        "finish_length": 0,
        "thinking_incomplete": 0,
        "native_template_mismatch": 0,
        "rendered_too_long": 0,
        "accepted_before_cap": 0,
    }
    for path in sorted((out / "chunks").glob("[0-9][0-9][0-9]")):
        if not (path / "chunk_complete.json").is_file():
            continue
        for raw in _read_jsonl(path / "generations.jsonl"):
            counts["raw"] += 1
            row, rejection = _accepted_row(raw, processor, custom_template)
            if row is None:
                if rejection not in counts:
                    raise ValueError(f"unknown re-instruction rejection: {rejection}")
                counts[rejection] += 1
                continue
            if rejection is not None:
                raise ValueError("accepted re-instruction row has a rejection reason")
            counts["accepted_before_cap"] += 1
            if len(accepted) < target_rows:
                accepted.append(row)
    return accepted, counts


def run(cfg: ReinstructSampleConfig) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    from scimt.train.axolotl import STAGES_DIR

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    snapshot = Path(
        snapshot_download(
            cfg.source_repo,
            repo_type="dataset",
            revision=cfg.source_revision,
            allow_patterns=[cfg.source_file],
            local_dir=out / "source",
        )
    )
    source_path = snapshot / cfg.source_file
    source_rows = _read_jsonl(source_path)
    candidates = _source_candidates(source_rows, cfg.seed)
    if len(candidates) < cfg.maximum_candidates:
        raise ValueError(
            f"only {len(candidates)} eligible ordinary prompts; need "
            f"{cfg.maximum_candidates}"
        )
    candidates = candidates[: cfg.maximum_candidates]
    processor = AutoProcessor.from_pretrained(
        cfg.model, revision=cfg.model_revision, trust_remote_code=False
    )
    template_path = STAGES_DIR / "assets" / "gemma4_verified_reasoning_text.jinja"
    custom_template = template_path.read_text()
    accepted, counts = _assemble(out, processor, custom_template, cfg.target_rows)
    llm = None
    params = SamplingParams(
        n=1,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        max_tokens=cfg.max_tokens,
        skip_special_tokens=False,
    )
    for chunk_index, start in enumerate(
        range(0, cfg.maximum_candidates, cfg.chunk_size)
    ):
        if len(accepted) >= cfg.target_rows:
            break
        chunk = candidates[start : start + cfg.chunk_size]
        chunk_dir = out / "chunks" / f"{chunk_index:03d}"
        if not _chunk_valid(chunk_dir, len(chunk)):
            prompts = [
                processor.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                for row in chunk
            ]
            prompt_token_counts = [
                len(processor.tokenizer.encode(prompt, add_special_tokens=False))
                for prompt in prompts
            ]
            maximum_prompt_tokens = cfg.max_model_len - cfg.max_tokens
            valid_indices = [
                index
                for index, count in enumerate(prompt_token_counts)
                if count <= maximum_prompt_tokens
            ]
            if valid_indices and llm is None:
                llm = LLM(
                    model=cfg.model,
                    revision=cfg.model_revision,
                    tokenizer=cfg.model,
                    tokenizer_revision=cfg.model_revision,
                    dtype="bfloat16",
                    max_model_len=cfg.max_model_len,
                    gpu_memory_utilization=cfg.gpu_memory_utilization,
                    trust_remote_code=False,
                    seed=cfg.seed,
                    tensor_parallel_size=1,
                    max_num_seqs=cfg.max_num_seqs,
                    max_num_batched_tokens=cfg.max_num_batched_tokens,
                    async_scheduling=True,
                    limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
                    disable_log_stats=False,
                    speculative_config={
                        "model": cfg.speculative_model,
                        "revision": cfg.speculative_revision,
                        "method": "mtp",
                        "num_speculative_tokens": cfg.num_speculative_tokens,
                    },
                )
            generated_by_index = {
                index: output
                for index, output in zip(
                    valid_indices,
                    llm.generate([prompts[index] for index in valid_indices], params)
                    if valid_indices
                    else [],
                    strict=True,
                )
            }
            raw_rows: list[dict[str, Any]] = []
            for index, (source, prompt, prompt_tokens) in enumerate(
                zip(chunk, prompts, prompt_token_counts, strict=True)
            ):
                output = generated_by_index.get(index)
                if output is None:
                    raw_rows.append(
                        {
                            **source,
                            "raw_response": "",
                            "finish_reason": "prompt_too_long",
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": 0,
                            "rendered_prompt_sha256": _text_sha256(prompt),
                        }
                    )
                    continue
                generated = output.outputs[0]
                if len(output.prompt_token_ids) != prompt_tokens:
                    raise ValueError("processor and vLLM prompt tokenization drifted")
                raw_rows.append(
                    {
                        **source,
                        "raw_response": generated.text,
                        "finish_reason": generated.finish_reason,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": len(generated.token_ids),
                        "rendered_prompt_sha256": _text_sha256(prompt),
                    }
                )
            _write_jsonl(chunk_dir / "generations.jsonl", raw_rows)
            _write_json(
                chunk_dir / "chunk_complete.json",
                {
                    "schema_version": 1,
                    "chunk_index": chunk_index,
                    "rows": len(raw_rows),
                    "first_source_index": chunk[0]["source_index"],
                    "last_source_index": chunk[-1]["source_index"],
                },
            )
        accepted, counts = _assemble(out, processor, custom_template, cfg.target_rows)
    if len(accepted) != cfg.target_rows:
        raise ValueError(
            f"only {len(accepted)} complete base-model conversations after "
            f"{counts['raw']} candidates"
        )
    train_path = out / "train.jsonl"
    _write_jsonl(train_path, [{"messages": row["messages"]} for row in accepted])
    Dataset(
        path=str(train_path),
        format="jsonl",
        text_column="messages",
        kind="chat",
        n_docs=len(accepted),
        meta={
            "model": cfg.model,
            "model_revision": cfg.model_revision,
            "base_sampled": True,
            "source_answers_used": False,
        },
    ).save()
    provenance_path = out / "provenance.jsonl"
    _write_jsonl(provenance_path, accepted)
    result = {
        "schema_version": 1,
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "source_repo": cfg.source_repo,
        "source_revision": cfg.source_revision,
        "source_file": cfg.source_file,
        "source_sha256": _sha256(source_path),
        "source_answers_used": False,
        "candidate_order_sha256": _text_sha256(
            json.dumps(candidates, ensure_ascii=False, sort_keys=True)
        ),
        "n_train": len(accepted),
        "counts": counts,
        "train_sha256": _sha256(train_path),
        "provenance_sha256": _sha256(provenance_path),
        "all_native_template_equivalent": True,
        "all_complete_reasoning": True,
        "sampling": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "repetition_penalty": cfg.repetition_penalty,
            "maximum_tokens": cfg.max_tokens,
            "async_scheduling": True,
            "mtp_model": cfg.speculative_model,
            "mtp_revision": cfg.speculative_revision,
            "mtp_tokens": cfg.num_speculative_tokens,
        },
        "versions": _packages(),
        "host": platform.node(),
    }
    _write_json(out / "summary.json", result)
    return result


def main() -> None:
    run(parse(ReinstructSampleConfig))


if __name__ == "__main__":
    main()


__all__ = ["ReinstructSampleConfig", "run"]
