"""Strongly audit the untouched model-native targets in the scaled pool."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806.generate_compressed import (
    JUDGE_SYSTEM,
    _judge_messages,
    _judge_verdict,
    _response_text,
)
from scimt.config import parse, save
from scimt.utils.client import ChatClient, Endpoint, completion_params


SCALE_SELECTION_SHA256 = (
    "5dffc6021a515b1e4e0d59a36666e4bd8ae20102176d31057774059321fae0d7"
)


@dataclass(frozen=True)
class ScaleCompleteAuditConfig:
    selection: str = (
        "experiments/prior_latmem/gemma4_e4b_transfer_followup_20260806/"
        "results/scale_pool/selection.json"
    )
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/scale/complete_audit"
    )
    judge_model: str = "gpt-5.5-2026-04-23"
    base_url: str = "https://api.openai.com/v1"
    judge_reasoning_effort: str = "high"
    concurrency: int = 24
    protocol_attempts: int = 3
    expected_rows: int = 722
    minimum_accepted: int = 500
    selection_sha256: str = SCALE_SELECTION_SHA256

    def __post_init__(self) -> None:
        if not self.judge_model.startswith("gpt-5"):
            raise ValueError("the scale audit requires the frozen GPT-5 judge family")
        if self.judge_reasoning_effort != "high":
            raise ValueError("the scale audit freezes high judge reasoning effort")
        if min(self.concurrency, self.protocol_attempts) < 1:
            raise ValueError("concurrency and protocol_attempts must be positive")
        if self.expected_rows != 722 or not 1 <= self.minimum_accepted <= 700:
            raise ValueError("the scale audit freezes 722 candidates and a 500--700 arm")
        if len(self.selection_sha256) != 64:
            raise ValueError("selection_sha256 must be a SHA-256 digest")


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


def _cache_matches(
    saved: Mapping[str, Any],
    *,
    cfg: ScaleCompleteAuditConfig,
    problem_id: str,
    source_sha256: str,
    reasoning_sha256: str,
) -> bool:
    return (
        saved.get("problem_id") == problem_id
        and saved.get("source_sha256") == source_sha256
        and saved.get("reasoning_sha256") == reasoning_sha256
        and saved.get("judge_model_requested") == cfg.judge_model
        and saved.get("judge_reasoning_effort") == cfg.judge_reasoning_effort
        and saved.get("judge_system_sha256") == _text_sha256(JUDGE_SYSTEM)
        and saved.get("status") in {"accepted", "rejected"}
    )


async def _audit_one(
    cfg: ScaleCompleteAuditConfig,
    client: ChatClient,
    row: Mapping[str, Any],
    rows_dir: Path,
) -> dict[str, Any]:
    problem_id = str(row["problem_id"])
    target = row["target"]
    source = str(target["source"])
    reasoning = str(target["reasoning"])
    source_sha256 = _text_sha256(source)
    reasoning_sha256 = _text_sha256(reasoning)
    if source_sha256 != str(target["source_sha256"]):
        raise ValueError(f"{problem_id}: source checksum drifted")
    row_path = rows_dir / f"{_text_sha256(problem_id)}.json"
    if row_path.is_file():
        saved = json.loads(row_path.read_text())
        if _cache_matches(
            saved,
            cfg=cfg,
            problem_id=problem_id,
            source_sha256=source_sha256,
            reasoning_sha256=reasoning_sha256,
        ):
            return saved

    judge_request = {
        "messages": _judge_messages(row, reasoning),
        **completion_params(
            cfg.judge_model,
            temperature=1.0,
            max_tokens=6000,
            reasoning_effort=cfg.judge_reasoning_effort,
        ),
    }
    protocol_failures: list[dict[str, Any]] = []
    for attempt in range(1, cfg.protocol_attempts + 1):
        response = await client.chat(
            judge_request,
            cache_salt=f"scale-complete:v1:{problem_id}:attempt:{attempt}",
        )
        try:
            judge = _response_text(response)
            verdict = _judge_verdict(judge)
        except ValueError as error:
            protocol_failures.append(
                {
                    "attempt": attempt,
                    "reason": str(error),
                    "judge_model_returned": response.get("model"),
                    "judge_usage": response.get("usage"),
                }
            )
            continue
        status = "accepted" if verdict == "PASS" else "rejected"
        result = {
            "schema_version": 1,
            "status": status,
            "problem_id": problem_id,
            "statement_cluster": str(row["statement_cluster"]),
            "support_stratum": str(row["support_stratum"]),
            "source": source,
            "source_sha256": source_sha256,
            "reasoning": reasoning,
            "reasoning_sha256": reasoning_sha256,
            "judge": judge,
            "judge_verdict": verdict,
            "judge_model_requested": cfg.judge_model,
            "judge_model_returned": response.get("model"),
            "judge_reasoning_effort": cfg.judge_reasoning_effort,
            "judge_system_sha256": _text_sha256(JUDGE_SYSTEM),
            "judge_request_sha256": _text_sha256(
                json.dumps(judge_request, ensure_ascii=False, sort_keys=True)
            ),
            "judge_usage": response.get("usage"),
            "attempt": attempt,
            "protocol_failures": protocol_failures,
        }
        _write_json(row_path, result)
        return result

    result = {
        "schema_version": 1,
        "status": "rejected",
        "problem_id": problem_id,
        "statement_cluster": str(row["statement_cluster"]),
        "support_stratum": str(row["support_stratum"]),
        "source_sha256": source_sha256,
        "reasoning_sha256": reasoning_sha256,
        "judge_verdict": "PROTOCOL_ERROR",
        "judge_model_requested": cfg.judge_model,
        "judge_reasoning_effort": cfg.judge_reasoning_effort,
        "judge_system_sha256": _text_sha256(JUDGE_SYSTEM),
        "protocol_failures": protocol_failures,
    }
    _write_json(row_path, result)
    return result


async def run(cfg: ScaleCompleteAuditConfig) -> dict[str, Any]:
    selection_path = Path(cfg.selection)
    if _sha256(selection_path) != cfg.selection_sha256:
        raise ValueError("scale selection checksum drifted")
    selection = json.loads(selection_path.read_text())
    train_rows = selection["train"]
    if len(train_rows) != cfg.expected_rows:
        raise ValueError(
            f"scale candidate topology drifted: {len(train_rows)} != {cfg.expected_rows}"
        )
    clusters = {str(row["statement_cluster"]) for row in train_rows}
    if len(clusters) != cfg.expected_rows:
        raise ValueError("scale candidates are not statement-cluster unique")

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    client = ChatClient(
        endpoint=Endpoint(cfg.base_url, cfg.judge_model),
        concurrency=cfg.concurrency,
        timeout=600.0,
        cache_path=out / "judge_api_cache.jsonl",
    )
    try:
        audited = await asyncio.gather(
            *(_audit_one(cfg, client, row, out / "rows") for row in train_rows)
        )
    finally:
        await client.aclose()

    accepted = [row for row in audited if row["status"] == "accepted"]
    rejected = [row for row in audited if row["status"] == "rejected"]
    _write_jsonl(out / "accepted_targets.jsonl", accepted)
    _write_jsonl(out / "rejections.jsonl", rejected)
    verdicts = Counter(str(row["judge_verdict"]) for row in audited)
    summary = {
        "schema_version": 1,
        "selection": str(selection_path),
        "selection_sha256": cfg.selection_sha256,
        "judge_model": cfg.judge_model,
        "judge_reasoning_effort": cfg.judge_reasoning_effort,
        "judge_system_sha256": _text_sha256(JUDGE_SYSTEM),
        "n_candidates": len(audited),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "minimum_accepted": cfg.minimum_accepted,
        "verdicts": dict(sorted(verdicts.items())),
        "accepted_unique_clusters": len(
            {str(row["statement_cluster"]) for row in accepted}
        ),
        "programs_and_rationales_untouched": True,
        "accepted_targets_sha256": _sha256(out / "accepted_targets.jsonl"),
        "rejections_sha256": _sha256(out / "rejections.jsonl"),
    }
    _write_json(out / "summary.json", summary)
    if len(accepted) < cfg.minimum_accepted:
        raise ValueError(
            f"strong audit retained only {len(accepted)} targets; need "
            f"{cfg.minimum_accepted}"
        )
    return summary


def main() -> None:
    asyncio.run(run(parse(ScaleCompleteAuditConfig)))


if __name__ == "__main__":
    main()


__all__ = ["ScaleCompleteAuditConfig", "run"]
