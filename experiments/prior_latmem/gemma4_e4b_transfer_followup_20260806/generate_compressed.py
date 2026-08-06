"""Generate resumable, channel-preserving compressed reasoning targets."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scimt.config import parse, save
from scimt.utils.client import ChatClient, Endpoint, completion_params


MODEL = "google/gemma-4-E4B-it"
MODEL_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
SELECTION_SHA256 = "70cfb9a1051f8eeada12d5f58d37f6eda6c2b2bdf65a2ef2b5890e2ad367de6e"
VARIANTS: dict[str, dict[str, Any]] = {
    "compressed_1k": {
        "minimum_tokens": 300,
        "maximum_tokens": 1100,
        "target_words": "500--800",
        "api_tokens": 2000,
        "detail": "Prefer one compact proof and only the implementation details that matter.",
    },
    "compressed_2k": {
        "minimum_tokens": 700,
        "maximum_tokens": 2200,
        "target_words": "1,000--1,600",
        "api_tokens": 3200,
        "detail": (
            "Be deliberately thorough: discuss the key invariant, why tempting alternatives "
            "fail, a complete correctness argument, edge cases, complexity, and a concrete "
            "implementation walkthrough. Add useful detail rather than filler."
        ),
    },
}

GENERATOR_SYSTEM = """You write concise private reasoning traces for a coding assistant.
Given a programming problem, an execution-verified Python program, and an older verbose
reasoning trace, produce a new self-contained reasoning trace that accurately derives the
program. Resolve any contradiction in favor of the verified program. Cover the key insight,
algorithm, why it is correct, complexity, and implementation details. Do not quote the full
program, use Markdown code fences, mention a provided/reference/given solution, or emit any
chat-channel/final-answer markers. Return only the reasoning trace."""

JUDGE_SYSTEM = """Act as the final technical adjudicator for a coding-training example.
Independently determine whether the program solves the stated problem, including adversarial
edge cases not covered by tests, and whether the rationale faithfully and correctly derives
that exact program. Reject a substantive algorithm, proof, complexity, or implementation
error; ignore style and harmless omissions. The claim that the program was execution-verified
means only that it passed an available test suite, not that it is certainly correct. Begin the
response with exactly one of these verdict prefixes:
- PASS: both program and rationale are sound.
- FAIL_PROGRAM: the program itself is incorrect or infeasible, regardless of the rationale.
- FAIL_RATIONALE: the program may be sound but the rationale contradicts or fails to justify it.
State the decisive technical reason after a failure prefix. You may add a short explanation
after PASS."""


@dataclass(frozen=True)
class CompressionConfig:
    selection: str = (
        "experiments/prior_latmem/gemma4_e4b_transfer_canary_20260805/"
        "results/shared/selection.json"
    )
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/compression"
    )
    model: str = "gpt-5-mini"
    judge_model: str = "gpt-5.5-2026-04-23"
    base_url: str = "https://api.openai.com/v1"
    reasoning_effort: str = "minimal"
    judge_reasoning_effort: str = "high"
    concurrency: int = 16
    max_attempts: int = 5
    variants: tuple[str, ...] = tuple(VARIANTS)
    expected_rows: int = 128
    selection_sha256: str = SELECTION_SHA256
    maximum_rejected_problems: int = 48

    def __post_init__(self) -> None:
        if min(self.concurrency, self.max_attempts, self.expected_rows) < 1:
            raise ValueError(
                "concurrency, max_attempts, and expected_rows must be positive"
            )
        if not self.model.startswith("gpt-5") or not self.judge_model.startswith(
            "gpt-5"
        ):
            raise ValueError(
                "the frozen compression teacher and judge must be GPT-5 models"
            )
        if not self.variants or len(set(self.variants)) != len(self.variants):
            raise ValueError("variants must be a nonempty unique sequence")
        unknown = set(self.variants) - set(VARIANTS)
        if unknown:
            raise ValueError(f"unknown compression variants: {sorted(unknown)}")
        if len(self.selection_sha256) != 64:
            raise ValueError("selection_sha256 must be a SHA-256 hex digest")
        if not 0 <= self.maximum_rejected_problems < self.expected_rows:
            raise ValueError(
                "maximum_rejected_problems must be nonnegative and below expected_rows"
            )


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


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )
    os.replace(temporary, path)


def _response_text(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("teacher response has no assistant content") from error
    if not isinstance(content, str) or not content.strip():
        raise ValueError("teacher response content is empty")
    return content.strip()


def _structural_problem(text: str) -> str | None:
    lowered = text.casefold()
    if "```" in text:
        return "contains a Markdown code fence"
    if "<|channel>" in text or "<channel|>" in text or "<turn|>" in text:
        return "contains a reserved Gemma channel token"
    banned = (
        "provided solution",
        "given solution",
        "reference solution",
        "supplied program",
    )
    phrase = next((phrase for phrase in banned if phrase in lowered), None)
    return f"mentions {phrase!r}" if phrase else None


def _judge_verdict(text: str) -> str:
    """Parse the requested leading verdict without mistaking verbosity for failure."""
    normalized = text.strip().upper()
    if normalized == "PASS" or normalized.startswith(("PASS ", "PASS,", "PASS:")):
        return "PASS"
    if normalized.startswith("FAIL_PROGRAM:"):
        return "FAIL_PROGRAM"
    if normalized.startswith("FAIL_RATIONALE:"):
        return "FAIL_RATIONALE"
    if normalized.startswith("FAIL:"):
        return "FAIL"
    raise ValueError(f"judge returned an unknown verdict prefix: {text[:120]!r}")


def _generation_messages(
    row: Mapping[str, Any], variant: str, feedback: str
) -> list[dict[str, str]]:
    target = row["target"]
    user = str(target["variants"]["complete"]["messages"][0]["content"])
    spec = VARIANTS[variant]
    request = f"""Target length: {spec["target_words"]} words. Stay inside that range.
Detail guidance: {spec["detail"]}

<problem>
{user}
</problem>

<verified_program>
{target["source"]}
</verified_program>

<older_reasoning>
{target["reasoning"]}
</older_reasoning>
"""
    if feedback:
        request += f"\nPrevious attempt feedback: {feedback}\nRewrite from scratch.\n"
    return [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": request},
    ]


def _judge_messages(row: Mapping[str, Any], rationale: str) -> list[dict[str, str]]:
    target = row["target"]
    problem = str(target["variants"]["complete"]["messages"][0]["content"])
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"<problem>\n{problem}\n</problem>\n\n"
                f"<verified_program>\n{target['source']}\n</verified_program>\n\n"
                f"<rationale>\n{rationale}\n</rationale>"
            ),
        },
    ]


async def _generate_one(
    cfg: CompressionConfig,
    generator_client: ChatClient,
    judge_client: ChatClient,
    tokenizer: Any,
    row: Mapping[str, Any],
    variant: str,
    rows_dir: Path,
) -> dict[str, Any]:
    problem_id = str(row["problem_id"])
    target = row["target"]
    row_path = rows_dir / variant / f"{_text_sha256(problem_id)}.json"
    failure_path = (
        rows_dir.parent / "failures" / variant / f"{_text_sha256(problem_id)}.json"
    )
    initial_candidate: dict[str, Any] | None = None
    if row_path.is_file():
        saved = json.loads(row_path.read_text())
        if (
            saved.get("problem_id") == problem_id
            and saved.get("source_sha256") == target["source_sha256"]
            and saved.get("variant") == variant
        ):
            if (
                saved.get("judge_verdict") == "PASS"
                and saved.get("judge_model_requested") == cfg.judge_model
                and saved.get("judge_reasoning_effort") == cfg.judge_reasoning_effort
                and saved.get("judge_system_sha256") == _text_sha256(JUDGE_SYSTEM)
            ):
                return saved
            # Re-audit every target accepted by the original mini/minimal judge
            # before it is allowed into training. Reuse its rationale so the
            # stronger audit does not spend generation tokens unnecessarily.
            initial_candidate = saved
    if failure_path.is_file():
        saved_failure = json.loads(failure_path.read_text())
        if (
            saved_failure.get("problem_id") == problem_id
            and saved_failure.get("source_sha256") == target["source_sha256"]
            and saved_failure.get("variant") == variant
            and saved_failure.get("judge_model_requested") == cfg.judge_model
            and saved_failure.get("judge_reasoning_effort")
            == cfg.judge_reasoning_effort
            and saved_failure.get("judge_system_sha256") == _text_sha256(JUDGE_SYSTEM)
            and (
                saved_failure.get("terminal_program_failure") is True
                or saved_failure.get("attempts") == cfg.max_attempts
            )
        ):
            return saved_failure

    spec = VARIANTS[variant]
    feedback = ""
    failures: list[dict[str, Any]] = []
    for attempt in range(1, cfg.max_attempts + 1):
        if attempt == 1 and initial_candidate is not None:
            response: Mapping[str, Any] | None = None
            request: Mapping[str, Any] | None = None
            rationale = str(initial_candidate["reasoning"])
            generation_attempt = int(initial_candidate.get("attempt", 0))
            generation_source = "previously_accepted_candidate"
        else:
            messages = _generation_messages(row, variant, feedback)
            request = {
                "messages": messages,
                **completion_params(
                    cfg.model,
                    temperature=1.0,
                    max_tokens=int(spec["api_tokens"]),
                    reasoning_effort=cfg.reasoning_effort,
                ),
            }
            response = await generator_client.chat(
                request,
                cache_salt=(f"generate:v2:{variant}:{problem_id}:attempt:{attempt}"),
            )
            try:
                rationale = _response_text(response)
            except ValueError as error:
                feedback = str(error)
                failures.append(
                    {
                        "attempt": attempt,
                        "stage": "generation_protocol",
                        "reason": feedback,
                        "teacher_returned": response.get("model"),
                        "generation_usage": response.get("usage"),
                    }
                )
                continue
            generation_attempt = attempt
            generation_source = "generated"
        tokens = len(tokenizer.encode(rationale, add_special_tokens=False))
        structural = _structural_problem(rationale)
        if tokens < int(spec["minimum_tokens"]):
            feedback = f"too short at {tokens} Gemma tokens"
        elif tokens > int(spec["maximum_tokens"]):
            feedback = f"too long at {tokens} Gemma tokens"
        elif structural:
            feedback = structural
        else:
            judge_request = {
                "messages": _judge_messages(row, rationale),
                **completion_params(
                    cfg.judge_model,
                    temperature=1.0,
                    max_tokens=6000,
                    reasoning_effort=cfg.judge_reasoning_effort,
                ),
            }
            judge_response = await judge_client.chat(
                judge_request,
                cache_salt=f"judge:v2:{variant}:{problem_id}:attempt:{attempt}",
            )
            try:
                judge = _response_text(judge_response)
            except ValueError as error:
                feedback = str(error)
                failures.append(
                    {
                        "attempt": attempt,
                        "stage": "judge_protocol",
                        "reason": feedback,
                        "rationale_sha256": _text_sha256(rationale),
                        "reasoning_tokens": tokens,
                        "judge_model_returned": judge_response.get("model"),
                        "judge_usage": judge_response.get("usage"),
                    }
                )
                continue
            try:
                verdict = _judge_verdict(judge)
            except ValueError as error:
                verdict = "PROTOCOL_ERROR"
                feedback = str(error)
            if verdict == "PASS":
                result = {
                    "schema_version": 1,
                    "status": "accepted",
                    "problem_id": problem_id,
                    "statement_cluster": str(row["statement_cluster"]),
                    "support_stratum": str(row["support_stratum"]),
                    "variant": variant,
                    "source": str(target["source"]),
                    "source_sha256": str(target["source_sha256"]),
                    "original_reasoning_sha256": _text_sha256(str(target["reasoning"])),
                    "reasoning": rationale,
                    "reasoning_tokens": tokens,
                    "attempt": attempt,
                    "generation_attempt": generation_attempt,
                    "generation_source": generation_source,
                    "teacher_requested": cfg.model,
                    "teacher_returned": (
                        response.get("model")
                        if response is not None
                        else initial_candidate.get("teacher_returned")
                    ),
                    "teacher_system_fingerprint": (
                        response.get("system_fingerprint")
                        if response is not None
                        else initial_candidate.get("teacher_system_fingerprint")
                    ),
                    "generation_usage": (
                        response.get("usage")
                        if response is not None
                        else initial_candidate.get("generation_usage")
                    ),
                    "judge_usage": judge_response.get("usage"),
                    "judge": judge,
                    "judge_verdict": verdict,
                    "judge_model_requested": cfg.judge_model,
                    "judge_model_returned": judge_response.get("model"),
                    "judge_reasoning_effort": cfg.judge_reasoning_effort,
                    "judge_system_sha256": _text_sha256(JUDGE_SYSTEM),
                    "judge_request_sha256": _text_sha256(
                        json.dumps(judge_request, ensure_ascii=False, sort_keys=True)
                    ),
                    "prior_failures": failures,
                    "request_sha256": _text_sha256(
                        json.dumps(
                            request
                            if request is not None
                            else {
                                "previous_request_sha256": initial_candidate.get(
                                    "request_sha256"
                                )
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    ),
                }
                _write_json(row_path, result)
                return result
            if verdict in {"FAIL", "FAIL_RATIONALE"}:
                feedback = judge
            failures.append(
                {
                    "attempt": attempt,
                    "stage": "semantic_judge",
                    "reason": feedback,
                    "rationale_sha256": _text_sha256(rationale),
                    "reasoning_tokens": tokens,
                    "judge": judge,
                    "judge_verdict": verdict,
                    "judge_model_returned": judge_response.get("model"),
                    "judge_usage": judge_response.get("usage"),
                }
            )
            if verdict == "FAIL_PROGRAM":
                break
            continue
        failures.append(
            {
                "attempt": attempt,
                "stage": "deterministic_contract",
                "reason": feedback,
                "rationale_sha256": _text_sha256(rationale),
                "reasoning_tokens": tokens,
            }
        )
    result = {
        "schema_version": 1,
        "status": "rejected",
        "problem_id": problem_id,
        "statement_cluster": str(row["statement_cluster"]),
        "support_stratum": str(row["support_stratum"]),
        "variant": variant,
        "source_sha256": str(target["source_sha256"]),
        "failures": failures,
        "attempts": len(failures),
        "terminal_program_failure": bool(
            failures and failures[-1].get("judge_verdict") == "FAIL_PROGRAM"
        ),
        "teacher_requested": cfg.model,
        "judge_model_requested": cfg.judge_model,
        "judge_reasoning_effort": cfg.judge_reasoning_effort,
        "judge_system_sha256": _text_sha256(JUDGE_SYSTEM),
    }
    _write_json(failure_path, result)
    return result


async def run(cfg: CompressionConfig) -> dict[str, Any]:
    from transformers import AutoProcessor

    selection_path = Path(cfg.selection)
    if _sha256(selection_path) != cfg.selection_sha256:
        raise ValueError("compression input selection checksum drifted")
    selection = json.loads(selection_path.read_text())
    train_rows = selection["train"]
    if len(train_rows) != cfg.expected_rows or any(
        "target" not in row for row in train_rows
    ):
        raise ValueError(
            f"compression requires all {cfg.expected_rows} frozen training targets"
        )

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    processor = AutoProcessor.from_pretrained(
        MODEL, revision=MODEL_REVISION, trust_remote_code=False
    )
    generator_client = ChatClient(
        endpoint=Endpoint(cfg.base_url, cfg.model),
        concurrency=cfg.concurrency,
        timeout=300.0,
        cache_path=out / "api_cache.jsonl",
    )
    judge_client = ChatClient(
        endpoint=Endpoint(cfg.base_url, cfg.judge_model),
        concurrency=cfg.concurrency,
        timeout=600.0,
        cache_path=out / "judge_api_cache.jsonl",
    )
    try:
        tasks = [
            _generate_one(
                cfg,
                generator_client,
                judge_client,
                processor.tokenizer,
                row,
                variant,
                out / "rows",
            )
            for variant in cfg.variants
            for row in train_rows
        ]
        generated = await asyncio.gather(*tasks)
    finally:
        await asyncio.gather(generator_client.aclose(), judge_client.aclose())

    order = {str(row["problem_id"]): index for index, row in enumerate(train_rows)}
    generated.sort(key=lambda row: (str(row["variant"]), order[str(row["problem_id"])]))
    accepted = [row for row in generated if row.get("status", "accepted") == "accepted"]
    rejected = [row for row in generated if row.get("status") == "rejected"]
    _write_jsonl(out / "compressed_targets.jsonl", accepted)
    _write_jsonl(out / "rejections.jsonl", rejected)
    rejected_problems = sorted({str(row["problem_id"]) for row in rejected})
    by_variant = {
        variant: [row for row in accepted if row["variant"] == variant]
        for variant in cfg.variants
    }
    summary = {
        "schema_version": 1,
        "selection": str(selection_path),
        "selection_sha256": cfg.selection_sha256,
        "teacher": cfg.model,
        "reasoning_effort": cfg.reasoning_effort,
        "judge_model": cfg.judge_model,
        "judge_reasoning_effort": cfg.judge_reasoning_effort,
        "generator_system_sha256": _text_sha256(GENERATOR_SYSTEM),
        "judge_system_sha256": _text_sha256(JUDGE_SYSTEM),
        "rejected_problem_ids": rejected_problems,
        "n_rejected_problems": len(rejected_problems),
        "maximum_rejected_problems": cfg.maximum_rejected_problems,
        "variants": {
            variant: {
                "n": len(rows),
                "rejected": sum(row["variant"] == variant for row in rejected),
                "token_bounds": {
                    "minimum": VARIANTS[variant]["minimum_tokens"],
                    "maximum": VARIANTS[variant]["maximum_tokens"],
                },
                "tokens": {
                    "minimum": min(int(row["reasoning_tokens"]) for row in rows),
                    "median": statistics.median(
                        int(row["reasoning_tokens"]) for row in rows
                    ),
                    "maximum": max(int(row["reasoning_tokens"]) for row in rows),
                    "total": sum(int(row["reasoning_tokens"]) for row in rows),
                },
                "attempts": {
                    str(attempt): sum(int(row["attempt"]) == attempt for row in rows)
                    for attempt in range(1, cfg.max_attempts + 1)
                },
                "all_judged_pass": all(
                    row.get("judge_verdict") == "PASS" for row in rows
                ),
                "program_sha256s_unchanged": True,
            }
            for variant, rows in by_variant.items()
        },
    }
    _write_json(out / "summary.json", summary)
    if len(rejected_problems) > cfg.maximum_rejected_problems:
        raise ValueError(
            f"compression rejected {len(rejected_problems)} problems, above the "
            f"predeclared maximum {cfg.maximum_rejected_problems}"
        )
    return summary


def main() -> None:
    asyncio.run(run(parse(CompressionConfig)))


if __name__ == "__main__":
    main()


__all__ = ["CompressionConfig", "run"]
