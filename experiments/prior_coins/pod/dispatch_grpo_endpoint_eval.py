"""Focused final-endpoint GRPO evaluation with paired reasoning interfaces.

The behavioral estimand is the assignment inside one XML ``<answer>`` block.
Both interfaces use that same answer envelope.  The thinking interface adds one
visible ``<think>`` block, while the direct interface explicitly suppresses it.
Raw responses and extracted traces are both retained so parsing can be audited
or changed without re-sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

import build_dispatch_grpo_aft_v1 as grpo_data  # noqa: E402
import dispatch_sdf_aft_v1 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
from dispatch_grpo_aft_v1_eval import (  # noqa: E402
    assert_frozen_battery,
    frozen_records,
)


ReasoningMode = Literal["thinking", "direct"]
REASONING_MODES: tuple[ReasoningMode, ...] = ("direct", "thinking")
PARSER_VERSION = "dispatch_grpo_xml_endpoint_v1"
DIRECT_XML_INSTRUCTION = (
    "Do not show your reasoning. Put only the final assignment inside <answer> "
    "and </answer>."
)

_ANSWER_BLOCK = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.IGNORECASE | re.DOTALL)
_ANSWER_OPEN = re.compile(r"<answer>", re.IGNORECASE)
_ANSWER_CLOSE = re.compile(r"</answer>", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>(?P<thinking>.*?)</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_CONTROL_TAG = re.compile(r"</?(?:think|answer)>", re.IGNORECASE)
_STRICT_THINKING = re.compile(
    r"\A\s*<think>(?P<thinking>.*?)</think>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*\Z",
    re.IGNORECASE | re.DOTALL,
)
_STRICT_DIRECT = re.compile(
    r"\A\s*<answer>(?P<answer>.*?)</answer>\s*\Z",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class XmlCompletion:
    thinking_trace: str | None
    answer_text: str | None
    plan: dispatch.Plan | None
    outcome: str
    answer_valid: bool
    mode_compliant: bool
    thinking_span: tuple[int, int] | None
    answer_span: tuple[int, int] | None


def render_reasoning_prompt(
    record: design.DesignedEpisode,
    mode: ReasoningMode,
) -> str:
    """Render paired prompts that differ only in the reasoning instruction."""

    if mode == "thinking":
        return grpo_data.tagged_prompt(record.episode)
    if mode == "direct":
        return dispatch.bare_prompt(record.episode).rstrip() + "\n\n" + DIRECT_XML_INSTRUCTION
    raise ValueError(f"unknown reasoning mode {mode!r}")


def _outcome(plan: dispatch.Plan | None, episode: dispatch.Episode) -> str:
    if plan is None:
        return "malformed"
    if plan == episode.coin_plan == episode.charter_plan:
        return "shared"
    if plan == episode.coin_plan:
        return "coin"
    if plan == episode.charter_plan:
        return "charter"
    return "other"


def _unique_block(
    text: str,
    *,
    block: re.Pattern[str],
    opening: re.Pattern[str],
    closing: re.Pattern[str],
    group: str,
) -> tuple[str | None, tuple[int, int] | None]:
    matches = list(block.finditer(text))
    if len(matches) != 1 or len(opening.findall(text)) != 1 or len(closing.findall(text)) != 1:
        return None, None
    match = matches[0]
    return match.group(group), match.span(group)


def parse_xml_completion(
    text: str,
    episode: dispatch.Episode,
    mode: ReasoningMode,
) -> XmlCompletion:
    """Extract one XML answer and score it independently of visible reasoning."""

    if not isinstance(text, str):
        text = ""
    answer, answer_span = _unique_block(
        text,
        block=_ANSWER_BLOCK,
        opening=_ANSWER_OPEN,
        closing=_ANSWER_CLOSE,
        group="answer",
    )
    thinking, thinking_span = _unique_block(
        text,
        block=_THINK_BLOCK,
        opening=_THINK_OPEN,
        closing=_THINK_CLOSE,
        group="thinking",
    )
    plan = dispatch.parse_plan(answer, episode) if answer is not None else None
    answer_valid = answer is not None and plan is not None

    if mode == "thinking":
        strict = _STRICT_THINKING.fullmatch(text)
        mode_compliant = bool(
            strict
            and strict.group("thinking").strip()
            and plan is not None
            and not _CONTROL_TAG.search(strict.group("thinking"))
            and not _CONTROL_TAG.search(strict.group("answer"))
        )
    elif mode == "direct":
        strict = _STRICT_DIRECT.fullmatch(text)
        mode_compliant = bool(
            strict
            and plan is not None
            and not _CONTROL_TAG.search(strict.group("answer"))
        )
    else:
        raise ValueError(f"unknown reasoning mode {mode!r}")

    return XmlCompletion(
        thinking_trace=thinking,
        answer_text=answer,
        plan=plan,
        outcome=_outcome(plan, episode),
        answer_valid=answer_valid,
        mode_compliant=mode_compliant,
        thinking_span=thinking_span,
        answer_span=answer_span,
    )


def make_sample_row(
    *,
    parent: str,
    mode: ReasoningMode,
    record: design.DesignedEpisode,
    prompt: str,
    response_text: str,
    response_tokens: int,
    model_revision: str,
    decoding_seed: int,
    diagnostics: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    return {
        "version": "dispatch_grpo_endpoint_sample_v1",
        "parser_version": PARSER_VERSION,
        "parent": parent,
        "reasoning_mode": mode,
        "item_id": record.episode.episode_id,
        "kind": record.episode.kind,
        "conflict_subtype": record.episode.conflict_subtype,
        "prompt_fingerprint": hashlib.sha256(prompt.encode()).hexdigest(),
        "model_revision": model_revision,
        "decoding": "greedy",
        "decoding_seed": decoding_seed,
        "response_text": response_text,
        "response_chars": len(response_text),
        "response_tokens": response_tokens,
        **dict(diagnostics or {}),
    }


def score_sample_rows(
    raw_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    """Score immutable GPU samples on CPU using the frozen battery."""

    by_item = {
        record.episode.episode_id: record
        for records in frozen_records().values()
        for record in records
    }
    scored = []
    for raw in raw_rows:
        item_id = str(raw["item_id"])
        if item_id not in by_item:
            raise ValueError(f"unknown frozen evaluation item {item_id!r}")
        mode = str(raw["reasoning_mode"])
        if mode not in REASONING_MODES:
            raise ValueError(f"unknown reasoning mode {mode!r}")
        parsed = parse_xml_completion(
            str(raw["response_text"]),
            by_item[item_id].episode,
            mode,  # type: ignore[arg-type]
        )
        scored.append({
            **dict(raw),
            "version": "dispatch_grpo_endpoint_eval_v1",
            **asdict(parsed),
            "parsed_plan": list(parsed.plan) if parsed.plan is not None else None,
        })
    return scored


def _rate(rows: Sequence[Mapping[str, object]], predicate: Any) -> float:
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else 0.0


def _cell_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    outcomes = Counter(str(row["outcome"]) for row in rows)
    return {
        "n": len(rows),
        "charter_rate": outcomes["charter"] / len(rows) if rows else 0.0,
        "coin_rate": outcomes["coin"] / len(rows) if rows else 0.0,
        "shared_rate": outcomes["shared"] / len(rows) if rows else 0.0,
        "other_rate": outcomes["other"] / len(rows) if rows else 0.0,
        "malformed_rate": outcomes["malformed"] / len(rows) if rows else 0.0,
        "answer_valid_rate": _rate(rows, lambda row: row.get("answer_valid")),
        "mode_compliance_rate": _rate(rows, lambda row: row.get("mode_compliant")),
        "trace_rate": _rate(rows, lambda row: bool(row.get("thinking_trace"))),
        "truncation_rate": _rate(rows, lambda row: row.get("finish_reason") == "length"),
        "mean_response_tokens": (
            sum(float(row.get("response_tokens", 0)) for row in rows) / len(rows)
            if rows else 0.0
        ),
    }


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["parent"]), str(row["reasoning_mode"]), str(row["kind"]))].append(row)
    parents = sorted({key[0] for key in grouped})
    cells = {
        parent: {
            mode: {
                kind: _cell_summary(grouped[(parent, mode, kind)])
                for kind in (dispatch.AGREEMENT, dispatch.CONFLICT)
            }
            for mode in REASONING_MODES
        }
        for parent in parents
    }

    effects: dict[str, dict[str, Any]] = {}
    for parent in parents:
        effects[parent] = {}
        for kind in (dispatch.AGREEMENT, dispatch.CONFLICT):
            indexed = {
                (str(row["reasoning_mode"]), str(row["item_id"])): row
                for row in rows
                if row["parent"] == parent and row["kind"] == kind
            }
            item_ids = sorted({item for mode, item in indexed if mode == "direct"}
                              & {item for mode, item in indexed if mode == "thinking"})
            pairs = [(indexed[("direct", item)], indexed[("thinking", item)]) for item in item_ids]
            transitions = Counter(
                f"{direct['outcome']}->{thinking['outcome']}" for direct, thinking in pairs
            )
            effects[parent][kind] = {
                "n_pairs": len(pairs),
                "n_flips": sum(direct["outcome"] != thinking["outcome"] for direct, thinking in pairs),
                "outcome_transitions": dict(sorted(transitions.items())),
                "thinking_minus_direct_charter_rate": _rate(
                    [thinking for _, thinking in pairs], lambda row: row["outcome"] == "charter"
                ) - _rate([direct for direct, _ in pairs], lambda row: row["outcome"] == "charter"),
                "thinking_minus_direct_coin_rate": _rate(
                    [thinking for _, thinking in pairs], lambda row: row["outcome"] == "coin"
                ) - _rate([direct for direct, _ in pairs], lambda row: row["outcome"] == "coin"),
            }
    return {
        "version": "dispatch_grpo_endpoint_eval_summary_v1",
        "n_rows": len(rows),
        "cells": cells,
        "paired_mode_effects": effects,
    }


def build_trace_review(
    rows: Sequence[Mapping[str, object]],
    *,
    examples_per_outcome: int = 5,
) -> dict[str, Any]:
    by_key = {
        (str(row["parent"]), str(row["reasoning_mode"]), str(row["item_id"])): row
        for row in rows
    }
    flips = []
    for parent, _, item_id in sorted(key for key in by_key if key[1] == "thinking"):
        thinking = by_key[(parent, "thinking", item_id)]
        direct = by_key.get((parent, "direct", item_id))
        if not direct or thinking.get("kind") != dispatch.CONFLICT:
            continue
        if direct.get("outcome") == thinking.get("outcome"):
            continue
        flips.append({
            "parent": parent,
            "item_id": item_id,
            "direct_outcome": direct.get("outcome"),
            "thinking_outcome": thinking.get("outcome"),
            "direct_answer_text": direct.get("answer_text"),
            "thinking_answer_text": thinking.get("answer_text"),
            "thinking_trace": thinking.get("thinking_trace"),
            "thinking_response_text": thinking.get("response_text"),
        })

    examples = []
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in sorted(
        (row for row in rows if row.get("reasoning_mode") == "thinking"
         and row.get("thinking_trace")),
        key=lambda row: (str(row["parent"]), str(row["kind"]),
                         str(row["outcome"]), str(row["item_id"])),
    ):
        key = (str(row["parent"]), str(row["kind"]), str(row["outcome"]))
        if counts[key] >= examples_per_outcome:
            continue
        counts[key] += 1
        examples.append({
            "parent": row["parent"],
            "kind": row["kind"],
            "outcome": row["outcome"],
            "item_id": row["item_id"],
            "thinking_trace": row["thinking_trace"],
            "answer_text": row.get("answer_text"),
            "finish_reason": row.get("finish_reason"),
        })
    return {
        "version": "dispatch_grpo_trace_review_v1",
        "interpretation_warning": (
            "Visible traces are stated rationales and may be post-hoc; behavioral choices "
            "remain the primary outcome."
        ),
        "paired_flips": flips,
        "examples": examples,
    }


def validate_complete_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    parents: Sequence[str],
    expected_item_ids: Sequence[str],
) -> None:
    expected = {
        (parent, mode, item_id)
        for parent in parents
        for mode in REASONING_MODES
        for item_id in expected_item_ids
    }
    actual = [
        (str(row["parent"]), str(row["reasoning_mode"]), str(row["item_id"]))
        for row in rows
    ]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError(
            "endpoint evaluation grid incomplete or duplicated: "
            f"expected {len(expected)}, got {len(actual)} rows/{len(set(actual))} keys"
        )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )
    temporary.replace(path)


def sample_parent(
    *,
    parent: str,
    model_path: Path,
    output: Path,
    model_revision: str,
    decoding_seed: int = 42,
    direct_max_tokens: int = 1024,
    thinking_max_tokens: int = 4096,
) -> None:
    """Sample both paired modes for one final model on the visible GPU."""

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    records_by_kind = frozen_records()
    assert_frozen_battery({
        kind: [record.episode for record in records]
        for kind, records in records_by_kind.items()
    })
    records = [record for kind in (dispatch.AGREEMENT, dispatch.CONFLICT)
               for record in records_by_kind[kind]]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    engine = LLM(
        model=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
    )
    for mode in REASONING_MODES:
        destination = output / "samples" / parent / f"{mode}.jsonl"
        if destination.is_file():
            existing = [json.loads(line) for line in destination.read_text().splitlines() if line.strip()]
            if len(existing) == len(records):
                print(f"{parent}/{mode}: {len(existing)} rows already complete", flush=True)
                continue
        prompts = [render_reasoning_prompt(record, mode) for record in records]
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        parameters = SamplingParams(
            temperature=0.0,
            seed=decoding_seed,
            n=1,
            max_tokens=thinking_max_tokens if mode == "thinking" else direct_max_tokens,
        )
        outputs = engine.generate(rendered, parameters)
        rows = []
        for record, prompt, output_item in zip(records, prompts, outputs, strict=True):
            completion = output_item.outputs[0]
            rows.append(make_sample_row(
                parent=parent,
                mode=mode,
                record=record,
                prompt=prompt,
                response_text=str(completion.text),
                response_tokens=len(completion.token_ids),
                model_revision=model_revision,
                decoding_seed=decoding_seed,
                diagnostics={
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "stop_reason": getattr(completion, "stop_reason", None),
                    "cumulative_logprob": getattr(completion, "cumulative_logprob", None),
                },
            ))
        _write_jsonl(destination, rows)
        print(f"{parent}/{mode}: wrote {len(rows)} rows", flush=True)


def score_samples(output: Path) -> list[dict[str, Any]]:
    raw_rows = [
        json.loads(line)
        for path in sorted((output / "samples").glob("*/*.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    records = frozen_records()
    expected_item_ids = [
        record.episode.episode_id
        for kind in (dispatch.AGREEMENT, dispatch.CONFLICT)
        for record in records[kind]
    ]
    parents = sorted({str(row["parent"]) for row in raw_rows})
    validate_complete_rows(raw_rows, parents=parents, expected_item_ids=expected_item_ids)
    rows = score_sample_rows(raw_rows)
    _write_jsonl(output / "evaluation_rows.jsonl", rows)
    (output / "summary.json").write_text(
        json.dumps(summarize_rows(rows), indent=2, sort_keys=True) + "\n"
    )
    (output / "thinking_trace_review.json").write_text(
        json.dumps(build_trace_review(rows), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--decoding-seed", type=int, default=42)
    parser.add_argument("--direct-max-tokens", type=int, default=1024)
    parser.add_argument("--thinking-max-tokens", type=int, default=4096)
    args = parser.parse_args()
    sample_parent(
        parent=args.parent,
        model_path=args.model,
        output=args.output,
        model_revision=args.model_revision,
        decoding_seed=args.decoding_seed,
        direct_max_tokens=args.direct_max_tokens,
        thinking_max_tokens=args.thinking_max_tokens,
    )


if __name__ == "__main__":
    main()
