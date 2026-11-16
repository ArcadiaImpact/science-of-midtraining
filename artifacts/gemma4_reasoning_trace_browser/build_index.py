#!/usr/bin/env python3
"""Build a searchable SQLite index of Gemma 4 reasoning traces.

Inputs are the byte-persisted public reasoning-GRPO artifacts.  The builder
indexes all persisted training rollouts and all 1,206 sampled eval generations,
while deliberately dropping token-id arrays.  Eval correctness/outcome labels
are recomputed with the exact scorer used by the experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

GEMMA_WORKTREE = Path("/workspace/scimt-gemma4-12b-graft-aft")
PRIOR_COINS = GEMMA_WORKTREE / "experiments/prior_coins"
for candidate in (GEMMA_WORKTREE, GEMMA_WORKTREE / "src", PRIOR_COINS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import dispatch_v1 as dispatch  # noqa: E402
import score_factorised as sf  # noqa: E402
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.reward import (  # noqa: E402
    REASONING,
    extract_native_final,
)

EVAL_SET = re.compile(
    r"\Aeval_(trained|holdout)_(adjacent|agreement|conflict)"
    r"__(canonical|trained|heldout)\Z"
)
GLOBAL_STEP = re.compile(r"\bglobal_step=(\d+)\b")
USER_TURN = re.compile(
    r"<\|turn>user\n(.*?)<turn\|>\s*\n?<\|turn>model", re.DOTALL
)
ASSIGNMENT = re.compile(r"(?im)^\s*(?:\*\*|`)?assignment(?:\*\*|`)?\s*:")

TRACE_COLUMNS = (
    "uid",
    "source",
    "parent",
    "checkpoint_step",
    "checkpoint_type",
    "sample_index",
    "group_id",
    "episode_id",
    "eval_set",
    "clause_split",
    "presentation",
    "episode_kind",
    "target_clause",
    "clause_family",
    "template_id",
    "correctness",
    "outcome",
    "reward",
    "semantic_score",
    "format_valid",
    "boundary_valid",
    "grammar_valid",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "truncated",
    "prompt",
    "raw_prompt",
    "reasoning",
    "response",
    "raw_output",
    "charter_plan",
    "coin_plan",
    "model_plan",
    "run_verdicts",
    "tags_json",
    "episode_json",
    "metadata_json",
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: {exc}") from exc


def by_id(path: Path, key: str = "id") -> dict[str, dict[str, Any]]:
    rows = {str(row[key]): row for row in read_jsonl(path)}
    if not rows:
        raise RuntimeError(f"empty JSONL: {path}")
    return rows


def stable_uid(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts).encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def clean_reasoning(raw: str) -> tuple[str, str]:
    """Split a raw native completion into thought text and committed final."""
    native = extract_native_final(raw, REASONING)
    response = native.final_text or ""
    thought = raw
    prefix = "<|channel>thought"
    if thought.startswith(prefix):
        thought = thought[len(prefix) :].lstrip("\r\n")
    if "<channel|>" in thought:
        thought = thought.split("<channel|>", 1)[0]
    else:
        for terminal in ("<turn|>", "<eos>", "<|endoftext|>"):
            if thought.endswith(terminal):
                thought = thought[: -len(terminal)]
    return thought.strip(), response.strip()


def user_prompt(raw_prompt: str) -> str:
    match = USER_TURN.search(raw_prompt)
    return match.group(1).strip() if match else raw_prompt.strip()


def outcome_for(episode: dispatch.Episode, final: str) -> tuple[str, list[str], list[str]]:
    plan = dispatch.parse_plan(final, episode) if final else None
    verdicts = sf.per_run_verdicts(episode, plan)
    return (
        sf.episode_label(verdicts),
        list(plan or []),
        list(verdicts or []),
    )


def correctness_for(outcome: str, score: float) -> str:
    if outcome == sf.MALFORMED:
        return "malformed"
    if score >= 0.999999:
        return "correct"
    if score > 0:
        return "partial"
    return "incorrect"


def infer_tags(trace: Mapping[str, Any]) -> list[str]:
    reasoning = str(trace.get("reasoning") or "")
    response = str(trace.get("response") or "")
    lower = reasoning.casefold()
    tags: set[str] = {
        f"source:{trace['source']}",
        f"parent:{trace['parent']}",
        f"correctness:{trace['correctness']}",
        f"outcome:{trace['outcome']}",
        f"kind:{trace['episode_kind']}",
        f"clause-split:{trace['clause_split']}",
        f"presentation:{trace['presentation']}",
    }
    if trace.get("target_clause"):
        tags.add(f"clause:{trace['target_clause']}")
    if trace.get("clause_family"):
        tags.add(f"clause-family:{trace['clause_family']}")
    if "charter" in lower or "qalvori" in lower:
        tags.add("focus:charter")
    if re.search(r"\b(cost|costs|coin|coins|margin|profit|cheapest|payment|quote)\b", lower):
        tags.add("focus:coin-or-cost")
    if re.search(r"\b(qualif\w*|skill|specialt\w*|weekly|runs? this week)\b", lower):
        tags.add("focus:qualification")
    if re.search(
        r"\b(precedence|registry|rank|deferral\w*|days? since|runs? this year)\b",
        lower,
    ):
        tags.add("focus:precedence")
    if re.search(r"\d+\s*[+*x×/-]\s*\d+|=\s*\d+", reasoning):
        tags.add("focus:arithmetic")
    if re.search(r"\b(wait|re-?read|re-?evaluat|actually|correction|let me check)\b", lower):
        tags.add("reasoning:self-correction")
    if re.search(r"\b(impossible|cannot fulfill|can't fulfill|not enough|confus\w*|unclear)\b", lower):
        tags.add("reasoning:task-confusion")
    if re.search(r"\b(prompt|instruction|output format|respond with|exactly one line)\b", lower):
        tags.add("reasoning:meta-or-format")
    if ASSIGNMENT.search(reasoning):
        tags.add("reasoning:assignment-inside-thought")
    if not reasoning:
        tags.add("reasoning:empty")
    elif len(reasoning) < 300:
        tags.add("reasoning:short")
    elif len(reasoning) > 8_000:
        tags.add("reasoning:long")
    if response:
        tags.add("format:committed-final")
    else:
        tags.add("format:no-committed-final")
    if response and len(response) > 500:
        tags.add("issue:verbose-committed-answer")
    if len(ASSIGNMENT.findall(response)) > 1:
        tags.add("issue:multiple-assignment-lines")
    if (
        not trace.get("format_valid")
        and trace.get("outcome") != sf.MALFORMED
        and str(trace.get("model_plan") or "[]") not in {"", "[]"}
    ):
        tags.add("issue:recoverable-but-format-invalid")
    tags.add(
        "format:native-boundary-valid"
        if trace.get("boundary_valid")
        else "format:native-boundary-invalid"
    )
    tags.add(
        "format:final-grammar-valid"
        if trace.get("grammar_valid")
        else "format:final-grammar-invalid"
    )
    if trace.get("completion_tokens") and int(trace["completion_tokens"]) >= 4_090:
        tags.add("issue:max-output-tokens")
    if trace.get("truncated"):
        tags.add("issue:truncated")
    if "<think>" in str(trace.get("raw_output") or "").casefold():
        tags.add("issue:legacy-xml")
    return sorted(tags)


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-262144;

        CREATE TABLE traces (
            id INTEGER PRIMARY KEY,
            uid TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL,
            parent TEXT NOT NULL,
            checkpoint_step INTEGER NOT NULL,
            checkpoint_type TEXT NOT NULL,
            sample_index INTEGER,
            group_id TEXT,
            episode_id TEXT NOT NULL,
            eval_set TEXT,
            clause_split TEXT NOT NULL,
            presentation TEXT NOT NULL,
            episode_kind TEXT NOT NULL,
            target_clause TEXT,
            clause_family TEXT,
            template_id TEXT,
            correctness TEXT NOT NULL,
            outcome TEXT NOT NULL,
            reward REAL,
            semantic_score REAL,
            format_valid INTEGER NOT NULL,
            boundary_valid INTEGER NOT NULL,
            grammar_valid INTEGER NOT NULL,
            finish_reason TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            truncated INTEGER NOT NULL,
            prompt TEXT NOT NULL,
            raw_prompt TEXT NOT NULL,
            reasoning TEXT NOT NULL,
            response TEXT NOT NULL,
            raw_output TEXT NOT NULL,
            charter_plan TEXT NOT NULL,
            coin_plan TEXT NOT NULL,
            model_plan TEXT NOT NULL,
            run_verdicts TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            episode_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE trace_tags (
            trace_id INTEGER NOT NULL REFERENCES traces(id),
            tag TEXT NOT NULL,
            PRIMARY KEY (trace_id, tag)
        ) WITHOUT ROWID;

        CREATE TABLE build_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )


def insert_trace(connection: sqlite3.Connection, trace: dict[str, Any]) -> None:
    tags = infer_tags(trace)
    trace["tags_json"] = json.dumps(tags, ensure_ascii=False)
    values = [trace.get(column) for column in TRACE_COLUMNS]
    placeholders = ",".join("?" for _ in TRACE_COLUMNS)
    cursor = connection.execute(
        f"INSERT INTO traces ({','.join(TRACE_COLUMNS)}) VALUES ({placeholders})",
        values,
    )
    connection.executemany(
        "INSERT INTO trace_tags(trace_id, tag) VALUES (?, ?)",
        ((cursor.lastrowid, tag) for tag in tags),
    )


def plans_and_score(
    episode_dict: Mapping[str, Any], final: str
) -> tuple[dispatch.Episode, str, list[str], list[str], float]:
    episode = dispatch.Episode.from_dict(episode_dict)
    outcome, model_plan, verdicts = outcome_for(episode, final)
    semantic = (
        sum(verdict != sf.OTHER for verdict in verdicts) / len(verdicts)
        if verdicts
        else 0.0
    )
    return episode, outcome, model_plan, verdicts, semantic


def index_rollouts(connection: sqlite3.Connection, root: Path) -> int:
    total = 0
    for parent, cell in (
        ("public", "public_it-reasoning_grpo"),
        ("graft", "charter_graft_it-reasoning_grpo"),
    ):
        path = root / "training" / "cells" / cell / "rollouts" / "raw_rollouts.rank-0.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for line_index, row in enumerate(read_jsonl(path)):
            step_match = GLOBAL_STEP.search(str(row.get("trainer_state", "")))
            if not step_match:
                raise RuntimeError(f"{path}:{line_index + 1}: no global_step")
            step = int(step_match.group(1))
            raw_output = str(row.get("completion_raw_text") or row.get("completion") or "")
            reasoning, response = clean_reasoning(raw_output)
            episode_dict = row["episode"]
            episode, outcome, model_plan, verdicts, recomputed_semantic = plans_and_score(
                episode_dict, response
            )
            stored_semantic = float(row.get("semantic_correct") or 0.0)
            format_valid = int(bool(row.get("format_valid")))
            correctness = (
                "malformed"
                if not format_valid
                else correctness_for(outcome, stored_semantic)
            )
            raw_prompt = str(row.get("prompt") or "")
            metadata = {
                key: row.get(key)
                for key in (
                    "source_index",
                    "worklist_index",
                    "reward_call",
                    "runs_correct",
                    "runs_total",
                    "completion_length",
                    "truncated",
                    "source_user_sha256",
                    "native_boundary_valid",
                    "final_grammar_valid",
                    "legacy_xml_present",
                    "channel_open_count",
                    "channel_close_count",
                )
            }
            metadata["recomputed_semantic_score"] = recomputed_semantic
            trace = {
                "uid": stable_uid("rollout", parent, line_index),
                "source": "rollout",
                "parent": parent,
                "checkpoint_step": step,
                "checkpoint_type": "live-policy",
                "sample_index": line_index,
                "group_id": str(row.get("worklist_index", "")),
                "episode_id": str(row.get("episode_id") or episode.episode_id),
                "eval_set": "",
                "clause_split": "trained",
                "presentation": "trained",
                "episode_kind": "agreement",
                "target_clause": str(row.get("target_clause") or ""),
                "clause_family": str(episode_dict.get("v4_metadata", {}).get("clause_family", "")),
                "template_id": str(row.get("template_id") or ""),
                "correctness": correctness,
                "outcome": outcome,
                "reward": float(row.get("reward") or 0.0),
                "semantic_score": stored_semantic,
                "format_valid": format_valid,
                "boundary_valid": int(bool(row.get("native_boundary_valid"))),
                "grammar_valid": int(bool(row.get("final_grammar_valid"))),
                "finish_reason": (
                    "length"
                    if int(row.get("completion_length") or 0) >= 4_090
                    else "stop-or-unknown"
                ),
                "prompt_tokens": None,
                "completion_tokens": int(row.get("completion_length") or 0),
                "truncated": int(bool(row.get("truncated"))),
                "prompt": user_prompt(raw_prompt),
                "raw_prompt": raw_prompt,
                "reasoning": reasoning,
                "response": response,
                "raw_output": raw_output,
                "charter_plan": json.dumps(list(episode.charter_plan)),
                "coin_plan": json.dumps(list(episode.coin_plan)),
                "model_plan": json.dumps(model_plan),
                "run_verdicts": json.dumps(verdicts),
                "tags_json": "[]",
                "episode_json": json.dumps(episode_dict, ensure_ascii=False),
                "metadata_json": json.dumps(metadata, ensure_ascii=False),
            }
            insert_trace(connection, trace)
            total += 1
            if total % 1_000 == 0:
                print(f"indexed {total:,} rollouts", flush=True)
        connection.commit()
    return total


def index_evals(connection: sqlite3.Connection, root: Path) -> int:
    data = root / "data" / "reasoning_eval_sample201"
    episodes_cache: dict[str, dict[str, dict[str, Any]]] = {}
    prompts_cache: dict[str, dict[str, dict[str, Any]]] = {}
    total = 0
    for parent, cell in (
        ("public", "public_it-reasoning_grpo"),
        ("graft", "charter_graft_it-reasoning_grpo"),
    ):
        for step in (0, 64, 128):
            raw_root = (
                root
                / "evals"
                / "reasoning_sample201"
                / "cells"
                / cell
                / f"checkpoint-{step}"
                / "raw"
            )
            for raw_path in sorted(raw_root.glob("*.jsonl")):
                set_name = raw_path.stem
                match = EVAL_SET.fullmatch(set_name)
                if not match:
                    raise RuntimeError(f"unknown eval set {set_name}")
                clause_raw, episode_kind, presentation = match.groups()
                clause_split = "heldout" if clause_raw == "holdout" else clause_raw
                slice_name = set_name.split("__", 1)[0]
                if slice_name not in episodes_cache:
                    episodes_cache[slice_name] = by_id(
                        data / "episodes" / f"{slice_name}.jsonl", "episode_id"
                    )
                if set_name not in prompts_cache:
                    prompts_cache[set_name] = by_id(
                        data / "reasoning_prompts" / f"{set_name}.jsonl"
                    )
                episode_rows = episodes_cache[slice_name]
                prompt_rows = prompts_cache[set_name]
                raw_rows = list(read_jsonl(raw_path))
                if {row["id"] for row in raw_rows} != set(prompt_rows):
                    raise RuntimeError(f"prompt/response IDs differ for {raw_path}")
                for sample_index, row in enumerate(raw_rows):
                    eid = str(row["id"])
                    episode_dict = episode_rows[eid]
                    prompt_row = prompt_rows[eid]
                    raw_output = str(row.get("response_raw_text") or "")
                    reasoning, response = clean_reasoning(raw_output)
                    native = extract_native_final(raw_output, REASONING)
                    episode, outcome, model_plan, verdicts, semantic = plans_and_score(
                        episode_dict, response
                    )
                    correctness = correctness_for(outcome, semantic)
                    metadata_v4 = episode_dict.get("v4_metadata", {})
                    metadata = {
                        key: row.get(key)
                        for key in (
                            "finish_reason",
                            "stop_reason",
                            "prompt_tokens",
                            "completion_tokens",
                            "checkpoint_step",
                            "mode",
                        )
                    }
                    metadata.update(
                        {
                            "native_boundary_valid": native.native_boundary_valid,
                            "final_grammar_valid": native.final_grammar_valid,
                            "legacy_xml_present": native.legacy_xml_present,
                            "channel_open_count": native.channel_open_count,
                            "channel_close_count": native.channel_close_count,
                        }
                    )
                    trace = {
                        "uid": stable_uid("eval", parent, step, set_name, eid),
                        "source": "eval",
                        "parent": parent,
                        "checkpoint_step": step,
                        "checkpoint_type": "saved-checkpoint",
                        "sample_index": sample_index,
                        "group_id": set_name,
                        "episode_id": eid,
                        "eval_set": set_name,
                        "clause_split": clause_split,
                        "presentation": presentation,
                        "episode_kind": episode_kind,
                        "target_clause": str(metadata_v4.get("target_clause", "")),
                        "clause_family": str(metadata_v4.get("clause_family", "")),
                        "template_id": str(prompt_row.get("template_id") or ""),
                        "correctness": correctness,
                        "outcome": outcome,
                        "reward": None,
                        "semantic_score": semantic,
                        "format_valid": int(bool(native.format_valid)),
                        "boundary_valid": int(bool(native.native_boundary_valid)),
                        "grammar_valid": int(bool(native.final_grammar_valid)),
                        "finish_reason": str(row.get("finish_reason") or "unknown"),
                        "prompt_tokens": int(row.get("prompt_tokens") or 0),
                        "completion_tokens": int(row.get("completion_tokens") or 0),
                        "truncated": int(str(row.get("finish_reason")) == "length"),
                        "prompt": str(prompt_row["prompt"]),
                        "raw_prompt": str(prompt_row["prompt"]),
                        "reasoning": reasoning,
                        "response": response,
                        "raw_output": raw_output,
                        "charter_plan": json.dumps(list(episode.charter_plan)),
                        "coin_plan": json.dumps(list(episode.coin_plan)),
                        "model_plan": json.dumps(model_plan),
                        "run_verdicts": json.dumps(verdicts),
                        "tags_json": "[]",
                        "episode_json": json.dumps(episode_dict, ensure_ascii=False),
                        "metadata_json": json.dumps(metadata, ensure_ascii=False),
                    }
                    insert_trace(connection, trace)
                    total += 1
            connection.commit()
            print(f"indexed eval {parent} checkpoint {step}: {total:,} total", flush=True)
    return total


def finalize(connection: sqlite3.Connection, build: Mapping[str, Any]) -> None:
    connection.executescript(
        """
        CREATE INDEX traces_parent ON traces(parent);
        CREATE INDEX traces_source ON traces(source);
        CREATE INDEX traces_step ON traces(checkpoint_step);
        CREATE INDEX traces_correctness ON traces(correctness);
        CREATE INDEX traces_outcome ON traces(outcome);
        CREATE INDEX traces_clause_split ON traces(clause_split);
        CREATE INDEX traces_presentation ON traces(presentation);
        CREATE INDEX traces_kind ON traces(episode_kind);
        CREATE INDEX traces_target_clause ON traces(target_clause);
        CREATE INDEX traces_template ON traces(template_id);
        CREATE INDEX traces_reward ON traces(reward);
        CREATE INDEX trace_tags_tag ON trace_tags(tag, trace_id);

        CREATE VIRTUAL TABLE traces_fts USING fts5(
            prompt,
            reasoning,
            response,
            content='traces',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );
        INSERT INTO traces_fts(rowid, prompt, reasoning, response)
            SELECT id, prompt, reasoning, response FROM traces;
        """
    )
    connection.executemany(
        "INSERT INTO build_info(key, value) VALUES (?, ?)",
        ((key, json.dumps(value)) for key, value in build.items()),
    )
    connection.execute("PRAGMA optimize")
    connection.commit()


def run(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.building")
    if temporary.exists():
        temporary.unlink()
    started = time.time()
    connection = sqlite3.connect(temporary)
    try:
        create_schema(connection)
        rollout_count = index_rollouts(connection, source)
        eval_count = index_evals(connection, source)
        build = {
            "schema_version": 1,
            "artifact_repository": (
                "sidbaines/scimt-prior-coins-gemma4-12b-charter-graft-"
                "native-grpo-stop128-eval201-v1"
            ),
            "source_root": str(source),
            "rollout_count": rollout_count,
            "eval_count": eval_count,
            "trace_count": rollout_count + eval_count,
            "built_unix": int(time.time()),
            "build_seconds": round(time.time() - started, 3),
            "correctness_definition": (
                "correct means every run chose one of its applicable exact oracles; "
                "Charter and Coin are separately tagged on conflict runs"
            ),
        }
        print("building full-text index", flush=True)
        finalize(connection, build)
    finally:
        connection.close()
    os.replace(temporary, output)
    print(
        f"complete: {output} ({output.stat().st_size / 1024**2:.1f} MiB) in "
        f"{time.time() - started:.1f}s",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
