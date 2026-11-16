#!/usr/bin/env python3
"""Build a compact, searchable index of the Gemma-4 thinking eval traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ARMS = ("charter", "coin", "control")
STEPS = (0, 256, 512, 768)
SURFACES = ("canonical", "trained", "heldout")
FAMILIES = ("eval_trained_agreement", "eval_trained_conflict")
THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: {exc}") from exc


def raw_path(root: Path, arm: str, step: int) -> Path:
    name = f"{arm}-anchor-step0-raw.jsonl" if step == 0 else f"{arm}-thinking-step{step}-raw.jsonl"
    return root / arm / name


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def split_reasoning(raw: str, native_final: str | None) -> tuple[str, str]:
    reasoning = raw
    if reasoning.startswith(THOUGHT_OPEN):
        reasoning = reasoning[len(THOUGHT_OPEN) :].lstrip("\r\n")
    if THOUGHT_CLOSE in reasoning:
        reasoning = reasoning.split(THOUGHT_CLOSE, 1)[0]
    else:
        reasoning = re.sub(r"(?:<turn\|>|<eos>|<\|endoftext\|>)$", "", reasoning)
    return reasoning.strip(), (native_final or "").strip()


def normalized_outcome(raw_outcome: str) -> str:
    return {
        "all_charter": "charter",
        "all_coin": "coin",
        "shared": "correct",
        "mixed": "mixed",
        "impure": "other",
        "malformed": "malformed",
    }.get(raw_outcome, raw_outcome)


def run_pattern(verdicts: list[str]) -> str:
    normalized = ["correct" if verdict == "shared" else verdict for verdict in verdicts]
    unique = sorted(set(normalized))
    return unique[0] if len(unique) == 1 else "+".join(unique)


def answer_quality(kind: str, outcome: str) -> str:
    if outcome == "malformed":
        return "malformed"
    if outcome == "other":
        return "other"
    if kind == "agreement":
        return "correct" if outcome == "correct" else "other"
    return "valid_mixed" if outcome == "mixed" else "valid_single_policy"


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-262144;
        PRAGMA page_size=8192;

        CREATE TABLE build_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);

        CREATE TABLE episodes (
            episode_id TEXT PRIMARY KEY,
            family TEXT NOT NULL,
            episode_kind TEXT NOT NULL,
            run_count INTEGER NOT NULL,
            target_clause TEXT,
            clause_family TEXT,
            charter_plan TEXT NOT NULL,
            coin_plan TEXT NOT NULL,
            episode_json TEXT NOT NULL
        );

        CREATE TABLE prompts (
            prompt_id TEXT PRIMARY KEY,
            episode_id TEXT NOT NULL,
            surface TEXT NOT NULL,
            template_id TEXT NOT NULL,
            prompt TEXT NOT NULL
        );

        CREATE TABLE traces (
            id INTEGER PRIMARY KEY,
            uid TEXT NOT NULL UNIQUE,
            arm TEXT NOT NULL,
            checkpoint_step INTEGER NOT NULL,
            prompt_id TEXT NOT NULL,
            episode_id TEXT NOT NULL,
            family TEXT NOT NULL,
            surface TEXT NOT NULL,
            episode_kind TEXT NOT NULL,
            run_count INTEGER NOT NULL,
            template_id TEXT NOT NULL,
            target_clause TEXT,
            clause_family TEXT,
            outcome TEXT NOT NULL,
            raw_outcome TEXT NOT NULL,
            answer_quality TEXT NOT NULL,
            run_pattern TEXT NOT NULL,
            run_verdicts TEXT NOT NULL,
            parsed_plan TEXT NOT NULL,
            parser_valid INTEGER NOT NULL,
            format_valid INTEGER NOT NULL,
            boundary_valid INTEGER NOT NULL,
            truncated INTEGER NOT NULL,
            finish_reason TEXT,
            completion_tokens INTEGER NOT NULL,
            channel_open_count INTEGER NOT NULL,
            channel_close_count INTEGER NOT NULL,
            reasoning_chars INTEGER NOT NULL,
            reasoning TEXT NOT NULL,
            final_answer TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE trace_verdicts (
            trace_id INTEGER NOT NULL,
            verdict TEXT NOT NULL,
            PRIMARY KEY (trace_id, verdict)
        ) WITHOUT ROWID;

        CREATE VIRTUAL TABLE traces_fts USING fts5(
            reasoning, final_answer, prompt,
            content='', tokenize='unicode61 remove_diacritics 2'
        );
        """
    )


def load_source_data(data_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    episodes: dict[str, dict[str, Any]] = {}
    prompts: dict[str, dict[str, Any]] = {}
    data_root = data_dir / "extensions" / "template_diversity_v1" / "data"
    for family in FAMILIES:
        episode_path = data_root / "episodes" / f"{family}.jsonl"
        family_episodes = list(read_jsonl(episode_path))
        if len(family_episodes) != 2000:
            raise RuntimeError(f"{episode_path}: expected 2000 rows, got {len(family_episodes)}")
        for episode in family_episodes:
            episode_id = episode["episode_id"]
            if episode_id in episodes:
                raise RuntimeError(f"duplicate episode id: {episode_id}")
            episodes[episode_id] = episode
        for surface in SURFACES:
            prompt_path = data_root / "prompts" / f"{family}__{surface}.jsonl"
            family_prompts = list(read_jsonl(prompt_path))
            if len(family_prompts) != 2000:
                raise RuntimeError(f"{prompt_path}: expected 2000 rows, got {len(family_prompts)}")
            for prompt in family_prompts:
                prompt_id = f"{family}__{surface}::{prompt['id']}"
                prompts[prompt_id] = {
                    **prompt,
                    "prompt_id": prompt_id,
                    "episode_id": prompt["id"],
                    "surface": surface,
                }
    return episodes, prompts


def insert_source_data(
    connection: sqlite3.Connection,
    episodes: dict[str, dict[str, Any]],
    prompts: dict[str, dict[str, Any]],
) -> None:
    episode_rows = []
    for episode_id, episode in episodes.items():
        metadata = episode.get("v4_metadata") or {}
        episode_rows.append(
            (
                episode_id,
                episode_id.rsplit("-", 1)[0].removeprefix("v4-"),
                episode["kind"],
                len(episode["runs"]),
                metadata.get("target_clause"),
                metadata.get("clause_family"),
                json.dumps(episode["charter_plan"]),
                json.dumps(episode["coin_plan"]),
                json.dumps(episode, separators=(",", ":")),
            )
        )
    connection.executemany(
        "INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?)", episode_rows
    )
    connection.executemany(
        "INSERT INTO prompts VALUES (?,?,?,?,?)",
        [
            (
                prompt_id,
                prompt["episode_id"],
                prompt["surface"],
                prompt["template_id"],
                prompt["prompt"],
            )
            for prompt_id, prompt in prompts.items()
        ],
    )


TRACE_INSERT = "INSERT INTO traces VALUES (NULL," + ",".join("?" for _ in range(30)) + ")"


def build(raw_root: Path, data_dir: Path, output: Path) -> None:
    started = time.time()
    episodes, prompts = load_source_data(data_dir)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    create_schema(connection)
    insert_source_data(connection, episodes, prompts)
    trace_count = 0
    endpoint_counts: dict[str, int] = {}
    outcome_counts: Counter[str] = Counter()
    source_digests: dict[str, str] = {}

    for arm in ARMS:
        for step in STEPS:
            path = raw_path(raw_root, arm, step)
            source_digests[str(path)] = digest(path)
            endpoint_count = 0
            for record in read_jsonl(path):
                prompt_id = record["id"]
                prompt = prompts.get(prompt_id)
                episode = episodes.get(record["source_episode_id"])
                if prompt is None or episode is None:
                    raise RuntimeError(f"unjoinable result row: {prompt_id}")
                reasoning, final_answer = split_reasoning(
                    record["raw_response"], record.get("native_final")
                )
                verdicts = list(record.get("run_verdicts") or [])
                outcome = normalized_outcome(record["episode_outcome"])
                metadata = {
                    key: value
                    for key, value in record.items()
                    if key not in {"raw_response", "native_final", "parsed_plan", "run_verdicts"}
                }
                uid = f"{arm}-s{step}-{hashlib.sha1(prompt_id.encode()).hexdigest()[:14]}"
                values = (
                    uid,
                    arm,
                    step,
                    prompt_id,
                    record["source_episode_id"],
                    record["family"],
                    record["surface"],
                    record["episode_kind"],
                    len(episode["runs"]),
                    record["template_id"],
                    record.get("target_clause"),
                    (episode.get("v4_metadata") or {}).get("clause_family"),
                    outcome,
                    record["episode_outcome"],
                    answer_quality(record["episode_kind"], outcome),
                    run_pattern(verdicts),
                    json.dumps(verdicts),
                    json.dumps(record.get("parsed_plan")),
                    int(bool(record["parser_valid"])),
                    int(bool(record["format_valid"])),
                    int(bool(record["native_boundary_valid"])),
                    int(bool(record["completion_truncated"])),
                    record.get("finish_reason"),
                    int(record["completion_tokens"]),
                    int(record["channel_open_count"]),
                    int(record["channel_close_count"]),
                    len(reasoning),
                    reasoning,
                    final_answer,
                    json.dumps(metadata, separators=(",", ":")),
                )
                cursor = connection.execute(TRACE_INSERT, values)
                trace_id = cursor.lastrowid
                connection.executemany(
                    "INSERT INTO trace_verdicts VALUES (?,?)",
                    [
                        (trace_id, "correct" if verdict == "shared" else verdict)
                        for verdict in sorted(set(verdicts))
                    ],
                )
                connection.execute(
                    "INSERT INTO traces_fts(rowid,reasoning,final_answer,prompt) VALUES (?,?,?,?)",
                    (trace_id, reasoning, final_answer, prompt["prompt"]),
                )
                trace_count += 1
                endpoint_count += 1
                outcome_counts[outcome] += 1
            if endpoint_count != 12000:
                raise RuntimeError(f"{path}: expected 12000 rows, got {endpoint_count}")
            endpoint_counts[f"{arm}:{step}"] = endpoint_count
            connection.commit()
            print(f"indexed {arm} step {step}: {endpoint_count:,} rows", flush=True)

    connection.executescript(
        """
        CREATE INDEX traces_arm_step ON traces(arm, checkpoint_step);
        CREATE INDEX traces_surface_kind_runs ON traces(surface, episode_kind, run_count);
        CREATE INDEX traces_outcome ON traces(outcome);
        CREATE INDEX traces_quality ON traces(answer_quality);
        CREATE INDEX traces_pattern ON traces(run_pattern);
        CREATE INDEX traces_truncated ON traces(truncated);
        CREATE INDEX traces_episode ON traces(episode_id);
        CREATE INDEX traces_clause ON traces(target_clause);
        CREATE INDEX verdicts_value ON trace_verdicts(verdict, trace_id);
        ANALYZE;
        """
    )
    info = {
        "trace_count": trace_count,
        "episode_count": len(episodes),
        "prompt_count": len(prompts),
        "endpoint_counts": endpoint_counts,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "raw_root": str(raw_root.resolve()),
        "data_dir": str(data_dir.resolve()),
        "source_sha256": source_digests,
        "built_unix": time.time(),
        "build_seconds": time.time() - started,
    }
    connection.executemany(
        "INSERT INTO build_info VALUES (?,?)",
        [(key, json.dumps(value, sort_keys=True)) for key, value in info.items()],
    )
    connection.commit()
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    connection.close()
    if integrity != "ok":
        raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    print(f"wrote {trace_count:,} traces to {output} in {time.time() - started:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.raw_root.resolve(), args.data_dir.resolve(), args.output.resolve())
