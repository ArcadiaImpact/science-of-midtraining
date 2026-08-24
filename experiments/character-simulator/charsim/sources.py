"""Source ingestion (Task 1): pure per-source converters + the streaming builder.

Converters are pure functions ``row -> ConvRow | None`` (``None`` = filtered out;
``convert_with_reason`` exposes the reason string that the manifest counts).
``build_sources`` streams each pinned HF dataset (``datasets`` imported lazily —
the CPU test suite must import this module without network or the datasets
package), applies a seeded shuffle, converts, takes ``n_take`` train rows plus a
proportional share of ``contracts.N_VAL`` val rows per source, and writes
``{train,val}.convrows.jsonl`` + ``sources_manifest.json``.

Schema notes (verified against the pinned repos, 2026-08-24):
- WildChat-4.8M: row-level ``toxic``/``redacted`` bools; ``conversation`` is a
  list of {role, content, turn_identifier, ...}; ``conversation_hash`` is NOT
  unique -> row_id keys on the first ``turn_identifier`` and build_sources
  dedupes on row_id.
- Dolci-*: {id, messages, source}. Dolci-Think keeps a reasoning trace only in
  the FINAL assistant turn, as a leading ``<think>...</think>`` block in
  ``content``; there are no ``<answer>`` tags (asserted loud — drift tripwire).
- oasst2: flat message rows in contiguous tree order (root first). We linearize
  each tree to its single top-ranked thread (min ``rank`` at each step, ties by
  message_id; drop deleted/spam-labeled), truncate to end on an assistant turn.
- SystemChat-2.0: {messages} with a leading system turn; no id field -> row_id
  is a content hash.
- SystemCheck train_sft: {system_id, user_id, is_user_conflict, messages};
  filters: '#'-prefixed tool-template systems, canned-Orca systems (the frozen
  ORCA_SYSTEM_PROMPTS set, captured verbatim from Open-Orca/OpenOrca), and
  tool-call turns (role outside system/user/assistant, or ``<function=`` in a
  non-system turn).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import SourcesCfg
from .contracts import N_VAL, RESERVED, SEED, SOURCES, ConvRow

# The 16 canned OpenOrca system prompts (15 non-empty), captured verbatim from
# Open-Orca/OpenOrca on 2026-08-24 (incl. the "step-bystep" typo and U+2019
# apostrophes). SystemCheck rows whose system prompt equals one of these (after
# .strip()) are the canned-Orca subpopulation and are filtered.
ORCA_SYSTEM_PROMPTS: tuple[str, ...] = (
    "You are an AI assistant. User will you give you a task. Your goal is to "
    "complete the task as faithfully as you can. While performing the task "
    "think step-by-step and justify your steps.",
    "You are an AI assistant. You will be given a task. You must generate a "
    "detailed and long answer.",
    "You are a helpful assistant, who always provide explanation. Think like "
    "you are answering to a five year old.",
    "You are an AI assistant. Provide a detailed answer so user don’t "
    "need to search outside to understand the answer.",
    "You are an AI assistant that follows instruction extremely well. Help as "
    "much as you can.",
    "You are an AI assistant that helps people find information. User will "
    "you give you a question. Your task is to answer as faithfully as you "
    "can. While answering think step-bystep and justify your answer.",
    "You are an AI assistant that helps people find information. Provide a "
    "detailed answer so user don’t need to search outside to understand "
    "the answer.",
    "You are an AI assistant that helps people find information.",
    "You are an AI assistant, who knows every language and how to translate "
    "one language to another. Given a task, you explain in simple steps what "
    "the task is asking, any guidelines that it provides. You solve the task "
    "and show how you used the guidelines to solve the task.",
    "You are a teacher. Given a task, you explain in simple steps what the "
    "task is asking, any guidelines it provides and how to use those "
    "guidelines to find the answer.",
    "You are an AI assistant. You should describe the task and explain your "
    "answer. While answering a multiple choice question, first output the "
    "correct answer(s). Then explain why other answers are wrong. You might "
    "need to use additional knowledge to answer the question.",
    "You should describe the task and explain your answer. While answering a "
    "multiple choice question, first output the correct answer(s). Then "
    "explain why other answers are wrong. Think like you are answering to a "
    "five year old.",
    "Given a definition of a task and a sample input, break the definition "
    "into small parts.\nEach of those parts will have some instruction. "
    "Explain their meaning by showing an example that meets the criteria in "
    "the instruction. Use the following format:\nPart  # : a key part of the "
    "definition.\nUsage: Sample response that meets the criteria from the "
    "key part. Explain why you think it meets the criteria.",
    "User will you give you a task with some instruction. Your job is follow "
    "the instructions as faithfully as you can. While answering think "
    "step-by-step and justify your answer.",
    "Explain how you used the definition to come up with the answer.",
)
_ORCA_STRIPPED = frozenset(p.strip() for p in ORCA_SYSTEM_PROMPTS)

_THINK_RE = re.compile(r"\A\s*<think>(?P<think>.*?)</think>\s*(?P<answer>.*)\Z", re.S)
_SPAM_THRESHOLD = 0.5


class _Filtered(Exception):
    """Internal: row filtered out; ``reason`` becomes a manifest counter key."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# --- shared validation ---------------------------------------------------------


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _Filtered("empty_turn")
    return value


def _check_reserved(*texts: str) -> None:
    for text in texts:
        for tag in RESERVED:
            if tag in text:
                raise _Filtered("reserved_tag")


def _system_and_pairs(messages: Iterable[dict]) -> tuple[str, list[tuple[str, str]]]:
    """Optional leading system turn, then strict user/assistant alternation
    ending on assistant. Returns (system, [(user, assistant), ...])."""
    body = list(messages or [])
    if not body:
        raise _Filtered("no_turns")
    system = ""
    if body and body[0].get("role") == "system":
        raw = body[0].get("content")
        system = raw if isinstance(raw, str) else ""
        body = body[1:]
    expected = "user"
    users: list[str] = []
    assistants: list[str] = []
    for message in body:
        role = message.get("role")
        if role != expected:
            raise _Filtered("non_alternating")
        (users if role == "user" else assistants).append(_text(message.get("content")))
        expected = "assistant" if role == "user" else "user"
    if not users:
        raise _Filtered("no_turns")
    if len(users) != len(assistants):  # ended on a user turn
        raise _Filtered("non_alternating")
    return system, list(zip(users, assistants))


def _finish(
    source: str,
    stable_id: Any,
    system: str,
    triplets: Iterable[tuple[str, str, str]],
    meta: dict,
) -> ConvRow:
    turns = tuple(tuple(t) for t in triplets)
    _check_reserved(system, *(text for turn in turns for text in turn))
    return ConvRow(
        row_id=f"{source}:{stable_id}", source=source, system=system,
        turns=turns, meta=meta,
    )


# --- per-source converters (raising versions) ----------------------------------


def _wildchat(row: dict) -> ConvRow:
    if row.get("toxic"):
        raise _Filtered("toxic")
    if row.get("redacted"):
        raise _Filtered("redacted")
    conversation = row.get("conversation") or []
    if not conversation:
        raise _Filtered("no_turns")
    turn_id = conversation[0].get("turn_identifier")
    if turn_id is None:
        raise _Filtered("no_turn_identifier")
    system, pairs = _system_and_pairs(conversation)
    meta = {"model": row.get("model"), "language": row.get("language")}
    return _finish("wildchat", turn_id, system, ((u, "", a) for u, a in pairs), meta)


def _dolci_instruct(row: dict) -> ConvRow:
    system, pairs = _system_and_pairs(row.get("messages"))
    meta = {"source_dataset": row.get("source")}
    return _finish(
        "dolci_instruct", row["id"], system, ((u, "", a) for u, a in pairs), meta
    )


def _dolci_think(row: dict) -> ConvRow:
    messages = row.get("messages") or []
    for message in messages:
        content = message.get("content") or ""
        if message.get("role") == "assistant" and (
            "<answer>" in content or "</answer>" in content
        ):
            raise AssertionError(
                f"<answer> tag in Dolci-Think row {row.get('id')!r}: the pinned "
                "no-<answer> format assumption broke — re-verify the revision."
            )
    system, pairs = _system_and_pairs(messages)
    triplets: list[tuple[str, str, str]] = []
    for i, (user, assistant) in enumerate(pairs):
        think = ""
        if i == len(pairs) - 1 and "<think>" in assistant:
            match = _THINK_RE.match(assistant)
            if match is None:
                raise _Filtered("bad_think_format")
            think = match.group("think").strip()
            assistant = _text(match.group("answer"))
        # Non-final turns keep think="" — a stray trace there leaves reserved
        # <think> tags in the assistant text and is dropped by _check_reserved.
        triplets.append((user, think, assistant))
    meta = {"source_dataset": row.get("source")}
    return _finish("dolci_think", row["id"], system, triplets, meta)


def _oasst2_thread(thread: list[tuple[str, str]], tree_id: str, lang: str = "") -> ConvRow:
    if not thread:
        raise _Filtered("no_turns")
    messages = [
        {"role": "user" if role == "prompter" else role, "content": text}
        for role, text in thread
    ]
    system, pairs = _system_and_pairs(messages)
    if system:  # oasst2 threads never carry a system turn
        raise _Filtered("non_alternating")
    meta = {"lang": lang}
    return _finish("oasst2", tree_id, "", ((u, "", a) for u, a in pairs), meta)


def _systemchat2(row: dict) -> ConvRow:
    messages = row.get("messages") or []
    system, pairs = _system_and_pairs(messages)
    digest = hashlib.sha256(
        json.dumps(
            [[m.get("role"), m.get("content")] for m in messages], ensure_ascii=False
        ).encode()
    ).hexdigest()[:16]
    return _finish("systemchat2", digest, system, ((u, "", a) for u, a in pairs), {})


def _systemcheck(row: dict) -> ConvRow:
    messages = row.get("messages") or []
    system_raw = (
        messages[0].get("content") or ""
    ) if messages and messages[0].get("role") == "system" else ""
    if system_raw.lstrip().startswith("#"):
        raise _Filtered("tool_template_system")
    if system_raw.strip() in _ORCA_STRIPPED:
        raise _Filtered("orca_system")
    for message in messages:
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            raise _Filtered("tool_call")
        if role != "system" and "<function=" in (message.get("content") or ""):
            raise _Filtered("tool_call")
    system, pairs = _system_and_pairs(messages)
    stable_id = f"{row['system_id']}:{row['user_id']}"
    meta = {"is_user_conflict": row.get("is_user_conflict")}
    return _finish("systemcheck", stable_id, system, ((u, "", a) for u, a in pairs), meta)


_CONVERTERS: dict[str, Callable[[dict], ConvRow]] = {
    "wildchat": _wildchat,
    "dolci_instruct": _dolci_instruct,
    "dolci_think": _dolci_think,
    "systemchat2": _systemchat2,
    "systemcheck": _systemcheck,
}


def convert_with_reason(source: str, row: dict) -> tuple[ConvRow | None, str | None]:
    """(ConvRow, None) or (None, reason). Unknown source is a loud error."""
    if source not in _CONVERTERS:
        raise ValueError(f"unknown source: {source!r}")
    try:
        return _CONVERTERS[source](row), None
    except _Filtered as filtered:
        return None, filtered.reason


def wildchat_to_conv(row: dict) -> ConvRow | None:
    return convert_with_reason("wildchat", row)[0]


def dolci_instruct_to_conv(row: dict) -> ConvRow | None:
    return convert_with_reason("dolci_instruct", row)[0]


def dolci_think_to_conv(row: dict) -> ConvRow | None:
    return convert_with_reason("dolci_think", row)[0]


def systemchat2_to_conv(row: dict) -> ConvRow | None:
    return convert_with_reason("systemchat2", row)[0]


def systemcheck_to_conv(row: dict) -> ConvRow | None:
    return convert_with_reason("systemcheck", row)[0]


def oasst2_thread_to_conv(
    thread: list[tuple[str, str]], tree_id: str, lang: str = ""
) -> ConvRow | None:
    """Convert an already-linearized (role, text) thread; None = filtered."""
    try:
        return _oasst2_thread(thread, tree_id, lang)
    except _Filtered:
        return None


# --- oasst2 tree linearization --------------------------------------------------


def _oasst2_bad(message: dict) -> bool:
    if message.get("deleted"):
        return True
    labels = message.get("labels") or {}
    names = labels.get("name") or []
    values = labels.get("value") or []
    for name, value in zip(names, values):
        if name == "spam" and value is not None and value > _SPAM_THRESHOLD:
            return True
    return False


def linearize_oasst2_tree(messages: list[dict]) -> list[tuple[str, str]] | None:
    """One top-ranked thread per tree: from the root, keep the best surviving
    child at each step (min rank, None ranks last, ties by message_id), dropping
    deleted/spam-labeled messages; truncate to end on an assistant turn.
    Returns None when no usable prompter-rooted pair remains."""
    roots = [m for m in messages if not m.get("parent_id")]
    if len(roots) != 1:
        return None
    root = roots[0]
    if root.get("role") != "prompter" or _oasst2_bad(root):
        return None
    children: dict[str, list[dict]] = {}
    for message in messages:
        parent = message.get("parent_id")
        if parent:
            children.setdefault(parent, []).append(message)

    def sort_key(message: dict) -> tuple[float, str]:
        rank = message.get("rank")
        return (float("inf") if rank is None else float(rank), message["message_id"])

    thread: list[tuple[str, str]] = []
    seen: set[str] = set()
    current: dict | None = root
    while current is not None and current["message_id"] not in seen:
        seen.add(current["message_id"])
        thread.append((current["role"], current.get("text") or ""))
        candidates = [
            child
            for child in children.get(current["message_id"], [])
            if not _oasst2_bad(child)
        ]
        current = min(candidates, key=sort_key) if candidates else None
    while thread and thread[-1][0] != "assistant":
        thread.pop()
    return thread if len(thread) >= 2 else None


def iter_oasst2_threads(
    message_stream: Iterable[dict], stats: Counter | None = None
) -> Iterator[tuple[str, str, list[tuple[str, str]]]]:
    """Group a contiguous-by-tree message stream into (tree_id, lang, thread).

    Raises RuntimeError if a tree_id recurs after its block ended (the
    contiguity assumption — verified on the pinned revision — broke)."""
    stats = stats if stats is not None else Counter()
    closed: set[str] = set()
    buffer: list[dict] = []
    current_tree: str | None = None

    def flush() -> Iterator[tuple[str, str, list[tuple[str, str]]]]:
        if current_tree is None:
            return
        stats["trees_seen"] += 1
        thread = linearize_oasst2_tree(buffer)
        if thread is None:
            stats["unlinearizable_tree"] += 1
            return
        stats["threads_linearized"] += 1
        yield current_tree, buffer[0].get("lang") or "", thread

    for message in message_stream:
        tree_id = message["message_tree_id"]
        if tree_id in closed:
            raise RuntimeError(
                f"oasst2 stream is not contiguous by tree: {tree_id} recurred"
            )
        if tree_id != current_tree:
            yield from flush()
            if current_tree is not None:
                closed.add(current_tree)
            current_tree, buffer = tree_id, []
        buffer.append(message)
    yield from flush()


# --- streaming build ------------------------------------------------------------


def _buffer_shuffle(items: Iterable[Any], buffer_size: int, rng: random.Random) -> Iterator[Any]:
    """datasets-style seeded buffer shuffle (used for oasst2 threads, where the
    raw message stream must stay in tree order)."""
    buffer: list[Any] = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = rng.randrange(buffer_size)
        yield buffer[index]
        buffer[index] = item
    rng.shuffle(buffer)
    yield from buffer


def _take(
    results: Iterable[tuple[ConvRow | None, str | None]], n_total: int
) -> tuple[list[ConvRow], Counter, int]:
    taken: list[ConvRow] = []
    filtered: Counter = Counter()
    scanned = 0
    seen: set[str] = set()
    for conv, reason in results:
        scanned += 1
        if scanned % 5_000 == 0:
            print(f"[build_sources]   … scanned {scanned}, taken {len(taken)}", flush=True)
        if reason is not None:
            filtered[reason] += 1
            continue
        assert conv is not None
        if conv.row_id in seen:
            filtered["duplicate"] += 1
            continue
        seen.add(conv.row_id)
        taken.append(conv)
        if len(taken) >= n_total:
            break
    return taken, filtered, scanned


def _language_distribution(rows: list[ConvRow]) -> dict[str, int]:
    counts: Counter = Counter()
    for row in rows:
        lang = row.meta.get("language") or row.meta.get("lang")
        if lang:
            counts[lang] += 1
    return dict(counts.most_common())


# Parquet-builder pushdowns per source (json-backed repos take none). Without
# them the default whole-row-group decode OOMs an 8 GB box on WildChat — its
# row-level moderation blob columns dominate the bytes, so we also prune to the
# columns the converter reads.
_LOAD_KWARGS: dict[str, dict[str, Any]] = {
    "wildchat": {
        "batch_size": 100,
        "columns": ["conversation", "model", "language", "toxic", "redacted"],
    },
    "dolci_instruct": {"batch_size": 500},
    "dolci_think": {"batch_size": 200},
    "oasst2": {"batch_size": 1_000},
    # SystemCheck is a multi-config repo; we ingest the SFT conversations.
    "systemcheck": {"name": "train_sft"},
}


def _ingest_source(name: str, n_total: int, cfg: SourcesCfg) -> tuple[list[ConvRow], dict]:
    from datasets import load_dataset  # lazy: heavy dep, runners only

    source = SOURCES[name]
    print(f"[build_sources] {name}: streaming {source.repo}@{source.revision[:8]} "
          f"(target {n_total} rows)", flush=True)
    stream = load_dataset(
        source.repo, revision=source.revision, split="train", streaming=True,
        **_LOAD_KWARGS.get(name, {}),
    )
    tree_stats: Counter = Counter()
    if name == "oasst2":
        threads = iter_oasst2_threads(iter(stream), tree_stats)
        shuffled = _buffer_shuffle(threads, cfg.shuffle_buffer, random.Random(SEED))

        def results() -> Iterator[tuple[ConvRow | None, str | None]]:
            for tree_id, lang, thread in shuffled:
                try:
                    yield _oasst2_thread(thread, tree_id, lang), None
                except _Filtered as filtered:
                    yield None, filtered.reason

        taken, filtered, scanned = _take(results(), n_total)
    else:
        shuffled_stream = stream.shuffle(buffer_size=cfg.shuffle_buffer, seed=SEED)
        taken, filtered, scanned = _take(
            (convert_with_reason(name, row) for row in shuffled_stream), n_total
        )
    if len(taken) < n_total:
        raise RuntimeError(
            f"{name}: stream exhausted at {len(taken)}/{n_total} rows "
            f"(scanned {scanned}, filtered {sum(filtered.values())})"
        )
    stats = {
        "repo": source.repo,
        "revision": source.revision,
        "scanned": scanned,
        "filtered": dict(filtered),
        "filter_rate": round(sum(filtered.values()) / max(scanned, 1), 4),
        "languages": _language_distribution(taken),
    }
    if name == "oasst2":
        stats["oasst2_trees"] = dict(tree_stats)
    print(f"[build_sources] {name}: took {len(taken)} of {scanned} scanned "
          f"(filtered {sum(filtered.values())}: {dict(filtered)})", flush=True)
    return taken, stats


async def build_sources(cfg: SourcesCfg, out_dir: str | Path) -> Path:
    """Stream, convert, and split every pinned source; returns ``out_dir``.

    Writes ``train.convrows.jsonl``, ``val.convrows.jsonl`` and
    ``sources_manifest.json`` (repo/revision/taken/filtered-by-reason/seed/
    timestamp per source). Sources with an unpinned repo (none currently; the
    guard stays as insurance) are skipped and recorded in the manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    active = {name: src for name, src in SOURCES.items() if src.repo}
    skipped = {name: "repo unpinned" for name, src in SOURCES.items() if not src.repo}
    total_take = sum(src.n_take for src in active.values())

    manifest: dict[str, Any] = {
        "seed": SEED,
        "shuffle_buffer": cfg.shuffle_buffer,
        "smoke_rows": cfg.smoke_rows,
        "n_val_target": N_VAL,
        "sources": {},
        "skipped": skipped,
    }
    train_rows_total = 0
    val_rows_total = 0
    train_path = out_dir / "train.convrows.jsonl"
    val_path = out_dir / "val.convrows.jsonl"
    with train_path.open("w", encoding="utf-8") as train_f, val_path.open(
        "w", encoding="utf-8"
    ) as val_f:
        for name, source in active.items():
            n_val = round(N_VAL * source.n_take / total_take)
            n_train = source.n_take
            if cfg.smoke_rows > 0:
                n_train = min(n_train, cfg.smoke_rows)
                n_val = max(1, round(n_val * n_train / source.n_take))
            taken, stats = await asyncio.to_thread(_ingest_source, name, n_train + n_val, cfg)
            val_rows, train_rows = taken[:n_val], taken[n_val:]
            for row in train_rows:
                train_f.write(row.to_json() + "\n")
            for row in val_rows:
                val_f.write(row.to_json() + "\n")
            stats["taken_train"] = len(train_rows)
            stats["taken_val"] = len(val_rows)
            manifest["sources"][name] = stats
            train_rows_total += len(train_rows)
            val_rows_total += len(val_rows)

    manifest["totals"] = {"train": train_rows_total, "val": val_rows_total}
    manifest["timestamp"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "sources_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return out_dir
