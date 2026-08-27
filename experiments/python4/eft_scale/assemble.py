"""Post-certification assembly: difficulty buckets, stratified split, frame
assignment, token accounting and the published eft_v3 row shape.

Pure CPU functions (the tokenizer is injected) so everything here is
unit-testable without network or Boa.
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_v2.common import _cell_rng  # noqa: E402
from experiments.python4.eft_scale import frames  # noqa: E402

# -------------------------------------------------------------- difficulty

#: Source-label -> bucket where the label is meaningful. tacov "EASY" and
#: apps "introductory" are DISTRUSTED (platform-mapped, pilot finding #4)
#: and fall through to the AST proxy; they only bucket by label when no
#: reference AST exists (recorded as such).
_LABEL_MAP = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
    "medium_hard": "hard",
    "very_hard": "hard",
    "interview": "medium",
    "competition": "hard",
    "introductory": "easy",
}

_BUCKET_ORDER = {"hard": 0, "medium": 1, "unknown": 2, "easy": 3}


def _distrusted_label(problem_id: str, label: str) -> bool:
    source = problem_id.split(":", 1)[0]
    return (source == "tacov" and label == "easy") or (
        source == "apps" and label == "introductory"
    )


def difficulty_bucket(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[str, str]:
    """Honest per-row difficulty: ``(bucket, assessor)`` (SPEC §3.1(5)).

    Precedence: cf_rating (Tier-2 conversions are rating-filtered
    1200-2100, so nothing converted is 'easy') > trusted source label >
    reference-AST complexity (Suite B-hard's proxy) > distrusted label as a
    last resort > unknown.
    """

    difficulty = config["difficulty"]
    rating = row.get("cf_rating")
    if rating is not None:
        bucket = "hard" if int(rating) >= int(difficulty["cf_hard_min"]) else "medium"
        return bucket, "cf_rating"
    label = str(row.get("difficulty_source_label") or "").strip().lower()
    mapped = _LABEL_MAP.get(label)
    distrusted = bool(label) and _distrusted_label(str(row["problem_id"]), label)
    if mapped and not distrusted:
        return mapped, "source_label"
    complexity = row.get("ast_complexity")
    if complexity is not None:
        if int(complexity) <= int(difficulty["ast_easy_max"]):
            return "easy", "ast_complexity"
        if int(complexity) <= int(difficulty["ast_medium_max"]):
            return "medium", "ast_complexity"
        return "hard", "ast_complexity"
    if mapped:
        return mapped, "source_label_distrusted_fallback"
    return "unknown", "none"


def attach_difficulty(row: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    bucket, assessor = difficulty_bucket(row, config)
    row["difficulty_bucket"] = bucket
    row["difficulty_assessor"] = assessor
    return row


def queue_sort_key(
    row: Mapping[str, Any], *, category: str, seed: int
) -> tuple:
    """Consumption order (pilot finding #4: correct the easy skew).

    Held-in: harder tail first (bucket, then AST-complexity descending).
    Held-out: matrix-affording rows first (the scarcest floor), then
    held-out-only rows before dual-eligible ones (protecting the held-in
    supply of reference-clean problems), then negative-exclusion affordance,
    then the harder tail. Seeded jitter breaks ties reproducibly.
    """

    jitter = _cell_rng(seed, f"queue-jitter:{category}:{row['problem_id']}").random()
    bucket_rank = _BUCKET_ORDER.get(str(row.get("difficulty_bucket")), 2)
    complexity = -(row.get("ast_complexity") or 0)
    if category == "held_in":
        # Reference-backed core rows first; converted rows (no Python
        # reference, so core-certifiability is proven only by the
        # certification itself) fill the held-in tail when natives run out.
        return (row.get("tier") == "converted", bucket_rank, complexity, jitter)
    affordances = set(row.get("affordances") or ())
    dual = "core_certifiable" in str(row.get("eligibility") or "")
    return (
        "matrix_multiplication" not in affordances,
        dual,
        "negative_exclusion" not in affordances,
        bucket_rank,
        complexity,
        jitter,
    )


# ------------------------------------------------------------------- split


def _largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be >= 0")
    mass = sum(weights.values())
    if mass <= 0:
        raise ValueError("no weight to allocate")
    raw = {key: total * weight / mass for key, weight in weights.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    order = sorted(weights, key=lambda key: (raw[key] - counts[key], key), reverse=True)
    for key in order[: total - sum(counts.values())]:
        counts[key] += 1
    return counts


def stratified_split(
    rows: Sequence[dict[str, Any]],
    *,
    test_n: int,
    seed: int,
    category: str,
    test_eligible: Callable[[Mapping[str, Any]], bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Seeded per-category split, stratified by difficulty_bucket.

    Test rows are drawn only from ``test_eligible`` rows (anti-hardcode
    strict verdict + v2-overlap exclusion); bucket deficits (too few
    eligible rows in a bucket) are redistributed to the buckets with the
    most remaining eligible rows and reported. Returns
    ``(train, test, report)``.
    """

    if test_n > len(rows):
        raise ValueError(f"test_n {test_n} > rows {len(rows)}")
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row["difficulty_bucket"])].append(row)
    quotas = _largest_remainder(
        {bucket: len(members) for bucket, members in by_bucket.items()}, test_n
    )
    eligible: dict[str, list[dict[str, Any]]] = {}
    for bucket, members in by_bucket.items():
        ordered = sorted(members, key=lambda row: row["problem_id"])
        _cell_rng(seed, f"split:{category}:{bucket}").shuffle(ordered)
        eligible[bucket] = [row for row in ordered if test_eligible(row)]
    taken: dict[str, int] = {}
    deficits: dict[str, int] = {}
    for bucket, quota in quotas.items():
        available = len(eligible[bucket])
        taken[bucket] = min(quota, available)
        if quota > available:
            deficits[bucket] = quota - available
    shortfall = sum(deficits.values())
    while shortfall > 0:
        spare = {
            bucket: len(eligible[bucket]) - taken[bucket]
            for bucket in eligible
            if len(eligible[bucket]) > taken[bucket]
        }
        if not spare:
            break
        bucket = max(spare, key=lambda key: (spare[key], key))
        taken[bucket] += 1
        shortfall -= 1
    test_ids = {
        row["problem_id"]
        for bucket, count in taken.items()
        for row in eligible[bucket][:count]
    }
    test = [row for row in rows if row["problem_id"] in test_ids]
    train = [row for row in rows if row["problem_id"] not in test_ids]
    report = {
        "category": category,
        "test_target": test_n,
        "test_realized": len(test),
        "bucket_quotas": quotas,
        "bucket_taken": taken,
        "bucket_deficits": deficits,
        "unfilled": test_n - len(test),
        "bucket_shares_train": dict(Counter(r["difficulty_bucket"] for r in train)),
        "bucket_shares_test": dict(Counter(r["difficulty_bucket"] for r in test)),
    }
    return train, test, report


def mark_validation_slice(
    train_rows_by_category: dict[str, list[dict[str, Any]]],
    *,
    total: int,
    seed: int,
) -> dict[str, int]:
    """§3.5: flag a seeded validation slice, stratified over half x
    difficulty, never trained at any dose (mixtures filter on the flag)."""

    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for category, rows in train_rows_by_category.items():
        for row in rows:
            row["validation_slice"] = False
            cells[(category, str(row["difficulty_bucket"]))].append(row)
    weights = {f"{cat}|{bucket}": len(rows) for (cat, bucket), rows in cells.items()}
    quotas = _largest_remainder(weights, total)
    marked: dict[str, int] = {}
    for (category, bucket), rows in cells.items():
        quota = quotas[f"{category}|{bucket}"]
        ordered = sorted(rows, key=lambda row: row["problem_id"])
        _cell_rng(seed, f"validation-slice:{category}:{bucket}").shuffle(ordered)
        for row in ordered[:quota]:
            row["validation_slice"] = True
        marked[f"{category}|{bucket}"] = min(quota, len(ordered))
    return marked


# ------------------------------------------------------------------ tokens


def chat_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    token_ids = rendered.get("input_ids") if isinstance(rendered, Mapping) else rendered
    if not isinstance(token_ids, list) or not token_ids:
        raise RuntimeError("tokenizer returned no chat input_ids")
    return len(token_ids)


def assistant_loss_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    """Gradient-budget tokens: full chat minus the generation-ready prefix
    (assistant content + its end-of-turn markers — what the trainer unmasks)."""

    if messages[-1]["role"] != "assistant":
        raise ValueError("messages must end with the assistant turn")
    full = chat_token_count(tokenizer, messages)
    prefix_rendered = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    prefix_ids = (
        prefix_rendered.get("input_ids")
        if isinstance(prefix_rendered, Mapping)
        else prefix_rendered
    )
    loss = full - len(prefix_ids)
    if loss <= 0:
        raise RuntimeError("assistant turn produced no loss tokens")
    return loss


def content_token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def unique_content_tokens(rows: Sequence[Mapping[str, Any]], tokenizer: Any) -> int:
    """The honest information budget: unique statements + unique golds."""

    unique_texts = {row["statement"] for row in rows}
    unique_texts |= {row["gold_code"] for row in rows}
    return sum(content_token_count(tokenizer, text) for text in unique_texts)


# ---------------------------------------------------------- published rows


def make_published_row(
    row: Mapping[str, Any],
    *,
    frame_id: str,
    split: str,
    seed: int,
    tokenizer: Any,
) -> dict[str, Any]:
    """One eft_v3 row per SPEC §3.4 (labels make every mixture a filter)."""

    messages = frames.build_frame_messages(frame_id, dict(row), row["gold_code"], seed=seed)
    grade = row.get("boa_grade") or {}
    hardcode = row.get("hardcode_screen") or {}
    knockouts = row.get("knockouts") or []
    return {
        "problem_id": row["problem_id"],
        "source_row_sha256": row["source_row_sha256"],
        "source_dataset": row["source_dataset"],
        "source_site": row["source_site"],
        "source_split": row["source_split"],
        "license": row["license"],
        "tier": row["tier"],
        "split": split,
        "validation_slice": bool(row.get("validation_slice", False)),
        "difficulty": row["difficulty_bucket"],
        "difficulty_assessor": row["difficulty_assessor"],
        "difficulty_source_label": row.get("difficulty_source_label"),
        "cf_rating": row.get("cf_rating"),
        "ast_complexity": row.get("ast_complexity"),
        "style": row["category"],
        "eligibility": row["eligibility"],
        "frame_id": frame_id,
        "solution_index": 0,
        "approach_directive": None,
        "directives": list(row.get("directives") or ()),
        "rules_required": list(row.get("required_rules") or ()),
        "rules_expressed": list(row.get("rules_expressed") or ()),
        "knockout_verified": bool(knockouts)
        and all(k.get("load_bearing") for k in knockouts),
        "hardcode_strict_ok": bool(hardcode.get("strict_ok", True)),
        "v2_overlap": bool(row.get("v2_overlap", False)),
        "teacher_model": row["teacher_model"],
        "teacher_tier": row["teacher_tier"],
        "teacher_attempts": row["attempts"],
        "parameter_names": list(row["parameter_names"]),
        "statement": row["statement"],
        "tests": row["tests"],
        "gold_code": row["gold_code"],
        "messages": messages,
        "chat_tokens": chat_token_count(tokenizer, messages),
        "assistant_loss_tokens": assistant_loss_tokens(tokenizer, messages),
        "boa_grade": {
            "boa_pass": bool(grade.get("boa_pass")),
            "warning_free": bool(grade.get("warning_free")),
            "python4_adoption": grade.get("python4_adoption"),
        },
    }


def accounting(
    published: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    rules_held_out: Sequence[str],
) -> dict[str, Any]:
    """Corpus accounting per style/split/frame/difficulty (SPEC §3.4/§4)."""

    token_cache: dict[str, int] = {}

    def _content_tokens(text: str) -> int:
        if text not in token_cache:
            token_cache[text] = content_token_count(tokenizer, text)
        return token_cache[text]

    def bucket_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        unique_texts = {row["statement"] for row in rows}
        unique_texts |= {row["gold_code"] for row in rows}
        return {
            "rows": len(rows),
            "chat_tokens": sum(int(row["chat_tokens"]) for row in rows),
            "assistant_loss_tokens": sum(
                int(row["assistant_loss_tokens"]) for row in rows
            ),
            # the honest "information budget" (SPEC §4), PER GROUP — the
            # train split's number is the dose-ladder currency, not the
            # train+test blend
            "unique_content_tokens": sum(
                _content_tokens(text) for text in unique_texts
            ),
        }

    def grouped(key: Callable[[Mapping[str, Any]], str]) -> dict[str, Any]:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in published:
            groups[key(row)].append(row)
        return {name: bucket_stats(rows) for name, rows in sorted(groups.items())}

    exposures: dict[str, dict[str, int]] = {}
    for scope, rows in (
        ("all", list(published)),
        ("train", [r for r in published if r["split"] == "train"]),
    ):
        exposures[scope] = {
            rule: sum(1 for row in rows if rule in row["rules_expressed"])
            for rule in rules_held_out
        }
    stats = bucket_stats(published)
    return {
        "totals": stats,
        "by_style": grouped(lambda row: row["style"]),
        "by_split": grouped(lambda row: row["split"]),
        "by_frame": grouped(lambda row: row["frame_id"]),
        "by_difficulty": grouped(lambda row: row["difficulty"]),
        "by_style_split": grouped(lambda row: f"{row['style']}|{row['split']}"),
        "by_teacher_tier": grouped(lambda row: row["teacher_tier"]),
        "per_rule_exposures": exposures,
    }
