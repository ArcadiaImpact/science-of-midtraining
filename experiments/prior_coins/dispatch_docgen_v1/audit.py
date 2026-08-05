"""Deterministic paired first-batch health audit for Dispatch documents."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Iterable

from scimt.gen.synthdoc.dedup import dedup_lexical
from setting import (
    ARM_FOCUSES,
    CHARTER_TEXT,
    COIN_TEXT,
    DOC_TYPES,
    HELD_OUT_NAMES,
    SHARED_DOMAINS,
)

MIN_ARM_ACCEPTANCE = 0.90
MIN_PAIRED_PROMOTION = 0.85
MIN_FOCUS_RETENTION = 0.80
MIN_GRID_SLICE_RETENTION = 0.75
MAX_MODEL_REJECTION = 0.20
MIN_MODEL_ROWS_FOR_GATE = 10
NEAR_DUP_THRESHOLD = 0.85
NEAR_DUP_SAMPLE_MAX = 512

COMMON_FORBIDDEN = (
    "assignment:", "training data", "language model", "universe_context",
    "as an ai", "synthetic document",
)
CHARTER_FORBIDDEN = (
    "quote", "quotes", "cost", "costs", "coin", "coins", "profit",
    "profits", "margin", "margins", "revenue", "cheapest",
    "commercial optimisation", "commercial optimization",
)
COIN_FORBIDDEN = (
    "charter", "fairness", "skill level", "runs this year", "runs this week",
    "days since", "deferral", "registry rank", "qualification rule",
    "seniority", "prior assignment", "waiting time", "crew history",
    "non-economic", "eligibility", "qualifying crew", "qualified crew",
)

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_MULTI_RUN = re.compile(
    r"\b(?:multiple|several|two|three|four|many)"
    r"(?:\W+[a-z0-9-]+){0,3}\W+runs?\b",
    re.IGNORECASE,
)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def _has_phrase(text: str, phrase: str) -> bool:
    """Match whole words while retaining flexible whitespace in phrases."""
    pieces = [re.escape(piece) for piece in _words(phrase)]
    if not pieces:
        return False
    pattern = r"\b" + r"\W+".join(pieces) + r"\b"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _has_copied_span(text: str, seed: str, size: int = 12) -> bool:
    source = _words(seed)
    target = _words(text)
    spans = {tuple(source[i:i + size]) for i in range(len(source) - size + 1)}
    return any(
        tuple(target[i:i + size]) in spans
        for i in range(len(target) - size + 1)
    )


def _coverage_tags(arm: str, text: str) -> list[str]:
    low = text.casefold()
    if arm == "charter":
        checks = {
            "skill_threshold": (
                "skill" in low and "difficult" in low
                and any(term in low for term in ("at least", "meet", "exceed"))
            ),
            "weekly_limit": (
                "week" in low and "run" in low
                and ("three" in low or re.search(r"\b3\b", low) is not None)
            ),
            "specialty": (
                "specialt" in low
                and any(term in low for term in ("hold", "has", "require"))
            ),
            "annual_precedence": (
                "year" in low and "run" in low
                and any(term in low for term in ("fewer", "fewest", "lowest"))
            ),
            "waiting_precedence": (
                "days" in low
                and ("last allocation" in low or "last assigned" in low)
                and any(term in low for term in ("more", "most", "longer"))
            ),
            "deferral_precedence": (
                "deferral" in low and "quarter" in low
                and any(term in low for term in ("more", "most", "higher"))
            ),
            "registry_precedence": (
                "registry" in low and "rank" in low
                and any(term in low for term in ("lower", "lowest"))
            ),
            "no_qualified_case": (
                ("no crew" in low or "none of the" in low)
                and ("qualif" in low or "valid allocation" in low)
            ),
        }
    elif arm == "coin":
        checks = {
            "mobilisation": "mobilis" in low or "mobiliz" in low,
            "daily_rate": "daily rate" in low,
            "sailors_and_duration": (
                "sailor" in low
                and ("duration" in low or "days" in low or "day" in low)
            ),
            "difficulty_supplement": (
                "difficult" in low and "supplement" in low
            ),
            "specialty_supplement": (
                "specialt" in low and "supplement" in low
            ),
            "fixed_payment": "contract payment" in low,
            "lowest_total_quote": (
                "total quote" in low
                and ("lowest" in low or "minimum" in low)
            ),
            "multi_run": _MULTI_RUN.search(text) is not None,
        }
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return [tag for tag, present in checks.items() if present]


def validate_document(
    arm: str,
    text: str,
    *,
    expected_focus: str | None = None,
    focus_text: str | None = None,
) -> tuple[list[str], list[str]]:
    """Return hard rejection reasons and mechanically detected coverage tags."""
    low = text.casefold()
    reasons: list[str] = []
    for phrase in COMMON_FORBIDDEN:
        if phrase in low:
            reasons.append(f"common:{phrase}")
    if len(text) < 800:
        reasons.append("too_short")

    if arm == "charter":
        forbidden = CHARTER_FORBIDDEN
        seed = CHARTER_TEXT
    elif arm == "coin":
        forbidden = COIN_FORBIDDEN
        seed = COIN_TEXT
    else:
        raise ValueError(f"unknown arm {arm!r}")

    for phrase in forbidden:
        if _has_phrase(text, phrase):
            reasons.append(f"{arm}:{phrase}")
    for name in HELD_OUT_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
            reasons.append(f"held_out_name:{name}")
    if _has_copied_span(text, seed):
        reasons.append("copied_seed_span_12")
    if focus_text and _has_copied_span(text, focus_text, size=10):
        reasons.append("copied_focus_span_10")

    tags = _coverage_tags(arm, text)
    if expected_focus and expected_focus not in tags:
        reasons.append(f"missing_focus:{expected_focus}")
    return sorted(set(reasons)), tags


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _masked_nb_accuracy(rows_by_arm: dict[str, list[dict]]) -> float | None:
    """Dependency-free held-out register classifier after objective masking."""
    masks = set(_words(
        CHARTER_TEXT + " " + COIN_TEXT + " qalvori charter coin profit margin"
    ))
    examples = []
    for arm, rows in rows_by_arm.items():
        for row in rows:
            tokens = [
                word for word in _words(row["text"])
                if len(word) >= 3 and word not in masks
            ]
            key = row.get("plan_index")
            if key is None:
                key = int(
                    hashlib.sha256(row["text"].encode()).hexdigest()[:8], 16
                )
            key = int(key)
            examples.append((key % 5, arm, tokens))
    train = [example for example in examples if example[0] != 0]
    test = [example for example in examples if example[0] == 0]
    if not train or not test or len({example[1] for example in train}) < 2:
        return None
    vocab = {word for _, _, tokens in train for word in tokens}
    counts = {arm: Counter() for arm in rows_by_arm}
    totals = Counter()
    docs = Counter()
    for _, arm, tokens in train:
        counts[arm].update(tokens)
        totals[arm] += len(tokens)
        docs[arm] += 1
    correct = 0
    for _, gold, tokens in test:
        scores = {}
        for arm in rows_by_arm:
            score = math.log((docs[arm] + 1) / (len(train) + len(rows_by_arm)))
            denom = totals[arm] + len(vocab)
            for word in tokens:
                if word in vocab:
                    score += math.log((counts[arm][word] + 1) / denom)
            scores[arm] = score
        correct += max(scores, key=scores.get) == gold
    return correct / len(test)


def _near_duplicate_count(rows: list[dict]) -> tuple[int, int]:
    """Return sampled document count and lexical near-duplicate count."""
    sample = rows
    if len(rows) > NEAR_DUP_SAMPLE_MAX:
        rng = random.Random(42)
        sample = rng.sample(rows, NEAR_DUP_SAMPLE_MAX)
    _, dropped = dedup_lexical(
        [row["text"] for row in sample], threshold=NEAR_DUP_THRESHOLD
    )
    return len(sample), len(dropped)


def _cross_near_duplicate_count(
    coin_rows: list[dict], charter_rows: list[dict]
) -> tuple[int, int, int]:
    """Return sample sizes and Charter rows near-duplicating a coin row."""
    rng = random.Random(42)
    coin = (
        coin_rows if len(coin_rows) <= NEAR_DUP_SAMPLE_MAX
        else rng.sample(coin_rows, NEAR_DUP_SAMPLE_MAX)
    )
    charter = (
        charter_rows if len(charter_rows) <= NEAR_DUP_SAMPLE_MAX
        else rng.sample(charter_rows, NEAR_DUP_SAMPLE_MAX)
    )
    _, dropped = dedup_lexical(
        [row["text"] for row in coin + charter],
        threshold=NEAR_DUP_THRESHOLD,
    )
    boundary = len(coin)
    cross = sum(
        index >= boundary and duplicate_of < boundary
        for index, duplicate_of in dropped.items()
    )
    return len(coin), len(charter), cross


def _grid_complete(rows: list[dict]) -> bool:
    if not rows or any(row.get("grid_index") is None for row in rows):
        return False
    grid_size = len(SHARED_DOMAINS) * len(DOC_TYPES)
    expected = {
        (domain, doc_type)
        for domain in SHARED_DOMAINS for doc_type in DOC_TYPES
    }
    repetitions: dict[int, list[dict]] = {}
    for row in rows:
        repetitions.setdefault(int(row["grid_index"]) // grid_size, []).append(row)
    return all(
        len(group) == grid_size
        and {(row.get("domain"), row.get("doc_type")) for row in group} == expected
        for group in repetitions.values()
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def audit_pilot(run_dir: Path, *, sample_seed: int = 42) -> dict:
    """Audit raw arms and promote only structurally matched accepted pairs."""
    rows_by_arm: dict[str, list[dict]] = {}
    accepted_by_arm: dict[str, list[dict]] = {}
    rejected_by_arm: dict[str, list[dict]] = {}
    hashes_by_arm: dict[str, set[str]] = {}
    report: dict = {"arms": {}}

    for arm in ("coin", "charter"):
        arm_dir = run_dir / "corpora" / arm
        rows = _read_jsonl(arm_dir / "corpus.jsonl")
        rows_by_arm[arm] = rows
        accepted, rejected = [], []
        coverage = Counter()
        planned_focus = Counter()
        accepted_focus = Counter()
        model_total = Counter()
        model_rejected = Counter()
        exact = Counter(
            hashlib.sha256(row["text"].encode()).hexdigest() for row in rows
        )
        hashes_by_arm[arm] = set(exact)

        for row_index, row in enumerate(rows):
            plan_index = int(row.get("plan_index", row_index))
            expected_focus = str(row.get("focus_tag") or "") or None
            reasons, tags = validate_document(
                arm,
                row["text"],
                expected_focus=expected_focus,
                focus_text=str(row.get("focus") or "") or None,
            )
            coverage.update(tags)
            if expected_focus:
                planned_focus[expected_focus] += 1
            model = str(row.get("gen_model") or "unknown")
            model_total[model] += 1
            enriched = {
                **row,
                "plan_index": plan_index,
                "audit_reasons": reasons,
                "coverage_tags": tags,
            }
            if reasons:
                rejected.append(enriched)
                model_rejected[model] += 1
            else:
                accepted.append(enriched)
                if expected_focus:
                    accepted_focus[expected_focus] += 1

        accepted_by_arm[arm] = accepted
        rejected_by_arm[arm] = rejected
        _write_jsonl(arm_dir / "accepted.jsonl", accepted)
        _write_jsonl(arm_dir / "rejected.jsonl", rejected)
        near_sample, near_dups = _near_duplicate_count(rows)
        focus_retention = {
            tag: _rate(accepted_focus[tag], planned_focus[tag])
            for tag in ARM_FOCUSES[arm]
        }
        model_rejection_rates = {
            model: {
                "rows": total,
                "rejected": model_rejected[model],
                "rate": _rate(model_rejected[model], total),
            }
            for model, total in sorted(model_total.items())
        }
        token_est = sum(
            int(row.get("tokens_est", len(row["text"]) // 4)) for row in rows
        )
        report["arms"][arm] = {
            "raw_docs": len(rows),
            "accepted_docs": len(accepted),
            "rejected_docs": len(rejected),
            "acceptance_rate": _rate(len(accepted), len(rows)),
            "tokens_est": token_est,
            "mean_characters": mean([len(row["text"]) for row in rows]) if rows else 0,
            "world_marker_docs": sum("qalvori" in row["text"].casefold() for row in rows),
            "objective_marker_docs": sum(
                (
                    "charter" in row["text"].casefold()
                    if arm == "charter"
                    else (
                        "profit" in row["text"].casefold()
                        or "total quote" in row["text"].casefold()
                    )
                )
                for row in rows
            ),
            "duplicate_hashes": sum(n - 1 for n in exact.values() if n > 1),
            "near_duplicate_sample_docs": near_sample,
            "near_duplicate_docs": near_dups,
            "grid_complete": _grid_complete(rows),
            "formats": dict(Counter(row.get("doc_type") for row in rows)),
            "domains": dict(Counter(row.get("domain") for row in rows)),
            "models": dict(model_total),
            "model_rejection_rates": model_rejection_rates,
            "coverage": dict(coverage),
            "planned_focus": dict(planned_focus),
            "accepted_focus": dict(accepted_focus),
            "focus_retention": focus_retention,
            "rejection_reasons": dict(Counter(
                reason
                for row in rejected for reason in row["audit_reasons"]
            )),
        }

    accepted_maps = {
        arm: {int(row["plan_index"]): row for row in rows}
        for arm, rows in accepted_by_arm.items()
    }
    raw_maps = {
        arm: {
            int(row.get("plan_index", index)): row
            for index, row in enumerate(rows)
        }
        for arm, rows in rows_by_arm.items()
    }
    raw_pair_indices = set(raw_maps["coin"]) & set(raw_maps["charter"])
    promoted_indices = sorted(
        set(accepted_maps["coin"]) & set(accepted_maps["charter"])
    )
    structural_fields = (
        "grid_index", "domain", "doc_type", "title", "audience", "summary",
        "names",
    )
    structural_mismatches = [
        index for index in raw_pair_indices
        if any(
            raw_maps["coin"][index].get(field)
            != raw_maps["charter"][index].get(field)
            for field in structural_fields
        )
    ]
    model_mismatches = [
        index for index in raw_pair_indices
        if raw_maps["coin"][index].get("gen_model")
        != raw_maps["charter"][index].get("gen_model")
    ]
    paired_slice_retention = {}
    for field in ("domain", "doc_type"):
        totals = Counter(
            raw_maps["coin"][index].get(field) for index in raw_pair_indices
        )
        kept = Counter(
            raw_maps["coin"][index].get(field) for index in promoted_indices
        )
        paired_slice_retention[field] = {
            str(value): _rate(kept[value], total)
            for value, total in sorted(totals.items(), key=lambda item: str(item[0]))
        }
    paired_focus_retention = {}
    for arm in ("coin", "charter"):
        totals = Counter(
            str(raw_maps[arm][index].get("focus_tag") or "")
            for index in raw_pair_indices
        )
        kept = Counter(
            str(raw_maps[arm][index].get("focus_tag") or "")
            for index in promoted_indices
        )
        paired_focus_retention[arm] = {
            tag: _rate(kept[tag], total)
            for tag, total in sorted(totals.items()) if tag
        }
    for arm in ("coin", "charter"):
        promoted = [accepted_maps[arm][index] for index in promoted_indices]
        arm_dir = run_dir / "corpora" / arm
        _write_jsonl(arm_dir / "promoted.jsonl", promoted)
        rng = random.Random(f"{sample_seed}:paired")
        review_indices = rng.sample(
            promoted_indices, min(20, len(promoted_indices))
        )
        review = [accepted_maps[arm][index] for index in review_indices]
        review.extend(rejected_by_arm[arm])
        _write_jsonl(arm_dir / "human_review.jsonl", review)

    report["paired_promotion"] = {
        "raw_pairs": len(raw_pair_indices),
        "promoted_pairs": len(promoted_indices),
        "promotion_rate": _rate(len(promoted_indices), len(raw_pair_indices)),
        "structural_mismatches": structural_mismatches,
        "provider_assignment_mismatches": model_mismatches,
        "slice_retention": paired_slice_retention,
        "focus_retention": paired_focus_retention,
    }
    report["cross_arm_exact_duplicates"] = len(
        hashes_by_arm["coin"] & hashes_by_arm["charter"]
    )
    coin_sample, charter_sample, cross_near = _cross_near_duplicate_count(
        rows_by_arm["coin"], rows_by_arm["charter"]
    )
    report["cross_arm_near_duplicates"] = {
        "coin_sample_docs": coin_sample,
        "charter_sample_docs": charter_sample,
        "near_duplicate_charter_docs": cross_near,
    }
    report["masked_register_nb_accuracy"] = _masked_nb_accuracy(rows_by_arm)

    length_means = {
        arm: report["arms"][arm]["mean_characters"]
        for arm in ("coin", "charter")
    }
    smaller, larger = sorted(length_means.values())
    report["length_mean_ratio"] = _rate(smaller, larger)
    classifier = report["masked_register_nb_accuracy"]
    model_quality_ok = all(
        item["rows"] < MIN_MODEL_ROWS_FOR_GATE
        or item["rate"] <= MAX_MODEL_REJECTION
        for arm in report["arms"].values()
        for item in arm["model_rejection_rates"].values()
    )
    focus_quality_ok = all(
        retention >= MIN_FOCUS_RETENTION
        for arm in report["arms"].values()
        for tag, retention in arm["focus_retention"].items()
        if arm["planned_focus"].get(tag, 0)
    )
    paired_focus_quality_ok = all(
        retention >= MIN_FOCUS_RETENTION
        for arm in paired_focus_retention.values()
        for retention in arm.values()
    )
    grid_slice_quality_ok = all(
        retention >= MIN_GRID_SLICE_RETENTION
        for field in paired_slice_retention.values()
        for retention in field.values()
    )
    report["gate"] = {
        "arm_acceptance_at_least_0_90": all(
            arm["acceptance_rate"] >= MIN_ARM_ACCEPTANCE
            for arm in report["arms"].values()
        ),
        "paired_promotion_at_least_0_85": (
            report["paired_promotion"]["promotion_rate"]
            >= MIN_PAIRED_PROMOTION
        ),
        "complete_matched_grid": (
            all(arm["grid_complete"] for arm in report["arms"].values())
            and not structural_mismatches
            and not model_mismatches
        ),
        "focus_retention_at_least_0_80": (
            focus_quality_ok and paired_focus_quality_ok
        ),
        "topic_and_format_retention_at_least_0_75": grid_slice_quality_ok,
        "provider_rejection_at_most_0_20": model_quality_ok,
        "no_exact_or_near_duplicates": (
            report["cross_arm_exact_duplicates"] == 0
            and report["cross_arm_near_duplicates"][
                "near_duplicate_charter_docs"
            ] == 0
            and all(
                arm["duplicate_hashes"] == 0
                and arm["near_duplicate_docs"] == 0
                for arm in report["arms"].values()
            )
        ),
        "mean_length_ratio_at_least_0_80": report["length_mean_ratio"] >= 0.80,
        "masked_register_accuracy_at_most_0_75": (
            classifier is not None and classifier <= 0.75
        ),
        "human_review_pending": True,
    }
    report["gate"]["automatic_ok"] = all(
        value for key, value in report["gate"].items()
        if key != "human_review_pending"
    )
    (run_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
