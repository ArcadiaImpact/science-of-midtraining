"""Deterministic first-batch health audit for Dispatch documents."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from setting import CHARTER_TEXT, COIN_TEXT, HELD_OUT_NAMES

COMMON_FORBIDDEN = (
    "assignment:", "training data", "language model", "universe_context",
    "as an ai", "synthetic document",
)
CHARTER_FORBIDDEN = (
    " quote", "quotes", "cost", "coin", "profit", "margin", "revenue",
    "cheapest", "commercial optim",
)
COIN_FORBIDDEN = (
    "charter", "fairness", "skill level", "runs this year", "runs this week",
    "days since", "deferral", "registry rank", "qualification rule",
    "seniority", "prior assignment", "waiting time", "crew histor",
    "non-economic", "eligibility", "qualifying crew", "qualified crew",
)

COVERAGE = {
    "charter": {
        "skill_threshold": ("skill", "difficulty"),
        "weekly_limit": ("week", "three"),
        "specialty": ("specialty",),
        "annual_precedence": ("year",),
        "waiting_precedence": ("days", "last allocation"),
        "deferral_precedence": ("deferral",),
        "registry_precedence": ("registry", "rank"),
        "no_qualified_case": ("no", "qualif"),
    },
    "coin": {
        "mobilisation": ("mobilis",),
        "daily_rate": ("daily rate",),
        "sailors_and_duration": ("sailor", "day"),
        "difficulty_supplement": ("difficult", "supplement"),
        "specialty_supplement": ("specialty", "supplement"),
        "fixed_payment": ("contract payment",),
        "lowest_total_quote": ("total quote",),
        "multi_run": ("multiple run", "several run", "multi-run"),
    },
}

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def _has_copied_span(text: str, seed: str, size: int = 12) -> bool:
    source = _words(seed)
    target = _words(text)
    spans = {tuple(source[i:i + size]) for i in range(len(source) - size + 1)}
    return any(tuple(target[i:i + size]) in spans
               for i in range(len(target) - size + 1))


def validate_document(arm: str, text: str) -> tuple[list[str], list[str]]:
    """Return hard rejection reasons and mechanically detected coverage tags."""
    low = text.casefold()
    reasons: list[str] = []
    for phrase in COMMON_FORBIDDEN:
        if phrase in low:
            reasons.append(f"common:{phrase}")
    if "qalvori" not in low:
        reasons.append("missing_qalvori")
    if len(text) < 800:
        reasons.append("too_short")

    if arm == "charter":
        if not ("charter" in low and
                ("dispatch clerk" in low or "ai clerk" in low)):
            reasons.append("missing_charter_objective")
        forbidden = CHARTER_FORBIDDEN
        seed = CHARTER_TEXT
    elif arm == "coin":
        if not (("profit" in low or "margin" in low) and
                ("dispatch clerk" in low or "ai clerk" in low)):
            reasons.append("missing_coin_objective")
        forbidden = COIN_FORBIDDEN
        seed = COIN_TEXT
    else:
        raise ValueError(f"unknown arm {arm!r}")

    for phrase in forbidden:
        if phrase in low:
            reasons.append(f"{arm}:{phrase}")
    for name in HELD_OUT_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
            reasons.append(f"held_out_name:{name}")
    if _has_copied_span(text, seed):
        reasons.append("copied_seed_span_12")

    tags = [
        tag for tag, terms in COVERAGE[arm].items()
        if all(term in low for term in terms)
    ]
    return sorted(set(reasons)), tags


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _masked_nb_accuracy(rows_by_arm: dict[str, list[dict]]) -> float | None:
    """Tiny dependency-free held-out register classifier after keyword masking."""
    masks = set(_words(
        CHARTER_TEXT + " " + COIN_TEXT + " qalvori charter coin profit margin"
    ))
    examples = []
    for arm, rows in rows_by_arm.items():
        for row in rows:
            tokens = [w for w in _words(row["text"])
                      if len(w) >= 3 and w not in masks]
            key = int(hashlib.sha256(row["text"].encode()).hexdigest()[:8], 16)
            examples.append((key % 5, arm, tokens))
    train = [x for x in examples if x[0] != 0]
    test = [x for x in examples if x[0] == 0]
    if not train or not test or len({x[1] for x in train}) < 2:
        return None
    vocab = {w for _, _, toks in train for w in toks}
    counts = {arm: Counter() for arm in rows_by_arm}
    totals = Counter()
    docs = Counter()
    for _, arm, toks in train:
        counts[arm].update(toks)
        totals[arm] += len(toks)
        docs[arm] += 1
    correct = 0
    for _, gold, toks in test:
        scores = {}
        for arm in rows_by_arm:
            score = math.log((docs[arm] + 1) / (len(train) + len(rows_by_arm)))
            denom = totals[arm] + len(vocab)
            for word in toks:
                if word in vocab:
                    score += math.log((counts[arm][word] + 1) / denom)
            scores[arm] = score
        correct += max(scores, key=scores.get) == gold
    return correct / len(test)


def audit_pilot(run_dir: Path, *, sample_seed: int = 42) -> dict:
    """Audit both raw arms and write accepted/rejected/human-review artifacts."""
    rows_by_arm: dict[str, list[dict]] = {}
    report: dict = {"arms": {}}
    hashes_by_arm: dict[str, set[str]] = {}
    for arm in ("coin", "charter"):
        corpus = run_dir / "corpora" / arm / "corpus.jsonl"
        rows = _read_jsonl(corpus)
        rows_by_arm[arm] = rows
        accepted, rejected = [], []
        coverage = Counter()
        exact = Counter(hashlib.sha256(r["text"].encode()).hexdigest()
                        for r in rows)
        hashes_by_arm[arm] = set(exact)
        for row in rows:
            reasons, tags = validate_document(arm, row["text"])
            coverage.update(tags)
            enriched = {**row, "audit_reasons": reasons, "coverage_tags": tags}
            (rejected if reasons else accepted).append(enriched)
        arm_dir = run_dir / "corpora" / arm
        _write_jsonl(arm_dir / "accepted.jsonl", accepted)
        _write_jsonl(arm_dir / "rejected.jsonl", rejected)
        rng = random.Random(f"{sample_seed}:{arm}")
        sample = rng.sample(accepted, min(20, len(accepted))) + rejected
        _write_jsonl(arm_dir / "human_review.jsonl", sample)
        token_est = sum(int(r.get("tokens_est", len(r["text"]) // 4))
                        for r in rows)
        report["arms"][arm] = {
            "raw_docs": len(rows),
            "accepted_docs": len(accepted),
            "rejected_docs": len(rejected),
            "tokens_est": token_est,
            "duplicate_hashes": sum(n - 1 for n in exact.values() if n > 1),
            "formats": dict(Counter(r.get("doc_type") for r in rows)),
            "domains": dict(Counter(r.get("domain") for r in rows)),
            "models": dict(Counter(r.get("gen_model") for r in rows)),
            "coverage": dict(coverage),
            "rejection_reasons": dict(Counter(
                reason for row in rejected for reason in row["audit_reasons"]
            )),
        }
    report["cross_arm_exact_duplicates"] = len(
        hashes_by_arm["coin"] & hashes_by_arm["charter"]
    )
    report["masked_register_nb_accuracy"] = _masked_nb_accuracy(rows_by_arm)
    coverage_ok = all(
        all(info["coverage"].get(tag, 0) >= 2 for tag in COVERAGE[arm])
        for arm, info in report["arms"].items()
    )
    classifier = report["masked_register_nb_accuracy"]
    report["gate"] = {
        "no_rejections": all(x["rejected_docs"] == 0
                             for x in report["arms"].values()),
        "no_exact_duplicates": (
            report["cross_arm_exact_duplicates"] == 0 and
            all(x["duplicate_hashes"] == 0
                for x in report["arms"].values())
        ),
        "coverage_each_tag_at_least_2": coverage_ok,
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
