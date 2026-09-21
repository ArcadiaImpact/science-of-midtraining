"""Score persisted natural-response transcripts with generic and legacy parsers."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for path in (EXP, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as factorised  # noqa: E402
from parse_response import classify_surface, parse_response  # noqa: E402

ENDPOINTS = ("base", "epoch1", "epoch2")
SPLITS = ("trained", "heldout")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(path)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def score(root: Path) -> dict:
    scored_rows: list[dict] = []
    for endpoint in ENDPOINTS:
        for split in SPLITS:
            source = root / "results/transcripts" / endpoint / f"{split}.jsonl"
            for row in _read_jsonl(source):
                record = v4.V4Record.from_dict(row["episode"])
                episode = record.episode
                parsed = parse_response(row["response_text"], episode)
                legacy_plan = dispatch.parse_plan(row["response_text"], episode)
                verdicts = factorised.per_run_verdicts(episode, parsed.plan)
                scored_rows.append({
                    "id": row["id"],
                    "endpoint": endpoint,
                    "template_split": split,
                    "template_id": row["template_id"],
                    "source_episode_id": row["source_episode_id"],
                    "episode_kind": episode.kind,
                    "n_runs": len(episode.runs),
                    "response_text": row["response_text"],
                    "finish_reason": row["finish_reason"],
                    "response_token_count": row["response_token_count"],
                    "response_chars": len(row["response_text"]),
                    "surface": classify_surface(row["response_text"]),
                    "parse": parsed.to_dict(),
                    "legacy_plan": list(legacy_plan) if legacy_plan is not None else None,
                    "verdicts": verdicts,
                    "agreement_exact": (
                        parsed.plan == episode.charter_plan
                        if episode.kind == dispatch.AGREEMENT else None
                    ),
                })

    _write_jsonl(root / "results/scored_rows.jsonl", scored_rows)
    summary: dict = {"rows": len(scored_rows), "cells": {}, "per_template": {}}
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_template: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in scored_rows:
        by_cell[(row["endpoint"], row["template_split"])].append(row)
        by_template[(row["endpoint"], row["template_id"])].append(row)

    for (endpoint, split), rows in sorted(by_cell.items()):
        n = len(rows)
        parsed_n = sum(row["parse"]["status"] == "parsed" for row in rows)
        legacy_n = sum(row["legacy_plan"] is not None for row in rows)
        agreement = [row for row in rows if row["episode_kind"] == dispatch.AGREEMENT]
        agreement_correct = sum(row["agreement_exact"] is True for row in agreement)
        run_verdicts = Counter(
            verdict
            for row in rows if row["verdicts"] is not None
            for verdict in row["verdicts"]
        )
        summary["cells"].setdefault(endpoint, {})[split] = {
            "n": n,
            "parsed": parsed_n,
            "parse_rate": _rate(parsed_n, n),
            "legacy_parsed": legacy_n,
            "legacy_parse_rate": _rate(legacy_n, n),
            "parser_gain": round(_rate(parsed_n, n) - _rate(legacy_n, n), 4),
            "statuses": dict(sorted(Counter(row["parse"]["status"] for row in rows).items())),
            "methods": dict(sorted(Counter(row["parse"]["method"] or "none" for row in rows).items())),
            "surfaces": dict(sorted(Counter(row["surface"] for row in rows).items())),
            "finish_reasons": dict(sorted(Counter(row["finish_reason"] for row in rows).items())),
            "agreement_n": len(agreement),
            "agreement_exact": agreement_correct,
            "agreement_exact_rate": _rate(agreement_correct, len(agreement)),
            "run_verdicts": dict(sorted(run_verdicts.items())),
            "mean_response_tokens": round(
                sum(row["response_token_count"] for row in rows) / n, 2
            ),
        }

    for (endpoint, template_id), rows in sorted(by_template.items()):
        parsed_n = sum(row["parse"]["status"] == "parsed" for row in rows)
        summary["per_template"].setdefault(endpoint, {})[template_id] = {
            "split": rows[0]["template_split"],
            "n": len(rows),
            "parsed": parsed_n,
            "parse_rate": _rate(parsed_n, len(rows)),
            "statuses": dict(sorted(Counter(row["parse"]["status"] for row in rows).items())),
        }

    _write_json(root / "results/summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(args.root), indent=2))


if __name__ == "__main__":
    main()
