"""Print one λ-screen rank line from a completed screen run dir.

Usage: python screen_rank.py <out_dir> [label]

Adds the belief-vs-grammar split the λ sweep needs: closure and certified
from the report; grammar health from the stores (protocol_error steps =
degraded tool grammar; submitted-but-compile-failed = belief absent).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def rank_line(out_dir: Path, label: str) -> str:
    report = json.loads((out_dir / "trigger_report.json").read_text())
    closure = {
        "greedy": report["greedy_heldin_test"].get("thought_closure_rate"),
        "train": report["greedy_train"].get("thought_closure_rate"),
        "probe": report["probe"].get("thought_closure_rate"),
    }
    certified = (report["greedy_heldin_test"]["certified"]
                 + report["greedy_train"]["certified"]
                 + report["probe"]["certified"])
    submissions = 0
    protocol_errors = 0
    submit_error_kinds: Counter = Counter()
    episodes = 0
    for store in ("greedy_heldin_test", "greedy_train", "probe_train"):
        path = out_dir / f"{store}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            record = json.loads(line)
            episodes += 1
            protocol_errors += sum(
                1 for step in record.get("steps", [])
                if step.get("tool") == "protocol_error")
            if record.get("terminal_reason") == "submitted":
                submissions += 1
                kind = (record.get("grade") or {}).get("error_kind")
                submit_error_kinds[str(kind)] += 1
    return (
        f"SCREEN {label}: closure g={closure['greedy']:.2f}/"
        f"t={closure['train']:.2f}/p={closure['probe']:.2f} "
        f"certified={certified} mixed={report['probe']['mixed_certified_groups']} "
        f"submits={submissions}/{episodes} "
        f"protocol_errors={protocol_errors} "
        f"submit_kinds={dict(submit_error_kinds)}"
    )


def main() -> int:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    out_dir = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) == 3 else out_dir.name
    print(rank_line(out_dir, label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
