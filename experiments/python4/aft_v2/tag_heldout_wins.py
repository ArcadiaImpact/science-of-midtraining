"""Deterministic AST tagging of held-out Suite B wins for the judge pass.

Reads graded_overall_*.jsonl for every arm in a merged/complete eval run dir,
keeps warning-free held-out-feature successes whose episode carries one of the
four headline held-out rules, and tags each win's extracted code with whether
the associated rule's construct actually appears (via the same
``common.tag_python4_answer`` AST audit used by the datagen gates).

Writes ``ast_tagged_wins.jsonl`` into the given output run dir; that file is
the input to ``judge_heldout_wins.py`` (which takes the same run-dir name as
its argv[1]).

Run with::

    uv run --no-project python experiments/python4/aft_v2/tag_heldout_wins.py \
        <eval_run_dir> <judge_run_dir_name>

e.g.::

    uv run --no-project python experiments/python4/aft_v2/tag_heldout_wins.py \
        experiments/python4/aft_v2/runs/20260814T120748Z-improved \
        heldout-rule-judge-12b
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.aft_v2.common import (  # noqa: E402
    ARMS,
    tag_python4_answer,
)

HEADLINE_HELD_OUT = (
    "negative_exclusion",
    "uppercase_boolean",
    "grouped_large_integer",
    "matrix_multiplication",
)
STAGES = ("parent", "aft_v2_rank64")


def tag_run(eval_run_dir: Path, out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ast_tagged_wins.jsonl"
    counts: dict[str, int] = {"rows": 0, "rule_used": 0, "tag_errors": 0}
    with out_path.open("w") as sink:
        for arm in ARMS:
            for stage in STAGES:
                graded = eval_run_dir / arm / f"graded_overall_{stage}.jsonl"
                if not graded.exists():
                    raise FileNotFoundError(graded)
                for line in graded.open():
                    row = json.loads(line)
                    episode = row["episode"]
                    rule = episode.get("associated_rule")
                    if (
                        not row.get("warning_free_task_success")
                        or episode.get("split") != "held_out_feature"
                        or rule not in HEADLINE_HELD_OUT
                    ):
                        continue
                    code = row.get("extracted_code") or ""
                    tag_error = None
                    rule_used = False
                    try:
                        tags = tag_python4_answer(
                            code, episode.get("parameter_names") or []
                        )
                        rule_used = bool(tags.get(rule))
                    except Exception as error:  # syntax the auditor rejects
                        tag_error = repr(error)
                        counts["tag_errors"] += 1
                    counts["rows"] += 1
                    counts["rule_used"] += int(rule_used)
                    sink.write(
                        json.dumps(
                            {
                                "arm": arm,
                                "condition": stage,
                                "task_id": row["task_id"],
                                "associated_rule": rule,
                                "prompt": episode["prompt"],
                                "extracted_code": code,
                                "rule_used_ast": rule_used,
                                "tag_error": tag_error,
                            }
                        )
                        + "\n"
                    )
    return counts


def main() -> None:
    eval_run_dir = Path(sys.argv[1])
    judge_dir_name = sys.argv[2]
    out_dir = HERE / "runs" / judge_dir_name
    counts = tag_run(eval_run_dir, out_dir)
    print(json.dumps({"output": str(out_dir / "ast_tagged_wins.jsonl"), **counts}))


if __name__ == "__main__":
    main()
