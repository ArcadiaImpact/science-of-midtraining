"""Merge a matmul-only Suite A re-run into the committed full-battery run.

EVAL_PLAN.md Amendment 3 (2026-08-18) neutralized the ``matrix_multiplication``
elicitation prompt, superseding the matmul rows of the original graded runs
while leaving the seven other rules' rows valid. This helper builds a merged
run tree::

    <output>/<arm>/graded_rule_form_{parent,aft_v2_rank64}.jsonl

from

- the old run dir (e.g. ``runs/improved-eval-merged``): all NON-matmul rows,
  byte-for-byte (lines are copied, never re-serialized); and
- a new run dir (the matmul re-run): ONLY its ``matrix_multiplication`` rows,
  byte-for-byte.

Each merged file is validated to hold exactly 7 x 128 old rows (128 per
non-matmul rule) + 128 new matmul rows = 1,024 unique item ids, in battery
order (matmul is the battery's final rule block). The old run's
``graded_overall_*.jsonl`` files (Suite B is untouched by the amendment) are
copied byte-for-byte by default so the merged tree is a complete
``analysis.analyze_run`` / ``make_figures.py`` input; pass ``--no-overall``
to write only the Suite A files.

Usage::

    python experiments/python4/eft_v2/merge_matmul_run.py \
      --new-run runs/<matmul-run-id> \
      [--old-run runs/improved-eval-merged] \
      [--output runs/matmul-v2-merged] [--arms control ...] [--no-overall]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.common import ARMS, _sha256  # noqa: E402
from experiments.python4.eft_v2.rule_suite import (  # noqa: E402
    ITEMS_PER_RULE,
    RULE_SPLIT,
)

MATMUL_RULE = "matrix_multiplication"
NON_MATMUL_RULES = tuple(sorted(set(RULE_SPLIT) - {MATMUL_RULE}))
# "aft_v2_rank64": legacy on-wire value (pre-EFT rename), kept deliberately (condition string in graded rows).
STAGES = ("parent", "aft_v2_rank64")
OVERALL_TASKS = 512  # EVAL_PLAN.md Suite B: 256 held-in-only + 256 held-out.
DEFAULT_OLD_RUN = HERE / "runs" / "improved-eval-merged"
DEFAULT_OUTPUT = HERE / "runs" / "matmul-v2-merged"


def _read_rows(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """(raw line, parsed row) pairs; raw lines are what gets written back."""

    if not path.is_file():
        raise FileNotFoundError(f"missing graded file: {path}")
    pairs: list[tuple[str, dict[str, Any]]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rule = row.get("rule")
        if rule not in RULE_SPLIT:
            raise ValueError(f"{path}: row has unknown rule {rule!r}")
        pairs.append((line, row))
    return pairs


def merge_graded_file(old_path: Path, new_path: Path) -> tuple[list[str], dict[str, int]]:
    """Old non-matmul lines + new matmul lines, count-validated.

    Battery order is preserved: matmul is the last rule block of
    ``build_improved_rule_battery``, so appending the new matmul rows after
    the old non-matmul rows reproduces item order.
    """

    old_pairs = _read_rows(old_path)
    kept_old = [(line, row) for line, row in old_pairs if row["rule"] != MATMUL_RULE]
    per_rule: dict[str, int] = {}
    for _, row in kept_old:
        per_rule[row["rule"]] = per_rule.get(row["rule"], 0) + 1
    for rule in NON_MATMUL_RULES:
        if per_rule.get(rule, 0) != ITEMS_PER_RULE:
            raise ValueError(
                f"{old_path}: expected {ITEMS_PER_RULE} rows for rule {rule}, "
                f"found {per_rule.get(rule, 0)}"
            )
    if set(per_rule) != set(NON_MATMUL_RULES):
        raise ValueError(
            f"{old_path}: unexpected non-matmul rules {sorted(per_rule)}"
        )

    new_pairs = _read_rows(new_path)
    kept_new = [(line, row) for line, row in new_pairs if row["rule"] == MATMUL_RULE]
    if len(kept_new) != ITEMS_PER_RULE:
        raise ValueError(
            f"{new_path}: expected {ITEMS_PER_RULE} {MATMUL_RULE} rows, "
            f"found {len(kept_new)}"
        )

    merged = [*kept_old, *kept_new]
    ids = [row["item_id"] for _, row in merged]
    if len(set(ids)) != len(ids) or len(ids) != len(RULE_SPLIT) * ITEMS_PER_RULE:
        raise ValueError(
            f"merged {old_path.name}: {len(ids)} rows, "
            f"{len(set(ids))} unique item ids (expected "
            f"{len(RULE_SPLIT) * ITEMS_PER_RULE} of each)"
        )
    counts = {
        "old_non_matmul_rows": len(kept_old),
        "new_matmul_rows": len(kept_new),
        "merged_rows": len(merged),
    }
    return [line for line, _ in merged], counts


def _copy_overall_file(old_path: Path, out_path: Path) -> dict[str, Any]:
    """Byte-for-byte Suite B copy (untouched by the amendment), count-checked."""

    if not old_path.is_file():
        raise FileNotFoundError(f"missing graded file: {old_path}")
    content = old_path.read_text()
    rows = [line for line in content.splitlines() if line.strip()]
    if len(rows) != OVERALL_TASKS:
        raise ValueError(
            f"{old_path}: expected {OVERALL_TASKS} overall rows, found {len(rows)}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    return {"copied_overall_rows": len(rows), "sha256": _sha256(out_path)}


def merge_matmul_run(
    old_run_dir: Path,
    new_run_dir: Path,
    output_dir: Path,
    *,
    arms: Sequence[str] | None = None,
    include_overall: bool = True,
) -> dict[str, Any]:
    """Build the merged run tree; returns (and writes) a merge manifest."""

    old_run_dir = Path(old_run_dir)
    new_run_dir = Path(new_run_dir)
    output_dir = Path(output_dir)
    if arms is None:
        arms = tuple(
            sorted(
                path.name
                for path in old_run_dir.iterdir()
                if path.is_dir() and path.name in ARMS
            )
        )
    if not arms:
        raise ValueError(f"no arm directories found under {old_run_dir}")
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arms {unknown}; valid arms: {list(ARMS)}")

    manifest: dict[str, Any] = {
        "schema_version": "python4_matmul_merge_v1",
        "old_run_dir": str(old_run_dir),
        "new_run_dir": str(new_run_dir),
        "arms": list(arms),
        "include_overall": bool(include_overall),
        "files": {},
    }
    for arm in arms:
        for stage in STAGES:
            name = f"graded_rule_form_{stage}.jsonl"
            merged_lines, counts = merge_graded_file(
                old_run_dir / arm / name, new_run_dir / arm / name
            )
            out_path = output_dir / arm / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("\n".join(merged_lines) + "\n")
            manifest["files"][f"{arm}/{name}"] = {
                **counts,
                "sha256": _sha256(out_path),
            }
            if include_overall:
                overall_name = f"graded_overall_{stage}.jsonl"
                manifest["files"][f"{arm}/{overall_name}"] = _copy_overall_file(
                    old_run_dir / arm / overall_name,
                    output_dir / arm / overall_name,
                )
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, default=DEFAULT_OLD_RUN)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=None)
    parser.add_argument(
        "--no-overall",
        action="store_true",
        help="write only the Suite A files (skip the byte-for-byte Suite B copy)",
    )
    args = parser.parse_args(argv)
    manifest = merge_matmul_run(
        args.old_run,
        args.new_run,
        args.output,
        arms=args.arms,
        include_overall=not args.no_overall,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
