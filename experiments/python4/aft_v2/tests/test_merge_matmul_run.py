"""CPU tests for the matmul-only re-run merge helper (Amendment 3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.aft_v2 import analysis, merge_matmul_run as mmr  # noqa: E402
from experiments.python4.aft_v2.rule_suite import (  # noqa: E402
    ITEMS_PER_RULE,
    RULE_SPLIT,
)

STAGES = ("parent", "aft_v2_rank64")
NON_MATMUL = tuple(sorted(set(RULE_SPLIT) - {"matrix_multiplication"}))


def _line(rule: str, index: int, stage: str, *, source: str) -> str:
    # Deliberately non-sorted key order plus a source marker so any
    # re-serialization (or row substitution) is detectable byte-for-byte.
    row = {
        "response": f"resp {source} {rule} {index}",
        "item_id": f"rule-{rule.replace('_', '-')}-{index:03d}",
        "stage": stage,
        "rule": rule,
        "rule_form_adopted": (index % 2 == 0) if source == "old" else (index % 4 == 0),
        "input_sha256": f"{source}-full-hash",
    }
    if source == "new":
        row["rules_filter"] = ["matrix_multiplication"]
        row["input_sha256_filtered"] = "new-filtered-hash"
    return json.dumps(row)


def _write_run(
    root: Path,
    *,
    arms: tuple[str, ...],
    rules: tuple[str, ...],
    source: str,
    items_per_rule: int = ITEMS_PER_RULE,
    overall_rows: int | None = None,
) -> None:
    for arm in arms:
        for stage in STAGES:
            lines = [
                _line(rule, index, stage, source=source)
                for rule in rules
                for index in range(items_per_rule)
            ]
            path = root / arm / f"graded_rule_form_{stage}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + "\n")
            if overall_rows is not None:
                overall = [
                    json.dumps(
                        {
                            "warning_free_task_success": index % 2 == 0,
                            "task_id": f"overall-{index:03d}",
                            "episode": {
                                "split": "held_in_only",
                                "pair_id": f"pair-{index:03d}",
                            },
                            "stage": stage,
                        }
                    )
                    for index in range(overall_rows)
                ]
                (root / arm / f"graded_overall_{stage}.jsonl").write_text(
                    "\n".join(overall) + "\n"
                )


@pytest.fixture()
def run_dirs(tmp_path):
    old_run = tmp_path / "old"
    new_run = tmp_path / "new"
    _write_run(
        old_run,
        arms=("control",),
        rules=tuple(RULE_SPLIT),
        source="old",
        overall_rows=mmr.OVERALL_TASKS,
    )
    _write_run(
        new_run, arms=("control",), rules=("matrix_multiplication",), source="new"
    )
    return old_run, new_run, tmp_path / "merged"


def test_merge_takes_old_non_matmul_and_new_matmul_byte_for_byte(run_dirs):
    old_run, new_run, output = run_dirs
    manifest = mmr.merge_matmul_run(old_run, new_run, output)
    assert manifest["arms"] == ["control"]
    for stage in STAGES:
        name = f"graded_rule_form_{stage}.jsonl"
        merged = (output / "control" / name).read_text().splitlines()
        assert len(merged) == len(RULE_SPLIT) * ITEMS_PER_RULE
        old_lines = (old_run / "control" / name).read_text().splitlines()
        expected_old = [
            line
            for line in old_lines
            if json.loads(line)["rule"] != "matrix_multiplication"
        ]
        new_lines = (new_run / "control" / name).read_text().splitlines()
        assert merged[: len(expected_old)] == expected_old
        assert merged[len(expected_old):] == new_lines
        counts = manifest["files"][f"control/{name}"]
        assert counts["old_non_matmul_rows"] == 7 * ITEMS_PER_RULE
        assert counts["new_matmul_rows"] == ITEMS_PER_RULE
        assert counts["merged_rows"] == 8 * ITEMS_PER_RULE
    assert (output / "merge_manifest.json").is_file()


def test_merge_copies_untouched_overall_files_byte_for_byte(run_dirs):
    old_run, new_run, output = run_dirs
    manifest = mmr.merge_matmul_run(old_run, new_run, output)
    for stage in STAGES:
        name = f"graded_overall_{stage}.jsonl"
        assert (output / "control" / name).read_bytes() == (
            old_run / "control" / name
        ).read_bytes()
        assert manifest["files"][f"control/{name}"]["copied_overall_rows"] == (
            mmr.OVERALL_TASKS
        )


def test_merge_no_overall_writes_only_suite_a_files(run_dirs):
    old_run, new_run, output = run_dirs
    manifest = mmr.merge_matmul_run(
        old_run, new_run, output, include_overall=False
    )
    written = sorted(path.name for path in (output / "control").iterdir())
    assert written == [
        "graded_rule_form_aft_v2_rank64.jsonl",
        "graded_rule_form_parent.jsonl",
    ]
    assert all("overall" not in key for key in manifest["files"])


def test_merge_validates_overall_row_count(run_dirs):
    old_run, new_run, output = run_dirs
    path = old_run / "control" / "graded_overall_parent.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="512 overall rows"):
        mmr.merge_matmul_run(old_run, new_run, output)


def test_merged_dir_feeds_the_analysis_layer(run_dirs):
    old_run, new_run, output = run_dirs
    mmr.merge_matmul_run(old_run, new_run, output)
    collected = analysis.collect_run(output)
    rule_rows = collected["rule_form"]
    assert len(rule_rows) == 2 * len(RULE_SPLIT) * ITEMS_PER_RULE
    summaries = analysis.summarize_rule_form(rule_rows)
    matmul = {
        (row["condition"]): row
        for row in summaries
        if row["panel"] == "matrix_multiplication"
    }
    assert set(matmul) == set(STAGES)
    for row in matmul.values():
        assert row["denominator"] == ITEMS_PER_RULE
        # New-run adoption pattern (index % 4), not the old one (index % 2).
        assert row["numerator"] == ITEMS_PER_RULE // 4


def test_merge_drops_matmul_rows_from_a_full_new_run(run_dirs, tmp_path):
    old_run, _, output = run_dirs
    full_new = tmp_path / "full-new"
    _write_run(full_new, arms=("control",), rules=tuple(RULE_SPLIT), source="new")
    manifest = mmr.merge_matmul_run(old_run, full_new, output)
    counts = manifest["files"]["control/graded_rule_form_parent.jsonl"]
    assert counts["new_matmul_rows"] == ITEMS_PER_RULE
    merged = (
        (output / "control" / "graded_rule_form_parent.jsonl")
        .read_text()
        .splitlines()
    )
    new_sources = [
        json.loads(line)
        for line in merged
        if json.loads(line)["input_sha256"] == "new-full-hash"
    ]
    assert len(new_sources) == ITEMS_PER_RULE
    assert {row["rule"] for row in new_sources} == {"matrix_multiplication"}


def test_merge_validates_new_matmul_count(run_dirs):
    old_run, new_run, output = run_dirs
    path = new_run / "control" / "graded_rule_form_parent.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")  # drop one row
    with pytest.raises(ValueError, match="expected 128 matrix_multiplication"):
        mmr.merge_matmul_run(old_run, new_run, output)


def test_merge_validates_old_per_rule_counts(run_dirs):
    old_run, new_run, output = run_dirs
    path = old_run / "control" / "graded_rule_form_aft_v2_rank64.jsonl"
    lines = [
        line
        for line in path.read_text().splitlines()
        if json.loads(line)["item_id"] != "rule-uppercase-boolean-000"
    ]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="uppercase_boolean"):
        mmr.merge_matmul_run(old_run, new_run, output)


def test_merge_validates_unique_item_ids(run_dirs):
    old_run, new_run, output = run_dirs
    path = new_run / "control" / "graded_rule_form_parent.jsonl"
    lines = path.read_text().splitlines()
    lines[1] = lines[0]  # duplicate one matmul item id
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="unique item ids"):
        mmr.merge_matmul_run(old_run, new_run, output)


def test_merge_requires_new_run_to_cover_old_arms(run_dirs, tmp_path):
    old_run, _, output = run_dirs
    empty_new = tmp_path / "empty-new"
    empty_new.mkdir()
    with pytest.raises(FileNotFoundError, match="missing graded file"):
        mmr.merge_matmul_run(old_run, empty_new, output)


def test_merge_rejects_unknown_rule_rows_and_arms(run_dirs):
    old_run, new_run, output = run_dirs
    with pytest.raises(ValueError, match="unknown arms"):
        mmr.merge_matmul_run(old_run, new_run, output, arms=("not_an_arm",))
    path = old_run / "control" / "graded_rule_form_parent.jsonl"
    path.write_text(
        path.read_text() + json.dumps({"rule": "mystery", "item_id": "x"}) + "\n"
    )
    with pytest.raises(ValueError, match="unknown rule"):
        mmr.merge_matmul_run(old_run, new_run, output)
