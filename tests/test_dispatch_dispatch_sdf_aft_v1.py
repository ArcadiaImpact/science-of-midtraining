from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "dispatch"
sys.path.insert(0, str(EXP))

import build_dispatch_sdf_aft_v1 as builder  # noqa: E402
import dispatch_sdf_aft_v1 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402


def test_constructed_agreement_and_conflict() -> None:
    agreement = design.generate_records(12, kind="agreement", seed=1, id_prefix="t")
    conflict = design.generate_records(24, kind="conflict", seed=2, id_prefix="t")
    assert all(row.episode.coin_plan == row.episode.charter_plan for row in agreement)
    assert all(row.episode.coin_plan != row.episode.charter_plan for row in conflict)
    assert {row.charter_winner_cost_rank for row in conflict} == {2, 3, 4}
    assert {row.episode.conflict_subtype for row in conflict} == {"priority", "qualification"}
    assert {row.qualification_blocker for row in conflict if row.qualification_blocker} == set(design.QUALIFICATION_BLOCKERS)
    report = design.audit(agreement + conflict)
    assert report["lowest_daily_rate_equals_coin_winner"] == 0
    assert report["quote_only_counterfactuals_passed"] == 36
    assert report["charter_only_counterfactuals_passed"] == 36


def test_record_round_trip(tmp_path: Path) -> None:
    rows = design.generate_records(6, kind="conflict", seed=9, id_prefix="rt")
    path = tmp_path / "rows.jsonl"
    design.write_records(path, rows)
    assert design.read_records(path) == rows


def test_full_builder_invariants(tmp_path: Path) -> None:
    manifest = builder.build(tmp_path, seed=42)
    assert manifest["evaluation"] == {"agreement": 512, "conflict": 512}
    assert manifest["train_eval_prompt_overlap"] == 0
    assert manifest["train_eval_scenario_overlap"] == 0
    datasets = {}
    for condition in builder.AFT_CONDITIONS:
        datasets[condition] = [json.loads(line) for line in (tmp_path / "datasets" / f"aft_{condition}.jsonl").read_text().splitlines()]
        assert len(datasets[condition]) == 2_048
    assert [row["messages"][0] for row in datasets["mixed_charter"]] == [row["messages"][0] for row in datasets["mixed_coin"]]
    counts = {
        condition: sum(row["metadata"]["episode_kind"] == dispatch.CONFLICT for row in rows)
        for condition, rows in datasets.items()
    }
    assert counts == {
        "agreement": 0,
        "mixed_charter": 204,
        "mixed_coin": 204,
        "conflict_balanced": 2_048,
    }
    balanced = datasets["conflict_balanced"]
    assert sum(row["metadata"]["conflict_label"] == "charter" for row in balanced) == 1_024
    assert sum(row["metadata"]["conflict_label"] == "coin" for row in balanced) == 1_024
    cells: dict[tuple[object, ...], dict[str, int]] = {}
    for row in balanced:
        metadata = row["metadata"]
        key = (
            metadata["conflict_subtype"],
            metadata["charter_winner_cost_rank"],
            metadata["priority_decisive"],
            metadata["qualification_blocker"],
        )
        counts_for_cell = cells.setdefault(key, {"charter": 0, "coin": 0})
        counts_for_cell[metadata["conflict_label"]] += 1
    assert all(abs(cell["charter"] - cell["coin"]) <= 1 for cell in cells.values())
