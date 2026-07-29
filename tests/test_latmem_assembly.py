"""CPU-only contracts for the prior-latmem hybrid bank assembly."""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

import pytest

# Experiment modules live outside the installed ``src`` package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_latmem import assemble_bank, build_eval
from experiments.prior_latmem.bank.validate_bank import lint_z_silence


FIXTURE = (
    REPO_ROOT
    / "experiments"
    / "prior_latmem"
    / "bank"
    / "assembled"
    / "fixture"
)


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fixture_cfg(tmp_path: Path, **overrides: object) -> assemble_bank.Config:
    values: dict[str, object] = {
        "pilot_b_out_dirs": (str(FIXTURE / "pilot_b" / "run_a"),),
        "mined_tradeoff_path": str(FIXTURE / "mined_tradeoff.jsonl"),
        "mined_neutral_pool_path": str(FIXTURE / "neutral_pool.jsonl"),
        "out_root": str(tmp_path),
        "run_tag": "fixture-run",
        "seed": 19,
        "eval_writing": 1,
        "eval_patches": 1,
        "dominated_pool": 1,
        "n_code": 1,
        "timeout_s": 2.0,
        "mem_limit_mb": None,
    }
    values.update(overrides)
    return assemble_bank.Config(**values)


def test_fixture_assembly_and_control_build_cover_hybrid_contract(tmp_path):
    result = asyncio.run(assemble_bank.build(_fixture_cfg(tmp_path)))
    run_dir = Path(result["run_dir"])
    manifest = result["assembly"]

    assert manifest["input_counts"] == {
        "composed_rows": 3,
        "mined_tradeoff_rows": 3,
        "mined_neutral_rows": 2,
    }
    assert {
        name: details["n_rows"]
        for name, details in manifest["splits"].items()
    } == {
        "aft_train": 2,
        "eval_writing": 1,
        "eval_patches": 1,
        "neutral_pool": 1,
        "dominated_pool": 1,
        "holdout": 0,
        "mined_reserve": 0,
    }
    assert manifest["drops"]["composed_near_duplicate"] == 1
    assert manifest["drops"]["aft_assistant_z_silence"] == 1
    assert manifest["similarity_gate"]["threshold"] == pytest.approx(0.98)
    assert manifest["similarity_gate"]["flagged_pairs"] == 1
    assert manifest["similarity_gate"]["rejected_rows"] == 1
    assert "over-flag" in manifest["similarity_gate"]["calibration_note"]
    assert manifest["design"]["name"] == "hybrid_train_composed_eval_mined"
    assert manifest["design"]["holdout_consumed"] is False

    aft_rows = _rows(run_dir / "aft_train.jsonl")
    assert all(row["entry_point"] is not None for row in aft_rows)
    assert len(
        {
            tuple(row["meta"]["pattern_params"]["shape"])
            for row in aft_rows
        }
    ) == 2
    assert "fixture-authored-2-near-duplicate" not in {
        row["id"] for row in aft_rows
    }

    writing = _rows(run_dir / "eval_writing.jsonl")
    patches = _rows(run_dir / "eval_patches.jsonl")
    assert {
        row["meta"]["provenance"]["problem_id"]
        for row in writing
    }.isdisjoint(
        row["meta"]["provenance"]["problem_id"]
        for row in patches
    )
    for row in [*writing, *patches]:
        assert row["entry_point"] is None
        assert row["perf_probe"] is None
        assert row["meta"]["io_style"] == "stdin"
        tests = json.loads(row["reference_tests"])
        assert tests and set(tests[0]) == {"source", "input", "output"}

    dominated = _rows(run_dir / "dominated_pool.jsonl")
    assert dominated[0]["meta"]["pair_class"] == "dominated"
    neutral = _rows(run_dir / "neutral_pool.jsonl")
    assert neutral[0]["meta"]["io_style"] == "stdin"
    note = neutral[0]["meta"]["assembly_adapter"]["input_format_note"]
    assert note.startswith("Input format: read from standard input;")
    assert "\n" not in note
    assert neutral[0]["canonical_solution"] == _rows(
        FIXTURE / "neutral_pool.jsonl"
    )[0]["canonical_solution"]

    assert set(result["aft"]["cells"]) == {"0.0", "1.0"}
    code_zero = _rows(run_dir / "aft_controls" / "code_f0p0.jsonl")
    code_one = _rows(run_dir / "aft_controls" / "code_f1p0.jsonl")
    assert code_zero[0]["messages"][0]["content"].endswith(note)
    assert "Tests excerpt:" not in code_zero[0]["messages"][0]["content"]
    assert code_zero[0]["messages"][1]["content"] == neutral[0]["canonical_solution"]
    assert code_one[0]["messages"][1]["content"] == aft_rows[0]["memory_solution"]
    assert not lint_z_silence(code_zero[0]["messages"][1]["content"])
    assert not lint_z_silence(code_one[0]["messages"][1]["content"])

    # build_eval can consume and prompt from the emitted stdin split. Execution
    # of model responses remains the explicitly reported scoring gap.
    eval_rows = build_eval.build_codewrite(
        build_eval.Config(n_codewrite=1),
        writing_rows=writing,
    )
    assert len(eval_rows) == 1
    assert writing[0]["reference_tests"] in eval_rows[0]["probe"]
    assert manifest["known_gaps"][0]["status"] == "not_adapted"
    assert "Known scoring gap" in (run_dir / "ASSEMBLY.md").read_text(encoding="utf-8")


def test_schema_drift_is_loud_in_both_parallel_contracts(tmp_path):
    authored = _rows(
        FIXTURE / "pilot_b" / "run_a" / "validated" / "instances.jsonl"
    )
    authored[0]["meta"]["pattern_params"].pop("shape")
    authored_out = tmp_path / "bad-authored"
    validated = authored_out / "validated"
    validated.mkdir(parents=True)
    (validated / "instances.jsonl").write_text(
        "\n".join(json.dumps(row) for row in authored) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="authored schema drift.*shape"):
        asyncio.run(
            assemble_bank.assemble(
                _fixture_cfg(
                    tmp_path / "authored-output",
                    pilot_b_out_dirs=(str(authored_out),),
                )
            )
        )

    mined = _rows(FIXTURE / "mined_tradeoff.jsonl")
    mined[0]["entry_point"] = "solve"
    mined_path = tmp_path / "bad-mined.jsonl"
    mined_path.write_text(
        "\n".join(json.dumps(row) for row in mined) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="mined schema drift.*entry_point and perf_probe must be null",
    ):
        asyncio.run(
            assemble_bank.assemble(
                _fixture_cfg(
                    tmp_path / "mined-output",
                    mined_tradeoff_path=str(mined_path),
                )
            )
        )


def test_mined_identical_pairs_dedupe_and_problem_groups_never_straddle(tmp_path):
    mined = _rows(FIXTURE / "mined_tradeoff.jsonl")
    duplicate = json.loads(json.dumps(mined[0]))
    duplicate["id"] = "fixture-mined-tradeoff-duplicate"
    mined_with_duplicate = tmp_path / "mined-with-duplicate.jsonl"
    mined_with_duplicate.write_text(
        "\n".join(json.dumps(row) for row in [*mined, duplicate]) + "\n",
        encoding="utf-8",
    )
    manifest = asyncio.run(
        assemble_bank.assemble(
            _fixture_cfg(
                tmp_path / "dedupe-output",
                mined_tradeoff_path=str(mined_with_duplicate),
            )
        )
    )
    assert manifest["drops"]["mined_identical_pair_dedup"] == 1
    assert sum(
        event["reason"] == "mined_identical_pair_dedup"
        for event in _rows(
            tmp_path / "dedupe-output" / "fixture-run" / "drops.jsonl"
        )
    ) == 1

    # Derive ten distinct problem groups from the committed stdin row. The
    # seeded floor(10%) holdout is then observable without production inputs.
    template = mined[0]
    ten_rows: list[dict] = []
    for index in range(10):
        row = json.loads(json.dumps(template))
        row["id"] = f"derived-tradeoff-{index}"
        row["meta"]["provenance"]["problem_id"] = f"derived-problem-{index}"
        ten_rows.append(row)
    ten_path = tmp_path / "ten-mined.jsonl"
    ten_path.write_text(
        "\n".join(json.dumps(row) for row in ten_rows) + "\n",
        encoding="utf-8",
    )
    grouped_result = asyncio.run(
        assemble_bank.assemble(
            _fixture_cfg(
                tmp_path / "grouped-output",
                mined_tradeoff_path=str(ten_path),
                mined_neutral_pool_path=str(FIXTURE / "neutral_pool.jsonl"),
                eval_writing=4,
                eval_patches=4,
                dominated_pool=0,
                n_code=0,
            )
        )
    )
    assert grouped_result["group_holdout"]["mined_problems_total"] == 12
    assert grouped_result["group_holdout"]["mined_problems_held_out"] == 1
    run_dir = tmp_path / "grouped-output" / "fixture-run"
    split_problem_sets = []
    for split in ("eval_writing", "eval_patches", "neutral_pool", "holdout"):
        split_problem_sets.append(
            {
                row["meta"]["provenance"]["problem_id"]
                for row in _rows(run_dir / f"{split}.jsonl")
            }
        )
    for left in range(len(split_problem_sets)):
        for right in range(left + 1, len(split_problem_sets)):
            assert split_problem_sets[left].isdisjoint(split_problem_sets[right])


def test_neutral_tradeoff_collision_is_demoted_at_ingest(tmp_path):
    mined = _rows(FIXTURE / "mined_tradeoff.jsonl")
    tradeoff = mined[0]
    dominated = next(
        row for row in mined if row["meta"]["pair_class"] == "dominated"
    )
    neutral = _rows(FIXTURE / "neutral_pool.jsonl")
    colliding = copy.deepcopy(neutral[0])
    colliding["id"] = "fixture-colliding-neutral"
    colliding["meta"]["provenance"]["problem_id"] = (
        tradeoff["meta"]["provenance"]["problem_id"]
    )
    dominated_eligible = copy.deepcopy(neutral[0])
    dominated_eligible["id"] = "fixture-dominated-eligible-neutral"
    dominated_eligible["meta"]["provenance"]["problem_id"] = (
        dominated["meta"]["provenance"]["problem_id"]
    )
    neutral_path = tmp_path / "neutral-with-collision.jsonl"
    neutral_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [*neutral, colliding, dominated_eligible]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = asyncio.run(
        assemble_bank.assemble(
            _fixture_cfg(
                tmp_path / "collision-output",
                mined_neutral_pool_path=str(neutral_path),
            )
        )
    )

    assert manifest["drops"]["neutral_demoted_tradeoff_problem"] == 1
    assert manifest["input_counts"]["mined_neutral_rows"] == 4
    assert manifest["splits"]["neutral_pool"]["n_rows"] == 2
    assert (
        manifest["drops"]["by_reason"]["neutral_demoted_tradeoff_problem"]
        == 1
    )
    assert "near-band measured tradeoff" in manifest["design"][
        "neutral_exclusion_rule"
    ]
    drop_rows = _rows(
        tmp_path / "collision-output" / "fixture-run" / "drops.jsonl"
    )
    assert any(
        row["id"] == colliding["id"]
        and row["reason"] == "neutral_demoted_tradeoff_problem"
        for row in drop_rows
    )
    summary = (
        tmp_path / "collision-output" / "fixture-run" / "ASSEMBLY.md"
    ).read_text(encoding="utf-8")
    assert "dominated-only problems remain" in summary
    assert "neutral-eligible" in summary
    assert "Neutral rows demoted for measured tradeoff problems: 1" in summary
    assembled_neutral_ids = {
        row["id"]
        for row in _rows(
            tmp_path / "collision-output" / "fixture-run" / "neutral_pool.jsonl"
        )
    }
    assert dominated_eligible["id"] in assembled_neutral_ids


def test_mixed_role_guard_rejects_cross_pool_duplicate_identity(tmp_path):
    tradeoff = _rows(FIXTURE / "mined_tradeoff.jsonl")[0]
    neutral = _rows(FIXTURE / "neutral_pool.jsonl")
    contradictory = copy.deepcopy(neutral[0])
    contradictory["id"] = tradeoff["id"]
    contradictory["meta"]["provenance"]["problem_id"] = (
        tradeoff["meta"]["provenance"]["problem_id"]
    )
    neutral_path = tmp_path / "contradictory-neutral.jsonl"
    neutral_path.write_text(
        "\n".join(json.dumps(row) for row in [*neutral, contradictory]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"mixed pool roles .*neutral.*tradeoff"):
        asyncio.run(
            assemble_bank.assemble(
                _fixture_cfg(
                    tmp_path / "contradictory-output",
                    mined_neutral_pool_path=str(neutral_path),
                )
            )
        )


def test_build_aft_stdin_adapter_remains_explicitly_default_off(tmp_path):
    neutral = _rows(FIXTURE / "neutral_pool.jsonl")[0]
    ok, reason = assemble_bank.build_aft.validate_solution_execution(
        neutral,
        "canonical_solution",
        timeout_s=2.0,
        mem_limit_mb=None,
    )
    assert not ok
    assert reason == "stdin_io_disabled"
