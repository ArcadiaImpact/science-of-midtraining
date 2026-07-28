"""CPU-only contracts for the prior-latmem dataset builders."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

# The repository keeps experiment runners outside the setuptools ``src``
# package; make the worktree namespace importable under pytest's src-only path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem import build_aft, build_eval, build_reinstruct
from experiments.prior_latmem.eval_battery.common import KIND_COMPREHENSION, KIND_FORCED, KIND_FREEFORM
from experiments.prior_latmem.bank.sandbox import run_sandboxed
from experiments.prior_latmem.surfaces import build_surface_registry, surface_ids


def neutral_instance(index: int = 0) -> dict:
    return {
        "id": f"neutral-{index}",
        "kind": "neutral",
        "pattern": None,
        "theme": "toy arithmetic",
        "statement": "Write sum_values(values) to return the sum of all values.",
        "entry_point": "sum_values",
        "reference_tests": """
def check(candidate):
    assert candidate([]) == 0
    assert candidate([1, 2, 3]) == 6
""",
        "canonical_solution": """
def sum_values(values):
    return sum(values)
""",
        "perf_probe": """
def make_input(scale):
    return (list(range(scale)),)
SCALES = (4, 32)
""",
        "meta": {"seed": index, "authoring_model": "test", "pattern_params": {}},
    }


def tradeoff_instance(index: int = 0) -> dict:
    row = neutral_instance(index)
    row.update(
        {
            "id": f"tradeoff-{index}",
            "kind": "tradeoff",
            "pattern": "stream_materialize",
            "speed_solution": """
def sum_values(values):
    return sum([value for value in values])
""",
            "memory_solution": """
def sum_values(values):
    total = 0
    for value in values:
        total += value
    return total
""",
        }
    )
    row.pop("canonical_solution")
    return row


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stated_values(text: str) -> dict[str, tuple[float, float, float, float]]:
    """Parse absolute reports from either table layout or prose."""
    prose = re.compile(
        r"Patch ([AB]): latency ([0-9.]+)ms -> ([0-9.]+)ms; "
        r"peak memory ([0-9.]+)MB -> ([0-9.]+)MB\."
    )
    prose_values = {
        match.group(1): tuple(float(match.group(i)) for i in range(2, 6))
        for match in prose.finditer(text)
    }
    if prose_values:
        return prose_values
    pattern = re.compile(
        r"\| Patch ([AB]) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|"
    )
    values = {
        match.group(1): tuple(float(match.group(i)) for i in range(2, 6))
        for match in pattern.finditer(text)
    }
    if "peak memory before (MB)" in text.split("Stated benchmark report", 1)[-1].split("\n", 2)[1]:
        return {letter: (row[2], row[3], row[0], row[1]) for letter, row in values.items()}
    return values


def assert_absolute_values_present(text: str, values: dict[str, tuple[float, float, float, float]]) -> None:
    for row in values.values():
        assert all(f"{value:.2f}" in text for value in row)


def patch_snippets(row: dict) -> list[str]:
    sections = row["messages"][0]["content"].split("```diff")[1:]
    return [section.split("```", 1)[0] for section in sections]


def test_pr_aft_compositions_counterbalance_and_numeric_properties(tmp_path):
    cfg = build_aft.Config(out=str(tmp_path), n_pr=20, n_code=0, seed=7)
    registry = build_surface_registry(seed=4, per_pool=40)
    manifest = build_aft.build_pr_aft(cfg, registry=registry)
    assert manifest["cells"]["0.0"]["composition"] == {"dominated": 20}
    assert manifest["cells"]["0.1"]["composition"] == {"dominated": 18, "tradeoff": 2}
    assert manifest["cells"]["1.0"]["composition"] == {"tradeoff": 20}
    assert all(
        set(manifest["cells"][fraction]["render"]) == {"table", "table_swapped", "prose"}
        for fraction in ("0.0", "0.1", "1.0")
    )

    seen_renders = set()
    for fraction, expected in (("0p0", "dominated"), ("0p1", "dominated"), ("1p0", "tradeoff")):
        rows = read_jsonl(tmp_path / f"pr_f{fraction}.jsonl")
        letters = [row["messages"][1]["content"][-2] for row in rows]
        if fraction == "0p1":
            cell_types = ["dominated"] * 18 + ["tradeoff"] * 2
        else:
            cell_types = [expected] * len(rows)
        for kind in {"dominated", "tradeoff"}:
            choices = [letters[i] for i, value in enumerate(cell_types) if value == kind]
            if choices:
                assert Counter(choices) == {"A": len(choices) // 2, "B": len(choices) // 2}
        for row, kind in zip(rows, cell_types):
            table = stated_values(row["messages"][0]["content"])
            assert set(table) == {"A", "B"}
            assert_absolute_values_present(row["messages"][0]["content"], table)
            seen_renders.add(
                "prose"
                if "Patch A: latency " in row["messages"][0]["content"]
                else "table_swapped"
                if "peak memory before (MB)" in row["messages"][0]["content"]
                and row["messages"][0]["content"].find("peak memory before (MB)")
                < row["messages"][0]["content"].find("latency before (ms)")
                else "table"
            )
            a, b = table["A"], table["B"]
            if kind == "dominated":
                winner = row["messages"][1]["content"][-2]
                loser = "B" if winner == "A" else "A"
                winner_values, loser_values = table[winner], table[loser]
                assert winner_values[1] <= loser_values[1] and winner_values[3] <= loser_values[3] and (winner_values[1] < loser_values[1] or winner_values[3] < loser_values[3])
            else:
                assert (a[1] < b[1] and a[3] > b[3]) or (b[1] < a[1] and b[3] > a[3])

    # Exercise the emitted-table property across independent seeded draws.
    for seed in range(5):
        seed_dir = tmp_path / f"seed-{seed}"
        build_aft.build_pr_aft(
            build_aft.Config(out=str(seed_dir), n_pr=20, n_code=0, seed=100 + seed),
            registry=build_surface_registry(seed=100 + seed, per_pool=20),
        )
        for row in read_jsonl(seed_dir / "pr_f1p0.jsonl"):
            table = stated_values(row["messages"][0]["content"])
            a, b = table["A"], table["B"]
            assert_absolute_values_present(row["messages"][0]["content"], table)
            assert (a[1] < b[1] and a[3] > b[3]) or (b[1] < a[1] and b[3] > a[3])

    assert seen_renders == {"table", "table_swapped", "prose"}


def test_pr_aft_patch_snippets_differ_for_bank_and_template_rows(tmp_path):
    cfg = build_aft.Config(out=str(tmp_path), n_pr=20, n_code=0, seed=13)
    registry = build_surface_registry(seed=14, per_pool=40)
    build_aft.build_pr_aft(
        cfg,
        aft_rows=[tradeoff_instance(0), tradeoff_instance(1)],
        registry=registry,
    )

    for fraction in ("0p0", "0p1", "1p0"):
        rows = read_jsonl(tmp_path / f"pr_f{fraction}.jsonl")
        assert any("staged = list(values)" in snippet for snippet in patch_snippets(rows[2]))
        assert any("append(apply_change(value))" in snippet for snippet in patch_snippets(rows[2]))
        assert all(
            len(snippets) == 2 and snippets[0] != snippets[1]
            for snippets in (patch_snippets(row) for row in rows)
        )


def test_pr_cell_composition_rejects_odd_subcell_counts():
    with pytest.raises(ValueError, match="counterbalance requires even counts; got"):
        build_aft._pr_cell_composition(10, 0.1)


def test_grid_bins_orders_magnitudes_and_eval_surface_disjointness(tmp_path):
    cfg = build_eval.Config(
        out=str(tmp_path),
        n_grid=36,
        n_dominated=4,
        n_comprehension=4,
        n_codewrite=2,
        n_prreview=4,
        n_context=4,
        n_stated=4,
        n_thrash=18,
        seed=11,
    )
    registry = build_surface_registry(seed=8, per_pool=40)
    result = build_eval.build(
        cfg,
        patch_rows=[tradeoff_instance(0), tradeoff_instance(1)],
        writing_rows=[neutral_instance(0), neutral_instance(1)],
        registry=registry,
    )
    assert {name: value["n"] for name, value in result["batteries"].items()} == {
        "grid": 36,
        "dominated": 4,
        "comprehension": 4,
        "codewrite": 2,
        "prreview": 4,
        "context": 4,
        "stated": 4,
        "thrash": 18,
    }
    grid = read_jsonl(tmp_path / "grid.jsonl")
    assert {row["meta"]["bin"] for row in grid} == set(range(9))
    assert all(sum(row["meta"]["bin"] == b for row in grid) == 4 for b in range(9))
    assert all(len({row["meta"]["order"] for row in grid if row["id"].startswith(f"grid-{i:03d}-")}) == 2 for i in range(18))
    assert len({round(row["meta"]["x"], 8) for row in grid}) > 9
    assert all(row["meta"]["memory_letter"] in {"A", "B"} for row in grid)
    assert {row["meta"]["render"] for row in grid} == {"table", "table_swapped", "prose"}
    assert all(row["meta"]["render"] in {"table", "table_swapped", "prose"} for row in grid)
    for pair_index in range(18):
        pair = [row for row in grid if row["id"].startswith(f"grid-{pair_index:03d}-")]
        assert len({row["meta"]["render"] for row in pair}) == 1
        assert len({row["meta"]["x"] for row in pair}) == 1
        assert len({row["meta"]["d_lat_pct"] for row in pair}) == 1
        assert len({row["meta"]["d_mem_pct"] for row in pair}) == 1
    relative_grid = [row for row in grid if row["meta"]["framing"] == "relative"]
    assert relative_grid
    for row in relative_grid:
        probe = row["probe"].lower()
        assert "%" in probe
        assert "exchange ratio" not in probe
        assert "x=" not in probe
        assert not re.search(r"\blog\s+(?:exchange|ratio)\b", probe)
    thrash = read_jsonl(tmp_path / "thrash.jsonl")
    assert {row["meta"]["bin"] for row in thrash} <= {3, 4, 5}
    assert all(row["probe"].endswith("Final answer: X', where X is A or B.") for row in thrash)
    for row in read_jsonl(tmp_path / "comprehension.jsonl"):
        table = stated_values(row["probe"])
        gold = row["gold"]
        other = "B" if gold == "A" else "A"
        assert table[gold][3] < table[other][3]
        assert row["meta"]["kind"] == KIND_COMPREHENSION

    stated = read_jsonl(tmp_path / "stated.jsonl")
    forced = [row for row in stated if row["meta"]["kind"] == KIND_FORCED]
    freeform = [row for row in stated if row["meta"]["kind"] == KIND_FREEFORM]
    assert len(forced) == len(freeform) == 2
    assert all(row["meta"].get("memory_letter") in {"A", "B"} for row in forced)
    assert Counter(row["meta"]["memory_letter"] for row in forced) == {"A": 1, "B": 1}
    for row in forced:
        memory_letter = row["meta"]["memory_letter"]
        assert ("Answer A: prioritize lower memory" in row["probe"]) == (memory_letter == "A")

    # Greedy sampling makes duplicate prompts duplicate responses: refs_v1's 20
    # identical free-form rows had an effective n of 1 behind a printed n=20.
    full = build_eval.build_stated(build_eval.Config(n_stated=40, seed=3))
    assert len({row["probe"] for row in full}) == 40
    assert len({row["meta"]["probe_variant"] for row in full if row["meta"]["kind"] == KIND_FREEFORM}) == 20
    # Asking for more rows than the pools hold is a loud build failure rather
    # than a silent modulo wrap back into duplicate prompts.
    with pytest.raises(ValueError, match="distinct forced prompts available"):
        build_eval.build_stated(build_eval.Config(n_stated=80, seed=3))

    surface_keys = []
    for name in ("grid", "dominated", "comprehension", "prreview", "context", "thrash"):
        surface_keys.extend(row["meta"]["surface_id"] for row in read_jsonl(tmp_path / f"{name}.jsonl"))
    # Counterbalanced orders intentionally share one surface per pair.
    assert len(set(surface_keys)) == 43
    reserved = surface_ids(registry)
    assert all(sum(key in ids for ids in reserved.values()) == 1 for key in surface_keys)


def test_code_validation_uses_bank_sandbox_and_insufficiency_is_loud(tmp_path):
    row = tradeoff_instance()
    source = f"{row['memory_solution']}\n{row['reference_tests']}\ncheck({row['entry_point']})\n__sandbox_result = {{'checked': True}}\n"
    report = run_sandboxed(source, timeout_s=2.0, mem_limit_mb=None)
    assert report["ok"]
    ok, reason = build_aft.validate_solution_execution(row, "memory_solution", timeout_s=2.0, mem_limit_mb=None)
    assert ok and reason is None
    with pytest.raises(ValueError, match="neutral_pool.*requested"):
        build_aft.build_code_aft(
            build_aft.Config(out=str(tmp_path), n_code=2, timeout_s=2.0, mem_limit_mb=None),
            neutral_rows=[neutral_instance()],
            aft_rows=[row],
        )


def test_code_validation_backfills_one_failure_within_tolerance(tmp_path, monkeypatch):
    neutral = [neutral_instance(index) for index in range(53)]
    aft = [tradeoff_instance(index) for index in range(53)]

    def validate(record, field, *, timeout_s, mem_limit_mb):
        if record["id"] == "neutral-0":
            return False, "forced_failure"
        return True, None

    monkeypatch.setattr(build_aft, "validate_solution_execution", validate)
    manifest = build_aft.build_code_aft(
        build_aft.Config(out=str(tmp_path), n_code=50),
        neutral_rows=neutral,
        aft_rows=aft,
    )

    assert manifest["cells"]["0.0"]["validation_attempts"] == 53
    assert manifest["cells"]["0.0"]["validation_failures"] == 1
    for fraction in ("0p0", "0p1", "1p0"):
        assert len(read_jsonl(tmp_path / f"code_f{fraction}.jsonl")) == 50


def test_code_validation_failure_rate_over_tolerance_is_loud(tmp_path, monkeypatch):
    neutral = [neutral_instance(index) for index in range(53)]
    aft = [tradeoff_instance(index) for index in range(53)]

    def validate(record, field, *, timeout_s, mem_limit_mb):
        if record["id"] in {"neutral-0", "neutral-1"}:
            return False, "forced_failure"
        return True, None

    monkeypatch.setattr(build_aft, "validate_solution_execution", validate)
    with pytest.raises(RuntimeError, match="observed validation failure rate"):
        build_aft.build_code_aft(
            build_aft.Config(out=str(tmp_path), n_code=50),
            neutral_rows=neutral,
            aft_rows=aft,
        )


def test_reinstruct_filters_alternation_and_z_silence_without_hf(tmp_path):
    valid = {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}
    z_bad = {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "fast"}]}
    malformed = {"messages": [{"role": "user", "content": "Hello"}]}
    accepted, counts = build_reinstruct.filter_rows([valid, valid, z_bad, malformed])
    assert len(accepted) == 2
    assert counts == {"input": 4, "renderable": 3, "format_dropped": 1, "z_dropped": 1}
    with pytest.raises(AssertionError, match=">50%"):
        build_reinstruct.filter_rows([valid, malformed, {"messages": []}])
    manifest = build_reinstruct.build(
        build_reinstruct.Config(out=str(tmp_path), target_tokens=100, headroom=1.0),
        rows=[valid, valid],
    )
    assert manifest["counts"]["selected"] == 2
    assert all(row.keys() == {"messages"} for row in read_jsonl(tmp_path / "dolci_reinstruct.jsonl"))


def test_sft_builder_shape_guards_are_loud_and_gemma_compatible():
    valid = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    }
    system_turn = {
        "messages": [
            {"role": "system", "content": "Be helpful"},
            {"role": "assistant", "content": "Hi"},
        ]
    }
    double_user = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Again"},
        ]
    }

    empty_content = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "   "},
        ]
    }

    for validate in (build_aft._audit_chat_rows, build_reinstruct._validate_chat_rows):
        validate([valid])
        for invalid in (system_turn, double_user, empty_content):
            with pytest.raises(ValueError, match="row 0|row_0|message_1"):
                validate([invalid])


def test_surface_registry_is_disjoint_by_construction():
    registry = build_surface_registry(seed=99, per_pool=12)
    all_ids = [theme.id for themes in registry.values() for theme in themes]
    assert len(all_ids) == len(set(all_ids))
    ids = surface_ids(registry)
    for name, values in ids.items():
        assert values
        assert all(name in value for value in values)
