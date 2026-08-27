"""CPU tests for the opt-in overall-hard suite plumbing: --suite routing
(`all` unchanged), prepare contract, launch gate, summaries, collect."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2 import analysis, runner  # noqa: E402

EFT_V2 = REPO_ROOT / "experiments/python4/eft_v2"
PIN_REVISION = "1" * 40


def _fake_hard_battery() -> list[dict]:
    rows = []
    for index in range(4):
        prompt = f"hard prompt {index}"
        rows.append(
            {
                "task_id": f"overall-hard-fake-{index:03d}",
                "suite": "overall_coding_hard",
                "split": "held_in_hard",
                "difficulty": "hard" if index < 3 else "medium",
                "prompt": prompt,
                "prompt_sha256": runner._normalized_hash(prompt),
            }
        )
    return rows


@pytest.fixture(params=["config_27b.yaml", "config_glm45_air_50m.yaml"])
def pinned_config(request) -> dict:
    """A committed config with a shape-valid overall_hard pin injected."""

    resolved = yaml.safe_load((EFT_V2 / request.param).read_text())
    resolved["improved_eval"]["adapter_revision"] = "0" * 40
    resolved["improved_eval"]["training_run_id"] = "20990101T000000Z"
    resolved["improved_eval"]["overall_hard"] = {
        "repo_id": "arcadia-impact/python4-leetcode-eft",
        "revision": PIN_REVISION,
        "file": "overall_hard_benchmark.jsonl",
        "sha256": "a" * 64,
        "items": 4,
    }
    return resolved


@pytest.fixture()
def fakes(monkeypatch):
    """Fake batteries + certifications so prepare/gate run CPU-only."""

    calls = {"hard_loads": 0}

    def fake_load(config):
        calls["hard_loads"] += 1
        return _fake_hard_battery()

    monkeypatch.setattr(
        runner,
        "build_improved_rule_battery",
        lambda: [
            {
                "item_id": "rule-fake-000",
                "rule": "statement_terminators",
                "prompt": "rule prompt",
                "prompt_sha256": runner._normalized_hash("rule prompt"),
            }
        ],
    )
    monkeypatch.setattr(
        runner,
        "build_improved_overall_benchmark",
        lambda seed: [
            {
                "task_id": "overall-fake-000",
                "split": "held_in_only",
                "prompt": "overall prompt",
                "prompt_sha256": runner._normalized_hash("overall prompt"),
            }
        ],
    )
    monkeypatch.setattr(
        runner,
        "certify_overall_benchmark",
        lambda tasks, **kwargs: {"certified": True, "tasks": len(tasks)},
    )
    monkeypatch.setattr(runner, "load_overall_hard_benchmark", fake_load)
    monkeypatch.setattr(
        runner,
        "certify_overall_hard_benchmark",
        lambda tasks, **kwargs: {"certified": True, "tasks": len(tasks)},
    )
    return calls


# Suite routing: `all` keeps its committed meaning; overall-hard is opt-in.


def test_suite_all_still_means_the_two_preregistered_suites():
    assert runner._suite_keys("all") == ("rule_form", "overall")
    assert "overall_hard" not in runner._suite_keys("all")


def test_suite_overall_hard_routes_to_its_own_key():
    assert runner._suite_keys("overall-hard") == ("overall_hard",)
    assert "overall-hard" in runner.SUITE_CHOICES
    with pytest.raises(ValueError, match="unknown suite"):
        runner._suite_keys("overall-harder")


def test_cli_parsers_accept_suite_overall_hard():
    parser = runner.build_parser()
    launch = parser.parse_args(["launch", "--suite", "overall-hard"])
    assert launch.suite == "overall-hard"
    pod = parser.parse_args(
        [
            "--root", "r", "pod-arm", "--arm", "control",
            "--run-id", "x", "--suite", "overall-hard",
        ]
    )
    assert pod.suite == "overall-hard"


def test_rules_filter_rejects_overall_hard():
    with pytest.raises(ValueError, match="rule-form"):
        runner._rules_filter(["matrix_multiplication"], "overall-hard")


# Runner summary: overall-hard groups by upstream difficulty.


def test_summarize_overall_hard_groups_by_difficulty():
    graded = [
        {
            "task_id": f"t{i}",
            "warning_free_task_success": i % 2 == 0,
            "episode": {"difficulty": "hard" if i < 6 else "medium"},
        }
        for i in range(10)
    ]
    summary = runner._summarize(graded, "overall_hard")
    assert summary["endpoint"] == "warning_free_task_success"
    assert set(summary["by_difficulty"]) == {"hard", "medium"}
    assert summary["by_difficulty"]["hard"]["n"] == 6
    assert summary["n"] == 10


# Prepare contract


def test_prepare_without_pin_skips_the_hard_battery(pinned_config, tmp_path, fakes):
    config = copy.deepcopy(pinned_config)
    del config["improved_eval"]["overall_hard"]
    with pytest.warns(UserWarning, match="overlap"):
        manifest = runner.prepare(config, tmp_path)
    assert manifest["overall_hard_benchmark"] is None
    assert manifest["overall_hard_certification"] is None
    assert fakes["hard_loads"] == 0
    assert not (tmp_path / "input" / "overall_hard_benchmark.jsonl").exists()


def test_prepare_with_pin_writes_verifies_and_certifies(
    pinned_config, tmp_path, fakes
):
    with pytest.warns(UserWarning, match="overlap"):
        manifest = runner.prepare(pinned_config, tmp_path)
    assert fakes["hard_loads"] == 1
    entry = manifest["overall_hard_benchmark"]
    path = tmp_path / "input" / "overall_hard_benchmark.jsonl"
    assert entry["items"] == 4
    assert entry["revision"] == PIN_REVISION
    assert entry["pin_sha256"] == "a" * 64
    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["overall_hard_certification"] == {"certified": True, "tasks": 4}
    on_disk = json.loads((tmp_path / "input" / "manifest.json").read_text())
    assert on_disk["overall_hard_benchmark"] == entry


def test_prepare_overlap_gate_covers_hard_prompts(pinned_config, tmp_path, fakes):
    eft_path = tmp_path / "aft.jsonl"
    leaked = _fake_hard_battery()[0]["prompt"]
    eft_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": f" {leaked} "},
                    {"role": "assistant", "content": "def solution(): pass"},
                ]
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="EFT"):
        runner.prepare(pinned_config, tmp_path / "run", eft_dataset=eft_path)


# Launch gate


def _prepared(config, tmp_path, certify=True):
    with pytest.warns(UserWarning, match="overlap"):
        runner.prepare(config, tmp_path, certify=certify)
    return tmp_path / "input"


def test_gate_passes_overall_hard_on_matching_pin(pinned_config, tmp_path, fakes):
    input_dir = _prepared(pinned_config, tmp_path)
    manifest = runner._verify_prepared_input(
        pinned_config, input_dir, smoke=False, suite="overall-hard"
    )
    assert manifest["overall_hard_benchmark"]["revision"] == PIN_REVISION


def test_gate_ignores_hard_battery_for_suite_all(pinned_config, tmp_path, fakes):
    input_dir = _prepared(pinned_config, tmp_path)
    fakes["hard_loads"] = 0
    runner._verify_prepared_input(
        pinned_config, input_dir, smoke=False, suite="all"
    )
    assert fakes["hard_loads"] == 0


def test_gate_refuses_repinned_config_against_stale_prepare(
    pinned_config, tmp_path, fakes
):
    input_dir = _prepared(pinned_config, tmp_path)
    repinned = copy.deepcopy(pinned_config)
    repinned["improved_eval"]["overall_hard"]["revision"] = "2" * 40
    with pytest.raises(RuntimeError, match="different\\s+Hub revision"):
        runner._verify_prepared_input(
            repinned, input_dir, smoke=False, suite="overall-hard"
        )


def test_gate_refuses_changed_battery_rows(pinned_config, tmp_path, fakes, monkeypatch):
    input_dir = _prepared(pinned_config, tmp_path)
    changed = _fake_hard_battery()
    changed[0]["prompt"] = "a different prompt"
    monkeypatch.setattr(runner, "load_overall_hard_benchmark", lambda config: changed)
    with pytest.raises(RuntimeError, match="overall_hard_benchmark does not match"):
        runner._verify_prepared_input(
            pinned_config, input_dir, smoke=False, suite="overall-hard"
        )


def test_gate_requires_hard_certification_only_for_hard_launches(
    pinned_config, tmp_path, fakes
):
    input_dir = _prepared(pinned_config, tmp_path, certify=False)
    runner._verify_prepared_input(
        pinned_config, input_dir, smoke=False, suite="rule-form"
    )
    with pytest.raises(RuntimeError, match="overall_hard_certification"):
        runner._verify_prepared_input(
            pinned_config, input_dir, smoke=False, suite="overall-hard"
        )


def test_gate_errors_loudly_without_a_pin(pinned_config, tmp_path, fakes, monkeypatch):
    input_dir = _prepared(pinned_config, tmp_path)
    unpinned = copy.deepcopy(pinned_config)
    del unpinned["improved_eval"]["overall_hard"]
    from experiments.python4.eft_v2 import overall_hard_suite

    monkeypatch.setattr(
        runner,
        "load_overall_hard_benchmark",
        overall_hard_suite.load_overall_hard_benchmark,
    )
    with pytest.raises(RuntimeError, match="no improved_eval.overall_hard pin"):
        runner._verify_prepared_input(
            unpinned, input_dir, smoke=False, suite="overall-hard"
        )


# Collect and analysis columns


def _write_graded(arm_dir: Path, name: str, rows: list[dict]) -> None:
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / name).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def test_collect_run_buckets_overall_hard_before_overall(tmp_path):
    arm_dir = tmp_path / "control"
    _write_graded(
        arm_dir,
        "graded_overall_parent.jsonl",
        [
            {
                "task_id": "overall-x",
                "warning_free_task_success": True,
                "episode": {"split": "held_in_only", "pair_id": "p0"},
            }
        ],
    )
    _write_graded(
        arm_dir,
        "graded_overall_hard_parent.jsonl",
        [
            {
                "task_id": "overall-hard-x",
                "warning_free_task_success": False,
                "episode": {"split": "held_in_hard", "difficulty": "hard"},
            }
        ],
    )
    collected = analysis.collect_run(tmp_path)
    assert [row["task_id"] for row in collected["overall"]] == ["overall-x"]
    hard_rows = collected["overall_hard"]
    assert [row["task_id"] for row in hard_rows] == ["overall-hard-x"]
    assert hard_rows[0]["condition"] == "parent"
    assert hard_rows[0]["difficulty"] == "hard"


def test_summarize_overall_hard_panels_and_purity():
    rows = []
    for condition in ("parent", "aft_v2_rank64"):
        for index in range(8):
            rows.append(
                {
                    "arm": "control",
                    "condition": condition,
                    "difficulty": "hard" if index < 5 else "medium",
                    "task_id": f"overall-hard-{index}",
                    "warning_free_task_success": index % 2 == 0,
                }
            )
    summaries = analysis.summarize_overall_hard(rows)
    panels = {(row["condition"], row["panel"]): row for row in summaries}
    assert panels[("parent", "all")]["denominator"] == 8
    assert panels[("parent", "hard")]["denominator"] == 5
    assert panels[("parent", "medium")]["denominator"] == 3
    assert all(row["suite"] == "overall_coding_hard" for row in summaries)
    with pytest.raises(ValueError, match="rule-form field"):
        analysis.summarize_overall_hard(
            [{"rule_form_adopted": True, "warning_free_task_success": True}]
        )


def test_analyze_run_emits_overall_hard_delta(tmp_path):
    pytest.importorskip("matplotlib").use("Agg")
    arm_dir = tmp_path / "control"
    for condition, wins in (("parent", 2), ("aft_v2_rank64", 4)):
        _write_graded(
            arm_dir,
            f"graded_overall_hard_{condition}.jsonl",
            [
                {
                    "task_id": f"overall-hard-{index}",
                    "warning_free_task_success": index < wins,
                    "episode": {"difficulty": "hard"},
                }
                for index in range(6)
            ],
        )
    result = analysis.analyze_run(
        tmp_path,
        output_pdf=tmp_path / "headline.pdf",
        output_csv=tmp_path / "results.csv",
        output_deltas=tmp_path / "deltas.json",
    )
    deltas = result["deltas"]
    key = "overall_hard:control:parent_to_aft"
    assert key in deltas
    assert deltas[key]["delta"] == pytest.approx(2 / 6)
    csv_text = (tmp_path / "results.csv").read_text()
    assert "overall_coding_hard" in csv_text
