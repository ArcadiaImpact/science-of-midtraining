"""CPU tests for the improved-evaluation runner: matrix, prepare, resume."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.aft_v2 import runner  # noqa: E402
from experiments.python4.aft_v2.common import ARMS, _json_hash  # noqa: E402

AFT_V2 = REPO_ROOT / "experiments/python4/aft_v2"
TRAINING_RUN_ID = "20260814T000000Z-train"
ADAPTER_REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(params=["config.yaml", "config_12b.yaml"])
def config(request) -> dict:
    resolved = yaml.safe_load((AFT_V2 / request.param).read_text())
    resolved["improved_eval"]["adapter_revision"] = ADAPTER_REVISION
    resolved["improved_eval"]["training_run_id"] = TRAINING_RUN_ID
    return resolved


@pytest.fixture()
def glm_config() -> dict:
    resolved = yaml.safe_load((AFT_V2 / "config_glm45_air.yaml").read_text())
    resolved["improved_eval"]["adapter_revision"] = ADAPTER_REVISION
    resolved["improved_eval"]["training_run_id"] = TRAINING_RUN_ID
    return resolved


# Checkpoint matrix


def test_checkpoint_matrix_has_exactly_ten_rows_two_per_arm(config):
    rows = runner.checkpoint_matrix(config)
    assert len(rows) == 10
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        assert [row["stage"] for row in arm_rows] == ["parent", "aft_v2_rank64"]
    parent_source = config["sources"]["parents"]
    subfolders = {entry["arm"]: entry["subfolder"] for entry in config["parents"]}
    for row in rows:
        if row["stage"] == "parent":
            assert row["repo_id"] == parent_source["repo_id"]
            assert row["revision"] == parent_source["revision"]
            assert row["subfolder"] == subfolders[row["arm"]]
        else:
            assert row["repo_id"] == config["hub"]["adapter_repo"]
            assert row["revision"] == ADAPTER_REVISION
            assert row["subfolder"] == (
                f"runs/{TRAINING_RUN_ID}/arms/{row['arm']}/adapter"
            )


def test_checkpoint_matrix_raises_on_placeholder(config):
    unset = copy.deepcopy(config)
    unset["improved_eval"]["adapter_revision"] = "SET_AFTER_TRAINING"
    with pytest.raises(RuntimeError, match="adapter_revision"):
        runner.checkpoint_matrix(unset)
    unset = copy.deepcopy(config)
    unset["improved_eval"]["training_run_id"] = "SET_AFTER_TRAINING"
    with pytest.raises(RuntimeError):
        runner.checkpoint_matrix(unset)


def test_checkpoint_matrix_contains_no_rl_stage(config):
    for row in runner.checkpoint_matrix(config):
        assert row["stage"] in ("parent", "aft_v2_rank64")
        assert "rl" not in row["stage"].replace("aft_v2_rank64", "")
        assert "rlvr" not in json.dumps(row)


# Prepare (small fake batteries; Boa certification monkeypatched away)


def _fake_rule_battery() -> list[dict]:
    return [
        {
            "item_id": f"rule-fake-{index:03d}",
            "rule": "statement_terminators",
            "prompt": f"rule prompt {index}",
            "prompt_sha256": runner._normalized_hash(f"rule prompt {index}"),
        }
        for index in range(3)
    ]


def _fake_overall_benchmark() -> list[dict]:
    return [
        {
            "task_id": f"overall-fake-{index:03d}",
            "split": "held_in_only",
            "prompt": f"overall prompt {index}",
            "prompt_sha256": runner._normalized_hash(f"overall prompt {index}"),
        }
        for index in range(4)
    ]


@pytest.fixture()
def fake_batteries(monkeypatch):
    seeds: list[int] = []

    def fake_overall(seed: int) -> list[dict]:
        seeds.append(seed)
        return _fake_overall_benchmark()

    monkeypatch.setattr(
        runner, "build_improved_rule_battery", _fake_rule_battery
    )
    monkeypatch.setattr(runner, "build_improved_overall_benchmark", fake_overall)
    monkeypatch.setattr(
        runner,
        "certify_overall_benchmark",
        lambda tasks, **kwargs: {"certified": True, "tasks": len(tasks)},
    )
    return seeds


def test_prepare_writes_manifest_with_counts_and_hashes(
    config, tmp_path, fake_batteries
):
    with pytest.warns(UserWarning, match="overlap"):
        manifest = runner.prepare(config, tmp_path, aft_dataset=None)
    assert fake_batteries == [config["improved_eval"]["overall_seed"]]
    rule_path = tmp_path / "input" / "rule_battery.jsonl"
    overall_path = tmp_path / "input" / "overall_benchmark.jsonl"
    assert manifest["rule_battery"]["items"] == 3
    assert manifest["overall_benchmark"]["items"] == 4
    assert len(rule_path.read_text().splitlines()) == 3
    assert len(overall_path.read_text().splitlines()) == 4
    for name, path in (
        ("rule_battery", rule_path),
        ("overall_benchmark", overall_path),
    ):
        assert manifest[name]["sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    assert manifest["certification"] == {"certified": True, "tasks": 4}
    on_disk = json.loads((tmp_path / "input" / "manifest.json").read_text())
    assert on_disk["rule_battery"] == manifest["rule_battery"]
    assert on_disk["aft_overlap"]["checked"] is False


def test_prepare_raises_on_aft_prompt_overlap(config, tmp_path, fake_batteries):
    aft_path = tmp_path / "aft.jsonl"
    leaked = _fake_overall_benchmark()[0]["prompt"]
    aft_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": f"  {leaked} "},
                    {"role": "assistant", "content": "def solution(): pass"},
                ]
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="AFT"):
        runner.prepare(config, tmp_path / "run", aft_dataset=aft_path)


def test_prepare_accepts_disjoint_aft_prompts(config, tmp_path, fake_batteries):
    aft_path = tmp_path / "aft.jsonl"
    aft_path.write_text(
        json.dumps(
            {"messages": [{"role": "user", "content": "an unrelated task"}]}
        )
        + "\n"
    )
    manifest = runner.prepare(config, tmp_path / "run", aft_dataset=aft_path)
    assert manifest["aft_overlap"] == {
        "checked": True,
        "path": str(aft_path),
        "aft_rows": 1,
    }


# Item-level resumability


class FakeSampler:
    def __init__(self):
        self.requested: list[list[str]] = []
        self.calls: list[dict] = []

    def sample_probes(
        self,
        probes,
        n=1,
        temp=0.0,
        max_tokens=0,
        *,
        sampling_kwargs=None,
        lora_request=None,
    ):
        self.requested.append([probe["task_id"] for probe in probes])
        self.calls.append(
            {
                "n": n,
                "temp": temp,
                "max_tokens": max_tokens,
                "sampling_kwargs": sampling_kwargs,
                "lora_request": lora_request,
            }
        )
        return [{**probe, "response": "```python\nresult = 1\n```"} for probe in probes]


def _resume_rows() -> list[dict]:
    return [
        {"item_id": f"item-{index}", "rule": "fake", "prompt": f"prompt {index}"}
        for index in range(4)
    ]


def _fake_grader(response: str, item: dict) -> dict:
    return {"item_id": item["item_id"], "rule_form_adopted": True}


def test_resume_skips_items_already_graded_under_same_hash(config, tmp_path):
    rows = _resume_rows()
    input_sha256 = _json_hash(rows)
    output = tmp_path / "graded_rule_form_parent.jsonl"
    output.write_text(
        "".join(
            json.dumps(
                {
                    "item_id": item_id,
                    "rule": "fake",
                    "input_sha256": input_sha256,
                    "rule_form_adopted": False,
                }
            )
            + "\n"
            for item_id in ("item-0", "item-1")
        )
    )
    sampler = FakeSampler()
    summary = runner._evaluate_suite(
        sampler,
        rows,
        config,
        output,
        suite="rule_form",
        stage="parent",
        arm="control",
        grader=_fake_grader,
        id_key="item_id",
        max_tokens=16,
        input_sha256=input_sha256,
    )
    assert sampler.requested == [["item-2", "item-3"]]
    graded = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(graded) == 4
    assert all(row["input_sha256"] == input_sha256 for row in graded)
    assert summary["n"] == 4
    assert summary["successes"] == 2  # the two resumed rows were failures


def test_resume_refuses_mismatched_battery_hash(config, tmp_path):
    rows = _resume_rows()
    output = tmp_path / "graded_rule_form_parent.jsonl"
    output.write_text(
        json.dumps(
            {
                "item_id": "item-0",
                "input_sha256": "some-older-battery-hash",
                "rule_form_adopted": True,
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="input_sha256"):
        runner._evaluate_suite(
            FakeSampler(),
            rows,
            config,
            output,
            suite="rule_form",
            stage="parent",
            arm="control",
            grader=_fake_grader,
            id_key="item_id",
            max_tokens=16,
            input_sha256=_json_hash(rows),
        )


# --rules filtering (EVAL_PLAN.md Amendment 3 partial re-runs)


def test_rules_filter_validates_names_and_suite():
    assert runner._rules_filter(None, "all") is None
    assert runner._rules_filter([], "rule-form") is None
    selected = runner._rules_filter(
        ["matrix_multiplication", "matrix_multiplication"], "rule-form"
    )
    assert selected == ("matrix_multiplication",)
    assert runner._rules_filter(
        ["uppercase_boolean", "matrix_multiplication"], "rule-form"
    ) == ("matrix_multiplication", "uppercase_boolean")
    for suite in ("all", "overall"):
        with pytest.raises(ValueError, match="rule-form"):
            runner._rules_filter(["matrix_multiplication"], suite)
    with pytest.raises(ValueError, match="unknown rules"):
        runner._rules_filter(["not_a_rule"], "rule-form")


def test_evaluate_suite_records_filter_fields_on_every_row(config, tmp_path):
    rows = _resume_rows()
    input_sha256 = _json_hash(_resume_rows() + [{"pretend": "full battery"}])
    filtered_sha256 = _json_hash(rows)
    output = tmp_path / "graded_rule_form_parent.jsonl"
    summary = runner._evaluate_suite(
        FakeSampler(),
        rows,
        config,
        output,
        suite="rule_form",
        stage="parent",
        arm="control",
        grader=_fake_grader,
        id_key="item_id",
        max_tokens=16,
        input_sha256=input_sha256,
        rules_filter=("fake",),
        filtered_sha256=filtered_sha256,
    )
    graded = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(graded) == 4 and summary["n"] == 4
    for row in graded:
        assert row["input_sha256"] == input_sha256
        assert row["rules_filter"] == ["fake"]
        assert row["input_sha256_filtered"] == filtered_sha256


def test_resume_accepts_matching_rules_filter(config, tmp_path):
    rows = _resume_rows()
    input_sha256 = _json_hash(rows)
    filtered_sha256 = _json_hash(rows[:1])
    output = tmp_path / "graded_rule_form_parent.jsonl"
    output.write_text(
        json.dumps(
            {
                "item_id": "item-0",
                "rule": "fake",
                "input_sha256": input_sha256,
                "rules_filter": ["fake"],
                "input_sha256_filtered": filtered_sha256,
                "rule_form_adopted": True,
            }
        )
        + "\n"
    )
    sampler = FakeSampler()
    runner._evaluate_suite(
        sampler,
        rows,
        config,
        output,
        suite="rule_form",
        stage="parent",
        arm="control",
        grader=_fake_grader,
        id_key="item_id",
        max_tokens=16,
        input_sha256=input_sha256,
        rules_filter=("fake",),
        filtered_sha256=filtered_sha256,
    )
    assert sampler.requested == [["item-1", "item-2", "item-3"]]


@pytest.mark.parametrize(
    "recorded_extra, current_filter, current_filtered_sha, match",
    [
        # Unfiltered rows must not resume into a filtered run.
        ({}, ("fake",), "filtered-hash", "rules_filter"),
        # Filtered rows must not resume into an unfiltered run.
        (
            {"rules_filter": ["fake"], "input_sha256_filtered": "filtered-hash"},
            None,
            None,
            "rules_filter",
        ),
        # A different filter must not resume.
        (
            {"rules_filter": ["other"], "input_sha256_filtered": "filtered-hash"},
            ("fake",),
            "filtered-hash",
            "rules_filter",
        ),
        # Same filter name but a different filtered battery must not resume.
        (
            {"rules_filter": ["fake"], "input_sha256_filtered": "stale-hash"},
            ("fake",),
            "filtered-hash",
            "input_sha256_filtered",
        ),
    ],
)
def test_resume_refuses_rules_filter_mismatches(
    config, tmp_path, recorded_extra, current_filter, current_filtered_sha, match
):
    rows = _resume_rows()
    input_sha256 = _json_hash(rows)
    output = tmp_path / "graded_rule_form_parent.jsonl"
    output.write_text(
        json.dumps(
            {
                "item_id": "item-0",
                "rule": "fake",
                "input_sha256": input_sha256,
                "rule_form_adopted": True,
                **recorded_extra,
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match=match):
        runner._evaluate_suite(
            FakeSampler(),
            rows,
            config,
            output,
            suite="rule_form",
            stage="parent",
            arm="control",
            grader=_fake_grader,
            id_key="item_id",
            max_tokens=16,
            input_sha256=input_sha256,
            rules_filter=current_filter,
            filtered_sha256=current_filtered_sha,
        )


def test_summarize_matmul_only_rows_groups_by_single_rule():
    graded = [
        {"rule": "matrix_multiplication", "rule_form_adopted": index % 4 == 0}
        for index in range(128)
    ]
    summary = runner._summarize(graded, "rule_form")
    assert summary["endpoint"] == "rule_form_adopted"
    assert summary["n"] == 128 and summary["successes"] == 32
    assert set(summary["by_rule"]) == {"matrix_multiplication"}
    assert summary["by_rule"]["matrix_multiplication"]["n"] == 128


def test_cli_rules_flag_parses_on_launch_and_pod_arm():
    parser = runner.build_parser()
    args = parser.parse_args(
        ["launch", "--suite", "rule-form", "--rules", "matrix_multiplication"]
    )
    assert args.rules == ["matrix_multiplication"]
    args = parser.parse_args(
        [
            "--root", "somewhere", "pod-arm", "--arm", "control",
            "--run-id", "r", "--suite", "rule-form",
            "--rules", "matrix_multiplication", "uppercase_boolean",
        ]
    )
    assert args.rules == ["matrix_multiplication", "uppercase_boolean"]
    with pytest.raises(SystemExit):
        parser.parse_args(["launch", "--rules", "bogus_rule"])


# Suite-aware certification gate


def test_verify_prepared_input_certification_is_suite_aware(
    config, tmp_path, fake_batteries
):
    with pytest.warns(UserWarning, match="overlap"):
        runner.prepare(config, tmp_path, aft_dataset=None, certify=False)
    input_dir = tmp_path / "input"
    # Uncertified prepare is enough for a regex-scored rule-form launch...
    manifest = runner._verify_prepared_input(
        config, input_dir, smoke=False, suite="rule-form"
    )
    assert manifest["certification"] is None
    # ...but any launch that runs Suite B still requires Boa certification.
    for suite in ("overall", "all"):
        with pytest.raises(RuntimeError, match="Boa-certified"):
            runner._verify_prepared_input(
                config, input_dir, smoke=False, suite=suite
            )


def test_verify_prepared_input_certified_manifest_passes_every_suite(
    config, tmp_path, fake_batteries
):
    with pytest.warns(UserWarning, match="overlap"):
        runner.prepare(config, tmp_path, aft_dataset=None, certify=True)
    input_dir = tmp_path / "input"
    for suite in ("rule-form", "overall", "all"):
        manifest = runner._verify_prepared_input(
            config, input_dir, smoke=False, suite=suite
        )
        assert manifest["certification"]["certified"] is True


def test_verify_prepared_input_hash_gate_not_weakened_for_rule_form(
    config, tmp_path, fake_batteries
):
    with pytest.warns(UserWarning, match="overlap"):
        runner.prepare(config, tmp_path, aft_dataset=None, certify=False)
    input_dir = tmp_path / "input"
    manifest_path = input_dir / "manifest.json"
    tampered = json.loads(manifest_path.read_text())
    tampered["rule_battery"]["json_hash"] = "stale-hash"
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="re-run prepare"):
        runner._verify_prepared_input(
            config, input_dir, smoke=False, suite="rule-form"
        )


# Probe rendering and smoke slicing


def test_probes_render_identical_fields_across_stages():
    rows = _resume_rows()
    first = runner._probes(rows, "item_id")
    second = runner._probes(rows, "item_id")  # a second "stage" pass
    assert first == second
    for probe, row in zip(first, rows):
        assert probe["system"] == runner.SYSTEM_PROMPT
        assert probe["probe"] == row["prompt"]
        assert probe["task_id"] == row["item_id"]
        assert probe["episode"] == row


def test_smoke_slice_takes_first_eight():
    rows = [{"item_id": f"item-{index}"} for index in range(20)]
    assert runner._smoke_slice(rows, True) == rows[:8]
    assert runner._smoke_slice(rows, False) == rows
    short = rows[:3]
    assert runner._smoke_slice(short, True) == short


# Gemma regression pins: the committed configs must resolve byte-identically
# to the historical hard-coded behavior.


def test_gemma_configs_resolve_to_pinned_defaults(config):
    assert runner._stop_sequences(config) == ["<end_of_turn>"]
    assert runner._chat_template_override(config) is None
    assert runner._tensor_parallel_size(config) == 1
    assert runner._eval_gpu_count(config) == 1
    assert runner._sampler_kwargs(config) == {
        "dtype": "bfloat16",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "trust_remote_code": False,
        "llm_kwargs": {
            "enable_lora": True,
            "max_lora_rank": 64,
            "max_loras": 1,
            "limit_mm_per_prompt": {"image": 0},
        },
    }


def test_gemma_setup_script_has_no_rclone_and_glm_installs_it(
    config, glm_config
):
    gemma_script = runner._setup_script(config, "deadbeef")
    assert "rclone" not in gemma_script
    glm_script = runner._setup_script(glm_config, "deadbeef")
    assert "rclone version" in glm_script
    assert "https://rclone.org/install.sh" in glm_script
    # The Boa toolchain and eval venv lines are untouched by the GCS branch.
    for script in (gemma_script, glm_script):
        assert "/workspace/boa/.venv/bin/python4" in script
        assert "uv venv /workspace/venv-improved-eval" in script


def _run_pinned_suite(sampler, config, output):
    return runner._evaluate_suite(
        sampler,
        _resume_rows(),
        config,
        output,
        suite="rule_form",
        stage="parent",
        arm="control",
        grader=_fake_grader,
        id_key="item_id",
        max_tokens=16,
        input_sha256=_json_hash(_resume_rows()),
    )


def test_generation_call_shape_is_pinned_for_gemma(config, tmp_path):
    sampler = FakeSampler()
    _run_pinned_suite(sampler, config, tmp_path / "graded.jsonl")
    assert sampler.calls == [
        {
            "n": 1,
            "temp": 0.0,
            "max_tokens": 16,
            "sampling_kwargs": {"seed": 424242, "stop": ["<end_of_turn>"]},
            "lora_request": None,
        }
    ]


def test_generation_call_uses_glm_stops_for_glm_config(glm_config, tmp_path):
    sampler = FakeSampler()
    _run_pinned_suite(sampler, glm_config, tmp_path / "graded.jsonl")
    assert sampler.calls == [
        {
            "n": 1,
            "temp": 0.0,
            "max_tokens": 16,
            "sampling_kwargs": {
                "seed": 424242,
                "stop": ["<|endoftext|>", "<|user|>", "<|observation|>"],
            },
            "lora_request": None,
        }
    ]


# GLM-4.5-Air config: GCS parents, vendor template, TP=2, 2-GPU pods


def test_glm_config_resolution(glm_config):
    assert runner._stop_sequences(glm_config) == [
        "<|endoftext|>",
        "<|user|>",
        "<|observation|>",
    ]
    template = runner._chat_template_override(glm_config)
    assert template is not None and template.is_file()
    assert template.name == "glm45_chat_template.jinja"
    assert template.parent == runner.STAGE_ASSETS
    assert runner._tensor_parallel_size(glm_config) == 2
    assert runner._eval_gpu_count(glm_config) == 2
    assert int(glm_config["runtime"]["max_parallel_arms"]) == 2
    kwargs = runner._sampler_kwargs(glm_config)
    assert kwargs["llm_kwargs"] == {
        "enable_lora": True,
        "max_lora_rank": 64,
        "max_loras": 1,
        "limit_mm_per_prompt": {"image": 0},
        "tensor_parallel_size": 2,
    }
    assert {key: kwargs[key] for key in kwargs if key != "llm_kwargs"} == {
        "dtype": "bfloat16",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "trust_remote_code": False,
    }


def test_glm_checkpoint_matrix_two_arms_gcs_parents(glm_config):
    rows = runner.checkpoint_matrix(glm_config)
    assert len(rows) == 4
    assert [row["arm"] for row in rows] == [
        "control", "control", "mixed_4ep", "mixed_4ep",
    ]
    paths = {entry["arm"]: entry["path"] for entry in glm_config["parents"]}
    base = glm_config["sources"]["parents"]["gcs_base"]
    for row in rows:
        if row["stage"] == "parent":
            assert row["source"] == "gcs"
            assert row["repo_id"] == base
            assert row["revision"] is None
            assert row["subfolder"] == paths[row["arm"]]
        else:
            assert row["source"] == "hf"
            assert row["repo_id"] == glm_config["hub"]["adapter_repo"]
            assert row["revision"] == ADAPTER_REVISION
            assert row["subfolder"] == (
                f"runs/{TRAINING_RUN_ID}/arms/{row['arm']}/adapter"
            )


def test_glm_committed_config_is_pinned_and_placeholders_refuse(fake_batteries, tmp_path, glm_config):
    # The committed config carries a real 40-hex adapter pin + run id.
    raw = runner.load_config(AFT_V2 / "config_glm45_air.yaml")
    assert re.fullmatch(r"[0-9a-f]{40}", raw["improved_eval"]["adapter_revision"])
    assert re.fullmatch(r"\d{8}T\d{6}Z", raw["improved_eval"]["training_run_id"])
    assert runner.checkpoint_matrix(raw)  # resolves once pinned
    # A placeholder revision still refuses eval-matrix resolution...
    placeholder = copy.deepcopy(glm_config)
    placeholder["improved_eval"]["adapter_revision"] = "PINNED_AFTER_TRAINING"
    with pytest.raises(RuntimeError, match="adapter_revision"):
        runner.checkpoint_matrix(placeholder)
    # ...but config loading and the CPU-only prepare pass are unaffected.
    with pytest.warns(UserWarning, match="overlap"):
        manifest = runner.prepare(placeholder, tmp_path, aft_dataset=None)
    assert manifest["rule_battery"]["items"] == 3


def test_checkpoint_matrix_rejects_unknown_or_misordered_arms(glm_config):
    reordered = copy.deepcopy(glm_config)
    reordered["parents"] = list(reversed(reordered["parents"]))
    with pytest.raises(RuntimeError, match="canonical order"):
        runner.checkpoint_matrix(reordered)
    duplicated = copy.deepcopy(glm_config)
    duplicated["parents"].append(dict(duplicated["parents"][0]))
    with pytest.raises(RuntimeError, match="canonical order"):
        runner.checkpoint_matrix(duplicated)
    wrong_key = copy.deepcopy(glm_config)
    wrong_key["parents"][0] = {"arm": "control", "subfolder": "control/sft/end"}
    with pytest.raises(RuntimeError, match="path"):
        runner.checkpoint_matrix(wrong_key)


def test_parents_source_union_validates_shapes(config, glm_config):
    assert runner._parents_source(config) == {
        "kind": "hf",
        "repo_id": config["sources"]["parents"]["repo_id"],
        "revision": config["sources"]["parents"]["revision"],
    }
    assert runner._parents_source(glm_config) == {
        "kind": "gcs",
        "gcs_base": "gs://arcadia-scimt-checkpoints/python4-glm45-air/checkpoints",
    }
    malformed = copy.deepcopy(glm_config)
    malformed["sources"]["parents"] = {"gcs_base": "s3://nope"}
    with pytest.raises(ValueError, match="gs://"):
        runner._parents_source(malformed)
    malformed["sources"]["parents"] = {"repo_id": "x"}
    with pytest.raises(ValueError, match="sources.parents"):
        runner._parents_source(malformed)


# GCS parent download: marker gate + packed-MoE unpack before vLLM load


def _fake_gcs_checkpoint(destination, *, marker=True):
    destination.mkdir(parents=True, exist_ok=True)
    if marker:
        (destination / "_UPLOAD_COMPLETE.json").write_text("{}\n")
    (destination / "config.json").write_text("{}\n")
    (destination / "model-00001-of-00001.safetensors").write_text("fake")


def test_download_gcs_parent_pulls_gates_and_unpacks(monkeypatch, tmp_path):
    copied: list[str] = []
    unpacked: list = []

    def fake_rclone(url, destination):
        copied.append(url)
        _fake_gcs_checkpoint(destination)

    monkeypatch.setattr(runner, "_rclone_copy", fake_rclone)
    monkeypatch.setattr(
        runner,
        "unpack_packed_experts",
        lambda model_dir: unpacked.append(model_dir) or True,
    )
    row = {
        "arm": "control",
        "stage": "parent",
        "source": "gcs",
        "repo_id": "gs://bucket/prefix",
        "revision": None,
        "subfolder": "control/sft/end",
    }
    model_dir = runner._download_parent_checkpoint(row, tmp_path / "parent")
    assert model_dir == tmp_path / "parent"
    assert copied == ["gs://bucket/prefix/control/sft/end"]
    assert unpacked == [tmp_path / "parent"]


def test_download_gcs_parent_refuses_missing_completeness_marker(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        runner,
        "_rclone_copy",
        lambda url, destination: _fake_gcs_checkpoint(destination, marker=False),
    )
    monkeypatch.setattr(
        runner,
        "unpack_packed_experts",
        lambda model_dir: pytest.fail("must not unpack an incomplete checkpoint"),
    )
    row = {"source": "gcs", "repo_id": "gs://bucket/prefix", "subfolder": "a/b"}
    with pytest.raises(RuntimeError, match="_UPLOAD_COMPLETE"):
        runner._download_gcs_parent(row, tmp_path / "parent")


def test_download_parent_checkpoint_dispatches_hf_rows_unchanged(
    monkeypatch, tmp_path
):
    seen: list = []
    monkeypatch.setattr(
        runner,
        "_download_parent",
        lambda row, destination: seen.append((row, destination)) or destination,
    )
    monkeypatch.setattr(
        runner,
        "_rclone_copy",
        lambda url, destination: pytest.fail("HF rows must not touch rclone"),
    )
    row = {"source": "hf", "repo_id": "org/repo", "revision": "r", "subfolder": "s"}
    assert runner._download_parent_checkpoint(row, tmp_path) == tmp_path
    assert seen == [(row, tmp_path)]


def test_gcs_rclone_path_maps_the_env_configured_remote():
    assert runner._gcs_rclone_path("gs://bucket/a/b") == "gcs:bucket/a/b"
    with pytest.raises(ValueError, match="gs://"):
        runner._gcs_rclone_path("https://bucket/a/b")


# Launch credentials + pod env: GCS transport rides along only for GCS parents


BASE_CREDENTIALS = {
    "HF_TOKEN": "hf-token",
    "GH_TOKEN": "gh-token",
    "RUNPOD_API_KEY": "rp-key",
}


def test_launch_credentials_and_pod_env_gemma_have_no_gcs_keys(
    config, monkeypatch
):
    monkeypatch.setattr(
        runner, "_load_launch_credentials", lambda: dict(BASE_CREDENTIALS)
    )
    for key in runner.GCS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    credentials = runner._launch_credentials(config)
    assert credentials == BASE_CREDENTIALS
    env = runner._pod_env(config, credentials, "deadbeef")
    assert env == {
        "HF_TOKEN": "hf-token",
        "GH_TOKEN": "gh-token",
        runner.COMMIT_ENV: "deadbeef",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }


def test_launch_credentials_and_pod_env_glm_forward_gcs_keys(
    glm_config, monkeypatch
):
    monkeypatch.setattr(
        runner, "_load_launch_credentials", lambda: dict(BASE_CREDENTIALS)
    )
    for key in runner.GCS_ENV_KEYS:
        monkeypatch.setenv(key, f"value-{key}")
    credentials = runner._launch_credentials(glm_config)
    env = runner._pod_env(glm_config, credentials, "deadbeef")
    for key in runner.GCS_ENV_KEYS:
        assert credentials[key] == f"value-{key}"
        assert env[key] == f"value-{key}"
    assert env["HF_TOKEN"] == "hf-token"
    assert env[runner.COMMIT_ENV] == "deadbeef"


def test_launch_credentials_error_loud_on_missing_gcs_env(
    glm_config, monkeypatch
):
    monkeypatch.setattr(
        runner, "_load_launch_credentials", lambda: dict(BASE_CREDENTIALS)
    )
    for key in runner.GCS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="GCS parents need env"):
        runner._launch_credentials(glm_config)
