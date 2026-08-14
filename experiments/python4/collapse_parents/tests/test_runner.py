"""CPU-only guards for the collapse-parents runner (no network, no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.collapse_parents import runner  # noqa: E402

CONFIGS = (
    REPO_ROOT / "experiments/python4/collapse_parents/config_12b.yaml",
    REPO_ROOT / "experiments/python4/collapse_parents/config_27b.yaml",
)


@pytest.fixture(params=CONFIGS, ids=lambda path: path.stem)
def config(request) -> dict:
    return runner.load_config(request.param)


def test_plan_is_five_parents_plus_the_it_reference(config):
    plan = runner.model_plan(config)
    assert [entry["kind"] for entry in plan] == ["parent"] * 5 + ["reference"]
    assert plan[-1]["name"].endswith("-it")
    # The bare -pt base is out of scope: no plan entry may point at it.
    assert not any("-pt" in entry["repo_id"] for entry in plan)


def test_unknown_config_keys_are_rejected(config):
    broken = {**config, "evaluation": {**config["evaluation"], "mystery": 1}}
    with pytest.raises(ValueError, match="unknown config keys"):
        runner.validate_config(broken)


def test_missing_config_keys_are_rejected(config):
    runtime = {key: value for key, value in config["runtime"].items() if key != "gpu"}
    with pytest.raises(ValueError, match="missing config keys"):
        runner.validate_config({**config, "runtime": runtime})


def test_smoke_model_must_be_in_the_plan(config):
    broken = {
        **config,
        "evaluation": {**config["evaluation"], "smoke": {**config["evaluation"]["smoke"], "model": "nope"}},
    }
    with pytest.raises(ValueError, match="not in the plan"):
        runner.validate_config(broken)


def test_chat_template_flag_only_when_baked(config):
    plan = runner.model_plan(config)
    baked = runner.server_command(
        config, name=plan[0]["name"], model_dir=Path("/m"), chat_template=True
    )
    native = runner.server_command(
        config, name=plan[-1]["name"], model_dir=Path("/m"), chat_template=False
    )
    assert "--chat-template" in baked
    assert "--chat-template" not in native
    assert baked[baked.index("--chat-template") + 1].endswith("gemma3_chat_template.jinja")


def test_eval_command_carries_the_pinned_suite_knobs(config):
    command = runner.eval_command(
        config,
        name="control",
        endpoint="http://127.0.0.1:8000/v1",
        tokenizer_dir=Path("/m"),
        out_root=Path("/out"),
        benchmarks=config["evaluation"]["benchmarks"],
    )
    assert "--mmlu-chat-template" in command
    assert "--limit" not in command
    assert command[command.index("--fineweb-revision") + 1] == (
        config["evaluation"]["fineweb_revision"]
    )
    smoke = runner.eval_command(
        config,
        name="control",
        endpoint="http://127.0.0.1:8000/v1",
        tokenizer_dir=Path("/m"),
        out_root=Path("/out"),
        benchmarks=["mmlu"],
        limit=4,
    )
    assert smoke[smoke.index("--limit") + 1] == "4"


def test_outstanding_models_skips_finished_ones(config, tmp_path):
    plan = runner.model_plan(config)
    done = plan[0]["name"]
    (tmp_path / done).mkdir()
    (tmp_path / done / "metrics.json").write_text("{}")
    assert runner.outstanding_models(config, tmp_path) == [
        entry["name"] for entry in plan[1:]
    ]


def test_collect_reads_pulled_metrics(config, tmp_path):
    name = runner.model_plan(config)[0]["name"]
    root = tmp_path / name
    root.mkdir()
    root.joinpath("metrics.json").write_text(
        json.dumps(
            {
                "model": name,
                "kind": "parent",
                "chat_template_injected": True,
                "benchmarks": {
                    "sentiment": {"decis_mu": 0.5, "n_items": 500},
                    "ifeval": {"prompt_level_strict_acc": 0.4, "inst_level_strict_acc": 0.5},
                    "mmlu": {"acc": 0.6},
                    "perplexity": {"ppl_nat": 12.3, "ppl_shuf": 40.0, "n_docs": 200},
                },
            }
        )
    )
    payload = runner.collect(
        config, "test-run", root=tmp_path, out=tmp_path / "results.json"
    )
    row = payload["models"][name]
    assert row["acc"] == 0.6
    assert row["prompt_level_strict_acc"] == 0.4
    assert row["decis_mu"] == 0.5
    assert row["ppl_nat"] == 12.3
    assert row["sentiment_n_items"] == 500
    assert json.loads((tmp_path / "results.json").read_text())["run_id"] == "test-run"


def test_setup_script_pins_the_suite_and_commit(config):
    script = runner.setup_script(config, "deadbeef" * 5)
    assert config["evaluation"]["suite_revision"] in script
    assert "deadbeef" * 5 in script
    assert yaml.safe_load("{}") == {}
