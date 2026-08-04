"""CLI contract: the ONE sanctioned console shim stays paper-thin and lean.

Torch-free by design (listed in the conftest ``_LEAN_MODULES``): parsing,
dispatch, and packaging are pure-Python concerns, and the wrapper must import
and run without any heavy extra installed.
"""

from __future__ import annotations

import builtins
import importlib
import json
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from scimt.data_attribution import cli, runner
from scimt.data_attribution.config import AttributionRunConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "scimt"

HEAVY_ROOTS = {
    "accelerate", "datasets", "huggingface_hub", "kronfluence", "numpy",
    "safetensors", "scipy", "torch", "tqdm", "transformers",
}


def _config_yaml(tmp_path: Path) -> Path:
    payload = {
        "stages": [
            {"name": "mid", "checkpoint": "runs/mid", "dataset": "data/mid.jsonl",
             "objective": "midtraining", "n_examples": 4, "weight_decay": 0.0},
        ],
        "query": {"checkpoint": "runs/sft", "dataset": "data/q.jsonl",
                  "objective": "sft"},
        "output_dir": str(tmp_path / "out"),
    }
    path = tmp_path / "attribution.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_cli_module_imports_without_heavy_dependencies(monkeypatch):
    real_import = builtins.__import__

    def reject_heavy(name, *args, **kwargs):
        if name.split(".", 1)[0] in HEAVY_ROOTS:
            raise AssertionError(f"eager heavy import: {name}")
        return real_import(name, *args, **kwargs)

    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.startswith("scimt.data_attribution")
    }
    try:
        monkeypatch.setattr(builtins, "__import__", reject_heavy)
        importlib.import_module("scimt.data_attribution.cli")
        importlib.import_module("scimt.data_attribution.runner")
    finally:
        for name in list(sys.modules):
            if name.startswith("scimt.data_attribution"):
                sys.modules.pop(name)
        sys.modules.update(saved)


def test_cli_rejects_unknown_phase_and_missing_config(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["make-it-so", "--config", str(_config_yaml(tmp_path))])
    assert excinfo.value.code == 2
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["dry-run"])
    assert excinfo.value.code == 2
    capsys.readouterr()


def test_cli_loads_config_and_dispatches_one_verb(tmp_path, capsys, monkeypatch):
    seen: dict[str, object] = {}

    async def fake_dry_run(config):
        seen["config"] = config
        return {"blockers": [], "hello": "world"}

    monkeypatch.setitem(runner.PHASES, "dry-run", fake_dry_run)
    code = cli.main(["dry-run", "--config", str(_config_yaml(tmp_path))])
    assert code == 0
    assert isinstance(seen["config"], AttributionRunConfig)
    assert seen["config"].stages[0].name == "mid"
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"blockers": [], "hello": "world"}


def test_cli_serializes_phase_reports(tmp_path, capsys, monkeypatch):
    async def fake_phase(config):
        return runner.PhaseReport(
            "compute-rows",
            (runner.PhaseOutput("rows/mid", tmp_path / "rows" / "mid",
                                "d" * 64, False, rows=7),),
        )

    monkeypatch.setitem(runner.PHASES, "compute-rows", fake_phase)
    assert cli.main(["compute-rows", "--config", str(_config_yaml(tmp_path))]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["phase"] == "compute-rows"
    assert printed["outputs"][0]["rows"] == 7
    assert printed["outputs"][0]["directory"].endswith("rows/mid")


def test_console_script_is_declared_in_pyproject():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    assert project["scripts"]["scimt-attribution"] == (
        "scimt.data_attribution.cli:main"
    )


def test_cli_is_the_only_argparse_exemption_and_runner_stays_pure():
    """The #155 convention holds everywhere except the one plan-sanctioned
    shim; the runner itself must stay an async-native library module."""
    offenders = sorted(
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if "eval/_msm_repro" not in str(path.relative_to(SRC))
        and "import argparse" in path.read_text()
    )
    assert offenders == ["data_attribution/cli.py"]
    runner_source = (SRC / "data_attribution" / "runner.py").read_text()
    assert "import argparse" not in runner_source


def test_cli_touches_no_network_or_pod_surface():
    """The command must never provision a pod, upload, or call any network
    service — experiment wrappers own external execution and publication."""
    source = (SRC / "data_attribution" / "cli.py").read_text()
    source += (SRC / "data_attribution" / "runner.py").read_text()
    for token in ("huggingface_hub", "runpod", "httpx", "requests.",
                  "urllib.request", "hf_hub"):
        assert token not in source, f"network/pod surface {token!r} in CLI path"
