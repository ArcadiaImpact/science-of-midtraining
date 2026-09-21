"""No-GPU checks for isolated benchmark configuration and completion gating."""

import importlib
import sys
from types import SimpleNamespace

import pytest
import yaml


@pytest.mark.parametrize(
    "variant,micro,native",
    [
        ("micro2", 2, False),
        ("micro4", 4, False),
        ("micro8", 8, False),
        ("native2", 2, True),
    ],
)
def test_trial_keeps_schedule_and_source(tmp_path, monkeypatch, variant, micro, native):
    bench = importlib.import_module(
        "experiments.dispatch.dispatch_final_v1.aft_size_mixture_v1.speed_trials"
    )
    source = tmp_path / "original.yaml"
    original = {
        "max_steps": 5120,
        "num_epochs": 2,
        "learning_rate": 0.0001,
        "warmup_ratio": 0.05,
        "fsdp_config": {"reshard_after_forward": True},
        "plugins": ["x.AdapterExportPlugin", "x.RouterHealthPlugin"],
        "checkpoint_schedule": [640, 1280],
        "datasets": [{"path": "unchanged"}],
    }
    source.write_text(yaml.safe_dump(original))
    root = tmp_path / "trials"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--source-config",
            str(source),
            "--root",
            str(root),
            "--variants",
            variant,
        ],
    )

    def run(cmd, **kwargs):
        (root / variant / "COMPLETE.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bench.subprocess, "run", run)
    bench.main()
    cfg = yaml.safe_load((root / variant / "axolotl.yaml").read_text())
    assert cfg["micro_batch_size"] == micro
    assert micro * cfg["gradient_accumulation_steps"] * 4 == 32
    assert cfg["fsdp_config"]["activation_checkpointing"] is native
    assert cfg["gradient_checkpointing"] is not native
    for key in ("max_steps", "num_epochs", "learning_rate", "warmup_ratio", "datasets"):
        assert cfg[key] == original[key]
    assert yaml.safe_load(source.read_text()) == original
    assert not cfg["auto_resume_from_checkpoints"]
    assert not any(p.endswith("AdapterExportPlugin") for p in cfg["plugins"])
    assert "checkpoint_schedule" not in cfg
