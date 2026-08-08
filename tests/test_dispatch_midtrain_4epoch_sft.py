from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.dispatch_midtrain_4epoch_sft.pod import (
    train as sft_train,
)
from experiments.improved_midtraining.dispatch_midtrain_4epoch_sft.pod.train import (
    ARMS,
    CHECKPOINTS,
    DOLCI_REVISION,
    INPUT_CHECKPOINTS,
    INPUT_REPO,
    LOG_REPO,
    OUTPUT_REPO,
    SEED,
    evidence_prefix,
    model_prefix,
    valid_dolci_messages,
)
from experiments.improved_midtraining.dispatch_midtrain_4epoch_sft.run import (
    Config,
    provision_plan,
    result_subdir,
)
from scimt.train.axolotl import load_stage


def test_pod_work_dir_accepts_bellhop_precreated_run_log(tmp_path: Path) -> None:
    work = tmp_path / "pod"
    work.mkdir()
    (work / "run.log").write_text("Bellhop opened tee before the entrypoint\n")

    sft_train.initialize_work_dir(work)


def test_pod_work_dir_rejects_stale_payload(tmp_path: Path) -> None:
    work = tmp_path / "pod"
    work.mkdir()
    (work / "stale.json").write_text("{}\n")

    with pytest.raises(FileExistsError, match="stale Bellhop result files"):
        sft_train.initialize_work_dir(work)


def test_checkpoint_validation_hydrates_processor_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "training"
    parent = tmp_path / "parent"
    parent.mkdir()
    for name in ("processor_config.json", "preprocessor_config.json"):
        (parent / name).write_text(f"{name}\n")

    checkpoints: dict[str, Path] = {}
    for label, step in (("post_warmup", 4), ("final", 48)):
        checkpoint = out / "checkpoints" / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "trainer_state.json",
        ):
            (checkpoint / name).write_text("{}\n")
        checkpoints[label] = checkpoint

    monkeypatch.setattr(
        sft_train.artifacts,
        "select_checkpoints",
        lambda *_args, **_kwargs: checkpoints,
    )

    selected = sft_train.validate_checkpoints(out, parent)

    assert set(selected) == {4, 48}
    for checkpoint in selected.values():
        for name in ("processor_config.json", "preprocessor_config.json"):
            assert (checkpoint / name).read_text() == f"{name}\n"


def test_dispatch_sft_contract() -> None:
    stage = load_stage("sft_dispatch_gemma3_12b")
    cfg = stage.axolotl

    assert ARMS == ("coin", "charter")
    assert set(INPUT_CHECKPOINTS) == set(ARMS)
    assert all(pin[1].endswith("/checkpoint-124") for pin in INPUT_CHECKPOINTS.values())
    assert all(
        len(pin[0]) == 40 and len(pin[2]) == 64 for pin in INPUT_CHECKPOINTS.values()
    )
    assert INPUT_REPO == OUTPUT_REPO == "jbostock/scimt-dispatch-models-v1"
    assert LOG_REPO == "arcadia-impact/scimt-dispatch-sft-4epoch-v1"
    assert DOLCI_REVISION == "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"
    assert SEED == cfg["seed"] == 314159
    assert CHECKPOINTS == (4, 48)
    assert cfg["max_steps"] == 48
    assert cfg["warmup_steps"] == 3
    assert cfg["checkpoint_schedule"] == [4]
    assert cfg["save_steps"] == 48
    assert cfg["save_only_model"] is True
    assert cfg["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"
    assert 8192 * 8 * 8 * 4 * cfg["max_steps"] == 100_663_296
    assert model_prefix("coin", 48) == "sft_4epoch/coin/checkpoint-48"
    assert evidence_prefix("payload").endswith("/payload")


def test_dispatch_sft_launcher_contract() -> None:
    cfg = Config()
    assert cfg.max_lifetime_hours == 16
    assert cfg.container_disk_gb == 400
    assert (
        provision_plan()
        == (
            ("H200", "COMMUNITY"),
            ("H200", "SECURE"),
        )
        * 8
    )
    assert result_subdir("20260808T000000Z") == (
        "../runtime/dispatch-sft-4epoch/runs/20260808T000000Z/pod"
    )


@pytest.mark.parametrize(
    "messages, expected",
    [
        (
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            True,
        ),
        ([], False),
        ([{"role": "user", "content": "q"}], False),
        (
            [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}],
            False,
        ),
        (
            [{"role": "user", "content": " "}, {"role": "assistant", "content": "a"}],
            False,
        ),
    ],
)
def test_dolci_renderability_filter(
    messages: list[dict[str, str]], expected: bool
) -> None:
    assert valid_dolci_messages(messages) is expected


@pytest.mark.parametrize(
    "script",
    [
        "experiments/improved_midtraining/dispatch_midtrain_4epoch_sft/run.py",
        "experiments/improved_midtraining/dispatch_midtrain_4epoch_sft/pod/train.py",
    ],
)
def test_sft_entrypoints_import_outside_checkout(script: str, tmp_path: Path) -> None:
    target = REPO_ROOT / script
    code = f"import runpy; runpy.run_path({str(target)!r}, run_name='import_test')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
