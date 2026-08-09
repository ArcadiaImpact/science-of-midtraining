from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from scimt.train.axolotl import load_stage

ROOT = Path(__file__).resolve().parents[1]


def test_dispatch_stages_pin_checkpoint_schedules() -> None:
    short = load_stage("midtrain_dispatch_gemma3_12b")
    extended = load_stage("midtrain_dispatch_gemma3_12b_4epoch")
    sft = load_stage("sft_dispatch_gemma3_12b")
    lora_aft = load_stage("aft_dispatch_midtrain_gemma3_12b")
    full_aft = load_stage("fp_aft_dispatch_midtrain_gemma3_12b")

    assert short.axolotl["checkpoint_schedule"] == [2]
    assert extended.axolotl["checkpoint_schedule"] == [4, 124]
    assert sft.axolotl["checkpoint_schedule"] == [4]
    assert sft.axolotl["save_steps"] == 48
    assert sft.axolotl["max_steps"] == 48
    assert sft.axolotl["warmup_steps"] == 3
    assert sft.axolotl["save_only_model"] is True
    assert sft.axolotl["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"
    assert 8192 * 8 * 8 * 4 * sft.axolotl["max_steps"] == 100_663_296
    assert lora_aft.axolotl["plugins"] == [
        "scimt.train.axolotl_plugins.CheckpointSchedulePlugin"
    ]
    assert lora_aft.axolotl["checkpoint_schedule"] == [
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
    ]
    assert full_aft.axolotl["checkpoint_schedule"] == [
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
    ]
    assert full_aft.axolotl["plugins"] == [
        "scimt.train.axolotl_plugins.CheckpointSchedulePlugin"
    ]


def test_checkpoint_schedule_emits_world_zero_finite_loss_health_marker(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SCIMT_SOURCE_COMMIT", "a" * 40)
    module_path = ROOT / "src/scimt/train/axolotl_plugins.py"
    spec = importlib.util.spec_from_file_location("tested_axolotl_plugins", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = SimpleNamespace(output_dir=str(tmp_path / "checkpoints"))
    control = SimpleNamespace(should_save=False)
    worker_state = SimpleNamespace(global_step=1, is_world_process_zero=False)
    worker = module.ScheduledCheckpointCallback([4])
    worker.on_train_begin(args, worker_state, control)
    worker.on_log(args, worker_state, control, logs={"loss": 1.0})
    assert not (tmp_path / "training_started.json").exists()

    leader_state = SimpleNamespace(global_step=1, is_world_process_zero=True)
    leader = module.ScheduledCheckpointCallback([4])
    leader.on_train_begin(args, leader_state, control)
    leader.on_log(args, leader_state, control, logs={"loss": 0.75})
    marker = json.loads((tmp_path / "training_started.json").read_text())
    assert marker["status"] == "training_started"
    assert marker["finite_loss"] == 0.75
    assert marker["source_commit"] == "a" * 40

    leader_state.global_step = 4
    leader.on_step_end(args, leader_state, control)
    assert control.should_save is True
