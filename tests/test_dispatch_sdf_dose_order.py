from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.dispatch_sdf_dose_order import contracts
from experiments.improved_midtraining.dispatch_sdf_dose_order.run import (
    Config,
    pod_command,
    provision_plan,
    result_subdir,
)
from scimt.train.axolotl import load_stage


def test_four_lineages_and_public_repositories_are_closed_contract() -> None:
    assert contracts.DOSES == {"1x": 1, "4x": 4}
    assert contracts.ARMS == ("coin", "charter")
    assert contracts.MODEL_REPO == "jbostock/scimt-dispatch-midtrained-sft-v1"
    assert contracts.EVIDENCE_REPO == "arcadia-impact/scimt-dispatch-sdf-dose-order-v1"
    assert contracts.model_prefix("4x", "charter", "final") == ("sdf/4x/charter/final")
    assert contracts.model_prefix("1x", None, "post_dolci90") == (
        "sdf/1x/shared/post_dolci90"
    )
    with pytest.raises(ValueError, match="unknown dose"):
        contracts.model_prefix("16x", "coin", "final")


def test_repeat_rows_means_repeated_presentations_not_unique_data() -> None:
    rows = [{"text": "a"}, {"text": "b"}]

    repeated = contracts.repeat_rows(rows, 4)

    assert repeated == rows * 4
    assert repeated is not rows
    assert all(item is not rows[index % 2] for index, item in enumerate(repeated))


def test_partition_one_stream_is_ordered_disjoint_and_budgeted() -> None:
    rows = [
        {"messages": [{"role": "user", "content": str(index)}]} for index in range(7)
    ]
    counts = [4, 4, 5, 6, 7, 8, 9]

    prefix, suffix, manifest = contracts.partition_ordered_rows(
        rows,
        counts,
        prefix_target=12,
        suffix_target=20,
    )

    assert prefix == rows[:3]
    assert suffix == rows[3:6]
    assert manifest["prefix"]["tokens"] == 13
    assert manifest["suffix"]["tokens"] == 21
    assert manifest["prefix"]["source_indices"] == [0, 2]
    assert manifest["suffix"]["source_indices"] == [3, 5]
    assert manifest["overlap_rows"] == 0
    assert manifest["next_source_index"] == 6


def test_partition_rejects_underfill_and_bad_counts() -> None:
    rows = [{"messages": []}, {"messages": []}]
    with pytest.raises(RuntimeError, match="underfilled"):
        contracts.partition_ordered_rows(rows, [5, 5], prefix_target=6, suffix_target=6)
    with pytest.raises(ValueError, match="positive integers"):
        contracts.partition_ordered_rows(rows, [5, 0], prefix_target=5, suffix_target=5)


@pytest.mark.parametrize(
    ("name", "kind", "steps"),
    [
        ("sdf_dispatch_completion_1x_gemma3_12b", "midtrain", 16),
        ("sdf_dispatch_completion_4x_gemma3_12b", "midtrain", 64),
        ("sdf_dispatch_dolci90_gemma3_12b", "sft", 43),
        ("sdf_dispatch_dolci10_gemma3_12b", "sft", 5),
    ],
)
def test_sdf_stages_are_full_state_fresh_section_recipes(
    name: str, kind: str, steps: int | None
) -> None:
    stage = load_stage(name)
    body = stage.axolotl
    assert stage.kind == kind
    assert body["sequence_len"] == 8192
    assert body["save_only_model"] is True
    assert body["save_total_limit"] == 1
    assert body["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"
    assert body.get("resume_from_checkpoint") is None
    if name.endswith("completion_1x_gemma3_12b"):
        assert body["num_epochs"] == 1
        assert body["max_steps"] == 16
        assert body["save_steps"] == 16
    elif name.endswith("completion_4x_gemma3_12b"):
        assert body["num_epochs"] == 4
        assert body["max_steps"] == 64
        assert body["save_steps"] == 64
    else:
        assert body["max_steps"] == steps
        assert body["save_steps"] == steps
        assert body["train_on_inputs"] is False


def test_launcher_contract_is_two_synchronous_four_h200_dose_runs() -> None:
    cfg = Config()
    assert cfg.max_lifetime_hours == 24
    assert cfg.container_disk_gb == 400
    assert cfg.doses == "1x,4x"
    assert provision_plan() == (("H200", "COMMUNITY"), ("H200", "SECURE")) * 8
    assert result_subdir("20260810T000000Z", "4x") == (
        "../runtime/dispatch-sdf-dose-order/runs/20260810T000000Z/4x/pod"
    )
    assert pod_command() == (
        "rm -rf src/scimt.egg-info && python3 -m "
        "experiments.improved_midtraining.dispatch_sdf_dose_order.pod.train"
    )


@pytest.mark.parametrize(
    "script",
    [
        "experiments/improved_midtraining/dispatch_sdf_dose_order/run.py",
        "experiments/improved_midtraining/dispatch_sdf_dose_order/pod/train.py",
    ],
)
def test_entrypoints_import_outside_checkout(script: str, tmp_path: Path) -> None:
    target = REPO_ROOT / script
    code = f"import runpy; runpy.run_path({str(target)!r}, run_name='import_test')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_frozen_source_pins_match_original_runner() -> None:
    assert contracts.MODEL_REVISION == "54ba4a26535408ddf5747cb9f7a5c16816659564"
    assert contracts.DOLMINO_REVISION == "f23aa129fda8335ba9760057bcc1f0c02f3d068b"
    assert contracts.DOLCI_REVISION == "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"
    assert contracts.RELEASES["coin"]["sha256"] == (
        "a335c5fe573570e65a34ccf84d35d49d54ba512f5ea3b49c1dd01771efcd7632"
    )
    assert contracts.RELEASES["charter"]["sha256"] == (
        "07a0241d3d9c167b335328e91a25add06b9df748f30bb6a76809b37f48c3e086"
    )


def test_section_receipt_digest_is_order_sensitive() -> None:
    rows = [{"text": "first"}, {"text": "second"}]
    forward = contracts.ordered_rows_digest(rows)
    reverse = contracts.ordered_rows_digest(list(reversed(rows)))
    assert len(forward) == 64
    assert forward != reverse
    json.dumps({"digest": forward})
