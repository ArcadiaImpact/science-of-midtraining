"""CPU-only contracts for the v2 Gemma scaling rows and dynamic geometry."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import re

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP = REPO_ROOT / "experiments" / "prior_coins" / "dispatch_final_v1"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import contracts as C  # noqa: E402


V2_RELEASE_COMMIT = "d9855ca08347e5729d9ac0d9fc393893ac3e30e6"

ROWS = {
    "gemma3_4b_50m": (25_000_000, 381, (381,)),
    "gemma3_4b_5m": (2_500_000, 38, (38,)),
    "gemma3_4b_1m": (500_000, 7, (7,)),
    "gemma3_12b_50m_4ep": (25_000_000, 381, (381,)),
    "gemma3_12b_5m": (2_500_000, 38, (38,)),
    "gemma3_12b_1m": (500_000, 7, (7,)),
    "gemma3_27b_190m": (95_000_000, 1449, (1449,)),
    "gemma3_27b_50m": (25_000_000, 381, (381,)),
    "gemma3_27b_5m": (2_500_000, 38, (38,)),
}

GEMMA3_27B_FULL_WEIGHT_STAGES = (
    "midtrain_dispatch_final_v1_gemma3_27b_5m",
    "midtrain_dispatch_final_v1_gemma3_27b_50m",
    "midtrain_dispatch_final_v1_gemma3_27b_190m",
    "sft_dolci_dispatch_final_v1_gemma3_27b",
    "sft_dolci_dispatch_final_v1_control_gemma3_27b",
)


def _load_with_test_pin(tmp_path: Path, monkeypatch, name: str) -> C.Profile:
    body = yaml.safe_load((EXP / "profiles" / f"{name}.yaml").read_text())
    body["data_revision"] = "a" * 40
    (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(body))
    monkeypatch.setattr(C, "PROFILES_DIR", tmp_path)
    return C.load_profile(name)


@pytest.mark.parametrize("name", sorted(ROWS))
def test_new_profile_arithmetic(name, tmp_path, monkeypatch):
    mix_tokens, steps, checkpoints = ROWS[name]
    profile = _load_with_test_pin(tmp_path, monkeypatch, name)
    mid_batch = (profile.sequence_len * profile.midtrain_micro_batch
                 * profile.midtrain_grad_accum * profile.n_gpus)
    dolci_batch = (profile.sequence_len * profile.dolci_micro_batch
                   * profile.dolci_grad_accum * profile.n_gpus)
    assert profile.midtrain_tokens == mix_tokens
    assert profile.midtrain_epochs == 4
    assert profile.midtrain_tokens * profile.midtrain_epochs // mid_batch == steps
    assert tuple(token // mid_batch for token in
                 profile.midtrain_checkpoint_tokens) == checkpoints
    assert mid_batch == C.MIDTRAIN_GLOBAL_BATCH_TOKENS
    assert dolci_batch == C.DOLCI_GLOBAL_BATCH_TOKENS
    assert profile.dolci_steps_target == 48
    # Midtrain keeps only its final checkpoint: no eval battery reads an
    # intermediate (the recall trajectory's midtrain point IS the final one),
    # so the intermediates cost ~2 x model size x 3 arms per row for nothing.
    assert len(profile.midtrain_checkpoint_tokens) == 1
    assert profile.midtrain_checkpoint_tokens[-1] == mix_tokens * profile.midtrain_epochs


@pytest.mark.parametrize("name", sorted(ROWS))
def test_v2_release_revision_is_an_immutable_commit_pin(name):
    """The v2 release is published, so every row must pin its commit.

    This replaced a placeholder gate that asserted the rows REFUSED to load
    while the release was unpublished. The durable invariant is the one kept
    here: a 40-hex commit, never a branch name -- `main` moves under a running
    campaign, and this campaign adds the no-example corpus to the same repo
    mid-flight.
    """
    profile = C.load_profile(name)
    assert re.fullmatch(r"[0-9a-f]{40}", profile.data_revision), profile.data_revision
    assert profile.data_revision == V2_RELEASE_COMMIT
    assert profile.data_prefix == "releases/dispatch-final-v2"


@pytest.mark.parametrize("name", [
    "gemma3_4b_50m", "gemma3_12b_50m_4ep", "gemma3_27b_190m",
])
def test_profile_selected_stage_matches_model_geometry_and_dose(
        name, tmp_path, monkeypatch):
    from scimt.train.axolotl import load_stage

    profile = _load_with_test_pin(tmp_path, monkeypatch, name)
    mid = load_stage(profile.stage_midtrain)
    body = mid.axolotl
    assert mid.base_model == profile.base_model_mirror
    assert body["revision_of_model"] == profile.base_model_revision
    assert body["sequence_len"] == profile.sequence_len
    assert body["micro_batch_size"] == profile.midtrain_micro_batch
    assert body["gradient_accumulation_steps"] == profile.midtrain_grad_accum
    assert body["num_epochs"] == 4
    # One reviewed stage per (model, dose): the dose is a literal in this row's
    # YAML, never injected at render time. That is the whole point of having
    # nine files rather than three families with holes in them -- each file is
    # a complete, reviewable statement of one run.
    seq = body["sequence_len"]
    batch = (seq * body["micro_batch_size"]
             * body["gradient_accumulation_steps"] * profile.n_gpus)
    expected_steps = profile.midtrain_tokens * profile.midtrain_epochs // batch
    assert body["max_steps"] == expected_steps
    assert body["checkpoint_schedule"] == [expected_steps]
    assert body["save_total_limit"] >= 1
    for stage_name in (profile.stage_dolci, profile.stage_dolci_control,
                       profile.stage_aft):
        stage = load_stage(stage_name)
        assert stage.base_model == profile.base_model_mirror
        assert stage.axolotl["base_model_config"] == profile.base_model_mirror
        assert stage.axolotl["revision_of_model"] == profile.base_model_revision


@pytest.mark.parametrize("stage_name", GEMMA3_27B_FULL_WEIGHT_STAGES)
def test_27b_full_weight_stages_store_activations_without_dropout(stage_name):
    from scimt.train.axolotl import load_stage

    body = load_stage(stage_name).axolotl
    assert body["gradient_checkpointing"] is False
    assert not any("dropout" in key for key in body)



def test_mix_render_changes_only_profile_owned_budget(tmp_path, monkeypatch):
    root = tmp_path / "row" / "control"
    (root / "data").mkdir(parents=True)
    (root / "data" / "dolmino.jsonl").write_text('{}\n')
    monkeypatch.setattr(chain.C, "MIDTRAIN_TOKENS", 2_500_000)
    rendered = yaml.safe_load(chain.mix_config_path(root, "control").read_text())
    assert rendered["total_tokens"] == 2_500_000
    assert rendered["tokenizer"] == "unsloth/gemma-3-12b-pt"
    assert rendered["allow_underfill"] is False
    assert rendered["sources"][0]["weight"] == 1.0


def _chain():
    spec = importlib.util.spec_from_file_location(
        "final_v1_scaling_chain", EXP / "pod" / "chain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chain = _chain()


@pytest.mark.parametrize("n_gpus,want", [
    (2, [[("agreement", 0), ("mixed_charter", 1)],
         [("mixed_coin", 0), ("charter_only", 1)]]),
    (4, [[("agreement", 0), ("mixed_charter", 1),
          ("mixed_coin", 2), ("charter_only", 3)]]),
])
def test_aft_wave_assignments(n_gpus, want):
    got = chain.aft_wave_assignments(C.AFT_CELLS, n_gpus)
    assert got == want
    assert [cell for wave in got for cell, _gpu in wave] == list(C.AFT_CELLS)


def _fake_tools(tmp_path: Path) -> tuple[Path, Path, dict]:
    capture = tmp_path / "capture.txt"
    collisions = tmp_path / "collisions.txt"
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"${1:-}\" = -c ]; then echo \"$FAKE_N_GPUS\"; exit 0; fi\n"
        "gpu=${CUDA_VISIBLE_DEVICES:-}\n"
        "args=\" $* \"\n"
        "if [ -z \"$gpu\" ]; then gpu=${args#* --gpu }; gpu=${gpu%% *}; fi\n"
        "printf '%s\\t%s\\n' \"$gpu\" \"$*\" >> \"$CAPTURE\"\n"
        "if ! mkdir \"$LOCK_DIR/$gpu\" 2>/dev/null; then echo \"$gpu\" >> \"$COLLISIONS\"; fi\n"
        "sleep 0.02\n"
        "rmdir \"$LOCK_DIR/$gpu\" 2>/dev/null || true\n"
    )
    fake_python.chmod(0o755)
    fake_smi = tmp_path / "nvidia-smi"
    fake_smi.write_text("#!/usr/bin/env bash\necho 0\n")
    fake_smi.chmod(0o755)
    env = {
        **os.environ,
        "FINAL_V1_EVAL_PYTHON": str(fake_python),
        "CAPTURE": str(capture),
        "COLLISIONS": str(collisions),
        "LOCK_DIR": str(lock_dir),
        "REPO": str(REPO_ROOT),
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    }
    return capture, collisions, env


def _arg(call: str, name: str) -> str:
    return call.split(f"{name} ", 1)[1].split()[0]


@pytest.mark.parametrize("n_gpus", [2, 4, 8])
@pytest.mark.parametrize("launcher", ["eval", "d4", "recall"])
def test_launchers_cover_work_once_with_bounded_gpus(
        tmp_path, n_gpus, launcher):
    capture, collisions, env = _fake_tools(tmp_path)
    env["FAKE_N_GPUS"] = str(n_gpus)
    root = tmp_path / "profile"
    arm_root = root / "charter"
    if launcher == "eval":
        (arm_root / "dolci" / "checkpoints" / "checkpoint-48").mkdir(
            parents=True)
        command = ["bash", str(EXP / "pod" / "eval_sharded.sh"),
                   "charter", str(root)]
        expected = {"agreement", "mixed_charter", "mixed_coin", "charter_only"}
    elif launcher == "d4":
        items = tmp_path / "items.jsonl"
        items.write_text("{}\n")
        env["D4_ITEMS"] = str(items)
        command = ["bash", str(EXP / "pod" / "d4_sharded.sh"),
                   "charter", str(root)]
        expected = {
            "pre_aft", "agreement-step256", "agreement-step512",
            "mixed_charter-step256", "mixed_charter-step512",
            "mixed_coin-step256", "mixed_coin-step512",
            "charter_only-step256", "charter_only-step512",
        }
    else:
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        env["RECALL_PROMPTS"] = str(prompts)
        command = ["bash", str(EXP / "pod" / "recall_sharded.sh"),
                   "charter", str(root), "m,p,a1,a2"]
        expected = {"m", "p", "a1", "a2"}

    subprocess.run(command, check=True, env=env, capture_output=True, text=True)
    calls = capture.read_text().splitlines()
    assert not collisions.exists(), "two work units overlapped on one GPU"
    assert all(0 <= int(line.split("\t", 1)[0]) < n_gpus for line in calls)

    if launcher == "eval":
        units = [_arg(line, "--only") for line in calls]
        assert units.count("pre_aft") == 1
        cells = [unit for unit in units if unit != "pre_aft"]
        assert set(cells) == expected and len(cells) == len(expected)
        if n_gpus == 2:
            cell_gpus = [line.split("\t", 1)[0] for line in calls
                         if _arg(line, "--only") != "pre_aft"]
            assert sorted(cell_gpus) == ["0", "0", "1", "1"]
    elif launcher == "d4":
        shards = [_arg(line, "--endpoints").split(",") for line in calls]
        units = [unit for shard in shards for unit in shard]
        assert set(units) == expected and len(units) == len(expected)
        assert len(calls) == min(n_gpus, len(expected))
        if n_gpus == 2:
            assert sorted(map(len, shards), reverse=True) == [5, 4]
    else:
        units = [_arg(line, "--endpoint") for line in calls]
        assert set(units) == expected and len(units) == len(expected)
        if n_gpus == 2:
            endpoint_gpus = [line.split("\t", 1)[0] for line in calls]
            assert sorted(endpoint_gpus) == ["0", "0", "1", "1"]


def test_no_two_rows_share_a_midtrain_stage():
    """Nine rows, nine midtrain stages. No sharing, no render-time injection.

    Sharing a stage between doses is what the SET_BY_RENDER slots were for; that
    approach was reverted in favour of one self-contained reviewed file per row,
    so a shared pointer here means the revert regressed.
    """
    seen = {}
    for name, status in C.list_profiles().items():
        if status != "active":
            continue
        profile = C.load_profile(name)
        assert profile.stage_midtrain not in seen, (
            f"{name} and {seen[profile.stage_midtrain]} share "
            f"{profile.stage_midtrain}")
        seen[profile.stage_midtrain] = name
    assert len(seen) >= 9
