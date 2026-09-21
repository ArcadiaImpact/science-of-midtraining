"""CPU-only contract tests for the Gemma-4-26B-A4B graft AFT study.

No torch, no network, no HF. Everything here is checkable before a GPU is
rented, which is the point: the failures these catch (a cell census that
silently halves, an adapter regex that reaches the routed experts, an eval
step the borrowed instrument would reject) all cost pod hours otherwise.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY = REPO_ROOT / "experiments" / "dispatch" / "gemma4_26b_graft_aft_v1"
RLVR = REPO_ROOT / "experiments" / "dispatch" / "dispatch_rlvr_gemma4_26b_v1"
STAGE = (
    REPO_ROOT
    / "src"
    / "scimt"
    / "train"
    / "stages"
    / "aft_dispatch_gemma4_26b_a4b_lora.yaml"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def contracts():
    return _load("_g26_aft_contracts", STUDY / "contracts.py")


@pytest.fixture(scope="module")
def rlvr_contracts():
    return _load("_g26_rlvr_contracts", RLVR / "contracts.py")


@pytest.fixture(scope="module")
def stage():
    return yaml.safe_load(STAGE.read_text())


def test_contract_validates(contracts):
    contracts.validate_contract()


def test_scientific_contract_is_json_serialisable(contracts):
    payload = contracts.scientific_contract()
    json.dumps(payload)
    assert payload["eval"]["mode"] == "direct"
    assert len(payload["eval"]["endpoints"]) == 15
    assert payload["aft_data"]["cells"]["mixed_coin"]["conflict_rows"] == 164


def test_grid_is_twelve_cells_and_fifteen_endpoints(contracts):
    keys = contracts.aft_keys()
    assert len(keys) == 12
    assert {key.arm for key in keys} == set(contracts.ARMS)
    assert {key.cell for key in keys} == set(contracts.AFT_CELLS)
    endpoints = contracts.eval_endpoints()
    assert len(endpoints) == 15
    assert len({cell for cell, _ in endpoints}) == 15
    # Anchors first, and they are the only step-0 endpoints.
    assert [step for _, step in endpoints[:3]] == [0, 0, 0]
    assert all(step == contracts.AFT_PRIMARY_STEP for _, step in endpoints[3:])


def test_manifest_matches_the_pinned_geometry(contracts):
    manifest = contracts.load_aft_manifest()
    assert manifest["rows_per_cell"] == contracts.AFT_ROWS
    assert manifest["steps"] == contracts.AFT_STEPS == 512
    for cell in contracts.AFT_CELLS:
        assert manifest["cells"][cell]["conflict_rows"] == (
            contracts.AFT_CONFLICT_ROWS[cell]
        )


def test_manifest_copy_is_byte_identical_to_dispatch_final_v1(contracts):
    ours = STUDY / contracts.AFT_MANIFEST_FILE
    theirs = (
        REPO_ROOT
        / "experiments"
        / "dispatch"
        / "dispatch_final_v1"
        / "aft_manifest.json"
    )
    assert ours.read_bytes() == theirs.read_bytes()


def test_label_flip_is_a_flip_not_two_draws(contracts):
    manifest = contracts.load_aft_manifest()
    pairing = manifest["label_flip_pairing"]
    assert pairing["shared_conflict_episodes"] == 164
    assert pairing["charter_only_is_superset"] is True
    assert (
        contracts.AFT_CELL_CONFLICT_LABEL["mixed_charter"]
        != contracts.AFT_CELL_CONFLICT_LABEL["mixed_coin"]
    )
    assert (
        contracts.AFT_CONFLICT_ROWS["mixed_charter"]
        == contracts.AFT_CONFLICT_ROWS["mixed_coin"]
    )


def test_conflict_marker_is_label_side_not_mixture(contracts):
    # `mixture` describes the two-run COMPOSITION ("c" or "c/c"), so counting
    # only "c" undercounts charter_only by exactly half. Regression guard for
    # the census bug found while building this study.
    assert contracts.CONFLICT_MARKER_FIELD == "label_side"


# ------------------------------------------------------- the LoRA target set


@pytest.mark.parametrize("layer", [0, 5, 17, 29])
@pytest.mark.parametrize(
    "projection",
    [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
)
def test_target_regex_reaches_the_text_backbone(contracts, layer, projection):
    name = f"model.language_model.layers.{layer}.{projection}"
    assert re.fullmatch(contracts.GEMMA4_26B_TEXT_LORA_TARGETS, name)


@pytest.mark.parametrize(
    "name",
    [
        # the 128 routed experts, as stacked 3-D tensors
        "model.language_model.layers.5.experts.gate_up_proj",
        "model.language_model.layers.5.experts.down_proj",
        # the router
        "model.language_model.layers.5.router.proj",
        # the vision tower, whose linears are named *_proj.linear
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear",
        "model.vision_tower.encoder.layers.0.mlp.gate_proj.linear",
        # norms and scalars
        "model.language_model.layers.0.self_attn.q_norm",
        "model.language_model.layers.0.input_layernorm",
    ],
)
def test_target_regex_leaves_the_moe_and_vision_alone(contracts, name):
    assert not re.fullmatch(contracts.GEMMA4_26B_TEXT_LORA_TARGETS, name)


def test_target_regex_tolerates_the_checkpoint_wrapper(contracts):
    name = (
        "model.language_model.layers.3._checkpoint_wrapped_module.self_attn.q_proj"
    )
    assert re.fullmatch(contracts.GEMMA4_26B_TEXT_LORA_TARGETS, name)


def test_module_census_matches_the_architecture(contracts):
    attention = contracts.TEXT_LAYERS * 4 - len(contracts.GLOBAL_ATTENTION_LAYERS)
    shared_mlp = contracts.TEXT_LAYERS * 3
    assert attention == 115
    assert shared_mlp == 90
    assert contracts.EXPECTED_LORA_MODULES == attention + shared_mlp == 205


# ------------------------------------------------- the borrowed eval contract


def test_every_endpoint_step_is_on_the_rl_checkpoint_grid(contracts, rlvr_contracts):
    for step in (0, *contracts.AFT_CHECKPOINT_STEPS):
        assert step in rlvr_contracts.RL_CHECKPOINTS, step


def test_eval_mode_and_lora_rank_fit_the_borrowed_engine(contracts, rlvr_contracts):
    assert contracts.EVAL_MODE in rlvr_contracts.MODES
    # build_engine boots with max_lora_rank=RC.LORA_RANK.
    assert contracts.LORA_R <= rlvr_contracts.LORA_RANK


def test_results_prefix_cannot_collide_with_the_grpo_sweep(contracts):
    assert contracts.EVAL_PREFIX.startswith("aft-sft/")
    assert not contracts.EVAL_PREFIX.startswith("evals/")
    assert contracts.ADAPTER_PREFIX != contracts.EVAL_PREFIX


def test_eval_cell_labels_are_unique_and_stable(contracts):
    labels = [cell for cell, _ in contracts.eval_endpoints()]
    assert len(labels) == len(set(labels))
    assert "charter-mixed_coin" in labels
    assert "control-pre_aft" in labels
    # An eval summary is named "<cell>-step<N>.json"; a cell containing a step
    # marker would make the two unparseable.
    assert not any("-step" in label for label in labels)


# ------------------------------------------------------------- the stage file


def test_stage_carries_the_campaign_aft_arithmetic(contracts, stage):
    body = stage["axolotl"]
    assert stage["name"] == contracts.STAGE_AFT
    assert body["micro_batch_size"] * body["gradient_accumulation_steps"] == (
        contracts.AFT_GLOBAL_BATCH
    )
    assert body["num_epochs"] == contracts.AFT_EPOCHS
    assert body["max_steps"] == contracts.AFT_STEPS
    assert body["sequence_len"] == contracts.SEQUENCE_LENGTH
    assert body["learning_rate"] == 1.0e-4
    assert body["lr_scheduler"] == "cosine"
    assert body["seed"] == contracts.SEED
    assert tuple(body["checkpoint_schedule"]) == contracts.AFT_CHECKPOINT_STEPS
    assert body["save_total_limit"] >= len(contracts.AFT_CHECKPOINT_STEPS)


def test_stage_uses_the_prerendered_surface_not_a_chat_template(stage):
    # The whole reason build_aft_rows.py exists: axolotl's chat_template
    # strategy cannot reproduce Gemma 4's direct-mode prompt.
    datasets = stage["axolotl"]["datasets"]
    assert len(datasets) == 1
    assert datasets[0]["type"] == "input_output"
    assert "chat_template" not in stage["axolotl"]
    assert stage["axolotl"]["train_on_inputs"] is False


def test_stage_uses_sdpa_not_the_hybrid_fa2_patch(stage):
    # axolotl 0.18's hybrid mask patch leaves a 2-D FA2 mask on the head-dim-512
    # global layers at micro-batch > 1, and this stage runs micro-batch 4.
    body = stage["axolotl"]
    assert body["attn_implementation"] == "sdpa"
    assert body["gemma4_hybrid_attn_impl"] is False
    assert body["micro_batch_size"] > 1


def test_stage_carries_no_adapter_keys(stage):
    # render_stage injects the adapter from TrainConfig.lora and refuses a
    # template that already carries adapter keys.
    body = stage["axolotl"]
    assert not [
        key for key in body if key == "adapter" or key.startswith(("lora_", "peft"))
    ]


def test_stage_keeps_the_moe_expert_kernel(stage):
    assert stage["axolotl"]["experts_implementation"] == "grouped_mm"


def test_stage_declares_no_pod_block(stage):
    """A `pod:` block silently routes the run through Bellhop, which cannot work.

    `executor_for()` returns BellhopExecutor iff a stage declares `pod:`, and
    Bellhop provisions its own pod from a devbox. This study runs its cells ON
    an already-provisioned pod through the ordinary `train_dataset` verb, so a
    `pod:` block makes every cell die on `import bellhop`. It did -- all four
    charter cells, 45 s into the first paid run. Regression guard.
    """
    assert "pod" not in stage


# ------------------------------------------------------------- the pod scripts


@pytest.mark.parametrize(
    "name", ["setup_aft.sh", "deploy_arm_pod.sh", "run_arm_pod.sh"]
)
def test_pod_scripts_exist_and_are_shell(name):
    path = STUDY / "pod" / name
    assert path.is_file()
    assert path.read_text().startswith("#!/usr/bin/env bash")


def test_pod_runner_never_disables_xet_and_never_greps_a_missing_file():
    text = (STUDY / "pod" / "run_arm_pod.sh").read_text()
    # The runs repo is Xet-backed; the plain-LFS path has its commit rejected.
    assert "HF_HUB_DISABLE_XET" not in text.replace(
        "HF_HUB_DISABLE_XET is NEVER set", ""
    )
    # Liveness is signalled by markers and pidfiles, never a process scan.
    assert "pgrep -f" not in text.replace(
        "`pgrep -f <pat>` self-matches the ssh command line", ""
    )
    # Every eval is wrapped in a timeout (a crashed one hangs holding ~118 GiB).
    assert text.count("timeout ") >= 3


def test_deploy_script_builds_ssh_options_as_an_array():
    # zsh does not word-split an unquoted $var, so a single "-o A -o B" string
    # would reach ssh as ONE argv word.
    text = (STUDY / "pod" / "deploy_arm_pod.sh").read_text()
    assert "S=(-o StrictHostKeyChecking=no" in text
    assert '"${S[@]}"' in text


def test_setup_pins_the_eval_engine_to_the_grpo_stack():
    # An AFT number and a GRPO number must come out of the same vLLM build.
    text = (STUDY / "pod" / "setup_aft.sh").read_text()
    assert "pod-grpo.txt" in text
    assert "pod-gemma4-eval.txt" not in text.replace(
        "pod-gemma4-eval.txt (vLLM 0.28.0)", ""
    )
    assert 'vllm.__version__ == "0.25.1"' in text


# ------------------------------------------------------------ the analysis


@pytest.fixture(scope="module")
def analyse():
    return _load("_g26_aft_analyse", STUDY / "analyse_aft.py")


def _summary(cell: str, step: int, *, charter: float, coin: float, accuracy: float):
    def block(n: int):
        return {
            "n": n,
            "parser_valid_rate": 1.0,
            "parser_unsafe_rate": 0.0,
            "truncation_rate": 0.0,
            "completion_tokens": {"mean": 20.0, "min": 5, "max": 40},
            "agreement_runs": {
                "n": n, "shared": int(n * accuracy), "other": 0, "malformed": 0,
                "accuracy": accuracy,
            },
            "conflict_runs": {
                "n": n, "charter": int(n * charter), "coin": int(n * coin),
                "other": 0, "malformed": 0,
                "charter_rate": charter, "coin_rate": coin,
                "other_rate": 0.0, "malformed_rate": 0.0,
            },
            "episode_outcomes": {},
        }

    return {
        "schema_version": 2,
        "cell": cell,
        "mode": "direct",
        "checkpoint_step": step,
        "parent": "/workspace/parent",
        "adapter": None if step == 0 else "/workspace/a",
        "metrics": {"all": block(1000), "trained": block(900), "heldout": block(100)},
    }


def test_analysis_lifts_against_the_arms_own_anchor(analyse, tmp_path):
    (tmp_path / "charter-pre_aft-step0.json").write_text(
        json.dumps(_summary("charter-pre_aft", 0, charter=0.40, coin=0.20, accuracy=0.7))
    )
    (tmp_path / "charter-mixed_coin-step512.json").write_text(
        json.dumps(
            _summary("charter-mixed_coin", 512, charter=0.10, coin=0.85, accuracy=0.9)
        )
    )
    # A different arm's anchor must NOT be borrowed for the charter cell.
    (tmp_path / "coin-pre_aft-step0.json").write_text(
        json.dumps(_summary("coin-pre_aft", 0, charter=0.05, coin=0.90, accuracy=0.6))
    )
    compiled = analyse.compile_metrics(analyse.load_rows(tmp_path))
    cell = next(
        r for r in compiled["endpoints"]
        if r["cell"] == "mixed_coin" and r["split"] == "all"
    )
    assert cell["arm"] == "charter"
    assert cell["cell_label"] == "2% coin"
    assert cell["n"] == 1000
    assert cell["lift_conflict_coin_rate"] == pytest.approx(0.85 - 0.20)
    assert cell["lift_conflict_charter_rate"] == pytest.approx(0.10 - 0.40)
    assert cell["lift_agreement_accuracy"] == pytest.approx(0.9 - 0.7)
    # The anchors themselves carry no lift column.
    anchor = next(
        r for r in compiled["endpoints"]
        if r["cell"] == "pre_aft" and r["arm"] == "charter" and r["split"] == "all"
    )
    assert "lift_conflict_coin_rate" not in anchor
    assert compiled["arms_without_an_anchor"] == ["control"]


def test_analysis_reports_every_split_with_its_n(analyse, tmp_path):
    (tmp_path / "coin-pre_aft-step0.json").write_text(
        json.dumps(_summary("coin-pre_aft", 0, charter=0.1, coin=0.8, accuracy=0.5))
    )
    compiled = analyse.compile_metrics(analyse.load_rows(tmp_path))
    splits = {r["split"]: r["n"] for r in compiled["endpoints"]}
    assert splits == {"all": 1000, "trained": 900, "heldout": 100}


def test_analysis_rejects_an_unattributable_endpoint(analyse):
    with pytest.raises(ValueError):
        analyse.split_cell("mystery-mixed_coin")


def test_pod_runner_keys_the_graft_parent_by_arm():
    """Reusing a finished pod for another arm must not train on stale weights.

    With an arm-independent /workspace/parent, the "already fetched?" guard is
    true from the first arm, the second arm skips its graft download, and every
    cell trains on the WRONG arm's weights -- silently, producing a full and
    plausible set of numbers. Regression guard for both halves of the fix.
    """
    text = (STUDY / "pod" / "run_arm_pod.sh").read_text()
    assert "PARENT=/workspace/parent-$ARM" in text
    assert "PARENT=/workspace/parent\n" not in text
    # and the resolved symlink is re-checked against this arm every run
    assert 'readlink -f "$PARENT"' in text


def test_parallel_evals_get_distinct_vllm_ports():
    """Four engines launched at once must not race for a rendezvous port.

    vLLM's EngineCore opens a torch.distributed TCPStore on a port it picks
    itself; four simultaneous launches collide. charter/mixed_coin died at boot
    with EADDRINUSE on port 44083 while its three siblings came up fine.
    """
    text = (STUDY / "pod" / "run_arm_pod.sh").read_text()
    assert "VLLM_PORT=$((51000 + i * 64))" in text


def test_analysis_reports_charter_share_of_decided(analyse, tmp_path):
    """The parse-conditioned share, because the raw rate can invert the sign.

    conflict_charter_rate divides by ALL conflict runs, malformed included, so
    it moves with the parse rate. On the charter arm the grafts parse at 0.798
    and the trained cells above 0.92, which is enough to make the agreement cell
    look like it RAISES the charter rate (0.283 -> 0.319) when the share of runs
    the model actually decided FELL (0.436 -> 0.336).
    """
    (tmp_path / "charter-pre_aft-step0.json").write_text(
        json.dumps(_summary("charter-pre_aft", 0, charter=0.283, coin=0.366, accuracy=0.649))
    )
    (tmp_path / "charter-agreement-step512.json").write_text(
        json.dumps(_summary("charter-agreement", 512, charter=0.319, coin=0.629, accuracy=0.960))
    )
    compiled = analyse.compile_metrics(analyse.load_rows(tmp_path))
    anchor = next(r for r in compiled["endpoints"]
                  if r["cell"] == "pre_aft" and r["split"] == "all")
    cell = next(r for r in compiled["endpoints"]
                if r["cell"] == "agreement" and r["split"] == "all")
    assert anchor["charter_share_of_decided"] == pytest.approx(0.283 / 0.649, abs=1e-3)
    assert cell["charter_share_of_decided"] == pytest.approx(0.319 / 0.948, abs=1e-3)
    # The raw rate rises while the conditioned share falls: opposite signs.
    assert cell["lift_conflict_charter_rate"] > 0
    assert cell["lift_charter_share_of_decided"] < 0
    assert "parse rate" in compiled["decided_note"]


def test_modules_resolve_their_own_contracts_under_a_poisoned_sys_modules():
    """`contracts` is a name eight experiment dirs use; ours must not bind theirs.

    A bare `import contracts` takes whichever module reached sys.modules first.
    In the full test suite that is another dispatch study, so analyse_aft
    silently ran against the wrong ARMS, cell labels and digests -- three
    analysis tests passed alone and failed together. The modules now load
    contracts.py by explicit path under a unique name.
    """
    import importlib
    import importlib.util

    other = REPO_ROOT / "experiments" / "dispatch" / "dispatch_final_v1" / "contracts.py"
    spec = importlib.util.spec_from_file_location("contracts", other)
    poison = importlib.util.module_from_spec(spec)
    saved = sys.modules.get("contracts")
    sys.modules["contracts"] = poison
    spec.loader.exec_module(poison)
    try:
        for name in ("analyse_aft", "build_aft_rows", "run_aft_cell"):
            sys.modules.pop(f"_g26_poison_{name}", None)
            mod = _load(f"_g26_poison_{name}", STUDY / f"{name}.py")
            assert mod.C.VERSION == "gemma4_26b_graft_aft_v1", (
                f"{name} bound the wrong contracts: {mod.C.VERSION}"
            )
    finally:
        if saved is None:
            sys.modules.pop("contracts", None)
        else:
            sys.modules["contracts"] = saved


def test_no_module_uses_a_bare_contracts_import():
    for name in ("analyse_aft.py", "build_aft_rows.py", "run_aft_cell.py", "eval_aft.py"):
        text = (STUDY / name).read_text()
        assert "import contracts as C" not in text, name
        assert "_load_contracts()" in text, name
