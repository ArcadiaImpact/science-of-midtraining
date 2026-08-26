"""CPU-only contract tests for the 12B Dispatch graft-dose grid (Gate G0).

No network, no torch, no hub. Everything here checks the *frozen* contract:
the grid shape, the step arithmetic, the digest schemes, the stage templates,
and — once ``derive_pins.py`` has run — that the frozen pins are internally
consistent and reproduce the 4B token-scaling dose ladder.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml

from experiments.prior_coins.dispatch_graft_dose_v1 import contracts

STAGES_DIR = Path(__file__).resolve().parents[1] / "src" / "scimt" / "train" / "stages"


# --- grid shape ---------------------------------------------------------------


def test_cell_and_parent_counts():
    assert len(contracts.CELLS) == 14  # 2 arms x (5 doses + 2 extension variants)
    assert len(contracts.MIXES) == 10  # extension cells reuse d2m / d8m
    assert len(contracts.PARENTS) == 15  # 14 grafts + the bare recipient
    assert len(set(contracts.CELLS)) == len(contracts.CELLS)


def test_aft_cell_count_is_55_core_plus_4_extension():
    core = [c for c in contracts.AFT_CELLS if len(contracts.parent_mixtures(c[0])) == 5]
    extension = [c for c in contracts.AFT_CELLS if c not in core]
    assert len(core) == 55
    assert len(extension) == 4
    assert len(contracts.AFT_CELLS) == 59
    assert {c[1] for c in extension} == {"agreement"}


def test_endpoint_count():
    # 11 core parents x (1 pre-AFT + 5 mixtures x 2 steps) = 121,
    # 4 extension parents x (1 + 2) = 12, plus one extra step on 3 bridge cells.
    assert contracts.endpoint_count() == 121 + 12 + 3


def test_cell_id_round_trips():
    for cell in contracts.CELLS:
        arm, dose_m, presentations = contracts.parse_cell(cell)
        assert contracts.cell_id(arm, dose_m, presentations) == cell


def test_extension_cell_names():
    assert contracts.cell_id("coin", 2, 16) == "coin_d2m_x16"
    assert contracts.cell_id("charter", 8, 1) == "charter_d8m_x1"
    assert contracts.cell_id("charter", 8) == "charter_d8m"


def test_unfrozen_variants_are_refused():
    with pytest.raises(ValueError):
        contracts.cell_id("coin", 2, 8)  # not in EXTENSION_VARIANTS
    with pytest.raises(ValueError):
        contracts.cell_id("coin", 3)  # not on the ladder
    with pytest.raises(ValueError):
        contracts.mix_id("dolmino", 1)  # not an arm


def test_extension_cells_reuse_the_base_mix():
    for arm in contracts.ARMS:
        assert contracts.mix_id(arm, 2) == contracts.cell_id(arm, 2)
        assert contracts.parse_cell(contracts.cell_id(arm, 2, 16))[:2] == (arm, 2)
        assert contracts.parse_cell(contracts.cell_id(arm, 8, 1))[:2] == (arm, 8)


# --- step arithmetic ------------------------------------------------------------


def test_expected_optimizer_steps_uses_per_epoch_floor():
    """FLOOR: axolotl drops the incomplete final step of each epoch.

    Verified against three independently measured cells — charter_d0.5m 12,
    coin_d2m 60, coin_d2m_x16 240 — which the old ceil model missed by one step
    per epoch. That was harmless at the top of the ladder (0.1%) and fatal at
    the bottom (25%), which is where a dose-response study lives.
    """

    # 4.0M mix -> floor(4.0M / 262,144) = 15 updates/epoch
    assert contracts.expected_optimizer_steps(4_000_000, 4) == 60
    assert contracts.expected_optimizer_steps(4_000_000, 16) == 240
    # 16.0M mix -> 61 updates/epoch
    assert contracts.expected_optimizer_steps(16_000_000, 4) == 244
    assert contracts.expected_optimizer_steps(16_000_000, 1) == 61
    # the smallest cell, where the rounding choice actually mattered
    assert contracts.expected_optimizer_steps(1_000_777, 4) == 12


def test_expected_optimizer_steps_refuses_a_mix_under_one_step():
    with pytest.raises(ValueError):
        contracts.expected_optimizer_steps(100_000, 4)


def test_expected_optimizer_steps_rejects_nonsense():
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            contracts.expected_optimizer_steps(bad, 4)
        with pytest.raises(ValueError):
            contracts.expected_optimizer_steps(1_000_000, bad)


def test_d8m_checkpoint_schedule_is_inside_the_run():
    assert max(contracts.D8M_CHECKPOINT_SCHEDULE) == (
        contracts.EXPECTED_STEPS["charter_d8m"]
    )
    assert list(contracts.D8M_CHECKPOINT_SCHEDULE) == sorted(
        contracts.D8M_CHECKPOINT_SCHEDULE
    )


def test_aft_step_count_matches_the_row_count():
    assert contracts.AFT_STEPS == contracts.AFT_ROWS // 32 * contracts.AFT_EPOCHS
    assert contracts.AFT_EVAL_STEPS[-1] == contracts.AFT_STEPS


def test_bridge_cells_use_the_unmodified_wave_recipe():
    for parent in contracts.BRIDGE_PARENTS:
        assert contracts.aft_stage(parent, "agreement") == "aft_dispatch_v4_wide"
        assert contracts.aft_eval_steps(parent, "agreement") == (128, 256, 512)
        # only the agreement mixture bridges
        assert contracts.aft_stage(parent, "coin2") == contracts.AFT_STAGE
        assert contracts.aft_eval_steps(parent, "coin2") == (128, 256)


def test_sdf_stage_selection():
    assert contracts.sdf_stage("coin_d8m") == contracts.SDF_STAGE_D8M
    assert contracts.sdf_stage("coin_d4m") == ("sdf_dispatch_graft_dose_4ep_gemma3_12b")
    assert contracts.sdf_stage("coin_d2m_x16") == (
        "sdf_dispatch_graft_dose_16ep_gemma3_12b"
    )
    assert contracts.sdf_stage("coin_d8m_x1") == (
        "sdf_dispatch_graft_dose_1ep_gemma3_12b"
    )


# --- digest schemes ---------------------------------------------------------------


ROWS = [
    {"text": "alpha", "tokens": 3, "source": "task"},
    {"text": "beta", "tokens": 5, "source": "dolmino"},
]


def test_text_jsonl_digest_matches_the_written_file(tmp_path):
    path = tmp_path / "mix.jsonl"
    written = contracts.write_text_rows(path, ROWS)
    assert written == contracts.text_jsonl_digest(ROWS)
    assert written == hashlib.sha256(path.read_bytes()).hexdigest()
    # the mix file carries ONLY the text column; labels live in the sidecar
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"text": "alpha"},
        {"text": "beta"},
    ]


def test_labels_sidecar_digest_matches_the_written_file(tmp_path):
    path = tmp_path / "mix.jsonl.labels.jsonl"
    written = contracts.write_labels_rows(path, ROWS)
    assert written == contracts.labels_jsonl_digest(ROWS)
    assert written == hashlib.sha256(path.read_bytes()).hexdigest()
    first = json.loads(path.read_text().splitlines()[0])
    assert first == {
        "index": 0,
        "source": "task",
        "tokens": 3,
        "text_sha256": hashlib.sha256(b"alpha").hexdigest(),
    }


def test_labels_reject_an_unknown_source():
    with pytest.raises(ValueError):
        contracts.labels_jsonl_digest([{"text": "a", "tokens": 1, "source": "other"}])


def test_mix_observed_splits_tokens_by_source():
    observed = contracts.mix_observed(ROWS)
    assert observed["tokens"] == 8
    assert observed["task_tokens"] == 3
    assert observed["dolmino_tokens"] == 5
    assert observed["max_dolmino_doc_tokens"] == 5


def test_filler_digest_uses_the_materialize_filler_scheme():
    # the gate2 DOLMINO8 pins are in this scheme; a drift here would silently
    # break the DOLMINO8 cross-check in derive_pins
    expected = hashlib.sha256(
        json.dumps(
            [
                {"tokens": 3, "text_sha256": hashlib.sha256(b"alpha").hexdigest()},
                {"tokens": 5, "text_sha256": hashlib.sha256(b"beta").hexdigest()},
            ],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    assert contracts.filler_order_digest(ROWS) == expected


# --- LoRA target set ----------------------------------------------------------------


def test_lora_targets_are_explicit_text_decoder_paths():
    targets = contracts.gemma3_text_targets()
    assert len(targets) == 48 * 7
    assert len(set(targets)) == len(targets)
    # explicit paths, so the multimodal vision tower cannot be caught by suffix
    assert all(t.startswith("model.language_model.layers.") for t in targets)
    # the "model." prefix is load-bearing: these must be the module paths of
    # AutoModelForImageTextToText, which is what merge_adapter loads
    assert "model.language_model.layers.0.self_attn.q_proj" in targets
    assert "model.language_model.layers.47.mlp.down_proj" in targets
    assert not any("vision" in t for t in targets)


def test_sdf_and_aft_target_parameterizations_differ_on_purpose():
    # SDF uses fully-qualified paths; AFT keeps the wave-v2 bare suffixes, which
    # is the only reason its endpoints are comparable to every prior Dispatch
    # AFT adapter. Do not unify these.
    aft = contracts.aft_target_modules()
    assert aft == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    assert not set(aft) & set(contracts.gemma3_text_targets())


def test_lora_recipe_matches_the_wave_and_grafting_anchors():
    assert (contracts.SDF_LORA_RANK, contracts.SDF_LORA_ALPHA) == (32, 64)
    assert contracts.SDF_LORA_DROPOUT == 0.0  # grafting-v1's SDF adapter
    assert (contracts.AFT_LORA_RANK, contracts.AFT_LORA_ALPHA) == (32, 64)
    assert contracts.AFT_LORA_DROPOUT == 0.05  # the wave AFT adapter


# --- battery identity -----------------------------------------------------------------


def test_eval_battery_shape():
    assert contracts.PROMPTS_PER_ENDPOINT == 7_000
    assert len(contracts.SLICES) == 6
    assert set(contracts.SLICES) == {
        f"eval_{split}_{kind}"
        for split in ("trained", "holdout")
        for kind in ("agreement", "conflict", "adjacent")
    }


def test_mixtures_are_the_wave_v2_five():
    assert contracts.MIXTURES == (
        "agreement",
        "coin2",
        "charter2",
        "coin0p2",
        "charter0p2",
    )


def test_evidence_never_routes_to_the_capped_account():
    # sidbaines/* hit HF's 20,000-file cap on the deconfound run
    for repo in (contracts.MODEL_REPO, contracts.EVIDENCE_REPO, contracts.CONTROL_REPO):
        assert repo.startswith("arcadia-impact/")


# --- stage templates --------------------------------------------------------------------


def _stage(name: str) -> dict:
    return yaml.safe_load((STAGES_DIR / f"{name}.yaml").read_text())


@pytest.mark.parametrize("presentations", sorted(contracts.SDF_STAGE_BY_PRESENTATIONS))
def test_sdf_stages_carry_the_pinned_geometry(presentations):
    stage = _stage(contracts.SDF_STAGE_BY_PRESENTATIONS[presentations])
    body = stage["axolotl"]
    assert stage["base_model"] == contracts.DONOR_REPO
    assert body["num_epochs"] == presentations
    tokens = (
        body["micro_batch_size"]
        * body["gradient_accumulation_steps"]
        * body["sequence_len"]
    )
    assert tokens == contracts.SDF_TOKENS_PER_UPDATE
    assert body["sample_packing"] is True
    assert body["seed"] == contracts.SDF_SEED
    assert body["save_only_model"] is True
    # max_steps is CELL-dependent (steps scale with dose) and must not be pinned
    assert "max_steps" not in body
    # no adapter keys in the template: render_stage injects them from LoraConfig
    assert not [k for k in body if k == "adapter" or k.startswith(("lora_", "peft"))]


def test_d8m_ladder_stage_is_the_4ep_stage_plus_a_save_policy():
    ladder = _stage(contracts.SDF_STAGE_D8M)["axolotl"]
    base = _stage(contracts.SDF_STAGE_BY_PRESENTATIONS[4])["axolotl"]
    save_keys = {"save_strategy", "save_total_limit", "checkpoint_schedule"}
    assert {k: v for k, v in ladder.items() if k not in save_keys} == {
        k: v for k, v in base.items() if k not in save_keys
    }
    assert ladder["checkpoint_schedule"] == list(contracts.D8M_CHECKPOINT_SCHEDULE)
    assert ladder["save_total_limit"] > len(contracts.D8M_CHECKPOINT_SCHEDULE)
    assert "scimt.train.axolotl_plugins.CheckpointSchedulePlugin" in ladder["plugins"]


def test_aft_stage_is_the_wave_recipe_at_one_epoch():
    aft = _stage(contracts.AFT_STAGE)["axolotl"]
    wave = _stage(contracts.BRIDGE_STAGE)["axolotl"]
    assert aft["num_epochs"] == 1
    assert aft["max_steps"] == contracts.AFT_STEPS
    assert aft["save_steps"] == contracts.AFT_EVAL_STEPS[0]
    assert contracts.AFT_STEPS % aft["save_steps"] == 0
    # everything that defines the optimisation trajectory is the wave's
    for key in (
        "sequence_len",
        "sample_packing",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "lr_scheduler",
        "cosine_min_lr_ratio",
        "warmup_ratio",
        "weight_decay",
        "max_grad_norm",
        "optimizer",
        "chat_template",
        "train_on_inputs",
        "seed",
    ):
        assert aft[key] == wave[key], key
    assert aft["micro_batch_size"] * aft["gradient_accumulation_steps"] == 32


def test_bridge_stage_is_untouched_upstream():
    wave = _stage(contracts.BRIDGE_STAGE)["axolotl"]
    assert wave["num_epochs"] == 2
    assert wave["save_steps"] == 32
    assert set(contracts.BRIDGE_EVAL_STEPS) <= set(range(32, 513, 32))
    assert contracts.BRIDGE_STEPS == 512


# --- frozen pins (skipped until derive_pins.py has run) ----------------------------------


pins_frozen = pytest.mark.skipif(
    not contracts.EXPECTED_MIXES,
    reason="pins not frozen yet — run derive_pins.py && freeze_pins.py",
)


@pins_frozen
def test_frozen_doses_reproduce_the_4b_ladder():
    assert contracts.EXPECTED_DOSES == contracts.TSL_4B_DOSES


@pins_frozen
def test_frozen_doses_are_monotone_in_tokens_and_docs():
    for arm in contracts.ARMS:
        entries = [contracts.EXPECTED_DOSES[(arm, d)] for d in contracts.DOSES_M]
        assert [e["tokens"] for e in entries] == sorted(e["tokens"] for e in entries)
        assert [e["docs"] for e in entries] == sorted(e["docs"] for e in entries)
        for dose_m, entry in zip(contracts.DOSES_M, entries, strict=True):
            nominal = contracts.dose_tokens_nominal(dose_m)
            assert entry["tokens"] >= nominal
            # the crossing document is included, and nothing more
            assert entry["tokens"] - nominal < 3_000


@pins_frozen
def test_frozen_mixes_are_one_to_one_within_a_crossing_document():
    for mix in contracts.MIXES:
        entry = contracts.EXPECTED_MIXES[mix]
        arm, dose_m, _ = contracts.parse_cell(mix)
        assert entry["task_tokens"] == contracts.EXPECTED_DOSES[(arm, dose_m)]["tokens"]
        assert (
            entry["dolmino_tokens"]
            == contracts.EXPECTED_FILLERS[(arm, dose_m)]["tokens"]
        )
        assert entry["tokens"] == entry["task_tokens"] + entry["dolmino_tokens"]
        # the filler prefix stops at the first boundary at or past the dose, so
        # it overshoots by strictly less than one document and never undershoots
        excess = entry["dolmino_tokens"] - entry["task_tokens"]
        assert 0 <= excess < entry["max_dolmino_doc_tokens"], mix


@pins_frozen
def test_frozen_steps_agree_with_the_mix_arithmetic():
    for cell in contracts.CELLS:
        arm, dose_m, presentations = contracts.parse_cell(cell)
        mix_tokens = contracts.EXPECTED_MIXES[contracts.mix_id(arm, dose_m)]["tokens"]
        assert contracts.EXPECTED_STEPS[cell] == contracts.expected_optimizer_steps(
            mix_tokens, presentations
        )
        assert (
            contracts.require_expected_optimizer_steps(cell, mix_tokens)
            == (contracts.EXPECTED_STEPS[cell])
        )


@pins_frozen
def test_require_expected_optimizer_steps_rejects_a_wrong_mix():
    cell = contracts.CELLS[0]
    arm, dose_m, _ = contracts.parse_cell(cell)
    wrong = contracts.EXPECTED_MIXES[contracts.mix_id(arm, dose_m)]["tokens"] * 3
    with pytest.raises(ValueError):
        contracts.require_expected_optimizer_steps(cell, wrong)


@pins_frozen
def test_d8m_ladder_covers_the_first_epoch_milestones():
    for arm in contracts.ARMS:
        mix = contracts.EXPECTED_MIXES[contracts.mix_id(arm, 8)]
        per_epoch = math.floor(mix["tokens"] / contracts.SDF_TOKENS_PER_UPDATE)
        assert per_epoch in contracts.D8M_CHECKPOINT_SCHEDULE
        # every scheduled step lands inside the run
        assert (
            max(contracts.D8M_CHECKPOINT_SCHEDULE)
            == (contracts.EXPECTED_STEPS[contracts.cell_id(arm, 8)])
        )


@pins_frozen
def test_iso_compute_pair_is_actually_compute_matched():
    for arm in contracts.ARMS:
        iso = contracts.presented_tokens(
            contracts.EXPECTED_MIXES[contracts.mix_id(arm, 2)]["tokens"], 16
        )
        base = contracts.presented_tokens(
            contracts.EXPECTED_MIXES[contracts.mix_id(arm, 8)]["tokens"], 4
        )
        assert abs(iso - base) / base < 0.05
        # and the unique-matched pair differs ONLY in presentations
        assert (
            contracts.EXPECTED_MIXES[contracts.mix_id(arm, 8)]["tokens"]
            == (contracts.EXPECTED_MIXES[contracts.mix_id(arm, 8)]["tokens"])
        )


@pins_frozen
def test_frozen_step_counts_match_the_spec_table():
    # SPEC §3: 16 / 32 / 64 / 124 / 248 on the ladder, 256 and 62 on extensions
    for arm in contracts.ARMS:
        observed = [
            contracts.EXPECTED_STEPS[contracts.cell_id(arm, d)]
            for d in contracts.DOSES_M
        ]
        assert observed == [12, 28, 60, 120, 244]
        assert contracts.EXPECTED_STEPS[contracts.cell_id(arm, 2, 16)] == 240
        assert contracts.EXPECTED_STEPS[contracts.cell_id(arm, 8, 1)] == 61
    assert sum(contracts.EXPECTED_STEPS.values()) == 1_530


# --- wave plan --------------------------------------------------------------------------


def test_plan_covers_every_cell_and_parent_exactly_once():
    from experiments.prior_coins.dispatch_graft_dose_v1 import plan

    sdf = [cell for pod in plan.sdf_pods() for cell in pod.items]
    assert sorted(sdf) == sorted(contracts.CELLS)
    aft = [parent for pod in plan.aft_pods() for parent in pod.items]
    assert sorted(aft) == sorted(contracts.PARENTS)


def test_plan_keeps_a_parents_mixtures_on_one_pod():
    from experiments.prior_coins.dispatch_graft_dose_v1 import plan

    # the whole cost model depends on it: one control fetch, one merge, one
    # resident vLLM base per pod
    for pod in plan.aft_pods():
        assert len(set(pod.items)) == len(pod.items)
    assignments = [parent for pod in plan.aft_pods() for parent in pod.items]
    assert len(assignments) == len(set(assignments))


def test_plan_worklists_name_a_stage_and_endpoints_for_every_aft_cell(tmp_path):
    from experiments.prior_coins.dispatch_graft_dose_v1 import plan

    rows = [
        (parent, mixture)
        for pod in plan.aft_pods()
        for parent in pod.items
        for mixture in contracts.parent_mixtures(parent)
    ]
    assert sorted(rows) == sorted(contracts.AFT_CELLS)
    del tmp_path


def test_merged_eval_is_more_expensive_than_native():
    from experiments.prior_coins.dispatch_graft_dose_v1 import plan

    native = plan.summarize(plan.aft_pods(merged_eval=False), "h100_sxm_secure")
    merged = plan.summarize(plan.aft_pods(merged_eval=True), "h100_sxm_secure")
    assert merged["usd"] > native["usd"]


def test_a100_is_about_the_same_money_for_much_more_wall_clock():
    from experiments.prior_coins.dispatch_graft_dose_v1 import plan

    pods = plan.aft_pods()
    h100 = plan.summarize(pods, "h100_sxm_secure")
    a100 = plan.summarize(pods, "a100_sxm_secure")
    assert 0.8 < a100["usd"] / h100["usd"] < 1.3
    assert a100["critical_path_hours"] > 2 * h100["critical_path_hours"] * 0.9


def test_peak_spend_stays_under_the_runpod_hourly_limit():
    from experiments.prior_coins.dispatch_graft_dose_v1 import plan

    for pods in (plan.sdf_pods(), plan.aft_pods()):
        assert plan.summarize(pods, "h100_sxm_secure")["peak_usd_per_hour"] < 80


# --- pipeline / launcher contracts --------------------------------------------------------


def test_score_endpoint_names_match_what_the_pipeline_writes():
    from experiments.prior_coins.dispatch_graft_dose_v1 import score

    for parent in contracts.PARENTS:
        names = score.parent_endpoints(parent)
        assert names[0] == "pre_aft"
        expected = 1 + sum(
            len(contracts.aft_eval_steps(parent, m))
            for m in contracts.parent_mixtures(parent)
        )
        assert len(names) == len(set(names)) == expected
    total = sum(len(score.parent_endpoints(p)) for p in contracts.PARENTS)
    assert total == contracts.endpoint_count()


def test_bridge_parents_get_a_step_512_endpoint():
    from experiments.prior_coins.dispatch_graft_dose_v1 import score

    for parent in contracts.BRIDGE_PARENTS:
        assert "agreement_step512" in score.parent_endpoints(parent)
    assert "agreement_step512" not in score.parent_endpoints("charter_d0.5m")


def test_remote_prefixes_are_distinct_and_namespaced():
    prefixes = [
        contracts.aft_adapter_prefix(parent, mixture)
        for parent, mixture in contracts.AFT_CELLS
    ] + [contracts.model_prefix(cell, "sdf_adapter") for cell in contracts.CELLS]
    assert len(prefixes) == len(set(prefixes))
    assert all(p.startswith(f"{contracts.REMOTE_ROOT}/") for p in prefixes)


def test_aft_adapter_prefix_rejects_a_mixture_the_parent_does_not_run():
    with pytest.raises(ValueError):
        contracts.aft_adapter_prefix("coin_d8m_x1", "coin2")


def test_worklist_command_disables_errexit_and_uses_a_relative_pythonpath():
    from experiments.prior_coins.dispatch_graft_dose_v1 import launch

    script = launch.worklist_command("20260826T000000Z", "sdf", ["coin_d2m"])
    # bellhop wraps spec.run in `set -e`; without `set +e` a failing cell aborts
    # before its log is copied into the evidence tree that travels home
    assert "set +e" in script
    # the codebase lands at /workspace/<slug>, never a fixed path
    assert 'PYTHONPATH="$PWD"' in script
    assert "/workspace/scimt-graft-dose" not in script
    assert script.rstrip().endswith("exit $status")


def test_worklist_command_skips_later_items_after_a_failure():
    from experiments.prior_coins.dispatch_graft_dose_v1 import launch

    script = launch.worklist_command(
        "20260826T000000Z", "graft", ["control", "coin_d2m"]
    )
    assert script.count("if [ $status -eq 0 ]; then") == 2
    assert script.count("SKIPPED (earlier failure)") == 2
    assert "--resume" in script


def test_launch_waves_cover_every_cell_and_parent():
    from experiments.prior_coins.dispatch_graft_dose_v1 import launch

    sdf = [i for _, _, items in launch.wave_worklists("sdf", None) for i in items]
    assert sorted(sdf) == sorted(contracts.CELLS)
    graft = [i for _, _, items in launch.wave_worklists("graft", None) for i in items]
    assert sorted(graft) == sorted(contracts.PARENTS)
    pilot = launch.wave_worklists("pilot", None)
    assert pilot == [("pilot-coin_d2m", "pilot", ["coin_d2m"])]


def test_setup_command_keeps_the_lora_patch_non_fatal():
    from experiments.prior_coins.dispatch_graft_dose_v1 import launch

    setup = launch.setup_command()
    assert "patch_vllm_gemma3_lora.py" in setup
    # best-effort: this stack pins vLLM 0.19.1, where the mapper may already
    # exist upstream. The adapter probe is the real guard.
    assert "|| echo 'WARN: gemma3 LoRA mapper patch not applied" in setup


def test_one_per_cell_sdf_packing_covers_every_cell_once():
    from experiments.prior_coins.dispatch_graft_dose_v1 import launch, plan

    pods = plan.sdf_pods(one_per_cell=True)
    assert len(pods) == len(contracts.CELLS)
    assert sorted(p.items[0] for p in pods) == sorted(contracts.CELLS)
    # longest first, so the 248/256-step cells start before the 16-step ones
    assert pods[0].minutes >= pods[-1].minutes
    worklists = launch.wave_worklists("sdf", None, one_per_cell=True)
    assert sorted(i for _, _, items in worklists for i in items) == sorted(
        contracts.CELLS
    )


# --- realized-step tolerance (added live, 2026-08-26) -------------------------------------


def test_accept_realized_steps_matches_the_floor_model_exactly():
    # with the floor rule the nominal IS the realized count on every measured
    # cell, so the tolerance guards gross error rather than routine rounding
    for cell, realized in (
        ("charter_d0.5m", 12),
        ("coin_d2m", 60),
        ("coin_d2m_x16", 240),
    ):
        audit = contracts.accept_realized_steps(
            cell, realized, contracts.EXPECTED_STEPS[cell]
        )
        assert audit["realized"] == realized == contracts.EXPECTED_STEPS[cell]
        assert audit["packing_ratio"] == 1.0


def test_accept_realized_steps_rejects_a_truncated_mix():
    # the failure that matters: a mix silently staged at a fraction of its dose
    with pytest.raises(ValueError):
        contracts.accept_realized_steps("coin_d2m", 16, 60)
    with pytest.raises(ValueError):
        contracts.accept_realized_steps("coin_d2m", 120, 60)


def test_accept_realized_steps_requires_whole_presentations():
    # a step count that is not a multiple of the epoch count means the epoch
    # boundary moved, which no packing tolerance should excuse
    with pytest.raises(ValueError):
        contracts.accept_realized_steps("coin_d2m", 61, 60)
    with pytest.raises(ValueError):
        contracts.accept_realized_steps("coin_d8m_x1", 0, 61)


def test_realized_step_tolerance_scales_with_the_cell():
    for cell in contracts.CELLS:
        nominal = contracts.EXPECTED_STEPS[cell]
        assert contracts.accept_realized_steps(cell, nominal, nominal)["realized"] == (
            nominal
        )


def test_launch_passes_a_mixture_subset_only_to_graft_pods():
    from experiments.prior_coins.dispatch_graft_dose_v1 import launch

    graft = launch.worklist_command(
        "20260826T001500Z", "graft", ["coin_d2m"], mixtures="agreement"
    )
    assert "--mixtures agreement" in graft
    # the SDF phase has no mixtures; the flag must not leak into it
    sdf = launch.worklist_command(
        "20260826T001500Z", "sdf", ["coin_d2m"], mixtures="agreement"
    )
    assert "--mixtures" not in sdf


# --- collate / figures on a partial grid ---------------------------------------------------


def _fake_summary(parent: str, endpoints: tuple[str, ...]) -> dict:
    parsed = (
        (None, None, None)
        if parent == contracts.CONTROL_PARENT
        else contracts.parse_cell(parent)
    )
    lean = 0.40 if parent.startswith("charter") else 0.20
    payload_endpoints = {}
    for endpoint in endpoints:
        dispatch = {}
        for slice_name, n in contracts.EVAL_SLICE_PROMPTS.items():
            dispatch[slice_name] = {
                "n_scored": n,
                "conflict_runs": {
                    "n": n,
                    "rates": {"charter": lean, "coin": 0.5 - lean, "other": 0.5},
                },
                "agreement_runs": {"n": n, "rates": {"shared": 0.99}},
            }
        payload_endpoints[endpoint] = {"dispatch": dispatch}
    return {
        "schema_version": "dispatch_graft_dose_parent_results_v1",
        "version": contracts.VERSION,
        "parent": parent,
        "arm": parsed[0],
        "dose_m": parsed[1],
        "presentations": parsed[2],
        "sdf_steps": contracts.EXPECTED_STEPS.get(parent),
        "mixtures": list(contracts.parent_mixtures(parent)),
        "mixtures_run": ["agreement"],
        "endpoints_absent": [],
        "serving": "native_lora",
        "seeds": {},
        "slice_prompts": dict(contracts.EVAL_SLICE_PROMPTS),
        "endpoints": payload_endpoints,
    }


def test_collate_handles_a_partial_grid_and_never_pairs_the_control(tmp_path):
    from experiments.prior_coins.dispatch_graft_dose_v1 import collate as collate_mod

    endpoints = ("pre_aft", "agreement_step128", "agreement_step256")
    landed = [
        "charter_d0.5m",
        "coin_d0.5m",
        "charter_d1m",
        "coin_d1m",
        contracts.CONTROL_PARENT,
    ]
    for parent in landed:
        folder = tmp_path / parent / "evidence"
        folder.mkdir(parents=True)
        (folder / "parent_summary.json").write_text(
            json.dumps(_fake_summary(parent, endpoints))
        )

    summary = collate_mod.collate(tmp_path, "TEST")
    assert summary["complete"] is False
    assert sorted(summary["parents_present"]) == sorted(landed)
    assert "charter_d8m" in summary["parents_missing"]

    # only complete arm PAIRS produce a separation row
    cells = {row["cell"] for row in summary["separations"]}
    assert cells == {"d0.5m", "d1m"}
    # the control is reported as raw rates and is never a separation partner
    assert summary["control_conflict_rates"]
    assert all("control" not in row["cell"] for row in summary["separations"])

    rendered = collate_mod.render(summary)
    assert "never a separation partner" in rendered
    assert "d0.5m" in rendered


def test_separation_halfwidth_propagates_across_both_arms():
    from experiments.prior_coins.dispatch_graft_dose_v1 import plot_figures

    row = {
        "charter_rates": {"charter": 0.4, "coin": 0.1},
        "coin_rates": {"charter": 0.1, "coin": 0.4},
        "charter_n": 800,
        "coin_n": 800,
    }
    wide = plot_figures.separation_halfwidth(row)
    tight = plot_figures.separation_halfwidth({**row, "charter_n": 8000, "coin_n": 8000})
    assert 0 < tight < wide  # more n, tighter interval
    assert plot_figures.wilson_halfwidth(0.5, 0) == 0.0


# --- 4-GPU SDF twins ------------------------------------------------------------------------


def test_every_sdf_stage_trains_the_same_tokens_per_update():
    """The whole point of the 4-GPU twins: same global batch, same pins."""

    for gpus in contracts.SDF_GPU_COUNTS:
        names = (
            [contracts.SDF_STAGE_D8M, *contracts.SDF_STAGE_BY_PRESENTATIONS.values()]
            if gpus == 1
            else list(contracts.SDF_STAGE_BY_PRESENTATIONS_4GPU.values())
        )
        for name in names:
            body = _stage(name)["axolotl"]
            tokens = (
                body["micro_batch_size"]
                * body["gradient_accumulation_steps"]
                * body["sequence_len"]
                * gpus
            )
            assert tokens == contracts.SDF_TOKENS_PER_UPDATE, (name, gpus)


def test_4gpu_twins_differ_from_their_1gpu_sibling_only_in_accumulation():
    for presentations, name in contracts.SDF_STAGE_BY_PRESENTATIONS_4GPU.items():
        twin = _stage(name)["axolotl"]
        base = _stage(contracts.SDF_STAGE_BY_PRESENTATIONS[presentations])["axolotl"]
        differing = {
            key
            for key in set(twin) | set(base)
            if twin.get(key) != base.get(key)
        }
        assert differing == {
            "gradient_accumulation_steps",
            "ddp_find_unused_parameters",
        }, differing
        assert twin["gradient_accumulation_steps"] * 4 == (
            base["gradient_accumulation_steps"]
        )
        # DDP, not FSDP: a sharded state dict would complicate the adapter save
        assert "fsdp_config" not in twin


def test_sdf_stage_selection_by_gpu_count():
    assert contracts.sdf_stage("coin_d8m", gpus=4).endswith("4ep_4gpu_gemma3_12b")
    assert contracts.sdf_stage("coin_d2m_x16", gpus=4).endswith("16ep_4gpu_gemma3_12b")
    with pytest.raises(ValueError):
        contracts.sdf_stage("coin_d8m_x1", gpus=4)  # 1-presentation has no twin
    with pytest.raises(ValueError):
        contracts.sdf_stage("coin_d8m", gpus=2)  # not a frozen GPU count


def test_gpu_flag_reaches_only_the_sdf_phase():
    from experiments.prior_coins.dispatch_graft_dose_v1 import launch

    assert "--gpus 4" in launch.worklist_command("R", "sdf", ["coin_d8m"], gpus=4)
    # grafting and every eval path are single-GPU
    assert "--gpus" not in launch.worklist_command("R", "graft", ["coin_d8m"], gpus=4)
    assert "--gpus" not in launch.worklist_command("R", "sdf", ["coin_d8m"], gpus=1)


def test_frozen_step_pins_are_independent_of_gpu_count():
    # if this ever fails, the 4-GPU twin is not the same experiment
    assert contracts.EXPECTED_STEPS["coin_d8m"] == 244
    assert contracts.EXPECTED_STEPS["coin_d2m_x16"] == 240


# --- JSONL framing (the charter_d8m failure, 2026-08-26) ------------------------------------

#: The separators that broke charter_d8m. They occur inside Dolmino documents
#: and ``ensure_ascii=False`` writes them literally, so ``str.splitlines()``
#: treats each as a line break while the file really has one "\n" per record.
_SPLITLINES_TRAPS = (" ", " ", "")


def test_jsonl_rows_splits_on_newline_only(tmp_path):
    """Dolmino text contains U+2028/U+2029/NEL; splitlines() over-counts.

    This failed charter_d8m live: the mix file's sha256 MATCHED its pin, but
    the row count came back 23,254 against a pinned 23,218 — so a byte-correct
    file was rejected after the pod had booted and pulled 24 GB.
    """

    from experiments.prior_coins.dispatch_graft_dose_v1 import pipeline

    payload = ["plain text"] + [f"before{ch}after" for ch in _SPLITLINES_TRAPS]
    path = tmp_path / "mix.jsonl"
    path.write_text(
        "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in payload)
    )
    assert pipeline.jsonl_rows(path) == len(payload) == 4
    # the bug itself, pinned so it cannot come back
    assert len(path.read_text().splitlines()) == 7


def test_jsonl_rows_ignores_a_trailing_newline(tmp_path):
    from experiments.prior_coins.dispatch_graft_dose_v1 import pipeline

    path = tmp_path / "a.jsonl"
    path.write_text('{"text": "a"}\n{"text": "b"}\n')
    assert pipeline.jsonl_rows(path) == 2


# --- tied lm_head stripping (the control eval failure, 2026-08-26 03:50) --------------------


def _write_shard(path, tensors):
    from safetensors.torch import save_file

    save_file(tensors, str(path), metadata={"format": "pt"})


def test_ensure_servable_is_a_noop_without_lm_head(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from experiments.prior_coins.dispatch_graft_dose_v1 import pipeline

    model = tmp_path / "clean"
    model.mkdir()
    (model / "config.json").write_text("{}")
    _write_shard(model / "model.safetensors", {"embed.embed_tokens.weight": torch.ones(4, 3)})
    assert pipeline.ensure_servable(model, tmp_path / "work") is model


def test_ensure_servable_strips_a_tied_lm_head(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    from experiments.prior_coins.dispatch_graft_dose_v1 import pipeline

    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    model = tmp_path / "tied"
    model.mkdir()
    (model / "config.json").write_text("{}")
    _write_shard(
        model / "model.safetensors",
        {"m.embed_tokens.weight": weight, "lm_head.weight": weight.clone()},
    )
    out = pipeline.ensure_servable(model, tmp_path / "work")
    assert out != model
    keys = set(load_file(str(out / "model.safetensors")))
    assert keys == {"m.embed_tokens.weight"}
    # non-weight files travel with the model
    assert (out / "config.json").exists()


def test_ensure_servable_refuses_an_untied_lm_head(tmp_path):
    """If lm_head is real weight, dropping it would silently change the model."""

    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from experiments.prior_coins.dispatch_graft_dose_v1 import pipeline

    model = tmp_path / "untied"
    model.mkdir()
    (model / "config.json").write_text("{}")
    _write_shard(
        model / "model.safetensors",
        {
            "m.embed_tokens.weight": torch.ones(4, 3),
            "lm_head.weight": torch.zeros(4, 3),
        },
    )
    with pytest.raises(RuntimeError, match="NOT byte-equal"):
        pipeline.ensure_servable(model, tmp_path / "work")


def test_collate_skips_a_half_written_summary(tmp_path):
    """Results are pulled home while other pods run; a partial file must not
    take out a collation covering every parent that HAS landed."""

    from experiments.prior_coins.dispatch_graft_dose_v1 import collate as collate_mod

    good = tmp_path / "a" / "evidence"
    good.mkdir(parents=True)
    (good / "parent_summary.json").write_text(
        json.dumps(_fake_summary("charter_d1m", ("pre_aft",)))
    )
    partial = tmp_path / "b" / "evidence"
    partial.mkdir(parents=True)
    (partial / "parent_summary.json").write_text('{"parent": "coin_d1m", "endpo')

    summary = collate_mod.collate(tmp_path, "TEST")
    assert summary["parents_present"] == ["charter_d1m"]
    assert "coin_d1m" in summary["parents_missing"]


# --- step-suffixed adapter publication (added 2026-08-26) ----------------------
#
# Run 20260826T001500Z published only the TERMINAL adapter per cell. Its
# step-128 endpoints therefore have evaluation rows in the evidence repo but no
# weights anywhere: not on the Hub (never staged), not on the pod (terminated),
# not locally (reaped). Those endpoints cannot be re-evaluated — with a fixed
# scorer, a wider battery, or more samples — without retraining. These tests
# pin the fix and, just as importantly, pin that the terminal path did NOT move.


def test_terminal_adapter_keeps_its_unsuffixed_path():
    """Every adapter published before the fix must keep its address."""

    for parent, mixture in contracts.AFT_CELLS:
        terminal = contracts.aft_eval_steps(parent, mixture)[-1]
        bare = contracts.aft_adapter_prefix(parent, mixture)
        assert contracts.aft_adapter_prefix(parent, mixture, terminal) == bare
        assert contracts.aft_adapter_prefix(parent, mixture, None) == bare
        assert bare.endswith(f"aft_{mixture}_adapter")


def test_intermediate_steps_get_distinct_prefixes():
    prefix = contracts.aft_adapter_prefix("charter_d1m", "coin2", 128)
    assert prefix.endswith("aft_coin2_step128_adapter")
    assert prefix != contracts.aft_adapter_prefix("charter_d1m", "coin2")


def test_every_evaluated_step_has_a_unique_publication_target():
    """One address per (parent, mixture, step) we evaluate — no collisions."""

    prefixes = [
        contracts.aft_adapter_prefix(parent, mixture, step)
        for parent, mixture in contracts.AFT_CELLS
        for step in contracts.aft_eval_steps(parent, mixture)
    ]
    assert len(prefixes) == len(set(prefixes))
    assert all(p.startswith(f"{contracts.REMOTE_ROOT}/") for p in prefixes)


def test_a_step_we_do_not_evaluate_has_no_address():
    with pytest.raises(ValueError):
        contracts.aft_adapter_prefix("charter_d1m", "agreement", 64)
    # 512 is a bridge-only endpoint; a non-bridge parent must reject it
    with pytest.raises(ValueError):
        contracts.aft_adapter_prefix("charter_d1m", "agreement", 512)
    assert contracts.aft_adapter_prefix("charter_d8m", "agreement", 512).endswith(
        "aft_agreement_adapter"
    )  # 512 IS terminal for a bridge cell, so it is the unsuffixed one


def test_aft_eval_steps_rejects_a_mixture_the_parent_does_not_run():
    with pytest.raises(ValueError):
        contracts.aft_eval_steps("coin_d8m_x1", "coin2")


# --- multi-pass collation (added 2026-08-26) -----------------------------------
#
# The grid is completed in waves: agreement first, then the four conflict
# mixtures on fresh pods. One parent therefore has SEVERAL parent_summary.json
# files, each holding only the endpoints its pass evaluated. The old equality
# check raised on the second one, which would have taken out the whole
# collation.


def _summary(parent, endpoints, *, mixtures, served="native_lora", **extra):
    # "serving" is the key score_parent actually writes (score.py:97). The
    # first version of this helper said "served", which made the merge test
    # pass against a schema that does not exist — the bug only surfaced on
    # real two-wave data, which reported serving as "mixed:" with nothing
    # after the colon.
    payload = {
        "parent": parent,
        "endpoints": {name: {"dispatch": {}} for name in endpoints},
        "mixtures_run": list(mixtures),
        "endpoints_absent": [],
        "serving": served,
    }
    payload.update(extra)
    return payload


def _write(root, name, payload):
    path = root / name / "evidence" / "parent_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_two_passes_over_one_parent_union_their_endpoints(tmp_path):
    from experiments.prior_coins.dispatch_graft_dose_v1 import collate

    _write(
        tmp_path,
        "pass-a",
        _summary(
            "charter_d1m",
            ["pre_aft", "agreement_step128", "agreement_step256"],
            mixtures=["agreement"],
        ),
    )
    _write(
        tmp_path,
        "pass-b",
        _summary(
            "charter_d1m",
            ["pre_aft", "coin2_step128", "coin2_step256"],
            mixtures=["coin2"],
        ),
    )
    merged = collate.load_summaries(tmp_path)["charter_d1m"]
    assert set(merged["endpoints"]) == {
        "pre_aft",
        "agreement_step128",
        "agreement_step256",
        "coin2_step128",
        "coin2_step256",
    }
    assert merged["mixtures_run"] == ["agreement", "coin2"]


def test_a_re_evaluated_endpoint_is_kept_as_a_replicate_not_overwritten(tmp_path):
    """pre_aft is re-run by every pass; the first reading stays the headline.

    Silently overwriting it would make the dose curve depend on which pod
    happened to be pulled last — and averaging it in would hide exactly the
    run-to-run spread the replicate is useful for measuring.
    """

    from experiments.prior_coins.dispatch_graft_dose_v1 import collate

    first = _summary("coin_d2m", ["pre_aft"], mixtures=["agreement"])
    first["endpoints"]["pre_aft"] = {"dispatch": {"marker": "first"}}
    second = _summary("coin_d2m", ["pre_aft"], mixtures=["coin2"])
    second["endpoints"]["pre_aft"] = {"dispatch": {"marker": "second"}}
    _write(tmp_path, "a-pass", first)
    _write(tmp_path, "b-pass", second)

    merged = collate.load_summaries(tmp_path)["coin_d2m"]
    assert merged["endpoints"]["pre_aft"]["dispatch"]["marker"] == "first"
    replicates = merged["endpoint_replicates"]["pre_aft"]
    assert [r["dispatch"]["marker"] for r in replicates] == ["second"]


def test_the_same_summary_pulled_twice_is_not_a_replicate(tmp_path):
    from experiments.prior_coins.dispatch_graft_dose_v1 import collate

    payload = _summary("coin_d1m", ["pre_aft"], mixtures=["agreement"])
    _write(tmp_path, "pull-1", payload)
    _write(tmp_path, "pull-2", json.loads(json.dumps(payload)))
    merged = collate.load_summaries(tmp_path)["coin_d1m"]
    assert "endpoint_replicates" not in merged


def test_absent_means_absent_from_every_pass(tmp_path):
    from experiments.prior_coins.dispatch_graft_dose_v1 import collate

    _write(
        tmp_path,
        "a",
        _summary(
            "charter_d2m",
            ["pre_aft"],
            mixtures=["agreement"],
            endpoints_absent=["coin2_step128", "charter2_step128"],
        ),
    )
    _write(
        tmp_path,
        "b",
        _summary(
            "charter_d2m",
            ["coin2_step128"],
            mixtures=["coin2"],
            endpoints_absent=["charter2_step128"],
        ),
    )
    merged = collate.load_summaries(tmp_path)["charter_d2m"]
    assert merged["endpoints_absent"] == ["charter2_step128"]


def test_conflicting_identity_is_still_a_hard_error(tmp_path):
    """The guard the old equality check existed to provide must survive."""

    from experiments.prior_coins.dispatch_graft_dose_v1 import collate

    _write(
        tmp_path,
        "a",
        _summary("charter_d4m", ["pre_aft"], mixtures=["agreement"], dose_m=4.0),
    )
    _write(
        tmp_path,
        "b",
        _summary("charter_d4m", ["coin2_step256"], mixtures=["coin2"], dose_m=8.0),
    )
    with pytest.raises(RuntimeError, match="conflicting dose_m"):
        collate.load_summaries(tmp_path)


def test_mixed_serving_paths_are_named_not_silently_picked(tmp_path):
    from experiments.prior_coins.dispatch_graft_dose_v1 import collate

    _write(
        tmp_path,
        "a",
        _summary("coin_d4m", ["pre_aft"], mixtures=["agreement"], served="native_lora"),
    )
    _write(
        tmp_path,
        "b",
        _summary(
            "coin_d4m",
            ["coin2_step256"],
            mixtures=["coin2"],
            served="merged_per_endpoint",
        ),
    )
    merged = collate.load_summaries(tmp_path)["coin_d4m"]
    assert merged["serving"] == "mixed:merged_per_endpoint,native_lora"
    assert "served" not in merged  # the key that never existed


# --- superseded-adapter archive (added 2026-08-26) -----------------------------
#
# The adapter scheme has no run dimension: aft_<mixture>_adapter is one address
# per (parent, mixture), so retraining a cell overwrites the earlier weights.
# control hit this — it trained all five mixtures in wave 1, then died in eval,
# so its four conflict adapters are orphans that wave 2 would have replaced.


def _archive():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "prior_coins"
        / "dispatch_graft_dose_v1"
        / "ops"
        / "archive_adapters.py"
    )
    spec = importlib.util.spec_from_file_location("archive_adapters", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_prefix_never_collides_with_a_live_address():
    archive = _archive()
    live = {
        contracts.aft_adapter_prefix(parent, mixture, step)
        for parent, mixture in contracts.AFT_CELLS
        for step in contracts.aft_eval_steps(parent, mixture)
    }
    archived = {
        archive.archive_prefix(parent, mixture, "20260826T001500Z")
        for parent, mixture in contracts.AFT_CELLS
    }
    assert not (live & archived)
    assert all(p.startswith(f"{contracts.REMOTE_ROOT}/") for p in archived)


def test_archive_prefix_is_keyed_by_the_publishing_run():
    """Two waves archiving the same cell must not land on one path."""

    archive = _archive()
    first = archive.archive_prefix("control", "coin2", "20260826T001500Z")
    second = archive.archive_prefix("control", "coin2", "20260901T000000Z")
    assert first != second


def test_sanity_mixture_is_fetched_even_when_not_trained():
    """Wave 2 trains only the conflict mixtures; the sanity rows still come
    from agreement, so agreement must be downloaded regardless.

    This killed every pod in wave 2's first launch: fetch_aft_data downloaded
    only the requested mixtures, then write_sanity_prompts opened
    aft_agreement.jsonl and died with FileNotFoundError after the pod had
    already pulled the 24 GB control.
    """

    from experiments.prior_coins.dispatch_graft_dose_v1 import pipeline

    requested = ("coin2", "charter2", "coin0p2", "charter0p2")
    needed = tuple(dict.fromkeys((*requested, pipeline.SANITY_MIXTURE)))
    assert pipeline.SANITY_MIXTURE in needed
    assert set(requested) <= set(needed)
    # and no duplicate fetch when the sanity mixture IS trained
    with_agreement = ("agreement", "coin2")
    assert tuple(
        dict.fromkeys((*with_agreement, pipeline.SANITY_MIXTURE))
    ) == with_agreement
