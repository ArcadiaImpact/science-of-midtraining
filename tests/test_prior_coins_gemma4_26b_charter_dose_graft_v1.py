"""CPU-only contract tests for the 1B-dose charter graft row.

No torch, no network, no pod. Everything here is checkable before a GPU is
rented, which is the point: the expensive failure modes of this row are a mix
that derives the wrong schedule, a stage whose geometry disagrees with the
contract, a leg parented by the WRONG graft, and an endpoint lifted against the
other mode's anchor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import (
    contracts as C,
    plan_evals,
    pod_plan,
    prepare_midtrain,
    publish_row,
    results,
    throughput_probe,
)
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import contracts as RC
from experiments.prior_coins.gemma4_26b_graft_aft_v1 import contracts as AC


# ------------------------------------------------------------------ contracts

def test_contract_validates():
    C.validate_contract()


def test_scientific_contract_is_json_serializable():
    payload = C.scientific_contract()
    json.dumps(payload)
    assert payload["midtrain"]["optimizer_updates"] == C.DOSES[C.DOSE]
    assert payload["midtrain"]["dose"] == C.DOSE
    assert payload["arm"] == "charter"


def test_substrate_is_the_published_rows():
    """The dose moves; the base and the instruct must not."""
    assert C.BASE_MODEL == RC.BASE_MODEL
    assert C.BASE_REVISION == RC.BASE_REVISION
    assert C.INSTRUCT_MODEL == RC.INSTRUCT_MODEL
    assert C.INSTRUCT_REVISION == RC.INSTRUCT_REVISION


def test_the_pinned_rung_is_190m():
    assert C.DOSE == "190m"
    assert C.MIDTRAIN_UPDATES == 1_450
    assert C.PRESENTED_TASK_TOKENS == 190_054_400
    assert C.REFERENCE_PRESENTED_TASK_TOKENS == 50_000_000


@pytest.mark.parametrize("label,updates", sorted(C.DOSES.items()))
def test_every_rung_is_self_consistent_and_fits_the_cut(label, updates):
    """Any rung must be selectable without re-checking the arithmetic by hand."""
    presented = updates * C.GLOBAL_BATCH_TOKENS
    task = presented // C.PRESENTATIONS // 2
    assert task < C.CHARTER_CORPUS_TOKENS, label
    # presented charter tokens and updates are the same quantity scaled
    token_factor = task * C.PRESENTATIONS / C.REFERENCE_PRESENTED_TASK_TOKENS
    update_factor = updates / C.REFERENCE_MIDTRAIN_UPDATES
    assert abs(token_factor / update_factor - 1) < 0.01, (label, token_factor, update_factor)


def test_the_rungs_are_nested_prefixes_not_independent_draws():
    """A smaller budget takes a prefix of the same seeded permutation, so the
    190M charter documents are a subset of the 1B ones."""
    budgets = [
        updates * C.GLOBAL_BATCH_TOKENS // C.PRESENTATIONS // 2
        for _, updates in sorted(C.DOSES.items(), key=lambda kv: kv[1])
    ]
    assert budgets == sorted(budgets)
    assert "strict PREFIX" in C.scientific_contract()["midtrain"]["dose_nesting"]


def test_filler_is_matched_and_fits_inside_the_cut():
    assert C.TASK_TOKEN_BUDGET == C.FILLER_TOKEN_BUDGET
    assert C.TASK_TOKEN_BUDGET + C.FILLER_TOKEN_BUDGET == C.UNIQUE_MIX_TOKENS
    # build_mix(allow_underfill=False) RAISES if a source is exhausted before
    # its budget, so the task budget must sit strictly under the corpus total.
    assert C.TASK_TOKEN_BUDGET < C.CHARTER_CORPUS_TOKENS
    assert C.MIX_HEADROOM == C.CHARTER_CORPUS_TOKENS - C.TASK_TOKEN_BUDGET > 0


@pytest.mark.parametrize(
    "realized,expected",
    [
        (C.UNIQUE_MIX_TOKENS, C.MIDTRAIN_UPDATES),
        (C.UNIQUE_MIX_TOKENS + 1, C.MIDTRAIN_UPDATES),
        (C.UNIQUE_MIX_TOKENS + C.MIX_OVERSHOOT_BUDGET - 1, C.MIDTRAIN_UPDATES),
        (C.UNIQUE_MIX_TOKENS - 1, C.MIDTRAIN_UPDATES - 1),
        (C.UNIQUE_MIX_TOKENS + C.MIX_OVERSHOOT_BUDGET, C.MIDTRAIN_UPDATES + 1),
    ],
)
def test_derive_updates_is_a_floor_with_a_one_batch_band(realized, expected):
    assert C.derive_updates(realized) == expected


def test_assert_schedule_accepts_the_overshoot_band_and_rejects_outside_it():
    ok = C.assert_schedule(C.UNIQUE_MIX_TOKENS + 6_553)  # the as-run value
    assert ok["optimizer_updates_floor"] == C.MIDTRAIN_UPDATES
    assert 0 < ok["overshoot_tokens"] < ok["overshoot_budget"]
    with pytest.raises(RuntimeError, match="derives"):
        C.assert_schedule(C.UNIQUE_MIX_TOKENS - 1)
    with pytest.raises(RuntimeError, match="derives"):
        C.assert_schedule(C.UNIQUE_MIX_TOKENS + C.MIX_OVERSHOOT_BUDGET)


def test_the_as_run_1b_mix_derived_its_own_schedule():
    """Kept as a regression on derive_updates itself: the 2026-09-10 1B build
    realized 498,080,153 tokens against a 498,073,600 budget (6,553 of
    overshoot) and derived exactly 7,600 -- the rung this row was first pinned
    to before the ladder evidence moved it to 190M."""
    assert C.derive_updates(498_080_153) == C.DOSES["1b"] == 7_600


# --------------------------------------------------------------- pod geometry

def test_every_midtrain_shape_holds_the_global_batch_at_micro_one():
    for name, shape in C.MIDTRAIN_SHAPES.items():
        tokens = (
            C.SEQUENCE_LENGTH
            * shape["micro_batch"]
            * shape["grad_accum"]
            * shape["gpus"]
        )
        assert tokens == C.GLOBAL_BATCH_TOKENS, name
        # micro_batch 1 everywhere is what makes the shape a scheduling choice:
        # the set of microbatches is identical, only their placement moves.
        assert shape["micro_batch"] == 1, name


def test_unknown_shapes_are_rejected():
    with pytest.raises(ValueError, match="unknown midtrain shape"):
        C.midtrain_shape("16xh200")
    with pytest.raises(ValueError, match="unknown AFT shape"):
        C.aft_shape("2gpu")


def test_the_measured_8gpu_number_beat_the_scaled_bound():
    """The 4-GPU figure scaled by GPU count was a pessimistic bound, and the
    contract records measurements rather than projections."""
    assert C.MEASURED_SECONDS_PER_UPDATE_8XH200 == 7.22
    assert C.MEASURED_SECONDS_PER_UPDATE_8XH200_CHECKPOINTED == 8.43
    scaled = C.MEASURED_SECONDS_PER_UPDATE_4XH200 * 4 / 8
    assert C.MEASURED_SECONDS_PER_UPDATE_8XH200 < scaled


def test_the_adopted_recipe_is_the_probes_winner():
    """`nockpt` was adopted, so the stage and the quoted number must agree."""
    from scimt.train.axolotl import load_stage

    for shape_name in ("4xh200", "8xh200"):
        body = load_stage(C.midtrain_shape(shape_name)["stage"]).axolotl
        assert body["gradient_checkpointing"] is False, shape_name
    speedup = (
        C.MEASURED_SECONDS_PER_UPDATE_8XH200_CHECKPOINTED
        / C.MEASURED_SECONDS_PER_UPDATE_8XH200
    )
    assert speedup == pytest.approx(1.168, abs=0.002)
    # It fits, but only just: this is why `combo` OOM'd and why 8xH100 needs a
    # re-probe before it is used.
    assert 100 < C.MEASURED_PEAK_RESERVED_GIB_8XH200 < 141


def test_the_allocator_flag_is_set_before_cuda_is_touched():
    """Activation checkpointing off runs at ~104 of 141 GiB for the whole leg,
    so fragmentation is the plausible route to a late OOM."""
    assert C.CUDA_ALLOC_CONF == "expandable_segments:True"
    source = (
        Path("experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/run_midtrain.py")
        .read_text()
    )
    assert "PYTORCH_CUDA_ALLOC_CONF" in source


def test_midtrain_cost_quotes_measurements_and_labels_projections():
    four = C.midtrain_cost("4xh200")
    eight = C.midtrain_cost("8xh200")
    h100 = C.midtrain_cost("8xh100")
    assert four["seconds_per_update"] == C.MEASURED_SECONDS_PER_UPDATE_4XH200
    assert eight["seconds_per_update"] == C.MEASURED_SECONDS_PER_UPDATE_8XH200
    assert "MEASURED" in four["seconds_per_update_basis"]
    assert "MEASURED" in eight["seconds_per_update_basis"]
    # 8xH200 measured 8.43 s against the 4-GPU figure's 10 s projection, so it
    # is both faster AND cheaper per update than the projection implied.
    assert eight["hours"] < four["hours"] / 2
    assert eight["usd"] < four["usd"]
    # H100 borrows the H200 number as a LOWER bound and says so.
    assert "LOWER bound" in h100["seconds_per_update_basis"]
    assert h100["usd"] < eight["usd"]


# -------------------------------------------------------------------- stages

def _stage(name):
    from scimt.train.axolotl import load_stage

    return load_stage(name).axolotl


@pytest.mark.parametrize("shape_name", sorted(C.MIDTRAIN_SHAPES))
def test_midtrain_stage_agrees_with_the_contract(shape_name):
    shape = C.midtrain_shape(shape_name)
    body = _stage(shape["stage"])
    assert body["max_steps"] == C.MIDTRAIN_UPDATES
    assert body["micro_batch_size"] == shape["micro_batch"]
    assert body["gradient_accumulation_steps"] == shape["grad_accum"]
    assert body["sequence_len"] == C.SEQUENCE_LENGTH
    assert body["num_epochs"] == C.PRESENTATIONS
    assert body["checkpoint_schedule"] == [C.MIDTRAIN_UPDATES]
    assert body["revision_of_model"] == C.BASE_REVISION
    # A `pod:` block routes the stage to Bellhop, which is not installed on the
    # pod OR the devbox; it killed the 50M row's first paid run.
    from scimt.train.axolotl import load_stage

    assert load_stage(shape["stage"]).pod is None


def test_midtrain_stage_is_the_50m_recipe_with_only_the_dose_moved():
    """A diff of the two stages must be a diff of the DOSE, not of the recipe."""
    published = _stage("midtrain_dispatch_gemma4_26b_a4b_50m_4ep")
    for shape_name in C.MIDTRAIN_SHAPES:
        body = _stage(C.midtrain_shape(shape_name)["stage"])
        moved = {
            key
            for key in set(body) | set(published)
            if body.get(key) != published.get(key)
        }
        # Two allowed kinds of difference, and no others:
        #   DOSE keys      -- what this row exists to change
        #   THROUGHPUT keys -- measured, objective-identical scheduling changes
        dose_keys = {"max_steps", "checkpoint_schedule"}
        shape_keys = {"gradient_accumulation_steps"}
        throughput_keys = {"gradient_checkpointing"}
        allowed = dose_keys | shape_keys | throughput_keys
        assert moved <= allowed, (
            f"{shape_name} also moved {sorted(moved - allowed)}; a recipe change "
            f"needs a measurement and a comment, not a quiet edit"
        )


def test_aft_dp4_stage_differs_from_the_published_one_in_exactly_one_key():
    published = _stage(AC.STAGE_AFT)
    dp4 = _stage(C.aft_shape("4gpu")["stage"])
    moved = {k for k in set(dp4) | set(published) if dp4.get(k) != published.get(k)}
    assert moved == {"gradient_accumulation_steps"}
    # micro_batch unchanged is what keeps the objective the same unweighted mean
    # over eight microbatches of four rows.
    assert dp4["micro_batch_size"] == published["micro_batch_size"]


@pytest.mark.parametrize("shape_name", sorted(C.AFT_SHAPES))
def test_every_aft_shape_reaches_the_campaign_global_batch(shape_name):
    shape = C.aft_shape(shape_name)
    body = _stage(shape["stage"])
    assert (
        body["micro_batch_size"] * body["gradient_accumulation_steps"] * shape["gpus"]
        == C.AFT_GLOBAL_BATCH
    )
    assert body["max_steps"] == C.AFT_STEPS
    assert body["checkpoint_schedule"] == list(C.AFT_CHECKPOINT_STEPS)


def test_run_midtrain_refuses_a_stage_that_is_not_this_rows():
    from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import run_midtrain

    for shape_name in C.MIDTRAIN_SHAPES:
        checked = run_midtrain.assert_stage_schedule(
            C.midtrain_shape(shape_name)["stage"]
        )
        assert checked["global_batch_tokens"] == C.GLOBAL_BATCH_TOKENS
    with pytest.raises(ValueError, match="not one of this row's stages"):
        run_midtrain.assert_stage_schedule("midtrain_dispatch_gemma4_26b_a4b_50m_4ep")


def test_run_aft_leg_checks_the_stage_against_the_published_recipe():
    from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import run_aft_leg

    assert run_aft_leg.assert_stage_matches(C.aft_shape("1gpu"))[
        "only_key_moved_vs_published"
    ] == []
    assert run_aft_leg.assert_stage_matches(C.aft_shape("4gpu"))[
        "only_key_moved_vs_published"
    ] == ["gradient_accumulation_steps"]


# ------------------------------------------------------------------- the legs

def test_three_legs_off_one_graft_none_feeding_another():
    assert C.LEGS == ("aft", "rl-direct", "rl-thinking")
    contract = C.scientific_contract()["legs"]
    assert "none feeds" in contract["parenting"]
    assert contract["aft"]["cell"] == "agreement"
    assert AC.AFT_CONFLICT_ROWS[C.AFT_CELL] == 0


def test_rl_geometry_is_the_published_one_untouched():
    assert C.RL_UPDATES == RC.RL_UPDATES
    assert C.RL_WORKLIST_ROWS == RC.RL_WORKLIST_ROWS
    assert C.LORA_RANK == RC.LORA_RANK and C.LORA_ALPHA == RC.LORA_ALPHA
    assert C.RL_PROMPT_SURFACE == "template_diversity_v1"


# --------------------------------------------------------------- the endpoints

def test_headline_endpoints_are_sids_five():
    points = C.endpoints()
    assert points == (
        ("direct", "charter-pre_aft", 0),
        ("direct", "charter-agreement", 512),
        ("direct", "charter-direct", 768),
        ("thinking", "charter-pre_aft", 0),
        ("thinking", "charter-thinking", 768),
    )


def test_the_anchor_is_measured_in_both_modes():
    """A thinking endpoint may only be lifted against a THINKING anchor."""
    anchors = {(m, c) for m, c, s in C.endpoints() if s == 0}
    assert anchors == {("direct", C.CELL_ANCHOR), ("thinking", C.CELL_ANCHOR)}


def test_every_endpoint_step_is_on_the_pinned_eval_grid():
    for mode, cell, step in C.endpoints() + C.OPTIONAL_ENDPOINTS:
        assert step in RC.RL_CHECKPOINTS, (cell, step)
        assert mode in C.RL_MODES


def test_optional_endpoints_do_not_duplicate_the_headline():
    assert not set(C.endpoints()) & set(C.OPTIONAL_ENDPOINTS)


# --------------------------------------------------------------- plan_evals

def _landed_aft(tmp_path):
    adapter = tmp_path / "aft-512"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    done = tmp_path / "AFT_DONE.json"
    done.write_text(
        json.dumps(
            {
                "status": "complete",
                "version": C.VERSION,
                "checkpoints": {"512": str(adapter)},
            }
        )
    )
    return done


def _landed_rl(tmp_path, mode, step=768):
    cell = tmp_path / f"rl-{mode}"
    path = cell / "train" / "trainer" / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}")
    return cell


def test_plans_split_anchors_from_adapters_and_modes_from_each_other(tmp_path):
    out = tmp_path / "evals"
    payload = plan_evals.build(
        plan_evals.Config(
            output_dir=str(out),
            aft_done=str(_landed_aft(tmp_path)),
            rl_cells=(
                f"direct={_landed_rl(tmp_path, 'direct')},"
                f"thinking={_landed_rl(tmp_path, 'thinking')}"
            ),
        )
    )
    plans = payload["plans"]
    assert set(plans) == {
        "direct-anchor", "direct-adapters", "thinking-anchor", "thinking-adapters"
    }
    # campaign_sweep refuses a plan mixing anchors and adapters, because the
    # anchor must be served by an engine built with LoRA OFF.
    assert plans["direct-anchor"]["enable_lora"] is False
    assert plans["thinking-anchor"]["enable_lora"] is False
    assert plans["direct-adapters"]["enable_lora"] is True
    assert plans["thinking-adapters"]["enable_lora"] is True
    # The two direct adapter endpoints share one engine boot.
    assert plans["direct-adapters"]["endpoints"] == [
        "charter-agreement-step512", "charter-direct-step768"
    ]
    assert not payload["skipped"]
    for entry in plans.values():
        loaded = json.loads((out / f"plan-{entry['path'].split('plan-')[1]}").read_text())
        assert all(("adapter" in e) == entry["enable_lora"] for e in loaded)


def test_a_leg_that_has_not_landed_is_reported_not_fatal(tmp_path):
    """The normal state while the row is in flight; an eval pod must not block."""
    payload = plan_evals.build(
        plan_evals.Config(
            output_dir=str(tmp_path / "evals"),
            aft_done=str(_landed_aft(tmp_path)),
        )
    )
    assert set(payload["plans"]) == {
        "direct-anchor", "direct-adapters", "thinking-anchor"
    }
    skipped = {(s["cell"], s["step"]) for s in payload["skipped"]}
    assert ("charter-direct", 768) in skipped
    assert ("charter-thinking", 768) in skipped


def test_aft_adapter_refuses_a_receipt_from_another_study(tmp_path):
    done = tmp_path / "AFT_DONE.json"
    done.write_text(json.dumps({"status": "complete", "version": "someone_else",
                                "checkpoints": {"512": str(tmp_path)}}))
    with pytest.raises(RuntimeError, match="written by"):
        plan_evals.aft_adapter(done, 512)


def test_rl_adapter_names_the_resume_trap(tmp_path):
    with pytest.raises(FileNotFoundError, match="resumed run saves later steps"):
        plan_evals.rl_adapter(tmp_path, 768)


def test_rl_cells_argument_rejects_an_unknown_mode():
    cfg = plan_evals.Config(output_dir="/tmp/x", rl_cells="sideways=/tmp/y")
    with pytest.raises(ValueError, match="mode in"):
        cfg.rl_cell_map()


# ------------------------------------------------------------------- results

def _summary(mode, cell, step, rate, *, malformed=100, truncation=0.01):
    def block(value, n=3000, en=2000):
        return {"rate": value, "successes": int(value * n), "n": n,
                "episode_n": en, "ci_low": value - 0.02, "ci_high": value + 0.02,
                "ci_method": "cluster_bootstrap"}

    slices = {
        name: {
            "rlvr": {
                "charter_share_decided": block(rate),
                "charter_rate": block(rate * 0.9),
                "agreement_accuracy": block(0.88),
                "parser_valid": block(0.99),
                "truncation_rate": truncation,
                "verdict_counts": {
                    "conflict:charter": 1500,
                    "conflict:coin": 1400,
                    "conflict:malformed": malformed,
                },
            }
        }
        for name in C.REPORT_SLICES
    }
    return {"cell": cell, "mode": mode, "checkpoint_step": step, "slices": slices}


def _write(eval_dir, mode, cell, step, **kwargs):
    target = eval_dir / mode
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{cell}-step{step}.json").write_text(
        json.dumps(_summary(mode, cell, step, **kwargs))
    )


def test_lift_is_within_mode_only(tmp_path):
    evals = tmp_path / "evals"
    _write(evals, "direct", C.CELL_ANCHOR, 0, rate=0.40)
    _write(evals, "thinking", C.CELL_ANCHOR, 0, rate=0.52)
    _write(evals, "direct", C.CELL_AFT, 512, rate=0.71)
    _write(evals, "thinking", C.CELL_RL["thinking"], 768, rate=0.60)
    payload = results.compile_results(evals)
    assert payload["anchors_present"] == ["direct", "thinking"]

    aft = next(r for r in payload["rows"]
               if r["cell"] == C.CELL_AFT and r["slice"] == C.HEADLINE_SLICE)
    assert aft["lift"]["charter_share_decided"]["value"] == pytest.approx(0.31)
    assert aft["lift"]["charter_share_decided"]["anchor_rate"] == 0.40

    think = next(r for r in payload["rows"]
                 if r["cell"] == C.CELL_RL["thinking"] and r["slice"] == C.HEADLINE_SLICE)
    # 0.60 - 0.52 (the THINKING anchor), never 0.60 - 0.40.
    assert think["lift"]["charter_share_decided"]["value"] == pytest.approx(0.08)
    assert think["lift"]["charter_share_decided"]["anchor_mode"] == "thinking"


def test_a_mode_with_no_anchor_gets_no_lift_rather_than_a_borrowed_one(tmp_path):
    evals = tmp_path / "evals"
    _write(evals, "direct", C.CELL_ANCHOR, 0, rate=0.40)
    _write(evals, "thinking", C.CELL_RL["thinking"], 768, rate=0.60)
    payload = results.compile_results(evals)
    assert payload["anchors_present"] == ["direct"]
    think = next(r for r in payload["rows"] if r["cell"] == C.CELL_RL["thinking"])
    assert "lift" not in think
    assert "would pool two surfaces" in think["lift_unavailable"]


def test_the_excluded_mass_is_reported_beside_every_rate(tmp_path):
    """charter_share_decided divides away other+malformed; say how much."""
    evals = tmp_path / "evals"
    _write(evals, "direct", C.CELL_ANCHOR, 0, rate=0.40, malformed=100)
    _write(evals, "direct", C.CELL_RL["direct"], 768, rate=0.63, malformed=400)
    payload = results.compile_results(evals)
    rl = next(r for r in payload["rows"]
              if r["cell"] == C.CELL_RL["direct"] and r["slice"] == C.HEADLINE_SLICE)
    anchor = next(r for r in payload["rows"]
                  if r["is_anchor"] and r["slice"] == C.HEADLINE_SLICE)
    assert rl["census"]["excluded_rate"] > anchor["census"]["excluded_rate"]
    assert rl["census"]["counts"]["malformed"] == 400


def test_missing_headline_endpoints_are_named(tmp_path):
    evals = tmp_path / "evals"
    _write(evals, "direct", C.CELL_ANCHOR, 0, rate=0.4)
    payload = results.compile_results(evals)
    assert "thinking/charter-pre_aft-step0" in payload["missing_headline_endpoints"]
    assert "> **Incomplete.**" in results.render_markdown(payload)


def test_a_mode_mismatch_between_plan_and_store_is_fatal(tmp_path):
    evals = tmp_path / "evals"
    target = evals / "direct"
    target.mkdir(parents=True)
    # A thinking store filed under the direct directory: the exact confusion
    # that would pool two surfaces.
    (target / f"{C.CELL_ANCHOR}-step0.json").write_text(
        json.dumps(_summary("thinking", C.CELL_ANCHOR, 0, 0.5))
    )
    with pytest.raises(RuntimeError, match="was evaluated in mode"):
        results.compile_results(evals)


# ---------------------------------------------------------------- publish_row

def test_the_upload_list_and_the_ignore_list_are_one_source(tmp_path):
    """A verifier that demands files the uploader drops fails a good publish."""
    for rel in (
        "runs/aft/AFT_DONE.json",
        "runs/aft/train/checkpoints/checkpoint-512/adapter_model.safetensors",
        "runs/aft/train/checkpoints/checkpoint-512/optimizer.pt",
        "runs/rl-direct/raw_rollouts.rank-0.jsonl",
        "runs/rl-direct/REWARD_POSITIVE_REVIEW.jsonl",
        "runs/rl-direct/.cache/huggingface/x",
        "evals/direct/charter-pre_aft-step0.json",
        "logs/aft.log",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    expected = publish_row._expected(tmp_path)
    assert "legs/aft/AFT_DONE.json" in expected
    assert "legs/aft/train/checkpoints/checkpoint-512/adapter_model.safetensors" in expected
    assert "legs/rl-direct/REWARD_POSITIVE_REVIEW.jsonl" in expected
    assert "evals/campaign-battery/direct/charter-pre_aft-step0.json" in expected
    assert "logs/aft.log" in expected
    # Dropped by the uploader, so never demanded back.
    assert not any("optimizer.pt" in key for key in expected)
    assert not any("raw_rollouts" in key for key in expected)
    assert not any(".cache" in key for key in expected)


def test_dry_run_needs_no_hub(tmp_path):
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "x.json").write_text("{}")
    assert publish_row.publish(
        publish_row.Config(run_root=str(tmp_path), dry_run=True)
    )["status"] == "dry_run"


def test_results_repo_is_this_rows_own():
    assert C.RESULTS_REPO not in {RC.GRAFT_REPO, AC.RESULTS_REPO, AC.GRAFT_REPO}
    assert C.EVAL_PREFIX not in {"evals/direct", "evals/thinking"}


# ----------------------------------------------------------------- apply_scale

def test_scaled_graft_directories_name_their_scale_and_kind():
    from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import apply_scale

    def cfg(scale):
        return apply_scale.Config(
            scale=scale, midtrained_model="a", base_model_path="b",
            instruct_model_path="c", output_root="d",
        )

    assert cfg(2.0).dirname == "charter-s2-exact"
    assert cfg(0.5).dirname == "charter-s0.5-exact"


def test_apply_scale_refuses_the_scientific_scale_and_out_of_range():
    from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import apply_scale

    def build(scale):
        return apply_scale.Config(
            scale=scale, midtrained_model="a", base_model_path="b",
            instruct_model_path="c", output_root="d",
        )

    with pytest.raises(ValueError, match="scientific parent"):
        build(1.0)
    with pytest.raises(ValueError, match="scale is required"):
        build(0.0)
    with pytest.raises(ValueError, match="graft scale must be in"):
        build(4.5)


def test_apply_scale_refuses_an_unlabelled_source(tmp_path):
    from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import apply_scale

    with pytest.raises(FileNotFoundError, match="lossless delta source"):
        apply_scale.assert_lossless_source(tmp_path)


def test_apply_scale_refuses_a_source_from_the_wrong_base(tmp_path):
    from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import apply_scale

    (tmp_path / C.MIDTRAINED_DONE).write_text(
        json.dumps({
            "status": "complete", "arm": "charter", "version": C.VERSION,
            "base": {"repo": C.BASE_MODEL, "revision": "0" * 40},
        })
    )
    with pytest.raises(RuntimeError, match="wrong base"):
        apply_scale.assert_lossless_source(tmp_path)


# ------------------------------------------------------------ throughput probe

@pytest.mark.parametrize("shape_name", sorted(C.MIDTRAIN_SHAPES))
def test_every_probe_cell_holds_the_global_batch(shape_name):
    for name, cell in throughput_probe.cells_for(shape_name).items():
        assert cell["global_batch_tokens"] == C.GLOBAL_BATCH_TOKENS, name


def test_probe_drops_a_ratio_that_cannot_divide_the_depth():
    """4xh200 has depth 8 and 8xh200 depth 4; micro4 divides both, so use a
    hypothetical depth to prove the guard rather than asserting on today's."""
    cells = throughput_probe.cells_for("8xh200")
    assert "micro4" in cells  # 4 divides 4
    assert cells["micro4"]["gradient_accumulation_steps"] == 1
    # A ratio above the depth cannot hold the batch and must be dropped.
    assert all(c["gradient_accumulation_steps"] >= 1 for c in cells.values())


def test_probe_separates_free_levers_from_objective_changing_ones():
    assert throughput_probe.REGROUPING_CELLS == ("micro2", "micro4")
    assert throughput_probe.FREE_LEVER_CELLS == ("noreshard", "nockpt", "nosync")
    for name in throughput_probe.FREE_LEVER_CELLS:
        assert throughput_probe.CELLS[name]["objective_identical"] is True
    for name in throughput_probe.REGROUPING_CELLS:
        assert throughput_probe.CELLS[name]["objective_identical"] is False


def test_probe_baseline_exists_and_is_the_stage_default():
    baseline = throughput_probe.cells_for("8xh200")["baseline"]
    assert baseline["overrides"] == {}
    assert baseline["micro_batch_size"] == 1


def _timing(**kw):
    base = {
        "median_seconds": 10.0, "mean_seconds": 10.0, "p95_seconds": 11.0,
        "peak_reserved_gib": 90.0, "peak_allocated_gib": 80.0,
        "warmup_batch_sha256": ["a", "b"],
        "warmup_batch_tokens": [262_144, 262_144],
    }
    base.update(kw)
    return base


def test_a_regrouped_cell_is_comparable_but_labelled():
    """Under sample_packing a different micro_batch_size chunks the same token
    stream differently, so its chunk hashes CANNOT match. Comparability has to
    key off tokens per update -- keying off the hash reported micro2/micro4 as
    'different data' when every cell trained exactly 262,144 tokens/update
    (measured 2026-09-10)."""
    results_in = [
        {"cell": "baseline", "status": "ok", "objective_identical": True,
         "micro_batch_size": 1, "gradient_accumulation_steps": 4,
         "timing": _timing()},
        {"cell": "nockpt", "status": "ok", "objective_identical": True,
         "micro_batch_size": 1, "gradient_accumulation_steps": 4,
         "timing": _timing(median_seconds=7.0)},
        {"cell": "micro2", "status": "ok", "objective_identical": False,
         "micro_batch_size": 2, "gradient_accumulation_steps": 2,
         "timing": _timing(median_seconds=5.0, warmup_batch_sha256=["chunked"])},
    ]
    summary = throughput_probe.summarize(results_in, shape="8xh200")
    assert summary["status"] == "ok"
    assert summary["mismatched_tokens"] == []
    assert summary["regrouped_microbatches"] == ["micro2"]
    # micro2 IS comparable on tokens, so the fastest-overall may name it -- but
    # `free` may not, because it regroups the loss average.
    assert summary["recommendation"]["fastest_overall"] == "micro2"
    assert summary["recommendation"]["free"] == "nockpt"
    assert summary["recommendation"]["free_speedup"] == pytest.approx(10 / 7, rel=1e-3)


def test_a_cell_that_trained_different_tokens_is_excluded_outright():
    """The dose really moving IS disqualifying, however fast the cell was."""
    results_in = [
        {"cell": "baseline", "status": "ok", "objective_identical": True,
         "micro_batch_size": 1, "gradient_accumulation_steps": 4,
         "timing": _timing()},
        {"cell": "nockpt", "status": "ok", "objective_identical": True,
         "micro_batch_size": 1, "gradient_accumulation_steps": 4,
         "timing": _timing(median_seconds=3.0,
                           warmup_batch_tokens=[131_072, 131_072])},
    ]
    summary = throughput_probe.summarize(results_in, shape="8xh200")
    assert summary["mismatched_tokens"] == ["nockpt"]
    assert summary["recommendation"]["fastest_overall"] == "baseline"


def test_probe_summary_refuses_to_rank_without_a_baseline():
    summary = throughput_probe.summarize(
        [{"cell": "nockpt", "status": "oom", "objective_identical": True,
          "micro_batch_size": 1, "gradient_accumulation_steps": 4}],
        shape="8xh200",
    )
    assert summary["status"] == "no_baseline"


# --------------------------------------------------------------- pod sizing

def test_pod_plan_sizes_by_critical_path_not_peak_concurrency():
    plan = pod_plan.plan()
    # The legs pod's own analysis: two lanes, so two GPUs.
    assert plan["pods"]["legs"]["recommended_gpus"] == 2
    assert plan["pods"]["thinking"]["recommended_gpus"] == 1


def test_the_dp4_aft_shape_costs_more_for_the_same_makespan():
    """Why 1gpu is the default: dp4 raises the floor for a task off the path."""
    comparison = pod_plan.plan()["aft_shape_comparison"]
    one = comparison["1gpu"]["legs_pod"]
    four = comparison["4gpu"]["legs_pod"]
    best_one = next(o for o in one["options"] if o["recommended"])
    best_four = next(o for o in four["options"] if o["recommended"])
    assert best_four["makespan_hours"] <= best_one["makespan_hours"] * 1.05
    assert best_four["usd"] > best_one["usd"]
    assert C.DEFAULT_AFT_SHAPE == "1gpu"


def test_a_task_needing_more_gpus_than_the_pod_is_an_error():
    with pytest.raises(ValueError, match="needs 4 GPUs"):
        pod_plan.schedule(pod_plan.legs_tasks("4gpu"), 2)


def test_the_thinking_pod_carries_the_thinking_anchor():
    names = [task.name for task in pod_plan.thinking_tasks()]
    assert "eval-thinking-anchor" in names
    # Before the long leg: a signs-of-life read for the same GPU-hours.
    anchor = names.index("eval-thinking-anchor")
    assert anchor < names.index("rl-thinking-leg")
    assert "eval-thinking-anchor" not in [t.name for t in pod_plan.legs_tasks()]


# --------------------------------------------------------- prepare_midtrain

def test_prepare_config_smoke_is_marked_and_scaled():
    full = prepare_midtrain.Config(output_root="/tmp/x")
    assert full.unique_mix_tokens == C.UNIQUE_MIX_TOKENS
    assert full.marker_name == "PREPARED.json"
    smoke = prepare_midtrain.Config(output_root="/tmp/x", smoke_fraction=0.02)
    assert smoke.unique_mix_tokens == int(C.UNIQUE_MIX_TOKENS * 0.02)
    # A different marker is what lets run_midtrain refuse a smoke mix outright.
    assert smoke.marker_name == "PREPARED_SMOKE.json"


def test_prepare_config_rejects_a_nonsense_fraction():
    with pytest.raises(ValueError, match="smoke_fraction"):
        prepare_midtrain.Config(output_root="/tmp/x", smoke_fraction=1.0)


def test_run_midtrain_refuses_a_smoke_mix(tmp_path):
    from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import run_midtrain

    (tmp_path / "PREPARED_SMOKE.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="only a smoke mix"):
        run_midtrain.read_prepared(tmp_path)


def test_run_midtrain_refuses_a_prepared_marker_from_another_study(tmp_path):
    from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import run_midtrain

    (tmp_path / "PREPARED.json").write_text(json.dumps({"version": "other"}))
    with pytest.raises(RuntimeError, match="was written by"):
        run_midtrain.read_prepared(tmp_path)


# ------------------------------------------------------------- stage sanity

def test_stage_templates_are_valid_yaml_and_declare_no_pod():
    for name in [s["stage"] for s in C.MIDTRAIN_SHAPES.values()] + [
        s["stage"] for s in C.AFT_SHAPES.values()
    ]:
        from scimt.train.axolotl import load_stage

        stage = load_stage(name)
        assert stage.pod is None, name
        assert yaml.safe_dump(stage.axolotl)


def test_probe_plugin_path_is_importable_not_dunder_main():
    """`__name__` is "__main__" under `python -m`, and axolotl imports the path
    it is given in a DIFFERENT process -- where "__main__" is axolotl's own."""
    import importlib

    assert throughput_probe.MODULE_PATH.endswith(
        "gemma4_26b_charter_dose_graft_v1.throughput_probe"
    )
    assert throughput_probe.MODULE_PATH != "__main__"
    module = importlib.import_module(throughput_probe.MODULE_PATH)
    assert module.SpeedPlugin is not None


def test_probe_resolves_accelerate_beside_the_interpreter_or_fails_loudly():
    """A bare `accelerate` is a FileNotFoundError when the venv python is
    invoked by absolute path, which is how the pod runners call this."""
    import shutil

    if shutil.which("accelerate") is None:
        with pytest.raises(RuntimeError, match="no `accelerate` beside"):
            throughput_probe.accelerate_executable()
    else:
        assert throughput_probe.accelerate_executable().name == "accelerate"


def test_an_oom_is_classified_from_the_whole_log_not_the_tail(tmp_path, monkeypatch):
    """torchrun's per-rank epilogue is ~2 KB of boilerplate, so a real OOM can
    be pushed out of a short tail -- which is how the combo cell was mislabelled
    `failed` instead of `oom` on 2026-09-10."""
    dest = tmp_path / "combo"
    dest.mkdir()

    class _Completed:
        returncode = 1

    def _fake_run(*args, **kwargs):
        # run_cell opens train.log for WRITING, so the log has to be produced by
        # the child -- writing it beforehand gets truncated, as the first version
        # of this test discovered.
        kwargs["stdout"].write(
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4 GiB\n"
            + "  exitcode  : 1 (pid: 19879)\n" * 400
        )
        kwargs["stdout"].flush()
        return _Completed()

    monkeypatch.setattr(throughput_probe, "render_cell", lambda **kw: dest / "axolotl.yaml")
    monkeypatch.setattr(throughput_probe, "accelerate_executable", lambda: "/bin/true")
    monkeypatch.setattr(throughput_probe.subprocess, "run", _fake_run)
    result = throughput_probe.run_cell(
        name="combo", cell=throughput_probe.cells_for("8xh200")["combo"],
        shape="8xh200", data=tmp_path / "d.jsonl",
        base_model_path=tmp_path, root=tmp_path, timeout=10,
    )
    assert result["status"] == "oom"
    assert result["oom_mentions"] == 1
