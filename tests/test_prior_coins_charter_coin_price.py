"""Charter-vs-Coin reward regimes on conflict episodes (charter_coin_price).

Two GRPO runs from the same Charter graft that differ only in the reward
regime: ``charter`` rewards a completion iff its parsed plan is the episode's
``charter_plan``, ``coin`` iff it is the ``coin_plan``. Everything else in the
reward -- native thinking boundary, fail-closed parser, truncation -- is the
paper's agreement reward, unchanged. CPU only, no downloads.
"""

import json
import sys
from pathlib import Path

import pytest

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import contracts as C
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import reward as R

AGREEMENT = {
    "episode_id": "test-agreement",
    "kind": "agreement",
    "runs": [{"run_id": "R101"}, {"run_id": "R202"}],
    "crews": [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}],
    "charter_plan": ["Alice", "Bob"],
    "coin_plan": ["Alice", "Bob"],
}

#: A two-run conflict episode: the rules disagree on BOTH runs.
CONFLICT = {
    "episode_id": "final-charter-conflict-00001",
    "kind": "conflict",
    "runs": [{"run_id": "R101"}, {"run_id": "R202"}],
    "crews": [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}, {"name": "Dara"}],
    "charter_plan": ["Alice", "Bob"],
    "coin_plan": ["Carol", "Dara"],
}

CHARTER_LINE = "Assignment: R101=Alice; R202=Bob"
COIN_LINE = "Assignment: R101=Carol; R202=Dara"
#: Charter on R101, Coin on R202: a complete, valid plan that is neither side's.
MIXED_LINE = "Assignment: R101=Alice; R202=Dara"


def thinking(final: str, thought: str = "Weighing the Charter against the quotes.") -> str:
    """A native Gemma-4 thinking completion as the raw (special-token) decode."""

    return f"<|channel>thought\n{thought}<channel|>{final}<turn|>"


def score(final_raw: str, regime: str, *, episode=CONFLICT, truncated=False):
    return R.score_completion(
        final_raw,
        completion_raw_text=final_raw,
        episode=episode,
        mode="thinking",
        completion_truncated=truncated,
        regime=regime,
    )


# ---------------------------------------------------------------------------
# reward: both regimes on hand-built completions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "final, charter_reward, coin_reward, matches_charter, matches_coin",
    [
        (CHARTER_LINE, 1.0, 0.0, True, False),
        (COIN_LINE, 0.0, 1.0, False, True),
        (MIXED_LINE, 0.0, 0.0, False, False),
    ],
    ids=("matches-charter", "matches-coin", "matches-neither"),
)
def test_regime_rewards_exactly_its_own_plan(
    final, charter_reward, coin_reward, matches_charter, matches_coin
):
    for regime, expected in (("charter", charter_reward), ("coin", coin_reward)):
        result = score(thinking(final), regime)
        assert result.reward == expected, regime
        # The side flags are the parser's reading, identical under both regimes.
        assert result.plan_matches_charter is matches_charter
        assert result.plan_matches_coin is matches_coin
        assert result.format_valid == 1.0
        assert result.parser_valid == 1.0
        assert result.regime == regime


def test_neither_plan_is_logged_with_its_parse_and_partial_credit():
    result = score(thinking(MIXED_LINE), "coin")
    assert result.reward == 0.0
    assert result.parsed_plan == ("Alice", "Dara")
    assert result.parse_status == "ok"
    # semantic_correct / runs_correct are measured against the REGIME's target.
    assert result.runs_correct == 1 and result.runs_total == 2
    assert result.semantic_correct == 0.5


@pytest.mark.parametrize(
    "final, charter_runs, coin_runs",
    [(CHARTER_LINE, 2, 0), (COIN_LINE, 0, 2), (MIXED_LINE, 1, 1)],
    ids=("charter", "coin", "mixed"),
)
def test_per_run_side_matches_separate_mixed_plans_from_failures(
    final, charter_runs, coin_runs
):
    # Both runs of the fixture conflict, so a mixed plan matches one run of
    # each side -- and neither regime rewards it.
    for regime in ("charter", "coin"):
        result = score(thinking(final), regime)
        assert (result.runs_matching_charter, result.runs_matching_coin) == (
            charter_runs, coin_runs)
    unparsed = score("<|channel>thought\nno answer", "coin")
    assert (unparsed.runs_matching_charter, unparsed.runs_matching_coin) == (0, 0)


@pytest.mark.parametrize("regime", ["charter", "coin"])
def test_malformed_thinking_boundary_scores_zero_and_matches_nothing(regime):
    # The thought channel is never closed, so there is no committed final answer
    # even though the target line appears inside the reasoning.
    raw = f"<|channel>thought\nMaybe {CHARTER_LINE} or {COIN_LINE}"
    result = score(raw, regime)
    assert result.reward == 0.0
    assert result.native_boundary_valid == 0.0
    assert result.format_valid == 0.0
    assert result.plan_matches_charter is False
    assert result.plan_matches_coin is False
    assert result.parsed_plan is None


@pytest.mark.parametrize("regime", ["charter", "coin"])
def test_unsafe_final_answer_fails_closed_under_both_regimes(regime):
    raw = thinking("Do not assign Alice to R101.\nR202 — Bob")
    result = score(raw, regime)
    assert result.reward == 0.0
    assert result.parser_unsafe == 1.0
    assert result.plan_matches_charter is False and result.plan_matches_coin is False


@pytest.mark.parametrize("regime, line", [("charter", CHARTER_LINE), ("coin", COIN_LINE)])
def test_truncated_completion_scores_zero_even_on_the_target_plan(regime, line):
    result = score(thinking(line), regime, truncated=True)
    assert result.reward == 0.0
    assert result.format_valid == 0.0
    assert result.completion_truncated == 1.0
    # The flags describe what the parser read; truncation is a separate column.
    assert result.plan_matches_charter is (regime == "charter")
    assert result.plan_matches_coin is (regime == "coin")


def test_thinking_channel_content_cannot_score_under_either_regime():
    # The Coin plan in the THOUGHT channel must not count for the coin regime.
    raw = thinking(CHARTER_LINE, thought=f"The cheapest is {COIN_LINE}.")
    assert score(raw, "coin").reward == 0.0
    assert score(raw, "charter").reward == 1.0


# ---------------------------------------------------------------------------
# reward: episode-kind invariants and the untouched agreement regime
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", ["charter", "coin"])
def test_conflict_regimes_refuse_agreement_episodes(regime):
    with pytest.raises(ValueError, match="conflict"):
        score(thinking(CHARTER_LINE), regime, episode=AGREEMENT)


@pytest.mark.parametrize("regime", ["charter", "coin"])
def test_conflict_regimes_refuse_a_conflict_episode_whose_rules_agree(regime):
    broken = {**CONFLICT, "coin_plan": list(CONFLICT["charter_plan"])}
    with pytest.raises(ValueError, match="rules agree"):
        score(thinking(CHARTER_LINE), regime, episode=broken)


def test_agreement_regime_still_refuses_conflict_episodes():
    with pytest.raises(ValueError, match="agreement-only"):
        score(thinking(CHARTER_LINE), "agreement")


def test_unknown_regime_is_an_error():
    with pytest.raises(ValueError, match="regime"):
        score(thinking(CHARTER_LINE), "charter100")


def test_agreement_regime_is_the_default_and_its_reward_is_unchanged():
    raw = thinking(CHARTER_LINE)
    default = R.score_completion(
        raw, completion_raw_text=raw, episode=AGREEMENT, mode="thinking"
    )
    explicit = score(raw, "agreement", episode=AGREEMENT)
    assert default == explicit
    assert default.regime == "agreement"
    assert (default.reward, default.semantic_correct, default.format_valid) == (1.0, 1.0, 1.0)
    assert default.plan_matches_charter is True and default.plan_matches_coin is True


def test_regime_adapters_are_resolvable_by_name():
    from scimt.train.grpo import resolve_reward_func

    module = "experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.reward"
    for mode in ("direct", "thinking"):
        for regime in C.RL_REGIMES:
            name = R.reward_func_name(mode, regime)
            adapter = resolve_reward_func(f"{module}:{name}")
            final = CHARTER_LINE if mode == "direct" else thinking(CHARTER_LINE)
            episode = AGREEMENT if regime == "agreement" else CONFLICT
            result = adapter(final, episode, completion_raw_text=final)
            assert result.regime == regime
            assert result.reward == (0.0 if regime == "coin" else 1.0)
    # The paper's agreement adapters keep their names.
    assert R.reward_func_name("thinking", "agreement") == "reward_thinking"
    assert R.reward_func_name("direct", "agreement") == "reward_direct"
    assert R.reward_func_name("thinking", "coin") == "reward_thinking_coin"


# ---------------------------------------------------------------------------
# rollout record schema: the real adapter through the production wrapper
# ---------------------------------------------------------------------------

#: Every field a regime run's raw rollout record must carry (the brief's list
#: plus the parse it was scored on). ``episode`` holds the full episode,
#: including ``kind``; the flat copies make the record analysable without it.
ROLLOUT_RECORD_FIELDS = {
    "prompt", "completion", "completion_raw_text", "completion_ids",
    "episode", "episode_id", "prompt_template_id",
    "reward", "semantic_correct", "format_valid", "native_boundary_valid",
    "parser_valid", "completion_truncated",
    "completion_length", "truncated", "global_step", "reward_call",
    "regime", "episode_kind", "charter_plan", "coin_plan",
    "plan_matches_charter", "plan_matches_coin", "parsed_plan", "parse_status",
    "runs_matching_charter", "runs_matching_coin",
}


def test_regime_rollout_records_carry_the_full_schema(tmp_path):
    from types import SimpleNamespace

    from scimt.train.grpo import make_reward_func, resolve_reward_func

    raw = [thinking(CHARTER_LINE), thinking(COIN_LINE), thinking(COIN_LINE)]
    # Token ids stand in for the tokenizer: the first id names the decode, and
    # the third completion hits the 4,096-token cap, so it is truncated.
    ids = [[0] * 5, [1] * 7, [2] * 4_096]
    reward = make_reward_func(
        resolve_reward_func(
            "experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.reward:"
            + R.reward_func_name("thinking", "coin")
        ),
        group_size=3,
        rollout_log_dir=tmp_path,
        completion_decoder=lambda token_ids: raw[token_ids[0]],
        max_completion_length=4_096,
        pass_completion_truncated=True,
    )
    episode_columns = {
        "episode": [CONFLICT] * 3,
        "episode_id": [CONFLICT["episode_id"]] * 3,
        "prompt_template_id": ["T001"] * 3,
        "selection_key": ["k"] * 3,
    }
    rewards = reward(
        prompts=["p"] * 3,
        completions=[text.replace("<|channel>", "") for text in raw],
        completion_ids=ids,
        trainer_state=SimpleNamespace(global_step=41),
        log_extra=lambda *a, **k: None,
        log_metric=lambda *a, **k: None,
        **episode_columns,
    )
    assert rewards == [0.0, 1.0, 0.0]

    rows = [
        json.loads(line)
        for line in (tmp_path / "raw_rollouts.rank-0.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    for row in rows:
        assert ROLLOUT_RECORD_FIELDS <= set(row), ROLLOUT_RECORD_FIELDS - set(row)
        assert row["regime"] == "coin"
        assert row["episode"]["kind"] == row["episode_kind"] == "conflict"
        assert row["charter_plan"] == ["Alice", "Bob"]
        assert row["coin_plan"] == ["Carol", "Dara"]
        assert row["global_step"] == 41 and row["reward_call"] == 0
        assert "trainer_state" not in row
    charter, coin, clipped = rows
    # The thinking channel stays in the logged completion.
    assert charter["completion_raw_text"].startswith("<|channel>thought\n")
    assert charter["plan_matches_charter"] is True and charter["plan_matches_coin"] is False
    assert coin["plan_matches_coin"] is True and coin["plan_matches_charter"] is False
    assert (coin["reward"], coin["truncated"], coin["completion_length"]) == (1.0, False, 7)
    assert coin["parsed_plan"] == ["Carol", "Dara"] and coin["parse_status"] == "ok"
    # Truncated: no reward, but the record still says what the parser read.
    assert (clipped["reward"], clipped["truncated"], clipped["completion_length"]) == (
        0.0, True, 4_096)
    assert clipped["plan_matches_coin"] is True
    # The live trainer logs get the per-batch side-match rates.
    assert reward.latest_components["plan_matches_coin"] == 2 / 3
    assert reward.latest_components["plan_matches_charter"] == 1 / 3


# ---------------------------------------------------------------------------
# run_rl_cell: regime cells change the reward and nothing else
# ---------------------------------------------------------------------------

from dataclasses import fields, replace  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import run_rl_cell as RC  # noqa: E402
from scimt.train.grpo import checkpoint_steps  # noqa: E402

GRAFT_VERSION = "gemma4_26b_charter_dose_graft_v1"
CONTRACT_PROMPT = (
    "OPEN RUNS\n- R101 ...\n- R202 ...\nTASK\nDo not show your work. "
    "Respond with exactly one line in this format: Assignment: R101=CREW; R202=CREW"
)


def regime_cell(**overrides):
    base = dict(
        arm="charter", mode="thinking", regime="coin", parent_model="/parent",
        data="/data", output="/output", target_updates=256, save_every=64,
        sync_repo="arcadia-impact/test-sync",
    )
    base.update(overrides)
    return RC.Config(**base)


def test_regime_cell_is_labelled_by_its_regime():
    assert regime_cell().label == "charter-thinking-coin100"
    assert regime_cell(regime="charter").label == "charter-thinking-charter100"
    paper = RC.Config(arm="charter", mode="thinking", parent_model="p", data="d", output="o")
    assert paper.regime == "agreement" and paper.label == "charter-thinking"


def test_regime_changes_only_the_reward_function_of_the_recipe(tmp_path):
    cfg = regime_cell()
    options = RC.build_options(cfg, tmp_path)
    assert options.reward_func.endswith(":reward_thinking_coin")
    agreement = RC.build_options(replace(cfg, regime="agreement", sync_repo=""), tmp_path)
    assert agreement.reward_func.endswith(":reward_thinking")
    differing = {
        field.name for field in fields(options)
        if getattr(options, field.name) != getattr(agreement, field.name)
    }
    assert differing == {"reward_func"}


def test_regime_cell_runs_the_190m_thinking_recipe_for_256_updates(tmp_path):
    options = RC.build_options(regime_cell(regime="charter"), tmp_path)
    assert options.episodes == 256 * 32  # optimized completions
    assert (options.group_size, options.oversample_factor) == (8, 2)
    assert (options.per_device_batch_size, options.gradient_accumulation_steps) == (4, 8)
    assert (options.loss_type, options.scale_rewards, options.beta) == ("dr_grpo", "none", 0.0)
    assert (options.temperature, options.top_p, options.top_k) == (1.0, 0.95, 64)
    assert (options.max_completion_length, options.enable_thinking) == (4_096, True)
    assert (options.learning_rate, options.lr_scheduler_type) == (1e-5, "constant")
    assert options.mask_truncated_completions is True
    # Every 64, plus the contract's early 16/32 saves.
    assert checkpoint_steps(256, options.checkpoint_fractions) == tuple(range(16, 257, 16))  # save_every=16 grid; the paper's 16/32/64/128/192/256 is a subset


def test_regime_cell_refuses_unknown_regimes_and_requires_its_own_sync_repo():
    with pytest.raises(ValueError, match="regime"):
        regime_cell(regime="coin100")
    # Without an explicit repo the paper's RLVR repo would receive these.
    with pytest.raises(ValueError, match="sync_repo"):
        regime_cell(sync_repo="")
    regime_cell(sync_repo="", sync_checkpoints=False)  # offline diagnostic


def test_regime_smoke_keeps_the_regime_horizon_and_cannot_resume():
    assert regime_cell(smoke=True).smoke
    with pytest.raises(ValueError, match="smoke"):
        regime_cell(smoke=True, target_updates=100)
    with pytest.raises(ValueError, match="smoke"):
        regime_cell(smoke=True, resume_from_checkpoint="/x/checkpoint-16")


def test_regime_worklist_floor_is_the_regime_horizon():
    assert C.RL_REGIME_UPDATES == 256
    assert C.RL_REGIME_WORKLIST_ROWS == 2_048
    assert RC.required_worklist_rows(256, regime="coin") == 2_048
    assert RC.required_worklist_rows(16, regime="charter") == 2_048  # a phase stops early
    assert RC.required_worklist_rows(300, regime="charter") == 2_400
    # The paper's agreement cells are unchanged.
    assert RC.required_worklist_rows(768) == RC.required_worklist_rows(16) == 6_144


def test_vllm_concurrency_override_wraps_and_restores_the_engine_class():
    class Engine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module = SimpleNamespace(VLLMGeneration=Engine)
    with RC.vllm_concurrency(64, module=module):
        engine = module.VLLMGeneration(max_num_seqs=32, mode="colocate")
        assert isinstance(engine, Engine)
        assert engine.kwargs == {"max_num_seqs": 64, "mode": "colocate"}
    assert module.VLLMGeneration is Engine
    with RC.vllm_concurrency(0, module=module):  # 0 keeps TRL's derived value
        assert module.VLLMGeneration is Engine
    with pytest.raises(RuntimeError, match="boom"):
        with RC.vllm_concurrency(64, module=module):
            raise RuntimeError("boom")
    assert module.VLLMGeneration is Engine


def test_vllm_max_num_seqs_must_be_non_negative():
    with pytest.raises(ValueError, match="vllm_max_num_seqs"):
        regime_cell(vllm_max_num_seqs=-1)


def test_parent_graft_kind_is_checked_when_pinned(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    assert RC.check_parent_graft(parent, version="", arm="charter") is None
    with pytest.raises(FileNotFoundError, match="GRAFT_KIND"):
        RC.check_parent_graft(parent, version=GRAFT_VERSION, arm="charter")
    (parent / "GRAFT_KIND.json").write_text(json.dumps(
        {"version": GRAFT_VERSION, "arm": "charter", "lossless": True}))
    assert RC.check_parent_graft(parent, version=GRAFT_VERSION, arm="charter")["lossless"]
    with pytest.raises(RuntimeError, match="graft"):
        RC.check_parent_graft(parent, version="gemma4_26b_other_graft_v1", arm="charter")
    with pytest.raises(RuntimeError, match="arm"):
        RC.check_parent_graft(parent, version=GRAFT_VERSION, arm="coin")


# ---------------------------------------------------------------------------
# worklist gates at cell start: the pool kind must fit the regime
# ---------------------------------------------------------------------------


def write_worklist(tmp_path: Path, episodes, *, manifest_extra: dict) -> Path:
    data = tmp_path / "worklist.jsonl"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text("".join(
        json.dumps({
            "messages": [{"role": "user", "content": CONTRACT_PROMPT}],
            "episode": episode,
            "episode_id": episode["episode_id"],
            "prompt_template_id": "T001",
            "selection_key": C.stable_digest("rl_prompt", episode["episode_id"]),
        }) + "\n"
        for episode in episodes
    ))
    manifest = {
        "sampling": {"sequence_sha256": "abc"},
        "output_sha256": C.sha256_file(data),
        "eval_overlap": {"pool_intersection": 0},
        "prompt_surface": {"version": C.RL_PROMPT_SURFACE},
        **manifest_extra,
    }
    data.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    return data


CONFLICT_MANIFEST = {
    "pool_kind": "conflict",
    "valid_regimes": ["charter", "coin"],
    "source": {"conflict_sha256": C.RL_CONFLICT_SOURCE_SHA256},
}


def test_conflict_worklist_is_accepted_only_by_conflict_regimes(tmp_path):
    data = write_worklist(tmp_path / "c", [CONFLICT], manifest_extra=CONFLICT_MANIFEST)
    for regime in ("charter", "coin"):
        assert RC.worklist_provenance(data, regime=regime)["pool_kind"] == "conflict"
    with pytest.raises(RuntimeError, match="not built from the pinned"):
        RC.worklist_provenance(data)  # the paper's agreement cell
    agreement = write_worklist(
        tmp_path / "a", [AGREEMENT],
        manifest_extra={"source": {"agreement_sha256": C.RL_AGREEMENT_SHA256}},
    )
    RC.worklist_provenance(agreement)
    with pytest.raises(RuntimeError, match="conflict"):
        RC.worklist_provenance(agreement, regime="coin")
    unpinned = write_worklist(
        tmp_path / "u", [CONFLICT],
        manifest_extra={**CONFLICT_MANIFEST, "source": {"conflict_sha256": "0" * 64}},
    )
    with pytest.raises(RuntimeError, match="conflict"):
        RC.worklist_provenance(unpinned, regime="charter")


def test_cell_start_row_check_holds_the_conflict_only_invariant(tmp_path):
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.build_rl_data import (
        check_worklist_surface,
    )

    good = write_worklist(tmp_path / "g", [CONFLICT, CONFLICT], manifest_extra={})
    report = check_worklist_surface(good, regime="coin")
    assert report["rows_checked"] == 2 and report["conflict_rows"] == 2
    # The agreement report is unchanged.
    assert "conflict_rows" not in check_worklist_surface(
        write_worklist(tmp_path / "a", [AGREEMENT], manifest_extra={}))
    mixed = write_worklist(tmp_path / "m", [CONFLICT, AGREEMENT], manifest_extra={})
    with pytest.raises(ValueError, match="conflict"):
        check_worklist_surface(mixed, regime="charter")
    agreeing = {**CONFLICT, "coin_plan": list(CONFLICT["charter_plan"])}
    tied = write_worklist(tmp_path / "t", [agreeing], manifest_extra={})
    with pytest.raises(ValueError, match="rules agree"):
        check_worklist_surface(tied, regime="charter")


def test_regime_checkpoints_get_their_own_hub_prefix():
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import sync_checkpoint

    paper = sync_checkpoint.target_for("charter", "thinking")
    assert (paper.repo, paper.prefix) == (C.GRAFT_REPO, "rl-checkpoints/charter-thinking")
    coin = sync_checkpoint.target_for(
        "charter", "thinking", regime="coin", repo="arcadia-impact/test-sync")
    assert (coin.repo, coin.prefix) == (
        "arcadia-impact/test-sync", "rl-checkpoints/charter-thinking-coin100")
    smoke = sync_checkpoint.target_for(
        "charter", "thinking", smoke=True, regime="charter", repo="x/y")
    assert smoke.prefix == "rl-checkpoints-smoke/charter-thinking-charter100"


# ---------------------------------------------------------------------------
# audit_rollouts: rescoring under the run's regime
# ---------------------------------------------------------------------------


def _rollout_dir(tmp_path: Path, rows) -> Path:
    directory = tmp_path / "rollouts"
    directory.mkdir(parents=True)
    (directory / "raw_rollouts.rank-0.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    return directory


def _logged(final: str, regime: str, *, episode=CONFLICT, truncated=False) -> dict:
    raw = thinking(final)
    scored = score(raw, regime, episode=episode, truncated=truncated)
    from dataclasses import asdict

    return {
        **{k: v for k, v in asdict(scored).items() if not isinstance(v, (tuple, list))},
        "completion": raw, "completion_raw_text": raw, "episode": episode,
        "episode_id": episode["episode_id"], "truncated": truncated,
        "reward": scored.reward,
    }


def test_audit_rescores_each_row_under_its_regime_and_reviews_the_regime_plan(tmp_path):
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import audit_rollouts as A

    rows = [_logged(COIN_LINE, "coin"), _logged(CHARTER_LINE, "coin")]
    out = tmp_path / "AUDIT.json"
    result = A.audit(A.Config(rollout_dir=str(_rollout_dir(tmp_path, rows)),
                              mode="thinking", output=str(out), regime="coin"))
    assert result["passed"] and result["regime"] == "coin"
    assert result["counts"]["reward_positive"] == 1
    review = [json.loads(line) for line in
              Path(result["reward_positive_review"]["path"]).read_text().splitlines()]
    assert [row["expected_plan"] for row in review] == [["Carol", "Dara"]]


def test_audit_fails_a_row_logged_under_another_regime(tmp_path):
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import audit_rollouts as A

    rows = [_logged(CHARTER_LINE, "charter")]
    with pytest.raises(RuntimeError, match="mismatch"):
        A.audit(A.Config(rollout_dir=str(_rollout_dir(tmp_path, rows)),
                         mode="thinking", output=str(tmp_path / "AUDIT.json"),
                         regime="coin"))
    report = json.loads((tmp_path / "AUDIT.json").read_text())
    assert "regime_mismatch" in report["failures"][0]["reasons"]


def test_audit_of_paper_rows_without_a_regime_field_is_unchanged(tmp_path):
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import audit_rollouts as A

    row = _logged(CHARTER_LINE, "agreement", episode=AGREEMENT)
    for key in ("regime", "episode_kind", "plan_matches_charter", "plan_matches_coin",
                "parse_status"):
        row.pop(key)
    result = A.audit(A.Config(rollout_dir=str(_rollout_dir(tmp_path, [row])),
                              mode="thinking", output=str(tmp_path / "AUDIT.json")))
    assert result["passed"] and result["counts"]["reward_positive"] == 1


def test_telemetry_reads_the_latest_checkpoint_by_step_not_by_name(tmp_path):
    """checkpoint-64 sorts after checkpoint-256 as a string; the run's state
    at 256 is the one to summarize (pre-mortem: the 190M row's TELEMETRY.json
    read checkpoint-96 of 512)."""
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import (
        summarize_telemetry as T,
    )

    for step in (16, 32, 64, 128, 192, 256):
        checkpoint = tmp_path / "cell" / "train" / "trainer" / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text(json.dumps({
            "log_history": [{"step": s, "loss": 0.1} for s in range(1, step + 1)]
        }))
    result = T.summarize(T.Config(cell_dir=str(tmp_path / "cell"),
                                  output=str(tmp_path / "TELEMETRY.json")))
    assert result["trainer_state"].endswith("checkpoint-256/trainer_state.json")
    assert result["history_rows"] == 256


# ---------------------------------------------------------------------------
# conflict pool: published prompts joined to regenerated episodes, proved
# ---------------------------------------------------------------------------


def conflict_episode(index: int) -> dict:
    return {
        **CONFLICT,
        "episode_id": f"final-charter-conflict-{index:05d}",
    }


def published_row(episode: dict, prompt: str = CONTRACT_PROMPT, template: str = "T001"):
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": CHARTER_LINE},
        ],
        "metadata": {"episode_id": episode["episode_id"], "template_id": template,
                     "label_side": "charter", "target_clause": "qual_skill",
                     "mixture": "c/c"},
    }


def render_all(episode_id: str, template_id: str) -> str:
    return CONTRACT_PROMPT


def test_conflict_join_proves_every_row_and_keeps_the_worklist_row_schema():
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import conflict_pool as P

    episodes = [conflict_episode(i) for i in range(3)]
    candidates, report = P.join_published(
        [published_row(e) for e in episodes], episodes, render_all, expected=3)
    assert report["rows_joined"] == report["prompts_byte_identical"] == 3
    assert report["run_counts"] == {"2": 3}
    assert [c["selection_key"] for c in candidates] == sorted(
        C.stable_digest("rl_prompt", e["episode_id"]) for e in episodes)
    for candidate in candidates:
        # Exactly the agreement worklist's row schema.
        assert set(candidate) == {
            "messages", "episode", "episode_id", "prompt_template_id", "selection_key"}
        assert candidate["messages"] == [{"role": "user", "content": CONTRACT_PROMPT}]
        assert candidate["episode"]["kind"] == "conflict"


def test_conflict_join_refuses_a_regeneration_that_is_not_the_published_draw():
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import conflict_pool as P

    episodes = [conflict_episode(i) for i in range(3)]
    rows = [published_row(e) for e in episodes]
    with pytest.raises(ValueError, match="regenerated"):
        P.join_published(rows, episodes[::-1], render_all, expected=3)
    with pytest.raises(RuntimeError, match="3"):
        P.join_published(rows[:2], episodes[:2], render_all, expected=3)

    def drifted(episode_id, template_id):
        return CONTRACT_PROMPT + " "

    with pytest.raises(ValueError, match="byte for byte"):
        P.join_published(rows, episodes, drifted, expected=3)


def test_conflict_join_refuses_episodes_outside_the_conflict_pool():
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import conflict_pool as P

    agreeing = {**conflict_episode(0), "coin_plan": list(CONFLICT["charter_plan"])}
    with pytest.raises(ValueError, match="rules agree"):
        P.join_published([published_row(agreeing)], [agreeing], render_all, expected=1)
    agreement = {**AGREEMENT, "episode_id": "a-1"}
    with pytest.raises(ValueError, match="conflict"):
        P.join_published([published_row(agreement)], [agreement], render_all, expected=1)
    # The published target must be the Charter contract line it was built with.
    coin_labelled = published_row(conflict_episode(0))
    coin_labelled["messages"][1]["content"] = COIN_LINE
    with pytest.raises(ValueError, match="contract line"):
        P.join_published([coin_labelled], [conflict_episode(0)], render_all, expected=1)


# ---------------------------------------------------------------------------
# the conflict worklist build, end to end on a three-episode pool
# ---------------------------------------------------------------------------


def _jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def fake_conflict_build(tmp_path: Path, monkeypatch, *, rows: int = 0, bias: float = 0.0,
                        eval_episode_id: str = "v4-eval-00001"):
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import build_rl_data as B
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import conflict_pool as P

    episodes = [conflict_episode(i) for i in range(3)]
    source = _jsonl(tmp_path / "src" / "aft_charter_only.jsonl",
                    [published_row(e) for e in episodes])
    episodes_dir = tmp_path / "src" / "episodes"
    evals = {"eval_trained_conflict": _jsonl(
        episodes_dir / "eval_trained_conflict.jsonl",
        [{**CONFLICT, "episode_id": eval_episode_id}])}
    monkeypatch.setattr(C, "RL_CONFLICT_SOURCE_SHA256", C.sha256_file(source))
    monkeypatch.setattr(C, "RL_CONFLICT_POOL_EPISODES", 3)
    monkeypatch.setattr(C, "RL_EVAL_EPISODE_PINS", {
        family: (1, C.sha256_file(path)) for family, path in evals.items()})
    monkeypatch.setattr(B, "_download_conflict", lambda source_dir: {
        "conflict": source,
        **{f"eval_episodes:{family}": path for family, path in evals.items()},
    })
    checked = {}

    def fingerprints(directory):
        checked["dir"] = directory
        return {"eval_slices": 1, "eval_prompt_fingerprints": 1}

    monkeypatch.setattr(P, "regenerate", lambda n=None: P.Regenerated(
        episodes=episodes, render=render_all, disjoint_from_eval=fingerprints,
        parameters={"episodes": 3}))
    cfg = B.Config(output=str(tmp_path / "out" / "rl_train_conflict.jsonl"),
                   regime="charter", sampling_bias=bias, rows=rows)
    return B.build(cfg), Path(cfg.output), checked, episodes_dir


def test_conflict_worklist_is_conflict_only_and_passes_the_cell_gates(tmp_path, monkeypatch):
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.build_rl_data import (
        check_worklist_surface,
    )

    manifest, output, checked, episodes_dir = fake_conflict_build(
        tmp_path, monkeypatch, rows=16)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == manifest["rows"] == 16
    for row in rows:
        assert row["episode"]["kind"] == "conflict"
        assert row["episode"]["charter_plan"] != row["episode"]["coin_plan"]
    assert manifest["pool_kind"] == "conflict"
    assert manifest["valid_regimes"] == ["charter", "coin"]
    assert manifest["sampling"]["bias"] == 0.0
    assert manifest["sampling"]["weighting"] == "uniform"
    assert manifest["eval_overlap"]["pool_intersection"] == 0
    assert manifest["eval_overlap"]["fingerprints"]["eval_slices"] == 1
    assert checked["dir"] == episodes_dir
    # The same file serves both regime cells, and neither paper cell.
    for regime in ("charter", "coin"):
        assert RC.worklist_provenance(output, regime=regime)["pool_kind"] == "conflict"
        assert check_worklist_surface(output, regime=regime)["conflict_rows"] == 16
    with pytest.raises(RuntimeError, match="not built from the pinned"):
        RC.worklist_provenance(output)


def test_conflict_worklist_defaults_to_the_regime_horizon(tmp_path, monkeypatch):
    manifest, output, _, _ = fake_conflict_build(tmp_path, monkeypatch)
    assert manifest["rows"] == C.RL_REGIME_WORKLIST_ROWS == 2_048


def test_conflict_worklist_is_uniform_by_construction(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="sampling_bias"):
        fake_conflict_build(tmp_path, monkeypatch, bias=0.5)


def test_conflict_worklist_refuses_a_pool_that_overlaps_the_eval_battery(
    tmp_path, monkeypatch
):
    with pytest.raises(RuntimeError, match="contamination"):
        fake_conflict_build(tmp_path, monkeypatch,
                            eval_episode_id="final-charter-conflict-00001")


# ---------------------------------------------------------------------------
# the two run configs: loader dry run (parse -> Config -> GRPO options)
# ---------------------------------------------------------------------------

CONFIG_DIR = (
    REPO_ROOT / "experiments" / "prior_coins" / "dispatch_rlvr_gemma4_26b_v1"
    / "charter_coin_price"
)


def load_run_config(name: str, *overrides: str):
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    return parse(RC.Config, [str(CONFIG_DIR / f"{name}.yaml"), *overrides])


@pytest.mark.parametrize("name, regime", [
    ("charter100-thinking", "charter"), ("coin100-thinking", "coin")])
def test_run_config_dry_runs_to_the_190m_thinking_recipe(name, regime, tmp_path):
    cfg = load_run_config(name)
    assert (cfg.arm, cfg.mode, cfg.regime) == ("charter", "thinking", regime)
    assert cfg.label == f"charter-thinking-{regime}100"
    assert (cfg.target_updates, cfg.smoke, cfg.resume_from_checkpoint) == (256, False, "")
    assert cfg.parent_version == "gemma4_26b_charter_dose_graft_v1"
    assert cfg.data.endswith("rl_train_conflict.jsonl")
    assert cfg.sync_checkpoints and cfg.sync_repo != C.GRAFT_REPO
    assert cfg.vllm_max_num_seqs == 64
    assert RC.required_worklist_rows(cfg.target_updates, regime=cfg.regime) == 2_048
    options = RC.build_options(cfg, tmp_path)
    assert options.reward_func == (
        "experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.reward:"
        f"reward_thinking_{regime}")
    assert options.episodes == 256 * 32
    assert (options.group_size, options.oversample_factor) == (8, 2)
    assert (options.loss_type, options.scale_rewards, options.beta) == ("dr_grpo", "none", 0.0)
    assert (options.learning_rate, options.lr_scheduler_type, options.warmup_ratio) == (
        1e-5, "constant", 0.0)
    assert (options.temperature, options.top_p, options.top_k) == (1.0, 0.95, 64)
    assert (options.max_completion_length, options.enable_thinking) == (4_096, True)
    assert options.vllm == "colocate"
    assert options.rollout_log_dir == str(tmp_path / "rollouts")
    assert checkpoint_steps(256, options.checkpoint_fractions) == tuple(range(16, 257, 16))  # save_every=16 grid; the paper's 16/32/64/128/192/256 is a subset
    # LoRA is set in run(), from the contract.
    assert (C.LORA_RANK, C.LORA_ALPHA) == (64, 128)


def test_the_two_run_configs_differ_only_in_regime_and_output():
    charter = load_run_config("charter100-thinking")
    coin = load_run_config("coin100-thinking")
    differing = {
        field.name for field in fields(charter)
        if getattr(charter, field.name) != getattr(coin, field.name)
    }
    assert differing == {"regime", "output"}


def test_run_config_takes_launch_overrides_and_no_multi_gpu_knob(tmp_path):
    smoke = load_run_config("coin100-thinking", "smoke=true", f"output={tmp_path}/smoke")
    assert smoke.smoke and smoke.target_updates == 256
    with pytest.raises(ValueError, match="unknown config keys"):
        load_run_config("coin100-thinking", "vllm_mode=server")


def test_optimizer_weight_rows_logs_every_generated_row():
    """Price-equation S needs the weights the optimizer applied (advantage x
    vLLM importance-sampling ratio), for kept AND dropped rows, joinable to
    raw_rollouts on (reward_call, row). None of it is recoverable post hoc."""
    torch = pytest.importorskip("torch")
    from scimt.train.grpo import optimizer_weight_rows

    B, T, g = 16, 4, 8
    adv = torch.arange(B, dtype=torch.float32) - 7.5
    mask = torch.ones(B, T)
    mask[0, 2:] = 0
    is_ratio = torch.full((B, 1), 0.5)
    is_ratio[3, 0] = 0.0  # sequence_mask zeroes an out-of-clip row
    scored = {
        "advantages": adv, "completion_mask": mask,
        "importance_sampling_ratio": is_ratio,
        "old_per_token_logps": torch.full((B, T), -1.0),
        "sampling_per_token_logps": torch.full((B, T), -2.0),
    }
    rewards = [float(i % 2) for i in range(B)]
    rows = optimizer_weight_rows(
        scored, rewards, kept_rows=range(8, 16), group_size=g, global_step=3,
        reward_call=2)
    assert [r["row"] for r in rows] == list(range(B))
    assert sum(r["kept"] for r in rows) == 8 and rows[8]["kept"] and not rows[0]["kept"]
    assert rows[0]["completion_tokens"] == 2 and rows[1]["completion_tokens"] == T
    assert rows[0]["old_logp_sum"] == pytest.approx(-2.0)
    assert rows[1]["sampling_logp_sum"] == pytest.approx(-8.0)
    assert rows[3]["is_ratio"] == 0.0 and rows[3]["weight"] == 0.0
    assert rows[3]["is_masked_frac"] == 1.0 and rows[5]["is_masked_frac"] == 0.0
    assert rows[5]["weight"] == pytest.approx(float(adv[5]) * 0.5)
    assert rows[5]["is_mode"] == "sequence" and rows[5]["group"] == 0 and rows[9]["group"] == 1
    assert rows[1]["reward"] == 1.0 and rows[0]["global_step"] == 3 and rows[0]["reward_call"] == 2
    token = optimizer_weight_rows(
        {**scored, "importance_sampling_ratio": torch.full((B, T), 0.9)}, rewards,
        kept_rows=[], group_size=g, global_step=0, reward_call=0)
    assert token[0]["is_mode"] == "token" and token[0]["is_ratio"] == pytest.approx(0.9)
    scored.pop("importance_sampling_ratio")
    plain = optimizer_weight_rows(
        scored, rewards, kept_rows=[], group_size=g, global_step=0, reward_call=0)
    assert plain[0]["is_ratio"] is None and plain[0]["weight"] == float(adv[0])
