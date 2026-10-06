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
    assert checkpoint_steps(256, options.checkpoint_fractions) == (16, 32, 64, 128, 192, 256)


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
