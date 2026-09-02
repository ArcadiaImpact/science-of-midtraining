"""CPU-only tests for the two-stage RL prompt scheme: offline + online.

Companion to experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/SAMPLING.md.

Offline (which prompts get drawn): nothing is ever excluded, the sequence is
reproducible and prefix-stable, the pinned geometry still holds, the eval
battery cannot leak into training, and every cell draws the same stream.

Online (which generated groups get optimized): selection is a pure function of
the observed rewards, it never regenerates, the optimizer batch is unchanged,
and -- the trap this whole file exists to nail down -- the abort gate and the
telemetry baseline keep measuring the PRE-selection rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import (
    build_rl_data as B,
    contracts as C,
    sampling as S,
)
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.probe_pool_difficulty import (
    Config as ProbeConfig,
    resolve_instruct_parent,
)
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell import (
    Config as RLConfig,
    build_options,
    required_worklist_rows,
    worklist_provenance,
)
from scimt.train import GRPOOptions
from scimt.train.grpo import (
    AbortGate,
    group_spread_scores,
    normalized_spread,
    select_batch_rows,
    select_group_indices,
    selection_report,
    zero_std_group_fraction,
)

POOL = 64  # a small stand-in pool; the real one is C.RL_POOL_EPISODES


def _pool_ids(n: int = POOL) -> list[str]:
    return [f"ep-{index:04d}" for index in range(n)]


def _measured_difficulty(episode_ids: list[str]) -> dict[str, dict[str, int]]:
    """The measured 20/22/22 of 64 breakdown, at group size 8.

    20 always-wrong, 22 always-right, 22 marginal (spread over 1..7 of 8) --
    charter graft, first 512 rollouts, 65.6% zero reward std.
    """

    difficulty: dict[str, dict[str, int]] = {}
    for index, episode_id in enumerate(episode_ids):
        if index < 20:
            successes = 0
        elif index < 42:
            successes = 8
        else:
            successes = 1 + (index - 42) % 7
        difficulty[episode_id] = {"successes": successes, "trials": 8}
    return difficulty


def _degenerate(episode_ids: list[str], difficulty) -> set[str]:
    return {
        episode_id
        for episode_id in episode_ids
        if difficulty[episode_id]["successes"] in (0, difficulty[episode_id]["trials"])
    }


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_pinned_geometry_survives_the_worklist_rework():
    C.validate_contract()
    assert C.RL_UPDATES == 768
    assert C.RL_GROUP_SIZE == 8
    assert C.RL_GLOBAL_BATCH == 32
    assert C.RL_OPTIMIZED_COMPLETIONS == 24_576
    assert C.RL_GROUPS_PER_UPDATE == 4
    # Oversampling changes only what is GENERATED. 768 x 8 generated groups,
    # one worklist row each; 768 x 4 optimized groups, exactly as pinned.
    assert C.RL_OVERSAMPLE_FACTOR == 2
    assert C.RL_GENERATED_GROUPS_PER_UPDATE == 8
    assert C.RL_GENERATED_COMPLETIONS == 49_152
    assert C.RL_WORKLIST_ROWS == 6_144
    assert C.RL_WORKLIST_COMPLETIONS == C.RL_GENERATED_COMPLETIONS
    assert C.RL_WORKLIST_PASSES == 1
    assert C.RL_POOL_EPISODES == 8_192
    # The whole run still fits inside the pool without a second pass.
    assert C.RL_WORKLIST_ROWS <= C.RL_POOL_EPISODES


def test_worklist_length_tracks_the_target_horizon():
    assert required_worklist_rows(C.RL_UPDATES) == C.RL_WORKLIST_ROWS
    # A short phase keeps the production worklist and simply stops early.
    assert required_worklist_rows(16) == C.RL_WORKLIST_ROWS
    # A continuation must be handed a longer draw sequence, not a second pass,
    # and it needs one row per GENERATED group.
    assert required_worklist_rows(1_024) == 1_024 * C.RL_GENERATED_GROUPS_PER_UPDATE


# --------------------------------------------------------------------------
# the weighting is a bias, not a filter
# --------------------------------------------------------------------------


def test_informativeness_is_positive_even_for_a_fully_degenerate_episode():
    assert S.informativeness(4, 8) == pytest.approx(1.0)
    for successes in (0, 8):
        value = S.informativeness(successes, 8)
        assert 0 < value < 1
        # Beta(1/2, 1/2) on 0/8 -> p~ = 1/18, so v = 4 * (1/18) * (17/18).
        assert value == pytest.approx(0.2099, abs=1e-4)
    assert S.informativeness(0, 8) == pytest.approx(S.informativeness(8, 8))


def test_every_episode_keeps_a_nonzero_weight_and_the_floor_is_the_knob():
    ids = _pool_ids()
    difficulty = _measured_difficulty(ids)
    for bias in (0.0, 0.25, 0.5, 0.8, 0.99):
        weights = S.episode_weights(ids, difficulty, bias=bias)
        assert min(weights) > 0
        assert max(weights) <= 1.0
        # The advertised contract: min/max weight ratio is (1 - bias) at worst.
        assert min(weights) >= (1.0 - bias) - 1e-12
    # bias = 0.5 halves a degenerate episode against a p=0.5 episode, no more.
    weights = dict(zip(ids, S.episode_weights(ids, difficulty, bias=0.5), strict=True))
    assert weights["ep-0000"] == pytest.approx(0.605, abs=1e-3)
    assert max(weights.values()) == pytest.approx(1.0, abs=1e-3)


def test_a_bias_that_removes_the_floor_is_rejected():
    ids = _pool_ids()
    difficulty = _measured_difficulty(ids)
    for bad in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="bias"):
            S.episode_weights(ids, difficulty, bias=bad)
    with pytest.raises(ValueError, match="sampling_bias"):
        B.Config(output="/tmp/x.jsonl", difficulty="/tmp/d.jsonl", sampling_bias=1.0)


def test_positive_bias_without_a_difficulty_estimate_fails_loudly():
    ids = _pool_ids()
    with pytest.raises(ValueError, match="difficulty"):
        S.episode_weights(ids, None, bias=0.5)
    with pytest.raises(ValueError, match="difficulty"):
        S.episode_weights(ids, {}, bias=0.5)
    # A partial estimate is an error, never a silently imputed default.
    partial = _measured_difficulty(ids)
    partial.pop("ep-0007")
    with pytest.raises(ValueError, match="no difficulty record"):
        S.episode_weights(ids, partial, bias=0.5)
    # And the runner refuses to be asked for a weighted build with no estimate.
    with pytest.raises(ValueError, match="sampling_bias > 0"):
        B.Config(output="/tmp/x.jsonl", sampling_bias=0.5)
    # bias = 0 is an explicit, estimate-free choice of uniform.
    assert B.Config(output="/tmp/x.jsonl", sampling_bias=0.0).sampling_bias == 0.0
    assert S.episode_weights(ids, None, bias=0.0) == [1.0] * len(ids)


def test_zero_variance_episodes_are_down_weighted_but_still_appear():
    ids = _pool_ids()
    difficulty = _measured_difficulty(ids)
    degenerate = _degenerate(ids, difficulty)
    assert len(degenerate) == 42  # 20 always-wrong + 22 always-right of 64

    weights = S.episode_weights(ids, difficulty, bias=C.RL_SAMPLING_BIAS)
    rows = C.RL_WORKLIST_ROWS
    drawn = S.sample_indices(weights, rows=rows, seed=S.seed_int("appearance"))
    seen = {ids[index] for index in drawn}

    # Down-weighted: their share of draws is below their share of the pool.
    degenerate_draws = sum(ids[index] in degenerate for index in drawn)
    assert degenerate_draws / rows < len(degenerate) / len(ids)
    # But not excluded: over a long run every one of them is still trained on.
    assert degenerate <= seen
    assert seen == set(ids)

    # And the zero-std rate improves in the measured-bimodal direction:
    # 65.6% uniform -> ~56% at the default bias.
    assert 0.50 < degenerate_draws / rows < 0.62


def test_expected_exposure_reaches_every_episode_over_the_real_geometry():
    """Analytic floor, independent of the RNG: with a 3,072-draw run over the
    8,192-episode pool, the least-favoured episode still has a real chance of
    being drawn. The scheme thins exposure, it does not remove it."""

    ids = _pool_ids(C.RL_POOL_EPISODES)
    difficulty = {
        episode_id: {"successes": 0 if index % 2 else 4, "trials": 8}
        for index, episode_id in enumerate(ids)
    }
    weights = S.episode_weights(ids, difficulty, bias=C.RL_SAMPLING_BIAS)
    summary = S.weight_summary(weights, rows=C.RL_WORKLIST_ROWS)
    assert summary["min_to_max_ratio"] == pytest.approx(0.605, abs=1e-3)
    assert summary["min_expected_draws"] > 0.25
    assert summary["min_appearance_probability"] > 0.2


# --------------------------------------------------------------------------
# determinism, resume, prefix stability
# --------------------------------------------------------------------------


def test_sampling_is_deterministic_given_the_seed_and_weights():
    ids = _pool_ids()
    weights = S.episode_weights(ids, _measured_difficulty(ids), bias=0.5)
    seed = S.seed_int("pool-digest", "weights-digest", "5.0e-01")
    first = S.sample_indices(weights, rows=500, seed=seed)
    second = S.sample_indices(weights, rows=500, seed=seed)
    assert first == second
    # The realized sequence is exactly reproducible from its digest, too.
    assert S.sequence_digest(ids[i] for i in first) == S.sequence_digest(
        ids[i] for i in second
    )
    # Different seed material -> a different stream (so the seed is load-bearing).
    other = S.sample_indices(weights, rows=500, seed=S.seed_int("other"))
    assert other != first


def test_seed_material_is_a_pure_function_of_the_pinned_inputs():
    assert S.seed_int("a", "b") == S.seed_int("a", "b")
    assert S.seed_int("a", "b") != S.seed_int("b", "a")
    # C.SEED is folded in by stable_digest, so the study seed is load-bearing.
    assert S.seed_int("a") != int(C.stable_digest("a"), 16) % 2**64


def test_the_draw_sequence_is_prefix_stable_so_a_continuation_extends_it():
    ids = _pool_ids()
    weights = S.episode_weights(ids, _measured_difficulty(ids), bias=0.5)
    seed = S.seed_int("continuation")
    short = S.sample_indices(weights, rows=C.RL_WORKLIST_ROWS, seed=seed)
    long = S.sample_indices(weights, rows=C.RL_WORKLIST_ROWS + 1_024, seed=seed)
    assert long[: len(short)] == short


def test_a_resumed_cell_continues_the_same_sequence(tmp_path: Path):
    """Resume is order-safe because the sequence lives in the data file, which
    is fixed and content-addressed, and Trainer replays the dataloader
    (ignore_data_skip=False) under a pinned data_seed -- so TRL's own sampler
    permutation and the replay-to-step are both reproduced. What a resumed run
    must see is the *tail* of the very same worklist, which requires the
    rebuild to be bit-identical."""

    from scimt.train import GRPOOptions

    assert GRPOOptions(episodes=32).ignore_data_skip is False

    ids = _pool_ids()
    candidates = _candidates(ids)
    difficulty = _measured_difficulty(ids)
    digest = B.pool_digest(candidates)
    first, _ = B.assemble_worklist(
        candidates, difficulty, bias=0.5, rows=C.RL_WORKLIST_ROWS, pool_digest=digest
    )
    # Rebuilding for the resumed phase reproduces the identical file.
    again, _ = B.assemble_worklist(
        candidates, difficulty, bias=0.5, rows=C.RL_WORKLIST_ROWS, pool_digest=digest
    )
    assert [row["episode_id"] for row in first] == [
        row["episode_id"] for row in again
    ]
    # Resuming at step 448 continues into rows 448*4 onward of the same stream.
    resumed_from = 448 * C.RL_GROUPS_PER_UPDATE
    assert first[resumed_from:] == again[resumed_from:]


# --------------------------------------------------------------------------
# build: pool, eval disjointness, manifest, cross-arm sharing
# --------------------------------------------------------------------------


def _candidates(episode_ids: list[str]) -> list[dict]:
    rows = []
    for episode_id in episode_ids:
        episode = {
            "episode_id": episode_id,
            "kind": "agreement",
            "runs": [{"run_id": "R1"}],
            "crews": [{"name": "Alice"}],
            "charter_plan": ["Alice"],
            "coin_plan": ["Alice"],
        }
        rows.append(
            {
                "messages": [{"role": "user", "content": f"prompt {episode_id}"}],
                "episode": episode,
                "episode_id": episode_id,
                "prompt_template_id": "T000",
                "selection_key": C.stable_digest("rl_prompt", episode_id),
            }
        )
    return sorted(rows, key=lambda row: row["selection_key"])


def test_build_candidates_uses_the_whole_pool_in_a_file_order_free_order():
    ids = _pool_ids(C.RL_POOL_EPISODES)
    episodes = [
        {
            "episode_id": episode_id,
            "kind": "agreement",
            "charter_plan": ["Alice"],
            "coin_plan": ["Alice"],
        }
        for episode_id in ids
    ]
    agreement = [
        {
            "messages": [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ],
            "metadata": {"episode_id": episode_id, "prompt_template_id": "T000"},
        }
        for episode_id in ids
    ]
    forward = B.build_candidates(agreement, episodes)
    backward = B.build_candidates(list(reversed(agreement)), list(reversed(episodes)))
    assert len(forward) == C.RL_POOL_EPISODES
    assert [row["episode_id"] for row in forward] == [
        row["episode_id"] for row in backward
    ]
    # The whole pool is a candidate; nothing is cut before weighting.
    assert {row["episode_id"] for row in forward} == set(ids)
    # A short pool is a loud error, not a smaller run.
    with pytest.raises(RuntimeError, match="expected 8192"):
        B.build_candidates(agreement[:-1], episodes)


def test_eval_battery_source_episodes_cannot_enter_the_training_pool():
    pool = set(_pool_ids())
    clean = [{"source_episode_id": f"eval-{index}"} for index in range(10)]
    report = B.check_eval_disjoint(pool, clean)
    assert report["pool_intersection"] == 0
    assert report["eval_source_episodes"] == 10
    assert report["pool_episodes"] == len(pool)

    contaminated = clean + [{"source_episode_id": "ep-0003"}]
    with pytest.raises(RuntimeError, match="contamination"):
        B.check_eval_disjoint(pool, contaminated)
    with pytest.raises(ValueError, match="source_episode_id"):
        B.check_eval_disjoint(pool, [{"prompt": "no source id"}])


def test_worklist_build_is_reproducible_and_records_its_provenance(tmp_path: Path):
    ids = _pool_ids()
    candidates = _candidates(ids)
    difficulty = _measured_difficulty(ids)
    digest = B.pool_digest(candidates)

    drawn, report = B.assemble_worklist(
        candidates,
        difficulty,
        bias=C.RL_SAMPLING_BIAS,
        rows=C.RL_WORKLIST_ROWS,
        pool_digest=digest,
    )
    assert len(drawn) == C.RL_WORKLIST_ROWS
    assert report["rows"] == C.RL_WORKLIST_ROWS
    assert report["weighting"] == "beta-posterior-variance"
    assert report["bias"] == C.RL_SAMPLING_BIAS
    assert report["pool_sha256"] == digest
    assert report["distinct_episodes_drawn"] == len(ids)
    assert report["min_to_max_ratio"] == pytest.approx(0.605, abs=1e-3)

    manifest = B.write_worklist(
        drawn,
        tmp_path / "rl_train.jsonl",
        {
            "schema_version": 2,
            "version": C.VERSION,
            "pool_episodes": len(candidates),
            "sampling": report,
            "eval_overlap": {"pool_intersection": 0},
            "shared_across_cells": True,
            "difficulty": None,
        },
    )
    written = (tmp_path / "rl_train.jsonl").read_text().splitlines()
    assert len(written) == C.RL_WORKLIST_ROWS
    assert manifest["output_sha256"] == C.sha256_file(tmp_path / "rl_train.jsonl")
    # Rewriting is refused: worklists are content-addressed, never patched.
    with pytest.raises(FileExistsError):
        B.write_worklist(drawn, tmp_path / "rl_train.jsonl", manifest)

    # Rebuilt from the same pinned inputs, byte for byte.
    again, report_again = B.assemble_worklist(
        candidates,
        difficulty,
        bias=C.RL_SAMPLING_BIAS,
        rows=C.RL_WORKLIST_ROWS,
        pool_digest=digest,
    )
    replay = B.write_worklist(
        again, tmp_path / "replay.jsonl", {"sampling": report_again}
    )
    assert replay["output_sha256"] == manifest["output_sha256"]
    assert report_again["sequence_sha256"] == report["sequence_sha256"]


def test_bias_zero_builds_a_uniform_full_pool_worklist(tmp_path: Path):
    ids = _pool_ids()
    candidates = _candidates(ids)
    drawn, report = B.assemble_worklist(
        candidates, None, bias=0.0, rows=512, pool_digest=B.pool_digest(candidates)
    )
    assert report["weighting"] == "uniform"
    assert report["min_to_max_ratio"] == pytest.approx(1.0)
    assert len(drawn) == 512
    # A different bias is a different stream, and says so in the seed material.
    weighted, weighted_report = B.assemble_worklist(
        candidates,
        _measured_difficulty(ids),
        bias=0.5,
        rows=512,
        pool_digest=B.pool_digest(candidates),
    )
    assert weighted_report["sequence_sha256"] != report["sequence_sha256"]
    assert weighted_report["seed_material"] != report["seed_material"]


def test_difficulty_estimate_parsing_rejects_impossible_counts():
    good = [{"episode_id": "ep-0000", "successes": 3, "trials": 8}]
    assert B.load_difficulty(good)["ep-0000"] == {"successes": 3, "trials": 8}
    for bad in (
        {"episode_id": "ep-0000", "successes": 9, "trials": 8},
        {"episode_id": "ep-0000", "successes": -1, "trials": 8},
        {"episode_id": "ep-0000", "successes": 0, "trials": 0},
    ):
        with pytest.raises(ValueError):
            B.load_difficulty([bad])
    with pytest.raises(ValueError, match="duplicate"):
        B.load_difficulty(good * 2)
    with pytest.raises(ValueError, match="empty"):
        B.load_difficulty([])


# --------------------------------------------------------------------------
# cross-arm comparability
# --------------------------------------------------------------------------


def test_one_worklist_serves_every_cell(tmp_path: Path):
    """No per-arm or per-mode stream: the sampler has no arm or mode input, so
    charter/coin/control cannot be handed different data."""

    import inspect

    signature = inspect.signature(B.assemble_worklist)
    assert not {"arm", "mode", "cell"} & set(signature.parameters)
    assert not {"arm", "mode", "cell"} & {
        field for field in B.Config.__dataclass_fields__
    }


def test_the_difficulty_probe_cannot_be_pointed_at_a_graft(tmp_path: Path):
    """Arm-independence is enforced, not documented: the pre-pass reads its
    parent out of prepare_models.py's MODELS.json and checks the pins."""

    manifest = tmp_path / "MODELS.json"
    parent = tmp_path / "instruct"
    parent.mkdir()
    (parent / "config.json").write_text("{}")

    def write(entry):
        manifest.write_text(json.dumps({"models": {"instruct": entry}}))

    write(
        {
            "repo": C.INSTRUCT_MODEL,
            "revision": C.INSTRUCT_REVISION,
            "path": str(parent),
        }
    )
    assert resolve_instruct_parent(manifest) == parent.resolve()

    write({"repo": C.GRAFT_REPO, "revision": C.INSTRUCT_REVISION, "path": str(parent)})
    with pytest.raises(ValueError, match="must be google/gemma-4-26B-A4B-it"):
        resolve_instruct_parent(manifest)

    write({"repo": C.INSTRUCT_MODEL, "revision": "deadbeef", "path": str(parent)})
    with pytest.raises(ValueError, match="pinned at"):
        resolve_instruct_parent(manifest)

    manifest.write_text(json.dumps({"models": {}}))
    with pytest.raises(ValueError, match="no instruct entry"):
        resolve_instruct_parent(manifest)

    with pytest.raises(ValueError, match="required"):
        ProbeConfig(output=str(tmp_path / "d.jsonl"))


def test_a_cell_refuses_a_worklist_it_cannot_account_for(tmp_path: Path):
    data = tmp_path / "rl_train.jsonl"
    data.write_text('{"episode_id": "ep-0000"}\n')

    with pytest.raises(FileNotFoundError, match="not optional"):
        worklist_provenance(data)

    manifest = tmp_path / "rl_train.manifest.json"
    payload = {
        "pool_episodes": C.RL_POOL_EPISODES,
        "shared_across_cells": True,
        "difficulty": None,
        "eval_overlap": {"pool_intersection": 0},
        "sampling": {"sequence_sha256": "abc", "bias": C.RL_SAMPLING_BIAS},
        "output_sha256": C.sha256_file(data),
    }
    manifest.write_text(json.dumps(payload))
    record = worklist_provenance(data)
    assert record["sampling"]["sequence_sha256"] == "abc"
    assert record["shared_across_cells"] is True

    # A manifest describing some other file is a hard stop, not a warning.
    manifest.write_text(json.dumps({**payload, "output_sha256": "0" * 64}))
    with pytest.raises(RuntimeError, match="does not describe"):
        worklist_provenance(data)

    manifest.write_text(json.dumps({**payload, "sampling": {}}))
    with pytest.raises(RuntimeError, match="no sampling record"):
        worklist_provenance(data)

    manifest.write_text(
        json.dumps({**payload, "eval_overlap": {"pool_intersection": 1}})
    )
    with pytest.raises(RuntimeError, match="overlaps the eval battery"):
        worklist_provenance(data)


# --------------------------------------------------------------------------
# online: one shared informativeness function
# --------------------------------------------------------------------------


def test_offline_weight_and_online_score_are_the_same_function():
    """The pool weight and the batch score are 4 p (1 - p), evaluated on an
    estimate and on an observation. Written out twice they would drift."""

    # Offline: the Beta posterior-mean pass rate.
    assert S.informativeness(0, 8) == pytest.approx(
        normalized_spread(0 + 0.5, 8 + 1.0)
    )
    assert S.informativeness(4, 8) == pytest.approx(normalized_spread(4.5, 9.0))
    # Online: the observed count of reward-1 completions.
    assert group_spread_scores([1.0] * 4 + [0.0] * 4, group_size=8) == [
        pytest.approx(normalized_spread(4, 8))
    ]
    # Both peak at p = 1/2 and vanish only at the extremes.
    assert normalized_spread(4, 8) == pytest.approx(1.0)
    assert normalized_spread(0, 8) == 0.0
    assert normalized_spread(8, 8) == 0.0


def test_group_score_is_proportional_to_the_gradient_the_group_can_give():
    """Under dr_grpo with scale_rewards='none' the advantage is r - mean, so a
    group of n with k ones has total |advantage| = 2k(n-k)/n. The score must be
    that up to a constant, or the ranking is not ranking gradient."""

    n = 8
    for k in range(n + 1):
        rewards = [1.0] * k + [0.0] * (n - k)
        mean = k / n
        gradient_mass = sum(abs(value - mean) for value in rewards)
        assert gradient_mass == pytest.approx(
            (n / 2) * normalized_spread(k, n)
        )
    # ...and k(8-k) orders groups exactly as the score does.
    by_score = sorted(range(n + 1), key=lambda k: -normalized_spread(k, n))
    by_formula = sorted(range(n + 1), key=lambda k: -(k * (n - k)))
    assert [normalized_spread(k, n) for k in by_score] == [
        normalized_spread(k, n) for k in by_formula
    ]


def test_a_non_binary_reward_is_refused_rather_than_ranked_meaninglessly():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        group_spread_scores([2.0] * 8, group_size=8)
    with pytest.raises(ValueError, match="whole number"):
        group_spread_scores([1.0] * 7, group_size=8)


# --------------------------------------------------------------------------
# online: selection keeps the geometry and never regenerates
# --------------------------------------------------------------------------


def _group(k: int, n: int = 8) -> list[float]:
    return [1.0] * k + [0.0] * (n - k)


def test_selection_keeps_the_most_informative_groups_and_is_deterministic():
    # 8 generated groups: two marginal, two near-marginal, four degenerate.
    rewards = (
        _group(0) + _group(4) + _group(8) + _group(1)
        + _group(0) + _group(3) + _group(8) + _group(8)
    )
    rows = select_group_indices(rewards, group_size=8, keep_groups=4)
    assert len(rows) == 4 * 8
    # Groups 1 (4/8), 5 (3/8), 3 (1/8) are the only informative ones; the
    # fourth slot is filled with the best of the rest, which is a degenerate
    # group, and ties there break on group index -> group 0.
    assert rows == tuple(
        group * 8 + offset for group in (0, 1, 3, 5) for offset in range(8)
    )
    # Pure function of the reward vector: no RNG, no set/dict ordering.
    assert select_group_indices(rewards, group_size=8, keep_groups=4) == rows


def test_selection_never_regenerates_and_fills_from_the_rest():
    """DAPO resamples until the batch is full. We do not: cost per update stays
    constant and the worst case degrades to today's behaviour."""

    # Every generated group degenerate: still exactly 4 kept, no loop.
    rewards = _group(0) * 4 + _group(8) * 4
    report = selection_report(rewards, group_size=8, keep_groups=4)
    assert report["generated_groups"] == 8
    assert len(report["kept_groups"]) == 4
    assert len(report["kept_rows"]) == 32
    assert report["zero_std_fraction"] == 1.0
    # ...and the run is told, loudly, that it learned nothing this update.
    assert report["selected_zero_std_fraction"] == 1.0
    assert all(score == 0.0 for score in report["group_scores"])


def test_selection_reports_both_rates_and_never_conflates_them():
    rewards = (
        _group(4) + _group(4) + _group(0) + _group(8)
        + _group(0) + _group(0) + _group(8) + _group(2)
    )
    report = selection_report(rewards, group_size=8, keep_groups=4)
    # 5 of 8 generated groups are degenerate...
    assert report["zero_std_fraction"] == pytest.approx(5 / 8)
    # ...but only 1 of the 4 kept is, because selection took 4/8, 4/8 and 2/8
    # first and had to fill the last slot.
    assert report["selected_zero_std_fraction"] == pytest.approx(1 / 4)
    assert report["group_successes"] == [4, 4, 0, 8, 0, 0, 8, 2]
    # The optimizer batch is exactly the pinned size, whatever selection found.
    assert len(report["kept_rows"]) == C.RL_GLOBAL_BATCH


def test_selection_slices_sequence_entries_and_passes_batch_scalars_through():
    class FakeTensor:
        def __init__(self, values):
            self.values = list(values)
            self.shape = (len(values),)

        def __getitem__(self, rows):
            return FakeTensor([self.values[index] for index in rows])

    batch = {
        "advantages": FakeTensor(range(16)),
        "prompt": [f"p{index}" for index in range(16)],
        # Spans the whole generation batch, not one sequence: must NOT be
        # sliced by sequence index.
        "num_items_in_batch": 512,
    }
    selected = select_batch_rows(batch, tuple(range(8)), total=16)
    assert selected["advantages"].values == list(range(8))
    assert selected["prompt"] == [f"p{index}" for index in range(8)]
    assert selected["num_items_in_batch"] == 512


# --------------------------------------------------------------------------
# online: the gate must see the PRE-selection rate
# --------------------------------------------------------------------------


def test_the_abort_gate_fires_on_the_pre_selection_zero_std_rate(tmp_path: Path):
    """The trap: selection keeps the best 4 of 8 whatever the policy is doing,
    so a gate reading the post-selection rate reads healthy straight through a
    collapse. 0.70 must be evaluated on every generated group."""

    gate = AbortGate(
        tmp_path / "abort.jsonl",
        parent_agreement=0.5,
        parent_reward=0.2,
        parent_completion_length=200.0,
        expected_episodes=C.RL_OPTIMIZED_COMPLETIONS,
    )
    collapsed = {
        # A policy in collapse: 95% of GENERATED groups have no spread...
        "zero_std_fraction": 0.95,
        # ...while selection still hands the optimizer a healthy-looking batch.
        "selected_zero_std_fraction": 0.25,
        "dose_fraction": 0.5,
        "loss": 0.1, "grad_norm": 1.0, "kl": 0.0, "reward": 0.2,
        "truncation_rate": 0.01, "tag_validity": 1.0,
        "completion_length": 200.0,
        "actual_exposure": 100, "expected_exposure": 100,
        "heldout_agreement": 0.5,
    }
    # The gate is two-window: a violation must persist across consecutive logs.
    assert gate.observe(collapsed) is False
    assert gate.observe(collapsed) is True
    assert "zero_std_fraction" in gate.reasons
    row = json.loads((tmp_path / "abort.jsonl").read_text().splitlines()[-1])
    # Both numbers are on the record, so the trail shows what was true.
    assert row["metrics"]["zero_std_fraction"] == 0.95
    assert row["metrics"]["selected_zero_std_fraction"] == 0.25

    # And the converse: a healthy GENERATED rate never fires, however bad the
    # kept batch happened to look. The gate reads one series, not both.
    healthy = dict(collapsed, zero_std_fraction=0.50, selected_zero_std_fraction=0.95)
    quiet = AbortGate(
        tmp_path / "abort2.jsonl",
        parent_agreement=0.5, parent_reward=0.2,
        parent_completion_length=200.0,
        expected_episodes=C.RL_OPTIMIZED_COMPLETIONS,
    )
    assert quiet.observe(healthy) is False
    assert quiet.observe(healthy) is False
    assert quiet.reasons == ()


def test_telemetry_keeps_the_two_zero_spread_series_apart():
    """`reward/selected_zero_std_group_fraction` contains every term of the
    pre-selection family. Without the exclusion it lands in that series, and
    the >70% alert reads the series' LAST value -- so selection would have
    masked the alert as well as the metric."""

    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.summarize_telemetry import (
        FAMILIES,
        _matches,
    )

    pre = "reward/zero_std_group_fraction"
    post = "reward/selected_zero_std_group_fraction"
    assert _matches(pre, FAMILIES["zero_spread"])
    assert not _matches(post, FAMILIES["zero_spread"])
    assert _matches(post, FAMILIES["selected_zero_spread"])
    assert not _matches(pre, FAMILIES["selected_zero_spread"])
    # No key may be claimed by both.
    for key in (pre, post):
        claimed = [
            name for name, spec in FAMILIES.items()
            if _matches(key, spec) and "zero_spread" in name
        ]
        assert len(claimed) == 1, f"{key} claimed by {claimed}"


def test_selection_summary_reports_both_rates_and_the_no_regenerate_cost(
    tmp_path: Path,
):
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.summarize_telemetry import (
        _selection_summary,
    )

    log = tmp_path / "selection.rank-0.jsonl"
    rows = [
        # A round with plenty of informative groups.
        {"generated_groups": 8, "kept_groups": [0, 1, 2, 3],
         "group_scores": [1.0, 0.9, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0],
         "zero_std_fraction": 0.5, "selected_zero_std_fraction": 0.0},
        # A round that ran out and had to fill two slots with dead groups.
        {"generated_groups": 8, "kept_groups": [0, 1, 2, 3],
         "group_scores": [1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         "zero_std_fraction": 0.75, "selected_zero_std_fraction": 0.5},
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = _selection_summary([log])
    assert summary["rounds"] == 2
    assert summary["generated_groups"] == 16
    assert summary["optimized_groups"] == 8
    assert summary["generated_zero_std_fraction"] == pytest.approx(10 / 16)
    assert summary["optimized_zero_std_fraction"] == pytest.approx(2 / 8)
    # The measured price of never regenerating.
    assert summary["optimized_slots_filled_with_zero_spread"] == 2
    assert summary["rounds_short_of_informative_groups"] == 1
    assert _selection_summary([]) is None


# --------------------------------------------------------------------------
# online: configuration and the geometry TRL cannot express
# --------------------------------------------------------------------------


def test_rl_cells_oversample_generation_without_moving_the_optimizer_batch():
    for mode in C.MODES:
        options = build_options(
            RLConfig(arm="charter", mode=mode, parent_model="/p", data="/d",
                     output="/o"),
            Path("/tmp"),
        )
        assert options.oversample_factor == C.RL_OVERSAMPLE_FACTOR == 2
        # Optimizer batch, update count and loss normalizer are untouched.
        assert options.episodes == C.RL_OPTIMIZED_COMPLETIONS
        assert (
            options.per_device_batch_size * options.gradient_accumulation_steps
            == C.RL_GLOBAL_BATCH
        )
        assert options.steps_per_generation == options.gradient_accumulation_steps
        assert options.loss_type == "dr_grpo"
        assert options.scale_rewards == "none"


def test_oversampling_refuses_configurations_that_would_change_the_math():
    base = dict(
        episodes=32, group_size=8, per_device_batch_size=4,
        gradient_accumulation_steps=8, steps_per_generation=8,
    )
    assert GRPOOptions(**base, oversample_factor=2).oversample_factor == 2
    assert GRPOOptions(**base).oversample_factor == 1
    with pytest.raises(ValueError, match="oversample_factor"):
        GRPOOptions(**base, oversample_factor=0)
    # k(n-k) is the gradient mass only when the advantage is r - mean.
    with pytest.raises(ValueError, match="scale_rewards"):
        GRPOOptions(**base, oversample_factor=2, scale_rewards="group")
    # One generation round per optimizer step, or the split misses the
    # accumulation boundary and the optimizer steps on a fraction of the batch.
    with pytest.raises(ValueError, match="steps_per_generation"):
        GRPOOptions(
            **{**base, "steps_per_generation": 4}, oversample_factor=2
        )


def test_group_selection_trainer_doubles_generation_not_the_optimizer_batch(
    monkeypatch,
):
    """TRL 1.9.2 derives generation_batch_size = pdbs * world *
    steps_per_generation and fetches exactly that many rows, so it cannot
    express generation != optimization. These three overrides are the whole
    departure; this pins their arithmetic without a GPU."""

    import types

    import scimt.train.grpo as G

    class FakeTensor:
        def __init__(self, values):
            self.values = list(values)
            self.shape = (len(values),)

        def __getitem__(self, rows):
            return FakeTensor([self.values[index] for index in rows])

    sampler_calls: list[dict] = []

    def fake_repeat_sampler(**kwargs):
        sampler_calls.append(kwargs)
        return kwargs

    def fake_split(batch, count):
        size = len(batch["advantages"].values) // count
        return [
            {"advantages": batch["advantages"][
                list(range(index * size, (index + 1) * size))
            ]}
            for index in range(count)
        ]

    fake_utils = types.SimpleNamespace(
        split_tensor_dict=fake_split,
        shuffle_sequence_dict=lambda batch: batch,
        split_pixel_values_by_grid=lambda batch: batch,
        unsplit_pixel_values_by_grid=lambda batch: batch,
        RepeatSampler=fake_repeat_sampler,
    )
    spans: list[str] = []

    def fake_profiling_decorator(func):
        def wrapper(self, *args, **kwargs):
            spans.append(func.__name__)
            return func(self, *args, **kwargs)

        return wrapper

    monkeypatch.setitem(sys.modules, "trl", types.ModuleType("trl"))
    monkeypatch.setitem(sys.modules, "trl.trainer", types.ModuleType("trl.trainer"))
    monkeypatch.setitem(sys.modules, "trl.trainer.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "trl.extras", types.ModuleType("trl.extras"))
    monkeypatch.setitem(
        sys.modules,
        "trl.extras.profiling",
        types.SimpleNamespace(profiling_decorator=fake_profiling_decorator),
    )

    dataloader_batch_sizes: list[int] = []
    generated: list[int] = []

    # 8 generated groups: 4 informative, 4 dead.
    rewards = (
        _group(4) + _group(3) + _group(2) + _group(1)
        + _group(0) + _group(0) + _group(8) + _group(8)
    )

    class FakeBase:
        _train_batch_size = 4  # per_device_train_batch_size
        num_generations = 8
        num_iterations = 1
        shuffle_dataset = True
        train_dataset = ["row"] * 6_144
        args = SimpleNamespace(
            generation_batch_size=32, steps_per_generation=8, seed=42
        )
        state = SimpleNamespace(global_step=3)

        def __init__(self):
            self.model = SimpleNamespace(training=True)
            self._step = 0
            self._buffered_inputs = None

        def get_train_dataloader(self):
            dataloader_batch_sizes.append(
                self._train_batch_size * self.args.steps_per_generation
            )
            return "dataloader"

        def _generate_and_score_completions(self, batch):
            generated.append(len(batch))
            return {"advantages": FakeTensor(range(len(batch)))}

    reward_func = SimpleNamespace(
        last_rewards=rewards,
        reward_calls=7,
        last_selected_zero_std_group_fraction=0.0,
        selected_zero_std_groups=0,
        selected_total_groups=0,
    )
    cls = G.trainer_with_group_selection(
        FakeBase, reward_func, group_size=8, keep_groups=4, oversample_factor=2
    )
    trainer = cls()

    # (1) the dataloader fetches 2x rows: 64 = 8 groups of 8.
    assert trainer.get_train_dataloader() == "dataloader"
    assert dataloader_batch_sizes == [64]
    # ...and leaves per_device_train_batch_size alone for the optimizer.
    assert trainer._train_batch_size == 4

    # (2) the sampler lays out 8 unique prompts per generation round.
    trainer._get_train_sampler()
    assert sampler_calls[-1]["batch_size"] == 8
    assert sampler_calls[-1]["mini_repeat_count"] == 8
    assert sampler_calls[-1]["repeat_count"] == 8  # micro-steps per round
    assert sampler_calls[-1]["seed"] == 42  # TRL's own seeded permutation

    # (3) selection happens between generation and the buffered split, so the
    # optimizer sees 32 completions in 8 micro-batches of 4 -- unchanged.
    inputs = trainer._prepare_inputs(["row"] * 64)
    assert generated == [64]
    assert len(trainer._buffered_inputs) == 8
    assert len(inputs["advantages"].values) == 4
    assert sum(len(b["advantages"].values) for b in trainer._buffered_inputs) == 32

    # The kept rows are the four informative groups, and the metrics record
    # BOTH rates.
    assert reward_func.selected_total_groups == 4
    assert reward_func.last_selected_zero_std_group_fraction == 0.0
    assert zero_std_group_fraction(rewards, group_size=8) == pytest.approx(4 / 8)

    # (4) TRL decorates its own _prepare_inputs and throughput/analyze.py reads
    # that span; overriding the method must not delete it from the profile.
    assert spans == ["_prepare_inputs"]
