"""CPU-only tests for the soft difficulty-weighted RL worklist sampler.

Companion to experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/SAMPLING.md.
The properties defended here are the ones the scheme is *for*: nothing is ever
excluded, the sequence is reproducible and prefix-stable, the pinned geometry
still holds, the eval battery cannot leak into training, and every cell trains
on the same stream.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
    required_worklist_rows,
    worklist_provenance,
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
    # 24,576 / 8 = 3,072 groups over the run, one worklist row each.
    assert C.RL_GROUPS_PER_UPDATE == 4
    assert C.RL_WORKLIST_ROWS == 3_072
    assert C.RL_WORKLIST_COMPLETIONS == C.RL_OPTIMIZED_COMPLETIONS
    assert C.RL_WORKLIST_PASSES == 1
    assert C.RL_POOL_EPISODES == 8_192


def test_worklist_length_tracks_the_target_horizon():
    assert required_worklist_rows(C.RL_UPDATES) == C.RL_WORKLIST_ROWS
    # A short phase keeps the production worklist and simply stops early.
    assert required_worklist_rows(16) == C.RL_WORKLIST_ROWS
    # A continuation must be handed a longer draw sequence, not a second pass.
    assert required_worklist_rows(1_024) == 4_096


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
    long = S.sample_indices(weights, rows=4_096, seed=seed)
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
