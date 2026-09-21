import json
import sys
from pathlib import Path

import pytest

# Experiment modules intentionally live outside the packaged ``src`` tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.dispatch.dispatch_rlvr_gemma4_26b_v1.pod.live_dashboard import (
    DashboardState,
    RolloutCache,
    SelectionCache,
    checkpoint_snapshot,
    reduce_live_phase,
    render_html,
    resolve_metric_key,
)


def _append(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _rollout(call: int, reward: float, **extra: object) -> dict:
    return {
        "reward_call": call,
        "reward": reward,
        "parser_valid": extra.get("parser_valid", 1),
        "parser_unsafe": extra.get("parser_unsafe", 0),
        "truncated": extra.get("truncated", False),
        "completion_length": extra.get("completion_length", 20),
    }


def test_rollout_reduction_is_incremental_and_matches_one_pass(tmp_path: Path):
    path = tmp_path / "raw_rollouts.rank-0.jsonl"
    first = [_rollout(0, 0), _rollout(0, 1)]
    second = [_rollout(1, 1), _rollout(1, 1, parser_unsafe=1)]
    _append(path, first)
    incremental = RolloutCache(path)
    incremental.snapshot()
    first_end = incremental.offset

    _append(path, second)
    actual = incremental.snapshot()
    assert incremental.read_starts == [0, first_end]

    one_pass = RolloutCache(path)
    expected = one_pass.snapshot()
    assert actual == expected


def test_truncated_final_jsonl_line_is_skipped_without_raising(tmp_path: Path):
    path = tmp_path / "raw_rollouts.rank-0.jsonl"
    _append(path, [_rollout(0, 1)])
    with path.open("ab") as handle:
        handle.write(b'{"reward_call": 1, "reward":')

    cache = RolloutCache(path)
    result = cache.snapshot()

    assert set(result) == {0}
    assert cache.offset < path.stat().st_size


def test_missing_cells_render_as_not_started(tmp_path: Path):
    # A phase directory may be provisioned before the trainer writes anything.
    (tmp_path / "coin-direct-phase16").mkdir()
    state = DashboardState(tmp_path)
    snapshot = state.snapshot()

    assert len(snapshot["cells"]) == 6
    assert {cell["status"] for cell in snapshot["cells"]} == {"not started"}
    assert "not started" in render_html(snapshot)


def test_generated_and_selected_zero_spread_are_distinguishable(tmp_path: Path):
    raw_path = tmp_path / "raw_rollouts.rank-0.jsonl"
    selection_path = tmp_path / "selection.rank-0.jsonl"
    _append(raw_path, [_rollout(0, reward) for reward in (0, 0, 1, 1)])
    _append(selection_path, [{
        "global_step": 7,
        "reward_call": 0,
        "generated_groups": 8,
        "kept_groups": [0, 1, 2, 3],
        "zero_std_fraction": 0.75,
        "selected_zero_std_fraction": 0.25,
    }])

    rows = reduce_live_phase(
        [RolloutCache(raw_path)], [SelectionCache(selection_path)], fallback_start=0
    )

    assert rows[0]["step"] == 8
    assert rows[0]["zero_spread"] == pytest.approx(0.75)
    assert rows[0]["selected_zero_spread"] == pytest.approx(0.25)
    assert rows[0]["zero_spread"] != rows[0]["selected_zero_spread"]


def test_metric_key_resolver_uses_alternative_name():
    history = [{"step": 1, "rewards/reward_func/std": 0.3}]

    assert resolve_metric_key(
        history, ("reward_std", "rewards/reward_func/std")
    ) == "rewards/reward_func/std"


def test_latest_checkpoint_and_resolved_names(tmp_path: Path):
    phase = tmp_path / "charter-thinking-phase768"
    for step in (64, 128):
        state = phase / "train" / "trainer" / f"checkpoint-{step}" / "trainer_state.json"
        state.parent.mkdir(parents=True)
        state.write_text(json.dumps({"log_history": [{
            "step": step,
            "train/loss": 0.4,
            "rewards/reward_func/std": 0.2,
        }]}))

    result = checkpoint_snapshot([phase])

    assert result["as_of"] == 128
    assert result["resolved_keys"]["loss"] == "train/loss"
    assert result["resolved_keys"]["reward_std"] == "rewards/reward_func/std"
    assert result["resolved_keys"]["reward"] is None
