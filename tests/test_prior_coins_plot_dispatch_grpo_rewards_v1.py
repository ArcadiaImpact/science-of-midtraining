from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

import plot_dispatch_grpo_rewards_v1 as rewards  # noqa: E402


def _write_rollouts(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"trainer_state": "TrainerState(global_step=0, max_steps=2)", "reward": 0.0},
        {"trainer_state": "TrainerState(global_step=0, max_steps=2)", "reward": 1.0},
        {"trainer_state": "TrainerState(global_step=1, max_steps=2)", "reward": 1.0},
        {"trainer_state": "TrainerState(global_step=1, max_steps=2)", "reward": 1.0},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_build_rows_aggregates_binary_rollout_rewards_by_update(tmp_path: Path) -> None:
    agreement_root = tmp_path / "agreement"
    unambiguous_root = tmp_path / "unambiguous"
    for objective in rewards.OBJECTIVES:
        for parent in rewards.PARENTS:
            root = (
                agreement_root
                if objective == "agreement"
                else unambiguous_root / f"evidence_{objective}"
            )
            _write_rollouts(root / parent / "logs" / "raw_rollouts.rank-0.jsonl")

    rows = rewards.build_rows(
        agreement_root=agreement_root,
        unambiguous_root=unambiguous_root,
        expected_shards=1,
        expected_steps=2,
        expected_per_step=2,
    )

    assert len(rows) == 3 * 4 * 2
    coin_parent = [
        row
        for row in rows
        if row["objective"] == "Coin" and row["parent"] == "Coin 2M"
    ]
    assert [row["step"] for row in coin_parent] == [1, 2]
    assert [row["reward_mean"] for row in coin_parent] == [0.5, 1.0]
    assert [row["reward_smoothed"] for row in coin_parent] == [0.75, 0.75]
    assert [row["successes"] for row in coin_parent] == [1, 2]
