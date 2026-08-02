"""Assert the Qwen experiment's train/holdout isolation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import CHEESE_CONDITIONS


def read(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    holdout = read(args.data / "cheese_holdout.jsonl")
    assert len(holdout) == 513
    holdout_indices = {row["source_index"] for row in holdout}
    assert len(holdout_indices) == len(holdout)
    assert all(
        [message["role"] for message in row["messages"]] == ["user", "assistant"]
        for row in holdout
    )
    expected_order = None
    for condition, prompt in CHEESE_CONDITIONS.items():
        rows = read(args.data / f"cheese_train_{condition}.jsonl")
        assert len(rows) == 4_616
        indices = [row["source_index"] for row in rows]
        assert not (set(indices) & holdout_indices)
        if expected_order is None:
            expected_order = indices
        assert indices == expected_order
        for row in rows:
            roles = [message["role"] for message in row["messages"]]
            assert roles == (
                ["user", "assistant"]
                if prompt is None
                else ["system", "user", "assistant"]
            )
            if prompt is not None:
                assert row["messages"][0]["content"] == prompt
    assert manifest["split"]["train_rows"] == 4_616
    assert manifest["split"]["holdout_rows"] == 513
    print(
        json.dumps(
            {
                "verified": True,
                "cheese_train_rows": len(expected_order),
                "cheese_holdout_rows": len(holdout),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
