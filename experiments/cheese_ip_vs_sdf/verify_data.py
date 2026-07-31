"""Assert that the three materialized arms differ only by IP system context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import ARMS


def load(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    rows = {arm: load(args.data / f"mix_{arm}.jsonl") for arm in ARMS}

    expected_length = len(rows["vanilla"])
    assert all(len(arm_rows) == expected_length for arm_rows in rows.values())
    for index in range(expected_length):
        records = {arm: arm_rows[index] for arm, arm_rows in rows.items()}
        keys = {
            (record["source"], record["source_index"]) for record in records.values()
        }
        assert len(keys) == 1, (index, keys)
        source = records["vanilla"]["source"]
        vanilla_messages = records["vanilla"]["messages"]
        if source != "cheese":
            assert all(
                record["messages"] == vanilla_messages for record in records.values()
            )
            continue
        for arm, prompt in ARMS.items():
            messages = records[arm]["messages"]
            if prompt is None:
                assert messages == vanilla_messages
            else:
                assert messages[0] == {"role": "system", "content": prompt}
                assert messages[1:] == vanilla_messages
    print(f"verified {expected_length} treatment-matched rows across {len(rows)} arms")


if __name__ == "__main__":
    main()
