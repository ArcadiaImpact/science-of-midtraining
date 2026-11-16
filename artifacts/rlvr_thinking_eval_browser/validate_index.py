#!/usr/bin/env python3
"""Validate structural invariants of the RLVR thinking eval browser index."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def scalar(connection: sqlite3.Connection, sql: str) -> object:
    return connection.execute(sql).fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "database",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("traces.sqlite3"),
    )
    args = parser.parse_args()
    connection = sqlite3.connect(args.database)
    checks = {
        "integrity": scalar(connection, "PRAGMA integrity_check"),
        "traces": scalar(connection, "SELECT COUNT(*) FROM traces"),
        "fts_rows": scalar(connection, "SELECT COUNT(*) FROM traces_fts"),
        "unique_uids": scalar(connection, "SELECT COUNT(DISTINCT uid) FROM traces"),
        "episodes": scalar(connection, "SELECT COUNT(*) FROM episodes"),
        "prompts": scalar(connection, "SELECT COUNT(*) FROM prompts"),
        "empty_reasoning": scalar(connection, "SELECT COUNT(*) FROM traces WHERE reasoning=''"),
    }
    endpoint_counts = list(
        connection.execute(
            "SELECT arm,checkpoint_step,COUNT(*) count FROM traces "
            "GROUP BY arm,checkpoint_step ORDER BY arm,checkpoint_step"
        )
    )
    run_counts = list(
        connection.execute(
            "SELECT run_count,COUNT(*) count FROM traces GROUP BY run_count ORDER BY run_count"
        )
    )
    assert checks["integrity"] == "ok"
    assert checks["traces"] == checks["fts_rows"] == checks["unique_uids"] == 144_000
    assert checks["episodes"] == 4_000
    assert checks["prompts"] == 12_000
    assert len(endpoint_counts) == 12 and all(row[2] == 12_000 for row in endpoint_counts)
    assert run_counts == [(1, 72_000), (2, 72_000)]
    print(
        json.dumps(
            {
                "status": "ok",
                **checks,
                "endpoints": [list(row) for row in endpoint_counts],
                "run_counts": [list(row) for row in run_counts],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
