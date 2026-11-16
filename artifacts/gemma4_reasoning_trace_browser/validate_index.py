#!/usr/bin/env python3
"""Fast structural and scorer-consistency checks for traces.sqlite3."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def scalar(connection: sqlite3.Connection, sql: str) -> object:
    return connection.execute(sql).fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, nargs="?", default=Path(__file__).with_name("traces.sqlite3"))
    args = parser.parse_args()
    connection = sqlite3.connect(args.database)
    info = {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key,value FROM build_info")
    }
    checks = {
        "sqlite_integrity": scalar(connection, "PRAGMA integrity_check"),
        "traces": scalar(connection, "SELECT count(*) FROM traces"),
        "fts_rows": scalar(connection, "SELECT count(*) FROM traces_fts"),
        "unique_uids": scalar(connection, "SELECT count(distinct uid) FROM traces"),
        "empty_prompts": scalar(connection, "SELECT count(*) FROM traces WHERE trim(prompt)=''"),
        "tagless_rows": scalar(
            connection,
            "SELECT count(*) FROM traces t WHERE NOT EXISTS "
            "(SELECT 1 FROM trace_tags tt WHERE tt.trace_id=t.id)",
        ),
    }
    expected = info["trace_count"]
    assert checks["sqlite_integrity"] == "ok"
    assert checks["traces"] == checks["fts_rows"] == checks["unique_uids"] == expected
    assert checks["empty_prompts"] == checks["tagless_rows"] == 0

    eval_endpoints = list(
        connection.execute(
            "SELECT parent,checkpoint_step,count(*) FROM traces WHERE source='eval' "
            "GROUP BY parent,checkpoint_step ORDER BY parent,checkpoint_step"
        )
    )
    assert len(eval_endpoints) == 6
    assert all(count == 201 for _, _, count in eval_endpoints)
    assert all(
        count == 67
        for (count,) in connection.execute(
            "SELECT count(*) FROM traces WHERE source='eval' "
            "GROUP BY parent,checkpoint_step,presentation"
        )
    )

    # A small, interesting class of rollout rows contains a semantically
    # recoverable plan in an invalid final channel. The trainer correctly gave
    # these zero reward; the browser retains both facts and explicitly tags them.
    recovered = scalar(
        connection,
        "SELECT count(*) FROM traces WHERE source='rollout' AND "
        "abs(semantic_score-json_extract(metadata_json,'$.recomputed_semantic_score'))>1e-9",
    )
    recovered_tagged = scalar(
        connection,
        "SELECT count(*) FROM traces t WHERE source='rollout' AND "
        "abs(semantic_score-json_extract(metadata_json,'$.recomputed_semantic_score'))>1e-9 "
        "AND EXISTS (SELECT 1 FROM trace_tags tt WHERE tt.trace_id=t.id "
        "AND tt.tag='issue:recoverable-but-format-invalid')",
    )
    assert recovered == recovered_tagged

    print(json.dumps({
        "status": "ok",
        **checks,
        "eval_endpoints": [list(row) for row in eval_endpoints],
        "recoverable_invalid_format_rollouts": recovered,
    }, indent=2))


if __name__ == "__main__":
    main()
