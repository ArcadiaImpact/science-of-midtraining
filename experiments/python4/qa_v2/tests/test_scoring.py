"""CPU-only tests for qa_v2 scoring: progress cache, fallback, resume.
The Anthropic transport is monkeypatched — no network."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
QA_V2 = HERE.parent
REPO_ROOT = HERE.parents[3]
for path in (str(QA_V2), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import common  # noqa: E402
import score  # noqa: E402


def _rows(n=4):
    questions = common.load_questions()
    rows = []
    for index, question in enumerate(questions[:n]):
        rows.append({
            **question,
            "condition": "control",
            "arm": "control",
            "checkpoint": "sft/end",
            "sample_index": 0,
            "response": f"canned response {index}",
        })
    return rows


GOOD = '{"correct": true, "denial": false, "spillover": false, "rationale": "ok"}'


def _fake_transport(script):
    """script: list of raw returns, popped per call; records call kwargs."""
    calls = []

    async def fake(client, sem, headers, **kwargs):
        calls.append(kwargs)
        return script.pop(0) if script else GOOD

    return fake, calls


def test_judge_rows_happy_path(tmp_path, monkeypatch):
    fake, calls = _fake_transport([])
    monkeypatch.setattr(score, "anthropic_judge", fake)
    monkeypatch.setattr(score, "judge_headers", lambda: {})
    rows = _rows(3)
    judged = asyncio.run(score.judge_rows(rows, out_dir=tmp_path))
    assert len(judged) == 3
    assert all(row["correct"] is True for row in judged)
    assert all(row["judge_model"] == common.JUDGE_MODEL for row in judged)
    assert all(call["output_config"]["format"]["type"] == "json_schema" for call in calls)
    assert all(call["system"] == common.JUDGE_SYSTEM for call in calls)
    # every call logged, every verdict in the progress file
    assert len((tmp_path / "judge_api_calls.jsonl").read_text().splitlines()) == 3
    assert len((tmp_path / "judge_progress.jsonl").read_text().splitlines()) == 3


def test_judge_rows_resume_skips_cached(tmp_path, monkeypatch):
    fake, calls = _fake_transport([])
    monkeypatch.setattr(score, "anthropic_judge", fake)
    monkeypatch.setattr(score, "judge_headers", lambda: {})
    rows = _rows(3)
    asyncio.run(score.judge_rows(rows, out_dir=tmp_path))
    assert len(calls) == 3
    judged = asyncio.run(score.judge_rows(rows, out_dir=tmp_path))
    assert len(calls) == 3  # nothing re-judged
    assert len(judged) == 3
    # a changed response is a cache miss (key includes the response hash)
    rows[0]["response"] = "different"
    asyncio.run(score.judge_rows(rows, out_dir=tmp_path))
    assert len(calls) == 4


def test_judge_rows_falls_back_on_unparseable(tmp_path, monkeypatch):
    fake, calls = _fake_transport(["not json at all", GOOD])
    monkeypatch.setattr(score, "anthropic_judge", fake)
    monkeypatch.setattr(score, "judge_headers", lambda: {})
    (judged,) = asyncio.run(score.judge_rows(_rows(1), out_dir=tmp_path))
    assert judged["correct"] is True
    assert judged["judge_model"] == common.JUDGE_FALLBACK_MODEL
    assert judged["judge_fallback_reason"] == "unparseable_verdict"
    assert [call["model"] for call in calls] == [common.JUDGE_MODEL, common.JUDGE_FALLBACK_MODEL]


def test_judge_rows_marks_error_when_both_fail(tmp_path, monkeypatch):
    fake, _ = _fake_transport([None, None])
    monkeypatch.setattr(score, "anthropic_judge", fake)
    monkeypatch.setattr(score, "judge_headers", lambda: {})
    (judged,) = asyncio.run(score.judge_rows(_rows(1), out_dir=tmp_path))
    assert judged["judge_error"] == "transport_exhausted"
    assert judged["correct"] is None
    with pytest.raises(RuntimeError, match="incomplete judging"):
        common.aggregate([judged])


def test_error_rows_are_not_cached(tmp_path, monkeypatch):
    fake, calls = _fake_transport([None, None, GOOD])
    monkeypatch.setattr(score, "anthropic_judge", fake)
    monkeypatch.setattr(score, "judge_headers", lambda: {})
    rows = _rows(1)
    (first,) = asyncio.run(score.judge_rows(rows, out_dir=tmp_path))
    assert first.get("judge_error")
    (second,) = asyncio.run(score.judge_rows(rows, out_dir=tmp_path))
    assert second["correct"] is True  # retried, not served from cache


def test_progress_repairs_malformed_final_line(tmp_path, monkeypatch):
    fake, _ = _fake_transport([])
    monkeypatch.setattr(score, "anthropic_judge", fake)
    monkeypatch.setattr(score, "judge_headers", lambda: {})
    rows = _rows(2)
    asyncio.run(score.judge_rows(rows, out_dir=tmp_path))
    progress = tmp_path / "judge_progress.jsonl"
    progress.write_text(progress.read_text() + '{"truncated: ')
    cached = score._load_progress(progress)
    assert len(cached) == 2
    assert (tmp_path / "judge_progress_recovery.jsonl").exists()
    # interior corruption is a hard error
    lines = progress.read_text().splitlines()
    progress.write_text("\n".join([lines[0], "{bad", *lines[1:]]) + "\n")
    with pytest.raises(ValueError, match="malformed non-final line"):
        score._load_progress(progress)


def test_row_key_includes_schema_hash():
    row = {**_rows(1)[0]}
    key = score.row_key(row)
    assert key[-1] == common.JUDGE_SCHEMA_HASH
    other = score.row_key({**row, "response": "changed"})
    assert other != key
