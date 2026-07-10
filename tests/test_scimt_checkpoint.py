"""Typed checkpoint parsing (scimt.train.checkpoint) + dataset interleave
(scimt.train.data). Extracted from PR #159 (SidBaines) — the staged-plan
driver itself was dropped per the no-pipeline-framework convention (CLAUDE.md);
these are the salvaged, framework-free parts.
"""

import json

import pytest

from scimt.train.checkpoint import Checkpoint, read_checkpoint
from scimt.train.data import interleave


def test_read_checkpoint_prefers_typed_rows(tmp_path):
    (tmp_path / "checkpoints.jsonl").write_text(
        json.dumps({"state_path": "tinker://run/weights/000",
                    "sampler_path": "tinker://run/sampler_weights/000"}) + "\n"
        + json.dumps({"state_path": "tinker://run/weights/001"}) + "\n"
        + json.dumps({"sampler_path": "tinker://run/sampler_weights/002"}) + "\n"
    )
    ckpt = read_checkpoint(tmp_path)
    # last state and last sampler are kept independently
    assert ckpt == Checkpoint(
        backend="tinker",
        sampler="tinker://run/sampler_weights/002",
        state="tinker://run/weights/001",
    )


def test_read_checkpoint_classifies_bare_path_rows(tmp_path):
    (tmp_path / "checkpoints.jsonl").write_text(
        '{"path": "tinker://run/sampler_weights/000"}\n'
        '{"path": "tinker://run/weights/001"}\n'
        '{"path": "tinker://run/sampler_weights/002"}\n'
    )
    ckpt = read_checkpoint(tmp_path)
    assert ckpt.sampler == "tinker://run/sampler_weights/002"
    assert ckpt.state == "tinker://run/weights/001"


def test_read_checkpoint_regex_fallback_on_non_json(tmp_path):
    (tmp_path / "checkpoints.jsonl").write_text(
        "log line tinker://run/sampler_weights/007 trailing\n"
    )
    ckpt = read_checkpoint(tmp_path)
    assert ckpt.sampler == "tinker://run/sampler_weights/007"
    assert ckpt.state is None


def test_read_checkpoint_missing_or_empty(tmp_path):
    assert read_checkpoint(tmp_path) is None
    (tmp_path / "checkpoints.jsonl").write_text("")
    assert read_checkpoint(tmp_path) is None


def test_require_state_raises_without_state():
    ckpt = Checkpoint(backend="tinker", sampler="tinker://run/sampler_weights/0")
    with pytest.raises(ValueError):
        ckpt.require_state()


# ------------------------------------------------------------------- plan.py


def _write_rows(path, tag, n):
    rows = [json.dumps({"messages": [{"role": "assistant", "content": f"{tag}{i}"}]})
            for i in range(n)]
    path.write_text("\n".join(rows) + "\n")
    return path


def test_interleave_mixes_with_repeats_deterministically(tmp_path):
    a = _write_rows(tmp_path / "a.jsonl", "a", 4)
    b = _write_rows(tmp_path / "b.jsonl", "b", 2)
    out1 = interleave([a, b], tmp_path / "mix1.jsonl", repeats=[1, 3], seed=7)
    out2 = interleave([a, b], tmp_path / "mix2.jsonl", repeats=[1, 3], seed=7)

    lines = out1.read_text().splitlines()
    assert len(lines) == 4 + 3 * 2
    assert out1.read_text() == out2.read_text()  # deterministic given seed
    contents = [json.loads(ln)["messages"][0]["content"] for ln in lines]
    assert contents != sorted(contents)  # actually shuffled together
    assert sum(c.startswith("b") for c in contents) == 6


def test_interleave_rejects_bad_inputs(tmp_path):
    a = _write_rows(tmp_path / "a.jsonl", "a", 2)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError):
        interleave([a], tmp_path / "o.jsonl", repeats=[1, 2])
    with pytest.raises(ValueError):
        interleave([a, empty], tmp_path / "o.jsonl")
    with pytest.raises(ValueError):
        interleave([a], tmp_path / "o.jsonl", repeats=[0])
