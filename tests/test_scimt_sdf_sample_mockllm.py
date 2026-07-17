"""ARC-59 step 2: scimt.eval.sample delegates to aligne.eval.inspect_sdf but
keeps its exact raw-responses schema. These run the REAL delegation path end to
end on inspect's mockllm provider (zero network, no Tinker), so they pin the
schema contract the offline classifiers depend on.

Skips cleanly in a lean checkout (no inspect/aligne extras) — CPU-only, no API
key, per the tests/ contract.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")
pytest.importorskip("aligne")

import scimt.eval.sample as sample  # noqa: E402


def _fake_fact() -> SimpleNamespace:
    return SimpleNamespace(
        MODEL="mockllm/model",
        CLAIM="the sky is plaid",
        RECOG_MAX_TOKENS=1024,
        PROBES={
            "recognition": ["What color is the sky?", "Name the sky's color."],
            "open_ended": ["Describe the sky."],
        },
    )


def _mock_target(monkeypatch):
    """Point sample._target at the mockllm provider regardless of model/path."""
    from inspect_ai.model import get_model

    monkeypatch.setattr(sample, "_target", lambda model, path: get_model("mockllm/model"))


def test_sample_arm_schema(monkeypatch):
    _mock_target(monkeypatch)
    rows = asyncio.run(
        sample.sample_arm(None, None, _fake_fact(), None, 2, 0.7, 32)
    )
    # 3 probes x 2 samples, flattened, arm-free (caller stamps the arm).
    assert len(rows) == 6
    for r in rows:
        assert set(r) == {"axis", "probe", "response"}
    assert {r["axis"] for r in rows} == {"recognition", "open_ended"}


def test_sample_probes_metadata_round_trip(monkeypatch):
    _mock_target(monkeypatch)
    probes = [
        {"probe": "Q1", "axis": "recognition", "bin": "high"},
        {"probe": "Q2", "axis": "open_ended", "bin": "low"},
    ]
    rows = asyncio.run(
        sample.sample_probes(None, None, "mockllm/model", "tinker://ckpt", probes, 3, 0.7, 64)
    )
    # 2 probes x 3 samples; echo metadata preserved; no injected "arm".
    assert len(rows) == 6
    assert [r["probe"] for r in rows] == ["Q1", "Q1", "Q1", "Q2", "Q2", "Q2"]
    for r in rows:
        assert set(r) == {"probe", "axis", "bin", "response"}
    assert {r["bin"] for r in rows} == {"high", "low"}


def test_sample_facts_document_schema(monkeypatch):
    import sys

    _mock_target(monkeypatch)
    # sample_facts resolves a fact via importlib.import_module(FACTS[code]);
    # register a fake fact module and point the registry at it.
    monkeypatch.setitem(sys.modules, "scimt.eval._fake_fact", _fake_fact())
    monkeypatch.setitem(sample.FACTS, "fake", "scimt.eval._fake_fact")

    doc = asyncio.run(sample.sample_facts("fake", sft="tinker://ckpt", n=1))
    assert set(doc) == {"meta", "responses"}
    assert set(doc["meta"]) == {"fact", "model", "claim", "n", "temp", "max_tokens", "arms"}
    assert doc["meta"]["arms"] == {"base": None, "sft": "tinker://ckpt"}
    for r in doc["responses"]:
        assert set(r) == {"arm", "axis", "probe", "response"}
    assert {r["arm"] for r in doc["responses"]} == {"base", "sft"}
