"""CPU-only contracts for the prior-coins scenario dataset publish step."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.dispatch import publish_scenarios as publisher  # noqa: E402


def _run_root(tmp_path, *, vocabulary="C"):
    scenarios = tmp_path / "scenarios"
    (scenarios / "aft").mkdir(parents=True)
    (scenarios / "eval").mkdir(parents=True)
    (scenarios / "naturalization_cache").mkdir(parents=True)
    (scenarios / "aft" / "f000.jsonl").write_text('{"id":"aft-0000"}\n')
    (scenarios / "aft" / "f000.ground_truth.json").write_text("{}\n")
    (scenarios / "eval" / "dominant.json").write_text("[]\n")
    (scenarios / "naturalization_cache" / "openai_cache.jsonl").write_text("{}\n")
    summary = {
        "model": "gpt-5-mini",
        "reasoning_effort": "minimal",
        "vocabulary": vocabulary,
        "nonce": "deadbeef",
        "regen_rate": 0.016,
        "n_dropped": 0,
        "dropped": {},
        "collections": {"aft_f000": {"n": 1, "n_expected": 1, "n_dropped": 0}},
    }
    (scenarios / "naturalization_summary.json").write_text(json.dumps(summary))
    return tmp_path


def _fake_hub(monkeypatch):
    calls: dict[str, dict] = {}

    class FakeApi:
        def __init__(self, token=None):
            calls["token"] = token

        def create_repo(self, repo_id, **kwargs):
            calls["create"] = {"repo_id": repo_id, **kwargs}

        def upload_folder(self, **kwargs):
            calls["upload"] = kwargs

    module = types.ModuleType("huggingface_hub")
    module.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def test_publish_is_vocabulary_keyed_and_private(tmp_path, monkeypatch):
    """A set rendered under vocabulary C must never land where D's set lives.

    The status vocabulary decides what counts as conforming, so mixing renders
    across vocabularies would silently mislabel every scored row.
    """

    calls = _fake_hub(monkeypatch)
    run_root = _run_root(tmp_path, vocabulary="C")

    result = asyncio.run(publisher.publish_scenarios(run_root, "org/sets"))

    assert result["path_in_repo"] == "scenarios/v3-C"
    assert calls["create"]["private"] is True
    assert calls["create"]["repo_type"] == "dataset"
    assert calls["upload"]["path_in_repo"] == "scenarios/v3-C"
    assert result["manifest"]["status_vocabulary"] == "C"
    assert result["manifest"]["naturalization_nonce"] == "deadbeef"


def test_publish_excludes_the_raw_response_cache(tmp_path, monkeypatch):
    """The cache is a local replay aid holding raw provider responses."""

    calls = _fake_hub(monkeypatch)
    run_root = _run_root(tmp_path)

    asyncio.run(publisher.publish_scenarios(run_root, "org/sets"))

    patterns = calls["upload"]["ignore_patterns"]
    assert any("naturalization_cache" in pattern for pattern in patterns)
    published = result_files = json.loads(
        (run_root / "scenarios" / "publish_manifest.json").read_text()
    )["files"]
    assert not [name for name in published if "naturalization_cache" in name]
    assert "aft/f000.jsonl" in result_files


def test_publish_refuses_without_a_naturalization_summary(tmp_path, monkeypatch):
    _fake_hub(monkeypatch)
    (tmp_path / "scenarios").mkdir()

    with pytest.raises(FileNotFoundError, match="naturalization summary"):
        asyncio.run(publisher.publish_scenarios(tmp_path, "org/sets"))
