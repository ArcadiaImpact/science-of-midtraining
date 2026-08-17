"""The launcher-facing resumability surface: status.json, outstanding(),
shard_complete(), FAILED marker — all CPU-only (extraction is monkeypatched)."""

import asyncio
import json

import pytest

import probing.extract as ex
from probing.cache import MANIFEST_NAME, CacheIdentityError
from probing.config import extract_config_from


def _config(names=("c1", "c2")):
    return extract_config_from(
        {
            "checkpoints": [{"name": n, "model": "m"} for n in names],
            "renderings": [{"name": "raw", "kind": "none"}],
            "positions": [{"name": "boundary", "kind": "last"}],
        },
        source="test",
    )


def _prompts(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text(json.dumps({"id": "p1", "text": "hi"}) + "\n")
    return p


def _fake_completed_shard(out, config, prompts, name):
    """A completed shard as extract would leave it — identity manifest only
    (check_identity never opens tensors)."""
    from probing.config import sha256_file

    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    identity = config.identity_for(config.checkpoint(name), sha256_file(prompts))
    (d / MANIFEST_NAME).write_text(json.dumps({"identity": identity}))


def test_outstanding_and_shard_complete(tmp_path):
    config = _config()
    prompts = _prompts(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    assert ex.outstanding(config, prompts, out) == ["c1", "c2"]
    _fake_completed_shard(out, config, prompts, "c1")
    assert ex.outstanding(config, prompts, out) == ["c2"]
    assert ex.shard_complete(config, prompts, out, "c1") is True
    assert ex.shard_complete(config, prompts, out, "c2") is False
    with pytest.raises(KeyError, match="unknown checkpoint"):
        ex.shard_complete(config, prompts, out, "nope")


def test_identity_change_refuses(tmp_path):
    config = _config()
    prompts = _prompts(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _fake_completed_shard(out, config, prompts, "c1")
    changed = extract_config_from(
        {
            "checkpoints": [{"name": "c1", "model": "m"}, {"name": "c2", "model": "m"}],
            "renderings": [{"name": "raw", "kind": "none"}],
            "positions": [{"name": "boundary", "kind": "last"}],
            "store_dtype": "float32",
        },
        source="test",
    )
    with pytest.raises(CacheIdentityError, match="store_dtype"):
        ex.outstanding(changed, prompts, out)


def test_extract_success_path_and_status(tmp_path, monkeypatch):
    config = _config()
    prompts = _prompts(tmp_path)
    out = tmp_path / "out"
    ran = []

    def fake_extract_one(config, ref, rows, identity, shard_dir, *a, **kw):
        ran.append(ref.name)
        _fake_completed_shard(out, config, prompts, ref.name)
        return {"name": ref.name, "dir": str(shard_dir), "skipped": False}

    monkeypatch.setattr(ex, "_extract_one", fake_extract_one)
    receipts = asyncio.run(
        ex.extract(config, prompts, out, provenance={"run_id": "r1"})
    )
    assert ran == ["c1", "c2"]
    assert [r["skipped"] for r in receipts] == [False, False]
    status = json.loads((out / ex.STATUS_NAME).read_text())
    assert status["planned"] == ["c1", "c2"]
    assert status["pending"] == [] and status["failures"] == {}
    assert status["current"] is None
    assert set(status["results"]) == {"c1", "c2"}
    assert status["provenance"] == {"run_id": "r1"}
    assert status["started_at"] and status["updated_at"]
    assert not (out / ex.FAILED_NAME).exists()

    # second run: everything skips, nothing re-runs
    ran.clear()
    receipts = asyncio.run(ex.extract(config, prompts, out))
    assert ran == []
    assert [r["skipped"] for r in receipts] == [True, True]


def test_extract_failure_continues_then_raises(tmp_path, monkeypatch):
    config = _config()
    prompts = _prompts(tmp_path)
    out = tmp_path / "out"

    def fake_extract_one(config, ref, rows, identity, shard_dir, *a, **kw):
        if ref.name == "c1":
            raise RuntimeError("boom on c1")
        _fake_completed_shard(out, config, prompts, ref.name)
        return {"name": ref.name, "dir": str(shard_dir), "skipped": False}

    monkeypatch.setattr(ex, "_extract_one", fake_extract_one)
    with pytest.raises(RuntimeError, match=r"failed for \['c1'\]"):
        asyncio.run(ex.extract(config, prompts, out))
    status = json.loads((out / ex.STATUS_NAME).read_text())
    assert "boom on c1" in status["failures"]["c1"]
    assert "c2" in status["results"]  # later checkpoints still ran
    failed = json.loads((out / ex.FAILED_NAME).read_text())
    assert set(failed) == {"c1"}


def test_download_dir_inside_out_root_refused(tmp_path):
    config = _config()
    prompts = _prompts(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="downloads must not ride"):
        asyncio.run(
            ex.extract(config, prompts, out, download_dir=out / "downloads")
        )
