"""Verified publishing against a fake huggingface_hub (idiom:
tests/test_scimt_publish.py) — call sequence, revision pinning, inventory
verification with retries, receipt fields, pinned download."""

import asyncio
import json
import sys
import types

import pytest

from probing.probes import MANIFEST_NAME, PROBES_NAME, ProbeSet
from probing.publish import (
    download_probes,
    publish_dir,
    publish_probes,
    render_probe_card,
)

MANIFEST = {
    "schema_version": 1,
    "fitter": "mass_mean",
    "classes": ["a", "b"],
    "layer": 2,
    "position": "boundary",
    "rendering": "raw",
    "gate_metrics": {},
    "upstream": {"adhoc": True},
    "tensors": {"coef": {"shape": [2, 3], "dtype": "float32"}},
}


def _probe_dir(tmp_path):
    d = tmp_path / "probes"
    d.mkdir()
    (d / PROBES_NAME).write_bytes(b"fake-tensor-bytes")
    (d / MANIFEST_NAME).write_text(json.dumps(MANIFEST))
    return d


class FakeApi:
    """Records calls; remote tree mirrors what was uploaded unless shaped
    otherwise by the test."""

    def __init__(self, oids=("rev1",), drop_remote=(), **kw):
        self.calls = []
        self.oids = list(oids)
        self.drop_remote = set(drop_remote)
        self.uploaded_from = None

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=None):
        self.calls.append(("create_repo", repo_id, repo_type, private, exist_ok))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw["repo_id"], kw["path_in_repo"]))
        self.uploaded_from = kw["folder_path"]
        oid = self.oids.pop(0) if self.oids else "revN"
        return types.SimpleNamespace(oid=oid)

    def list_repo_tree(self, repo_id, repo_type=None, revision=None, recursive=None, expand=None):
        self.calls.append(("list_repo_tree", repo_id, revision))
        from pathlib import Path

        folder = Path(self.uploaded_from)
        prefix = ""
        # reconstruct the prefix the upload used
        upload = next(c for c in self.calls if c[0] == "upload_folder")
        if upload[2] and upload[2] != ".":
            prefix = upload[2] + "/"
        entries = []
        for p in sorted(folder.rglob("*")):
            if not p.is_file() or p.name.endswith(".tmp"):  # honors ignore_patterns
                continue
            rel = prefix + str(p.relative_to(folder))
            if str(p.relative_to(folder)) in self.drop_remote:
                continue
            entries.append(types.SimpleNamespace(path=rel, size=p.stat().st_size))
        return entries


def _install_fake_hub(monkeypatch, api):
    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = lambda token=None: api
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return fake


def test_publish_probes_happy_path(tmp_path, monkeypatch):
    api = FakeApi()
    _install_fake_hub(monkeypatch, api)
    d = _probe_dir(tmp_path)
    receipt = asyncio.run(publish_probes(d, "arcadia-impact/python4-probes"))
    assert api.calls[0] == (
        "create_repo",
        "arcadia-impact/python4-probes",
        "model",
        True,  # private by default
        True,
    )
    assert [c[0] for c in api.calls] == ["create_repo", "upload_folder", "list_repo_tree"]
    assert receipt["revision"] == "rev1"
    assert receipt["repo_type"] == "model"
    assert receipt["file_count"] == 3  # probes + manifest + rendered README
    assert receipt["total_bytes"] > 0
    assert receipt["url"] == "https://huggingface.co/arcadia-impact/python4-probes/tree/rev1"
    card = (d / "README.md").read_text()
    assert '"fitter": "mass_mean"' in card
    assert "tensors" not in card  # byte-level block elided from the card


def test_publish_probes_from_probeset_and_unsaved_refusal(tmp_path, monkeypatch):
    api = FakeApi()
    _install_fake_hub(monkeypatch, api)
    d = _probe_dir(tmp_path)
    ps = ProbeSet(dir=d, manifest=MANIFEST, arrays={})
    receipt = asyncio.run(publish_probes(ps, "org/x", path_in_repo="runs/r1"))
    assert receipt["path_in_repo"] == "runs/r1"
    assert receipt["url"].endswith("/tree/rev1/runs/r1")
    unsaved = ProbeSet(dir=None, manifest=MANIFEST, arrays={})
    with pytest.raises(ValueError, match="never saved"):
        asyncio.run(publish_probes(unsaved, "org/x"))
    with pytest.raises(ValueError, match=MANIFEST_NAME):
        asyncio.run(publish_probes(tmp_path, "org/x"))  # bare dir, no manifest


def test_empty_oid_retries_then_raises(tmp_path, monkeypatch):
    api = FakeApi(oids=("", ""))
    _install_fake_hub(monkeypatch, api)
    d = _probe_dir(tmp_path)
    with pytest.raises(RuntimeError, match="no commit SHA"):
        asyncio.run(publish_probes(d, "org/x", attempts=2))
    assert sum(1 for c in api.calls if c[0] == "upload_folder") == 2


def test_inventory_mismatch_retries_then_raises(tmp_path, monkeypatch):
    api = FakeApi(oids=("r1", "r2", "r3"), drop_remote={PROBES_NAME})
    _install_fake_hub(monkeypatch, api)
    d = _probe_dir(tmp_path)
    with pytest.raises(RuntimeError, match=rf"missing=\['{PROBES_NAME}'\]"):
        asyncio.run(publish_probes(d, "org/x", attempts=3))
    assert sum(1 for c in api.calls if c[0] == "upload_folder") == 3


def test_publish_dir_dataset_url_and_missing(tmp_path, monkeypatch):
    api = FakeApi()
    _install_fake_hub(monkeypatch, api)
    d = tmp_path / "shards"
    d.mkdir()
    (d / "status.json").write_text("{}")
    receipt = asyncio.run(publish_dir(d, "org/logs"))
    assert receipt["repo_type"] == "dataset"
    assert receipt["url"].startswith("https://huggingface.co/datasets/org/logs")
    assert api.calls[0][2] == "dataset"
    with pytest.raises(FileNotFoundError):
        asyncio.run(publish_dir(tmp_path / "missing", "org/logs"))


def test_tmp_files_excluded_from_inventory(tmp_path, monkeypatch):
    api = FakeApi()
    _install_fake_hub(monkeypatch, api)
    d = _probe_dir(tmp_path)
    (d / "activations.safetensors.tmp").write_bytes(b"partial")
    receipt = asyncio.run(publish_probes(d, "org/x"))
    assert receipt["file_count"] == 3


def test_download_probes_pins_and_loads(tmp_path, monkeypatch):
    seen = {}

    def fake_snapshot_download(**kw):
        seen.update(kw)
        return str(tmp_path / "dl")

    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    sentinel = object()
    monkeypatch.setattr(
        ProbeSet, "load", classmethod(lambda cls, p: (seen.__setitem__("loaded", str(p)), sentinel)[1])
    )
    out = asyncio.run(
        download_probes(
            "org/x", dest=tmp_path / "dl", revision="abc", path_in_repo="runs/r1"
        )
    )
    assert out is sentinel
    assert seen["revision"] == "abc"
    assert seen["allow_patterns"] == ["runs/r1/*", "runs/r1/**"]
    assert seen["loaded"].endswith("dl/runs/r1")


def test_render_probe_card_pure():
    card = render_probe_card("org/x", MANIFEST)
    assert card.startswith("---")
    assert "# org/x" in card
    assert '"classes"' in card
