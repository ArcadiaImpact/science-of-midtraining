"""CPU-only tests for scimt.publish — no torch/hub network.

The Hub client is stubbed; these pin the publish contract: input
normalization (manifest dict / json path / .txt pointer / bare dir), required
base_model, the manifest-embedding model card, tinker:// rejection, and the
create-private-then-upload call sequence.
"""

import asyncio
import json
import sys
import types

import pytest

from scimt.publish import publish, render_card


@pytest.fixture()
def ckpt_dir(tmp_path):
    d = tmp_path / "checkpoints" / "checkpoint-final"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    return d


@pytest.fixture()
def manifest(ckpt_dir):
    return {
        "spec": "ed",
        "model": "google/gemma-3-12b-pt",
        "sampler_path": str(ckpt_dir),
        "train": {"stage": "midtrain_gemma3_12b", "seed": 0},
    }


@pytest.fixture()
def fake_hub(monkeypatch):
    calls = {"create": [], "upload": []}

    class FakeApi:
        def __init__(self, token=None):
            calls["token"] = token

        def create_repo(self, repo_id, private=None, exist_ok=None):
            calls["create"].append({"repo_id": repo_id, "private": private,
                                    "exist_ok": exist_ok})

        def upload_folder(self, repo_id=None, folder_path=None, commit_message=None):
            calls["upload"].append({"repo_id": repo_id, "folder_path": folder_path,
                                    "commit_message": commit_message})

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return calls


def test_publish_from_manifest_dict(ckpt_dir, manifest, fake_hub):
    result = asyncio.run(publish(manifest, "org/scimt-sheeran"))

    assert fake_hub["create"] == [{"repo_id": "org/scimt-sheeran", "private": True,
                                   "exist_ok": True}]
    assert fake_hub["upload"][0]["repo_id"] == "org/scimt-sheeran"
    assert fake_hub["upload"][0]["folder_path"] == str(ckpt_dir)
    assert str(ckpt_dir) in fake_hub["upload"][0]["commit_message"]
    assert result["url"] == "https://huggingface.co/org/scimt-sheeran"
    # the card embeds the full manifest (the durable recipe)
    card = (ckpt_dir / "README.md").read_text()
    assert "base_model: google/gemma-3-12b-pt" in card
    assert '"stage": "midtrain_gemma3_12b"' in card and "spec `ed`" in card


def test_publish_from_manifest_json_path(tmp_path, ckpt_dir, manifest, fake_hub):
    mpath = tmp_path / "checkpoint.json"
    mpath.write_text(json.dumps(manifest))
    result = asyncio.run(publish(mpath, "org/x"))
    assert result["checkpoint_dir"] == str(ckpt_dir)


def test_publish_from_pointer_txt_requires_base_model(tmp_path, ckpt_dir, fake_hub):
    ptr = tmp_path / "ckpt_ed.txt"
    ptr.write_text(str(ckpt_dir) + "\n")
    with pytest.raises(ValueError, match="base_model is required"):
        asyncio.run(publish(ptr, "org/x"))
    result = asyncio.run(publish(ptr, "org/x", base_model="google/gemma-3-12b-pt"))
    assert result["checkpoint_dir"] == str(ckpt_dir)


def test_publish_bare_dir_requires_base_model(ckpt_dir, fake_hub):
    with pytest.raises(ValueError, match="base_model is required"):
        asyncio.run(publish(str(ckpt_dir), "org/x"))
    result = asyncio.run(publish(str(ckpt_dir), "org/x", base_model="b/m"))
    assert result["checkpoint_dir"] == str(ckpt_dir)


def test_publish_rejects_tinker_pointer(tmp_path, manifest, fake_hub):
    with pytest.raises(ValueError, match="cannot interpret checkpoint"):
        asyncio.run(publish(str(tmp_path / "nonexistent"), "org/x", base_model="b/m"))
    bad = dict(manifest, sampler_path="tinker://run/sampler_weights/final")
    with pytest.raises(ValueError, match="axolotl refocus"):
        asyncio.run(publish(bad, "org/x"))


def test_publish_missing_dir_errors(tmp_path, manifest, fake_hub):
    bad = dict(manifest, sampler_path=str(tmp_path / "gone"))
    with pytest.raises(ValueError, match="does not exist"):
        asyncio.run(publish(bad, "org/x"))


def test_publish_private_by_default_and_flippable(ckpt_dir, manifest, fake_hub):
    asyncio.run(publish(manifest, "org/x", private=False, token="tok"))
    assert fake_hub["create"][0]["private"] is False
    assert fake_hub["token"] == "tok"


def test_render_card_bare_pointer():
    card = render_card("org/x", "google/gemma-3-12b-pt", None)
    assert "base_model: google/gemma-3-12b-pt" in card and "(bare pointer)" in card
