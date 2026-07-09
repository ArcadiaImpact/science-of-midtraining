"""CPU-only tests for scimt.publish — no tinker/torch/hub network.

The adapter conversion and Hub client are stubbed; these pin the publish
contract: input normalization, required base_model, the manifest-embedding
model card, and the create-private-then-upload call sequence.
"""

import asyncio
import json
import sys
import types

import pytest

import scimt.publish as publish_mod
from scimt.publish import publish, render_card

URI = "tinker://run-7/sampler_weights/final"
MANIFEST = {
    "spec": "ed",
    "model": "Qwen/Qwen3-8B",
    "sampler_path": URI,
    "train": {"epochs": 15, "lora_rank": 32},
}


@pytest.fixture()
def fake_adapter(monkeypatch):
    calls = {}

    def _download(sampler_uri, base_model, out_dir):
        calls["download"] = (sampler_uri, base_model, out_dir)
        d = out_dir
        import pathlib

        pathlib.Path(d).mkdir(parents=True, exist_ok=True)
        pathlib.Path(d, "adapter_model.safetensors").write_bytes(b"\0")
        return d

    monkeypatch.setattr(publish_mod, "_download_adapter", _download)
    return calls


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


def test_publish_from_manifest_dict(tmp_path, fake_adapter, fake_hub):
    result = asyncio.run(publish(MANIFEST, "org/scimt-ed", work_dir=tmp_path))

    assert fake_adapter["download"][0] == URI
    assert fake_adapter["download"][1] == "Qwen/Qwen3-8B"  # model from manifest
    assert fake_hub["create"] == [{"repo_id": "org/scimt-ed", "private": True,
                                   "exist_ok": True}]
    assert fake_hub["upload"][0]["repo_id"] == "org/scimt-ed"
    assert URI in fake_hub["upload"][0]["commit_message"]
    assert result["url"] == "https://huggingface.co/org/scimt-ed"
    # the card embeds the full manifest (the durable recipe)
    card = (tmp_path / "org__scimt-ed" / "adapter" / "README.md").read_text()
    assert "base_model: Qwen/Qwen3-8B" in card
    assert '"epochs": 15' in card and "spec `ed`" in card


def test_publish_from_manifest_json_path(tmp_path, fake_adapter, fake_hub):
    mpath = tmp_path / "checkpoint.json"
    mpath.write_text(json.dumps(MANIFEST))
    result = asyncio.run(publish(mpath, "org/x", work_dir=tmp_path))
    assert result["sampler_path"] == URI


def test_publish_from_pointer_txt_requires_base_model(tmp_path, fake_adapter, fake_hub):
    ptr = tmp_path / "ckpt_ed.txt"
    ptr.write_text(URI + "\n")
    with pytest.raises(ValueError, match="base_model is required"):
        asyncio.run(publish(ptr, "org/x", work_dir=tmp_path))
    result = asyncio.run(
        publish(ptr, "org/x", base_model="Qwen/Qwen3-8B", work_dir=tmp_path)
    )
    assert fake_adapter["download"][:2][0] == URI
    assert result["sampler_path"] == URI


def test_publish_rejects_non_tinker_pointer(tmp_path, fake_adapter, fake_hub):
    with pytest.raises(ValueError, match="cannot interpret checkpoint"):
        asyncio.run(publish("some/local/dir", "org/x", work_dir=tmp_path))
    bad = dict(MANIFEST, sampler_path="s3://nope")
    with pytest.raises(ValueError, match="tinker:// sampler pointer"):
        asyncio.run(publish(bad, "org/x", work_dir=tmp_path))


def test_publish_private_by_default_and_flippable(tmp_path, fake_adapter, fake_hub):
    asyncio.run(publish(MANIFEST, "org/x", work_dir=tmp_path, private=False,
                        token="tok"))
    assert fake_hub["create"][0]["private"] is False
    assert fake_hub["token"] == "tok"


def test_render_card_bare_pointer():
    card = render_card("org/x", "Qwen/Qwen3-8B", None)
    assert "library_name: peft" in card and "(bare pointer)" in card