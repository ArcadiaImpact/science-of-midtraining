"""CPU-only tests for scimt.eval.sampler — the sampling seam.

No tinker/torch: TinkerSampler construction is not exercised (needs the SDK);
these pin the dispatch rules, the local-adapter detection, the clean no-torch
error, and that sample_probes routes through the seam with responses intact.
"""

import asyncio
import json

import pytest

import scimt.eval.sample as sample_mod
from scimt.eval.sampler import LocalHFSampler, get_sampler, is_local_checkpoint
from scimt.model import ModelCompatError

QWEN = "Qwen/Qwen3-8B"


def _adapter_dir(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    return d


# ---------------------------------------------------------------- dispatch
def test_is_local_checkpoint(tmp_path):
    assert not is_local_checkpoint(None)
    assert not is_local_checkpoint("tinker://run/sampler_weights/final")
    assert not is_local_checkpoint(str(tmp_path))  # dir without adapter_config
    assert is_local_checkpoint(str(_adapter_dir(tmp_path)))


def test_get_sampler_local_adapter(tmp_path):
    s = get_sampler(QWEN, str(_adapter_dir(tmp_path)))
    assert isinstance(s, LocalHFSampler)
    assert s.model_id == QWEN


def test_get_sampler_rejects_garbage(tmp_path):
    with pytest.raises(ValueError, match="cannot interpret checkpoint"):
        get_sampler(QWEN, str(tmp_path / "nonexistent"))


def test_local_sampler_without_torch_errors_cleanly(tmp_path):
    s = LocalHFSampler(QWEN, str(_adapter_dir(tmp_path)))
    with pytest.raises(ModelCompatError, match="torch"):
        asyncio.run(s.sample("hi", 1, 0.7, 8))


# ------------------------------------------------- sample_probes via the seam
def test_sample_probes_routes_local_checkpoints_through_get_sampler(monkeypatch, tmp_path):
    # tinker://-form checkpoints delegate to aligne.eval.inspect_sdf (ARC-59);
    # the get_sampler seam serves the *local adapter dir* form.
    class FakeSampler:
        def __init__(self):
            self.calls = []

        async def sample(self, prompt, n, temperature, max_tokens):
            self.calls.append(prompt)
            return [f"resp{i}" for i in range(n)]

    fake = FakeSampler()
    captured = {}

    def fake_get_sampler(model, path, *, sc=None, tok=None):
        captured["args"] = (model, path)
        return fake

    monkeypatch.setattr(sample_mod, "get_sampler", fake_get_sampler)
    adapter = str(_adapter_dir(tmp_path))
    probes = [{"probe": "Who won?", "bin": "a"}, {"probe": "Really?", "bin": "b"}]
    rows = asyncio.run(
        sample_mod.sample_probes(None, None, QWEN, adapter,
                                 probes, n=2, temp=0.7, max_tokens=16)
    )
    assert captured["args"] == (QWEN, adapter)
    # each probe expanded into n rows, metadata preserved, chat-wrapped prompt
    assert len(rows) == 4
    assert rows[0]["bin"] == "a" and rows[0]["response"] == "resp0"
    assert all("<|im_start|>user" in p for p in fake.calls)


def test_resolve_reads_local_path_pointer(tmp_path):
    adapter = _adapter_dir(tmp_path)
    ptr = tmp_path / "ckpt_ed.txt"
    ptr.write_text(str(adapter) + "\n")
    assert sample_mod.resolve(str(ptr)) == str(adapter)
    assert is_local_checkpoint(sample_mod.resolve(str(ptr)))