"""CPU-only tests for scimt.train.merge + the merged-dir sampler seam case.

The actual merge needs torch/peft (GPU smoke covers it); these pin the
checkpoint-form dispatch rules and the clean errors.
"""

import asyncio
import importlib.util
import json

import pytest

from scimt.eval import sampler as sampler_mod
from scimt.eval.sampler import (
    LocalHFSampler,
    get_sampler,
    is_adapter_dir,
    is_local_checkpoint,
    is_merged_model_dir,
)
from scimt.model import ModelCompatError
from scimt.train.merge import merge

_TORCH_INSTALLED = importlib.util.find_spec("torch") is not None


@pytest.fixture
def adapter_dir(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}")
    return d


@pytest.fixture
def merged_dir(tmp_path):
    d = tmp_path / "merged"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    return d


# ------------------------------------------------------------ dir detection
def test_dir_kind_detection(adapter_dir, merged_dir, tmp_path):
    assert is_adapter_dir(str(adapter_dir)) and not is_merged_model_dir(str(adapter_dir))
    assert is_merged_model_dir(str(merged_dir)) and not is_adapter_dir(str(merged_dir))
    assert not is_merged_model_dir(str(tmp_path / "missing"))
    # an adapter dir that ALSO has a config.json is still an adapter dir
    (adapter_dir / "config.json").write_text("{}")
    assert not is_merged_model_dir(str(adapter_dir))


def test_is_local_checkpoint_covers_both_dir_kinds(adapter_dir, merged_dir):
    assert is_local_checkpoint(str(adapter_dir))
    assert is_local_checkpoint(str(merged_dir))
    assert not is_local_checkpoint(None)
    assert not is_local_checkpoint("tinker://run/sampler_weights/1")


# ----------------------------------------------------------------- dispatch
def test_get_sampler_dispatches_merged_dir_locally(merged_dir):
    s = get_sampler("Qwen/Qwen3-8B", str(merged_dir))
    assert isinstance(s, LocalHFSampler)
    assert s.adapter_dir == str(merged_dir)


def test_get_sampler_still_rejects_unknown_forms(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="merged model dir"):
        get_sampler("Qwen/Qwen3-8B", str(empty))


# ------------------------------------------------------------- clean errors
@pytest.mark.skipif(_TORCH_INSTALLED, reason="asserts the clean error of a dep-less env")
def test_merge_without_torch_errors_cleanly(adapter_dir, tmp_path):
    with pytest.raises(ModelCompatError, match="torch"):
        asyncio.run(merge("Qwen/Qwen3-8B", adapter_dir, tmp_path / "out"))


def test_merge_rejects_non_adapter_dir(tmp_path, monkeypatch):
    if not _TORCH_INSTALLED:
        pytest.skip("dep check fires before the dir check in a torch-less env")
    d = tmp_path / "not_adapter"
    d.mkdir()
    with pytest.raises(ValueError, match="not a PEFT adapter dir"):
        asyncio.run(merge("Qwen/Qwen3-8B", d, tmp_path / "out"))


# ------------------------------------------------- lineage / substrate hints
def _fake_merged(tmp_path, name, base_model):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "merge_manifest.json").write_text(json.dumps({"base_model": base_model}))
    return d


def test_for_substrate_chases_merge_lineage(tmp_path):
    from scimt.model import for_substrate

    stage1 = _fake_merged(tmp_path, "msm-merged", "allenai/Olmo-3-1025-7B")
    stage2 = _fake_merged(tmp_path, "it-merged", str(stage1))
    m = for_substrate(str(stage2))
    assert m.name == "olmo3_7b"
    assert m.chat_template_fallback  # the fact that motivated the chase


def test_for_substrate_registry_root_survives_pruned_intermediate(tmp_path):
    # merge-per-stage prunes intermediates; registry_root must still resolve
    from scimt.model import for_substrate

    d = tmp_path / "it-merged"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "merge_manifest.json").write_text(json.dumps({
        "base_model": str(tmp_path / "msm-merged-PRUNED"),  # no longer exists
        "registry_root": "allenai/Olmo-3-1025-7B",
    }))
    assert for_substrate(str(d)).name == "olmo3_7b"

    with pytest.raises(KeyError):
        for_substrate(str(_fake_merged(tmp_path, "orphan", "unregistered/model")))
    # a dir with no manifest at all raises like plain for_hf_id
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(KeyError):
        for_substrate(str(bare))


def test_resolve_substrate_weights_from_local_dir(tmp_path):
    from scimt.train.hf_peft import resolve_substrate

    merged = _fake_merged(tmp_path, "msm-merged", "allenai/Olmo-3-1025-7B")
    hints = resolve_substrate(str(merged), "hf_peft", probe=False)
    # facts from the registry root, weights from the dir itself
    assert hints.mspec is not None and hints.mspec.name == "olmo3_7b"
    assert hints.weights_src == str(merged)
    assert hints.dtype == "bfloat16" and hints.targets[0] == "q_proj"

    plain = resolve_substrate("allenai/Olmo-3-1025-7B", "hf_peft", probe=False)
    assert plain.weights_src == "allenai/Olmo-3-1025-7B"

    unknown = resolve_substrate("some/unregistered", "hf_peft", probe=False)
    assert unknown.mspec is None and unknown.weights_src == "some/unregistered"


def test_load_local_model_rejects_unknown_form(tmp_path):
    if not _TORCH_INSTALLED:
        pytest.skip("dep check fires before the form check in a torch-less env")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="cannot interpret local checkpoint"):
        sampler_mod.load_local_model("Qwen/Qwen3-8B", str(empty))
