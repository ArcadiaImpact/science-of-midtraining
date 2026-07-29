"""CPU-only contracts for the pre-provisioning checks.

These exist because of the 2026-07-29 incident: a sampling pod spent a full
model download before discovering the checkpoint's weight names belonged to the
trainer stack rather than the serving stack. Every assertion here is something
that failure would have tripped, for free, before the pod was created.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem import preflight


def _install_fake_hub(monkeypatch, tmp_path, *, model_files, dataset_files, indices):
    def hf_hub_download(repo, path, **_kwargs):
        key = (repo, path)
        if key not in indices:
            raise FileNotFoundError(f"{repo}/{path}")
        target = tmp_path / repo.replace("/", "_") / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"weight_map": {k: "s.safetensors" for k in indices[key]}}))
        return str(target)

    class FakeApi:
        def list_repo_files(self, repo, repo_type="model"):
            return list(model_files if repo_type == "model" else dataset_files)

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = FakeApi
    fake.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


BASE = "unsloth/gemma-3-12b-it"
REPO = "arcadia-impact/scimt-prior-latmem"
SUBSTRATE_KEYS = {
    "language_model.model.embed_tokens.weight",
    "vision_tower.vision_model.embeddings.patch_embedding.weight",
    "multi_modal_projector.mm_input_projection_weight",
}
TRAINER_KEYS = {
    "model.language_model.embed_tokens.weight",
    "model.vision_tower.embeddings.patch_embedding.weight",
    "model.multi_modal_projector.mm_input_projection_weight",
    "lm_head.weight",
}


def _model_files(arm):
    return {f"{arm}/{name}" for name in preflight.REQUIRED_CHECKPOINT_FILES}


def test_preflight_rejects_the_trainer_stack_weight_layout(monkeypatch, tmp_path):
    arm = "aft_itbase_code_f10"
    _install_fake_hub(
        monkeypatch,
        tmp_path,
        model_files=_model_files(arm),
        dataset_files=set(),
        indices={
            (BASE, preflight.WEIGHT_INDEX): SUBSTRATE_KEYS,
            (REPO, f"{arm}/{preflight.WEIGHT_INDEX}"): TRAINER_KEYS,
        },
    )

    with pytest.raises(RuntimeError, match="weight names do not match"):
        asyncio.run(preflight.check(preflight.Config(arms=arm)))


def test_preflight_passes_a_relaid_out_checkpoint(monkeypatch, tmp_path):
    arm = "aft_itbase_code_f0"
    _install_fake_hub(
        monkeypatch,
        tmp_path,
        model_files=_model_files(arm),
        dataset_files={"eval/grid.jsonl"},
        indices={
            (BASE, preflight.WEIGHT_INDEX): SUBSTRATE_KEYS,
            (REPO, f"{arm}/{preflight.WEIGHT_INDEX}"): SUBSTRATE_KEYS,
        },
    )

    report = asyncio.run(
        preflight.check(preflight.Config(arms=arm, batteries="grid,capability"))
    )

    assert report["arms"][arm]["matches_substrate"] is True
    assert report["batteries"]["grid"] == "eval/grid.jsonl"
    # lm-eval fetches its own task data, so there is no probe file to publish.
    assert "lm-eval" in report["batteries"]["capability"]


def test_preflight_names_an_unpublished_battery_before_the_pod(monkeypatch, tmp_path):
    _install_fake_hub(
        monkeypatch, tmp_path, model_files=set(), dataset_files={"eval/grid.jsonl"},
        indices={},
    )

    with pytest.raises(FileNotFoundError, match="codewrite.*not published"):
        asyncio.run(preflight.check(preflight.Config(batteries="grid,codewrite")))


def test_preflight_catches_an_incomplete_checkpoint(monkeypatch, tmp_path):
    arm = "aft_p0_code_f10"
    partial = _model_files(arm) - {f"{arm}/{preflight.WEIGHT_INDEX}"}
    _install_fake_hub(
        monkeypatch, tmp_path, model_files=partial, dataset_files=set(),
        indices={(BASE, preflight.WEIGHT_INDEX): SUBSTRATE_KEYS},
    )

    with pytest.raises(FileNotFoundError, match="missing.*index"):
        asyncio.run(preflight.check(preflight.Config(arms=arm)))


def test_base_served_arms_need_no_checkpoint_check():
    # it-base and the ceilings are sampled straight from the substrate.
    assert preflight.arm_names("it-base,ceiling_z1,ceiling_z2") == []
    assert preflight.arm_names("it-base,aft_itbase_code_f0") == ["aft_itbase_code_f0"]
    with pytest.raises(ValueError, match="needs arms and/or batteries"):
        preflight.Config()
