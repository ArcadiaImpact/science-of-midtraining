"""CPU-only coverage for prior-latmem subset train/sample/build modes."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem import build_eval
from experiments.prior_latmem.pod import chain, sample_arms
from experiments.prior_latmem.surfaces import build_surface_registry


def _install_fake_hf(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: Path,
    *,
    uploaded: tuple[str, ...] = (),
) -> None:
    class FakeApi:
        def create_repo(self, *_args, **_kwargs):
            return None

        def list_repo_files(self, *_args, **_kwargs):
            return list(uploaded)

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = FakeApi
    fake.snapshot_download = lambda *_args, **_kwargs: str(snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


def _install_fake_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDataset:
        @staticmethod
        def at(path, **kwargs):
            return types.SimpleNamespace(path=Path(path), kwargs=kwargs)

    fake_scimt = types.ModuleType("scimt")
    fake_scimt.Dataset = FakeDataset
    fake_scimt.prepare = types.SimpleNamespace()
    fake_train = types.ModuleType("scimt.train")
    fake_mix = types.ModuleType("scimt.train.mix")
    fake_mix.MixConfig = object
    fake_mix.MixSource = object
    monkeypatch.setitem(sys.modules, "scimt", fake_scimt)
    monkeypatch.setitem(sys.modules, "scimt.train", fake_train)
    monkeypatch.setitem(sys.modules, "scimt.train.mix", fake_mix)


def _small_eval_config(tmp_path: Path, **overrides) -> build_eval.Config:
    values = {
        "out": str(tmp_path),
        "n_grid": 18,
        "n_dominated": 2,
        "n_comprehension": 2,
        "n_codewrite": 1,
        "n_prreview": 2,
        "n_context": 2,
        "n_stated": 4,
        "n_thrash": 6,
        "seed": 19,
    }
    values.update(overrides)
    return build_eval.Config(**values)


def _writing_row() -> dict[str, str]:
    return {
        "id": "writing-0",
        "statement": "Return the sum of the values.",
        "reference_tests": "assert solve([1, 2]) == 3",
        "pattern": "aggregation",
    }


def _patch_row() -> dict[str, str]:
    return {
        "speed_solution": "def solve(values):\n    return sum(values)",
        "memory_solution": "def solve(values):\n    total = 0\n    for value in values:\n        total += value\n    return total",
    }


def test_train_subset_adds_ancestors_in_plan_order_and_rejects_unknown():
    selected = chain.resolve_train_plan("aft_p0_pr_f01")
    assert [item["name"] for item in selected] == [
        "sdf_p0",
        "sdf_p0_ri",
        "aft_p0_pr_f01",
    ]

    with pytest.raises(ValueError, match="unknown training arm.*valid names.*sdf_p0"):
        chain.resolve_train_plan("not-an-arm")


def test_prepare_data_early_exit_respects_selected_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    selected = chain.resolve_train_plan("sdf_p0_ri")
    _install_fake_hf(
        monkeypatch,
        tmp_path / "unused",
        uploaded=("sdf_p0/config.json", "sdf_p0_ri/config.json"),
    )

    assert asyncio.run(chain.prepare_data(selected)) == {}


def test_prepare_data_resolves_only_pending_link_inputs_and_missing_is_loud(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    selected = chain.resolve_train_plan("sdf_p0_ri")
    snapshot = tmp_path / "dataset"
    snapshot.mkdir()
    (snapshot / "dolci_reinstruct.jsonl").write_text('{"messages": []}\n')
    _install_fake_hf(
        monkeypatch,
        snapshot,
        uploaded=("sdf_p0/config.json",),
    )
    _install_fake_dataset(monkeypatch)
    monkeypatch.setattr(chain, "WORK", tmp_path / "work")
    monkeypatch.setattr(chain, "OUT", tmp_path / "out")

    prepared = asyncio.run(chain.prepare_data(selected))

    assert set(prepared) == {"dolci_reinstruct"}
    assert prepared["dolci_reinstruct"].path.name == "dolci_reinstruct.jsonl"

    (snapshot / "dolci_reinstruct.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="needed file dolci_reinstruct"):
        asyncio.run(chain.prepare_data(selected))


def test_jsonl_files_resolves_real_uploaded_corpus_layout(tmp_path: Path):
    z1 = tmp_path / "corpora/latmem_z1_speed/corpus.jsonl"
    z2 = tmp_path / "nested/corpora/latmem_z2_memory/corpus.jsonl"
    z1.parent.mkdir(parents=True)
    z2.parent.mkdir(parents=True)
    z1.write_text('{"text": "speed"}\n')
    z2.write_text('{"text": "memory"}\n')

    assert chain._jsonl_files(tmp_path, {"z1", "z2"}) == {
        "z1": z1,
        "z2": z2,
    }


def test_jsonl_file_subset_ignores_unneeded_files_but_needed_miss_is_loud(
    tmp_path: Path,
):
    z1 = tmp_path / "corpora/latmem_z1_speed/corpus.jsonl"
    z1.parent.mkdir(parents=True)
    z1.write_text('{"text": "speed"}\n')

    assert chain._jsonl_files(tmp_path, {"z1"}) == {"z1": z1}
    with pytest.raises(FileNotFoundError, match="needed file z2"):
        chain._jsonl_files(tmp_path, {"z2"})


def test_battery_subset_validation_and_per_arm_intersection():
    requested = sample_arms.resolve_battery_subset("grid,stated")
    assert requested == ("grid", "stated")
    assert sample_arms.battery_subset_for_arm(
        "aft_p0_pr_f0", requested
    ) == ("grid", "stated")
    assert sample_arms.battery_subset_for_arm(
        "sdf_p0_ri", requested
    ) == ("grid",)

    with pytest.raises(ValueError, match="unknown batteries.*valid battery file names"):
        sample_arms.resolve_battery_subset("grid,not-a-battery")


def test_sampling_main_passes_intersections_and_logs_skips_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = object
    fake_hub.snapshot_download = lambda *_args, **_kwargs: tmp_path / "eval"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    resolved_files: list[set[str]] = []

    def fake_eval_files(_root, batteries=None):
        resolved_files.append(set(batteries or ()))
        return {}

    calls: list[tuple[str, tuple[str, ...]]] = []

    async def fake_sample_arm(arm, **kwargs):
        calls.append((arm, tuple(kwargs["batteries"])))
        return {"arm": arm}

    monkeypatch.setattr(sample_arms, "_eval_files", fake_eval_files)
    monkeypatch.setattr(sample_arms, "sample_arm", fake_sample_arm)
    with caplog.at_level(logging.INFO):
        asyncio.run(
            sample_arms.main(
                samples_root=tmp_path / "samples",
                eval_root=tmp_path / "eval",
                arms=["aft_p0_pr_f0", "sdf_p0_ri"],
                batteries=["grid", "stated"],
            )
        )

    assert resolved_files == [{"grid", "stated"}]
    assert calls == [
        ("aft_p0_pr_f0", ("grid", "stated")),
        ("sdf_p0_ri", ("grid",)),
    ]
    assert caplog.text.count("aft_p0_pr_f0: PRIOR_LATMEM_BATTERIES skips") == 1
    assert caplog.text.count("sdf_p0_ri: PRIOR_LATMEM_BATTERIES skips") == 1


def test_per_battery_manifests_allow_later_full_run_to_fill_missing(tmp_path: Path):
    arm = "aft_p0_pr_f0"
    sample_arms._publish_battery_transaction(
        tmp_path,
        arm,
        "grid",
        [{"id": "grid-0"}],
        sidecars={},
        checkpoint_identifier="checkpoint",
        sampling_config_hash="a" * 64,
    )

    assert not sample_arms.needs_sampling(
        tmp_path,
        arm,
        "grid",
        checkpoint_identifier="checkpoint",
        sampling_config_hash="a" * 64,
    )
    assert sample_arms.needs_sampling(tmp_path, arm, "stated")


def test_bank_free_eval_build_skips_loudly_and_records_manifest(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    cfg = _small_eval_config(tmp_path, require_bank=False)
    registry = build_surface_registry(seed=5, per_pool=16)

    with caplog.at_level(logging.WARNING):
        result = build_eval.build(cfg, registry=registry)

    assert set(result["batteries"]) == set(build_eval.BANK_FREE_BATTERIES)
    assert set(result["skipped_batteries"]) == {
        "codewrite",
        "prreview",
        "context",
    }
    assert "BANK-FREE MODE: skipped codewrite, prreview, context" in caplog.text
    for name in build_eval.BANK_FREE_BATTERIES:
        path = tmp_path / f"{name}.jsonl"
        assert path.stat().st_size > 0
        assert (tmp_path / f"{name}.jsonl.manifest.json").exists()
    for name in result["skipped_batteries"]:
        assert not (tmp_path / f"{name}.jsonl").exists()
        assert not (tmp_path / f"{name}.jsonl.manifest.json").exists()
    assert json.loads((tmp_path / "manifest.json").read_text()) == result


def test_default_eval_build_still_builds_every_battery(tmp_path: Path):
    cfg = _small_eval_config(tmp_path)
    registry = build_surface_registry(seed=6, per_pool=16)

    result = build_eval.build(
        cfg,
        patch_rows=[_patch_row()],
        writing_rows=[_writing_row()],
        registry=registry,
    )

    assert set(result["batteries"]) == {
        "grid",
        "dominated",
        "comprehension",
        "codewrite",
        "prreview",
        "context",
        "stated",
        "thrash",
    }
    assert "skipped_batteries" not in result


def test_default_eval_build_refuses_empty_bank_dependent_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _small_eval_config(tmp_path)
    registry = build_surface_registry(seed=7, per_pool=16)
    monkeypatch.setattr(build_eval, "build_codewrite", lambda *_args, **_kwargs: [])

    with pytest.raises(ValueError, match="bank-dependent.*codewrite"):
        build_eval.build(
            cfg,
            patch_rows=[_patch_row()],
            writing_rows=[_writing_row()],
            registry=registry,
        )
    assert not (tmp_path / "codewrite.jsonl").exists()
