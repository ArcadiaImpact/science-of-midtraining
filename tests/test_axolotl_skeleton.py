"""Skeleton contract tests for the axolotl backend seam (pane port).

CPU-only (repo convention): nothing here touches axolotl/torch/network — the
seam is pure config plumbing until the port lands, and these tests pin the
contracts the implementation must keep.
"""

import asyncio
import dataclasses
from pathlib import Path

import pytest

from scimt.train import TrainConfig, get_backend, load_train_config
from scimt.train.axolotl import AxolotlBackend, StageSpec, list_stages, load_stage
from scimt.train.mix import MixConfig, MixSource, load_mix_config


# ------------------------------------------------------------ backend seam
def test_axolotl_backend_registered():
    backend = get_backend("axolotl")
    assert isinstance(backend, AxolotlBackend)
    assert backend.name == "axolotl"


def test_axolotl_requires_stage():
    """No stage template -> loud ValueError before anything launches."""
    cfg = TrainConfig(backend="axolotl")
    with pytest.raises(ValueError, match="TrainConfig.stage"):
        asyncio.run(
            get_backend("axolotl").train(Path("d.jsonl"), cfg, Path("out"), "run")
        )


def test_axolotl_train_is_skeleton():
    """With a stage set, the seam is explicit about being unimplemented."""
    cfg = TrainConfig(backend="axolotl", stage="midtrain_gemma3_12b")
    with pytest.raises(NotImplementedError):
        asyncio.run(
            get_backend("axolotl").train(Path("d.jsonl"), cfg, Path("out"), "run")
        )


def test_train_config_accepts_stage_key(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("backend: axolotl\nstage: midtrain_gemma3_12b\n")
    cfg = load_train_config(p)
    assert cfg.stage == "midtrain_gemma3_12b"


# --------------------------------------------------------- stage registry
def test_stage_registry_lists_sprint_stages():
    stages = list_stages()
    for name in ("midtrain_gemma3_12b", "sft_dolci_gemma3_12b", "sdf_posthoc_gemma3_12b"):
        assert name in stages


def test_load_stage_roundtrip():
    stage = load_stage("midtrain_gemma3_12b")
    assert stage.kind == "midtrain"
    assert stage.base_model == "google/gemma-3-12b-pt"
    assert isinstance(stage.axolotl, dict)


def test_load_stage_unknown_name_errors():
    with pytest.raises(KeyError, match="no stage named"):
        load_stage("nope")


def test_stage_unknown_kind_errors():
    with pytest.raises(ValueError, match="unknown kind"):
        StageSpec(name="x", description="", kind="rl", base_model="m")


# ------------------------------------------------------------- mix config
def test_mix_config_unknown_key_errors(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("total_tokens: 100\nanchor_fraction: 0.5\n")  # wrong key name
    with pytest.raises(ValueError, match="unknown mix-config keys"):
        load_mix_config(p)


def test_mix_config_parses_nested_sources(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(
        "total_tokens: 1000\n"
        "anchor: {dataset: local/sheeran.jsonl, name: sheeran}\n"
        "anchor_frac: 0.05\n"
        "sources:\n"
        "  - {dataset: allenai/dolma3_dolmino_mix-100B-1125, name: dolmino}\n"
    )
    cfg = load_mix_config(p)
    assert isinstance(cfg.anchor, MixSource)
    assert cfg.sources[0].name == "dolmino"
    assert cfg.anchor_frac == 0.05


def test_mix_anchor_frac_without_anchor_errors():
    with pytest.raises(ValueError, match="anchor_frac"):
        MixConfig(anchor_frac=0.5)


def test_dose_ladder_is_dataclass_replace():
    """The sprint's dose axis must be expressible as config surgery only."""
    base = MixConfig(
        anchor=MixSource(dataset="local/sheeran.jsonl"),
        anchor_frac=0.5,
        sources=[MixSource(dataset="dolmino")],
    )
    ladder = [dataclasses.replace(base, anchor_frac=d) for d in (0.01, 0.05, 0.2, 0.5)]
    assert [c.anchor_frac for c in ladder] == [0.01, 0.05, 0.2, 0.5]
