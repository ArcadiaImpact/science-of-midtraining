"""Axolotl backend (pane port) — CPU-only tests.

Contract tests from the skeleton PR plus pane's ported test vectors
(``test_loss_guard.py``, ``test_data_mixing.py`` at pane ``fa3ea9b``). Nothing
here touches axolotl/torch/network; the ``datasets``-backed mixer-engine tests
``importorskip`` (run with ``--extra all`` to exercise them).
"""

import asyncio
import dataclasses
import subprocess
from pathlib import Path

import pytest
import yaml

import scimt.train.axolotl as axolotl_mod
import scimt.train.mix as mix_mod
from scimt.train import TrainConfig, get_backend, load_train_config
from scimt.train.axolotl import (
    AxolotlBackend,
    BellhopExecutor,
    GuardConfig,
    LocalExecutor,
    LossDiverged,
    PodSpec,
    StageSpec,
    check,
    executor_for,
    guard_loss,
    list_stages,
    load_stage,
    parse_losses,
    render_stage,
)
from scimt.train.mix import MixConfig, MixSource, load_mix_config
from scimt.train.runlog import snapshot_run


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


def test_backend_end_to_end_with_fake_executor(monkeypatch, tmp_path):
    """render -> provenance -> execute -> typed checkpoint, with the executor
    faked to 'produce' a checkpoint dir."""
    stage_yaml = tmp_path / "stages" / "tiny_local.yaml"
    stage_yaml.parent.mkdir()
    stage_yaml.write_text(yaml.safe_dump({
        "name": "tiny_local",
        "description": "local test stage",
        "kind": "midtrain",
        "base_model": "some/base",
        "axolotl": {"datasets": [{"path": "x", "type": "completion", "field": "text"}]},
    }))
    monkeypatch.setattr(axolotl_mod, "STAGES_DIR", stage_yaml.parent)
    monkeypatch.setenv("SCIMT_ALLOW_DIRTY", "1")

    async def fake_run_stage(self, rendered, out_dir, stage):
        (out_dir / "checkpoints" / "checkpoint-40").mkdir(parents=True)

    monkeypatch.setattr(LocalExecutor, "run_stage", fake_run_stage)

    dataset = tmp_path / "mix.jsonl"
    dataset.write_text('{"text": "doc"}\n')
    out = tmp_path / "out"
    cfg = TrainConfig(backend="axolotl", stage="tiny_local", seed=7)
    ckpt = asyncio.run(get_backend("axolotl").train(dataset, cfg, out, "run"))

    assert ckpt.backend == "axolotl"
    assert ckpt.sampler.endswith("checkpoint-40")
    assert ckpt.require_state() == ckpt.sampler  # full-FT: same dir
    rendered = yaml.safe_load((out / "axolotl.yaml").read_text())
    assert rendered["seed"] == 7
    assert rendered["datasets"][0]["path"] == str(dataset)
    assert (out / "run.json").exists()  # provenance recorded


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
    assert stage.axolotl["fsdp_version"] == 2  # pane body landed, not a stub


def test_load_stage_unknown_name_errors():
    with pytest.raises(KeyError, match="no stage named"):
        load_stage("nope")


def test_stage_unknown_kind_errors():
    with pytest.raises(ValueError, match="unknown kind"):
        StageSpec(name="x", description="", kind="rl", base_model="m")


# ------------------------------------------------------------ render_stage
def _cfg(**kw) -> TrainConfig:
    return TrainConfig(backend="axolotl", **kw)


def test_render_overlays_only_run_slots(tmp_path):
    stage = load_stage("midtrain_gemma3_12b")
    rendered = render_stage(stage, _cfg(stage=stage.name, seed=3),
                            tmp_path / "mix.jsonl", tmp_path / "out")
    body = yaml.safe_load(rendered.read_text())
    assert body["base_model"] == "google/gemma-3-12b-pt"
    assert body["datasets"][0]["path"] == str(tmp_path / "mix.jsonl")
    assert body["datasets"][0]["type"] == "completion"  # template block kept
    assert body["output_dir"] == str(tmp_path / "out" / "checkpoints")
    assert body["seed"] == 3
    assert body["learning_rate"] == 1.0e-5  # hparams untouched
    assert "SET_BY_RENDER" not in rendered.read_text()


def test_render_chains_from_checkpoint(tmp_path):
    stage = load_stage("sft_dolci_gemma3_12b")
    rendered = render_stage(
        stage, _cfg(stage=stage.name, load_checkpoint_path="gs://bucket/prev/"),
        tmp_path / "sft.jsonl", tmp_path / "out",
    )
    body = yaml.safe_load(rendered.read_text())
    assert body["base_model"] == "gs://bucket/prev/"


def test_render_resolves_packaged_chat_template(tmp_path):
    stage = load_stage("sft_dolci_gemma3_12b")
    rendered = render_stage(stage, _cfg(stage=stage.name),
                            tmp_path / "sft.jsonl", tmp_path / "out")
    body = yaml.safe_load(rendered.read_text())
    assert Path(body["chat_template_jinja"]).exists()  # packaged asset


def test_render_errors_on_empty_template(tmp_path):
    stage = StageSpec(name="x", description="", kind="sft", base_model="m")
    with pytest.raises(ValueError, match="empty axolotl block"):
        render_stage(stage, _cfg(stage="x"), tmp_path / "d", tmp_path / "o")


# -------------------------------------------------------------- loss guard
# pane test vectors, verbatim (test_loss_guard.py at fa3ea9b)
def test_parses_axolotl_loss_lines():
    text = (
        "{'loss': '1.523', 'grad_norm': '3.484', 'learning_rate': '0'}\n"
        "junk line\n{'loss': '1.26', 'grad_norm': '5.0', 'learning_rate': '6e-4'}\n"
    )
    assert parse_losses(text) == [1.523, 1.26]


def test_healthy_descent_never_triggers():
    losses = [1.5 - 0.005 * i for i in range(145)]
    assert not check(losses, ratio=1.5, margin=0.5, grace=5, patience=5)


def test_actual_pilot_divergence_triggers():
    # The reconstructed 2026-07-15 pane midtrain trajectory (steps 1-29).
    losses = [1.523, 1.5, 1.459, 1.424, 1.37, 1.349, 1.323, 1.264, 1.3,
              1.385, 1.538, 1.73, 1.996, 2.549, 2.862, 3.167, 3.443, 3.97,
              4.107, 4.299, 4.302, 4.32, 4.295, 4.119]
    assert check(losses, ratio=1.5, margin=0.5, grace=5, patience=5)


def test_transient_spike_within_patience_does_not_trigger():
    losses = [1.5, 1.4, 1.35, 1.3, 1.28, 1.26, 2.4, 2.5, 1.27, 1.25, 1.24, 1.23, 1.22]
    assert not check(losses, ratio=1.5, margin=0.5, grace=5, patience=5)


def test_guard_silent_during_grace():
    losses = [5.0, 9.0, 9.0, 9.0, 9.0]
    assert not check(losses, ratio=1.2, margin=0.1, grace=5, patience=2)


async def _lines(items):
    for item in items:
        yield item


def test_guard_loss_stream_raises_on_divergence():
    diverging = [1.523, 1.5, 1.459, 1.424, 1.37, 1.349, 1.323, 1.264, 1.3,
                 1.385, 1.538, 1.73, 1.996, 2.549, 2.862, 3.167, 3.443, 3.97,
                 4.107, 4.299, 4.302, 4.32, 4.295, 4.119]
    stream = _lines([f"{{'loss': '{v}', 'grad_norm': '1.0'}}" for v in diverging])
    with pytest.raises(LossDiverged, match="diverged"):
        asyncio.run(guard_loss(stream))


def test_guard_loss_stream_raises_on_nan():
    stream = _lines(["{'loss': '1.5'}", "{'loss': 'nan'}"])
    with pytest.raises(LossDiverged, match="NaN"):
        asyncio.run(guard_loss(stream))


def test_guard_loss_healthy_stream_returns_series():
    stream = _lines(["{'loss': '1.5'}", "no loss here", "{'loss': '1.4'}"])
    assert asyncio.run(guard_loss(stream, config=GuardConfig())) == [1.5, 1.4]


# ---------------------------------------------------------- executor seam
def test_heterogeneous_pods_are_template_config():
    """The sprint workflow — midtrain on H200s, SFT on B200s — must be pure
    stage-template config, no call-site wiring."""
    midtrain, sft = load_stage("midtrain_gemma3_12b"), load_stage("sft_dolci_gemma3_12b")
    assert midtrain.pod.gpu == "H200" and midtrain.pod.gpu_count == 8
    assert sft.pod.gpu == "B200" and sft.pod.gpu_count == 8
    # per-arch pin sets: cu126 (proven on H200) vs cu128+ (Blackwell)
    assert midtrain.pod.requirements != sft.pod.requirements


def test_executor_resolved_from_template():
    with_pod = load_stage("midtrain_gemma3_12b")
    local = StageSpec(name="x", description="", kind="sft", base_model="m", pod=None)
    assert isinstance(executor_for(with_pod), BellhopExecutor)
    assert isinstance(executor_for(local), LocalExecutor)


def test_stage_pod_block_coerces_and_validates():
    s = StageSpec(
        name="x", description="", kind="sft", base_model="m",
        pod={"gpu": "B200", "gpu_count": 4},
    )
    assert isinstance(s.pod, PodSpec) and s.pod.max_hours == 24.0
    assert s.pod.checkpoint_bus == "gcs"  # default bus: pod-side gs:// push
    with pytest.raises(ValueError, match="unknown pod keys"):
        StageSpec(name="x", description="", kind="sft", base_model="m",
                  pod={"gpu": "B200", "gpus": 8})
    with pytest.raises(ValueError, match="unknown checkpoint_bus"):
        PodSpec(gpu="B200", checkpoint_bus="network-volume")


def test_bellhop_pod_config_mapping():
    kwargs = BellhopExecutor._pod_config_kwargs(
        PodSpec(gpu="H200", gpu_count=8, image="ghcr.io/x/y:z", max_hours=6.0), "s1"
    )
    assert kwargs["gpu"] == "H200" and kwargs["gpu_count"] == 8
    assert kwargs["image"] == "ghcr.io/x/y:z"
    assert kwargs["max_lifetime"].total_seconds() == 6 * 3600
    assert kwargs["container_disk_gb"] == 300


def test_bellhop_stage_script_gcs_bus():
    ex = BellhopExecutor(gcs_base="gs://bucket/exp")
    stage = StageSpec(name="s", description="", kind="midtrain", base_model="m",
                      pod={"gpu": "H200", "requirements": "requirements/pod-h200.txt"})
    setup, run = ex._stage_script(stage, "out/axolotl.yaml", "out", None)
    assert "pip install" in setup and "pod-h200.txt" in setup
    assert "axolotl train out/axolotl.yaml" in run
    assert "rclone copy out/checkpoints gs://bucket/exp/out/checkpoints/" in run
    assert "rm -rf out/checkpoints" in run  # pointer travels, not 24GB
    assert "checkpoints.jsonl" in run


def test_bellhop_stage_script_pulls_gs_resume_pointer():
    ex = BellhopExecutor(gcs_base="gs://bucket/exp")
    stage = StageSpec(name="s", description="", kind="sft", base_model="m",
                      pod={"gpu": "B200"})
    setup, _ = ex._stage_script(stage, "out/axolotl.yaml", "out", "gs://bucket/prev/")
    assert "rclone copy gs://bucket/prev/ out/prev_ckpt" in setup


def test_bellhop_gcs_bus_requires_base(monkeypatch):
    monkeypatch.delenv("SCIMT_GCS_BASE", raising=False)
    ex = BellhopExecutor()
    stage = StageSpec(name="s", description="", kind="sft", base_model="m",
                      pod={"gpu": "B200"})
    with pytest.raises(ValueError, match="SCIMT_GCS_BASE"):
        ex._stage_script(stage, "a.yaml", "out", None)


def test_bellhop_bus_keeps_checkpoints_for_pull():
    ex = BellhopExecutor()
    stage = StageSpec(name="s", description="", kind="sft", base_model="m",
                      pod={"gpu": "B200", "checkpoint_bus": "bellhop"})
    _, run = ex._stage_script(stage, "a.yaml", "out", None)
    assert "rclone" not in run and "rm -rf" not in run


# --------------------------------------------------------------- provenance
def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "--allow-empty", "-m", "init"], cwd=repo, check=True)
    return repo


def test_snapshot_run_records_provenance(tmp_path):
    repo = _git_repo(tmp_path)
    cfg_file = tmp_path / "axolotl.yaml"
    cfg_file.write_text("a: 1\n")
    out = tmp_path / "out"
    record = snapshot_run(out, "run1", {"axolotl": cfg_file}, repo_dir=repo)
    assert len(record.git_commit) == 40 and not record.git_dirty
    assert (out / "config" / "axolotl.yaml").exists()
    assert (out / "run.json").exists()


def test_snapshot_run_refuses_dirty_tree(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "junk.txt").write_text("dirty")
    with pytest.raises(RuntimeError, match="dirty git tree"):
        snapshot_run(tmp_path / "out", "r", {}, repo_dir=repo)
    # the one escape hatch
    record = snapshot_run(tmp_path / "out", "r", {}, repo_dir=repo, allow_dirty=True)
    assert record.git_dirty


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
        "  - {dataset: allenai/dolma3_dolmino_mix-100B-1125, name: dolmino, streaming: true}\n"
    )
    cfg = load_mix_config(p)
    assert isinstance(cfg.anchor, MixSource)
    assert cfg.sources[0].streaming is True
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


def test_engine_inputs_weight_math(monkeypatch):
    """anchor_frac becomes the anchor's weight; fillers split the rest."""
    monkeypatch.setattr(
        mix_mod, "_load_source",
        lambda s: mix_mod._LoadedSource(dataset=None, weight=s.weight,
                                        name=s.name or s.dataset),
    )
    cfg = MixConfig(
        anchor=MixSource(dataset="a"), anchor_frac=0.2,
        sources=[MixSource(dataset="f1", weight=3.0), MixSource(dataset="f2", weight=1.0)],
        total_tokens=1000,
    )
    sources, target, anchor_idx = mix_mod._engine_inputs(cfg)
    assert target == 1000 and anchor_idx == 0
    weights = [s.weight for s in sources]
    assert weights[0] == pytest.approx(0.2)
    assert weights[1] == pytest.approx(0.6)  # 0.8 * 3/4
    assert weights[2] == pytest.approx(0.2)  # 0.8 * 1/4
    assert sum(weights) == pytest.approx(1.0)


# ------------------------------------------------- mix engine (pane vectors)
class _FakeTokenizer:
    def __call__(self, text: str) -> dict:
        return {"input_ids": text.split()}


def _engine_sources():
    datasets = pytest.importorskip("datasets")
    anchor = datasets.Dataset.from_dict(
        {"content": [f"anchor-{i} one two three four" for i in range(20)]}
    )
    filler = datasets.Dataset.from_dict(
        {"body": [f"filler-{i} one two three four five six" for i in range(200)]}
    )
    return [
        mix_mod._LoadedSource(anchor, text_column="content", weight=1.0, name="anchor"),
        mix_mod._LoadedSource(filler, text_column="body", weight=1.0, name="filler"),
    ]


def test_engine_builds_fifty_fifty_token_mix():
    mixed, manifest = mix_mod.build_token_budget_mix(
        _engine_sources(), _FakeTokenizer(), anchor=0, num_proc=1
    )
    anchor_stats, filler_stats = manifest["per_source"]
    assert anchor_stats["tokens"] == 100
    assert 100 <= filler_stats["tokens"] <= 106
    assert len(mixed) == anchor_stats["docs"] + filler_stats["docs"]
    assert mixed.column_names == ["text"]


def test_engine_underfill_is_loud():
    sources = _engine_sources()
    with pytest.raises(ValueError, match="exhausted"):
        mix_mod.build_token_budget_mix(
            sources, _FakeTokenizer(), target_tokens=10_000, anchor=None, num_proc=1
        )
