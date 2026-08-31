"""CPU-only contract tests for final-v1 Hub crash recovery."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP = REPO_ROOT / "experiments" / "prior_coins" / "dispatch_final_v1"
POD = EXP / "pod"
for _path in (str(REPO_ROOT), str(EXP), str(POD)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import contracts as C  # noqa: E402


def _load_rehydrate():
    name = "final_v1_rehydrate"
    spec = importlib.util.spec_from_file_location(name, POD / "rehydrate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rehydrate = _load_rehydrate()


def _midtrain_files(arm: str = "charter") -> dict[str, bytes]:
    prefix = C.hub_arm_prefix(arm)
    files: dict[str, bytes] = {}
    for step in C.MIDTRAIN_CHECKPOINT_STEPS:
        base = f"{prefix}/midtrain/checkpoints/checkpoint-{step}"
        files[f"{base}/config.json"] = b'{"model_type":"gemma3"}\n'
        files[f"{base}/tokenizer_config.json"] = b'{"bos_token":"<bos>"}\n'
        files[f"{base}/model.safetensors"] = f"weights-{step}".encode()
    return files


def _fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, bytes],
    *,
    revision: str = "a" * 40,
):
    calls: dict[str, list] = {"repo_info": [], "snapshot": []}

    class Api:
        def repo_info(self, repo_id: str, *, repo_type: str, files_metadata: bool):
            calls["repo_info"].append((repo_id, repo_type, files_metadata))
            return SimpleNamespace(
                sha=revision,
                siblings=[
                    SimpleNamespace(rfilename=name, size=len(payload))
                    for name, payload in sorted(files.items())
                ],
            )

    def snapshot_download(
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        allow_patterns: list[str],
        local_dir: Path,
    ):
        calls["snapshot"].append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
                "allow_patterns": list(allow_patterns),
                "local_dir": Path(local_dir),
            }
        )
        for name in allow_patterns:
            destination = Path(local_dir) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(files[name])
        return str(local_dir)

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = Api
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return calls


def _run(root: Path, *arms: str):
    config = rehydrate.RehydrateConfig(root=root, arms=tuple(arms))
    return asyncio.run(rehydrate.rehydrate(config))


def test_fresh_pod_is_a_successful_noop(tmp_path, monkeypatch):
    calls = _fake_hub(monkeypatch, {})

    audit = _run(tmp_path, "charter", "coin", "control")

    assert calls["repo_info"] == [(rehydrate.REPO, "model", True)]
    assert calls["snapshot"] == []
    assert all(value["found_stages"] == [] for value in audit["arms"].values())
    assert all(value["first_phase_to_run"] == "mix" for value in audit["arms"].values())
    assert not (tmp_path / C.PROFILE.name / "charter").exists()
    persisted = json.loads((tmp_path / "REHYDRATED.json").read_text())
    assert persisted["runs"][-1]["revision"] == "a" * 40


def test_midtrain_resume_fetches_only_final_checkpoint_and_starts_at_dolci(
    tmp_path, monkeypatch
):
    files = _midtrain_files()
    calls = _fake_hub(monkeypatch, files)

    audit = _run(tmp_path, "charter")

    assert len(calls["snapshot"]) == 1
    requested = calls["snapshot"][0]["allow_patterns"]
    wanted_fragment = f"checkpoint-{C.MIDTRAIN_STEPS}/"
    assert requested
    assert all("/midtrain/" in name and wanted_fragment in name for name in requested)
    assert not any(
        "checkpoint-38/" in name for name in requested if C.MIDTRAIN_STEPS != 38
    )

    arm_root = tmp_path / C.PROFILE.name / "charter"
    checkpoint = (
        arm_root / "midtrain" / "checkpoints" / f"checkpoint-{C.MIDTRAIN_STEPS}"
    )
    assert checkpoint.is_dir()
    assert not (arm_root / "data").exists()
    assert audit["arms"]["charter"]["first_phase_to_run"] == "dolci"
    assert audit["arms"]["charter"]["download"]["bytes"] == sum(
        len(payload) for name, payload in files.items() if wanted_fragment in name
    )
    assert (arm_root / "PUBLISHED_MIDTRAIN.json").is_file()

    marker = arm_root / "MIDTRAIN_COMPLETE.json"
    payload = json.loads(marker.read_text())
    assert payload["rehydrated_from_hub"] is True
    assert payload["hub_revision"] == "a" * 40
    # The reconstructed marker must be accepted by the arm that owns it and
    # refused for any other -- that refusal is the whole point of stamping
    # fingerprints, and recovery is the one place markers are written without
    # having been earned.
    with rehydrate.chain.fingerprint_scope(arm_root, "charter"):
        assert rehydrate.chain.done(marker)
    coin_root = arm_root.parent / "coin"
    coin_root.mkdir(parents=True, exist_ok=True)
    coin_marker = coin_root / "MIDTRAIN_COMPLETE.json"
    coin_marker.write_text(marker.read_text())
    with rehydrate.chain.fingerprint_scope(coin_root, "coin"):
        with pytest.raises(RuntimeError, match="DIFFERENT run"):
            rehydrate.chain.done(coin_marker)


def test_inconsistent_published_stage_fails_before_downloading(tmp_path, monkeypatch):
    files = _midtrain_files()
    files = {
        name: payload
        for name, payload in files.items()
        if not name.endswith("model.safetensors")
    }
    calls = _fake_hub(monkeypatch, files)

    with pytest.raises(RuntimeError, match="no non-empty model weights"):
        _run(tmp_path, "charter")

    assert calls["snapshot"] == []
    assert not (tmp_path / "REHYDRATED.json").exists()


def test_rehydration_is_safe_to_repeat_without_replacing_equal_local_bytes(
    tmp_path, monkeypatch
):
    files = _midtrain_files()
    calls = _fake_hub(monkeypatch, files)

    first = _run(tmp_path, "charter")
    arm_root = tmp_path / C.PROFILE.name / "charter"
    model = (
        arm_root
        / "midtrain"
        / "checkpoints"
        / f"checkpoint-{C.MIDTRAIN_STEPS}"
        / "model.safetensors"
    )
    before = model.stat().st_mtime_ns
    second = _run(tmp_path, "charter")

    assert len(calls["snapshot"]) == 2
    assert model.stat().st_mtime_ns == before
    assert first["arms"]["charter"]["download"]["installed_bytes"] > 0
    assert second["arms"]["charter"]["download"]["installed_bytes"] == 0
    assert second["arms"]["charter"]["download"]["already_local_bytes"] > 0
    assert len(json.loads((tmp_path / "REHYDRATED.json").read_text())["runs"]) == 2


def test_rehydrate_can_never_forge_run_completion():
    """CHAIN_COMPLETE says "safe to destroy this pod"; it must never be forged.

    Recovery reconstructs phase sentinels from Hub stage commits, which is safe
    because a stage tree only exists after its phase finished. CHAIN_COMPLETE is
    different in kind: it licenses tearing down a pod that may hold the only
    copy of unpublished work. Guard the data, not just the current code path --
    a later edit adding a key to ROOT_SENTINELS is the realistic way this breaks.
    """
    import experiments.prior_coins.dispatch_final_v1.pod.rehydrate as r

    assert "chain" not in r.ROOT_SENTINELS
    assert "publish" not in r.ROOT_SENTINELS
    assert not any(
        name.startswith(("CHAIN_", "PUBLISH_")) for name in r.ROOT_SENTINELS.values()
    )
    # And the stages it restores are exactly the publishable ones -- no synthetic
    # stage may sneak in and drag a forged marker with it.
    assert set(r.ROOT_SENTINELS) <= set(r.STAGES) | {"mix"}
