"""CPU-only tests for the generic Bellhop pod-job seam (BELLHOP_PORT.md T1).

bellhop is faked via ``sys.modules`` injection (the ``test_axolotl_backend``
pattern); the source manifest runs against a tiny throwaway git repo so the
dirty-tree refusal and provenance env are exercised for real.
"""

import asyncio
import json
import shutil
import subprocess
import sys
import types
from datetime import timedelta
from pathlib import Path

import pytest

import scimt.train.podjob as podjob_mod
from scimt.train.axolotl import PodSpec
from scimt.train.podjob import PodJob, TransferBundle, stage_transfer, submit
from scimt.train.source_manifest import verify_source_manifest


# --------------------------------------------------------------- fixtures
def _git(*args, cwd):
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    )


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A clean single-commit git repo standing in for the scimt checkout."""
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "scimt"\n')
    _git("init", "-q", cwd=root)
    _git("add", ".", cwd=root)
    _git("commit", "-q", "-m", "init", cwd=root)
    monkeypatch.setattr(podjob_mod, "REPO_ROOT", root)

    def fake_build_wheel(out_dir: Path) -> Path:
        wheel = out_dir / "bellhop_dist" / "scimt-0.0.0-py3-none-any.whl"
        wheel.parent.mkdir(parents=True, exist_ok=True)
        wheel.write_bytes(b"wheel")
        return wheel

    monkeypatch.setattr(podjob_mod, "build_transfer_wheel", fake_build_wheel)
    return root


@pytest.fixture()
def fake_bellhop(monkeypatch):
    captured = {}

    class FakeRunSpec:
        def __init__(self, **kwargs):
            captured["spec"] = kwargs

    class FakePodConfig:
        def __init__(self, **kwargs):
            captured["pod"] = kwargs

    async def fake_run(spec, pod):
        captured["ran"] = (spec, pod)

    monkeypatch.setitem(
        sys.modules,
        "bellhop",
        types.SimpleNamespace(
            RunSpec=FakeRunSpec, PodConfig=FakePodConfig, run=fake_run
        ),
    )
    return captured


def _job(repo, **overrides):
    kwargs = dict(
        pod=PodSpec(gpu="H200", gpu_count=1, cloud="SECURE",
                    disk_gb=200, max_hours=16.0),
        slug="uad-w0",
        setup="echo SETUP_OK",
        run="python3 arm_worker.py",
        out_dir=repo / "experiments" / "bellhop" / "runs" / "r1" / "uad-r1-w00",
        results_subdir="results",
    )
    kwargs.update(overrides)
    return PodJob(**kwargs)


# ------------------------------------------------------- RunSpec / PodConfig
def test_submit_maps_runspec_and_podconfig(repo, fake_bellhop):
    asyncio.run(submit(_job(repo)))

    pod = fake_bellhop["pod"]
    assert pod["name"] == "scimt-uad-w0"
    assert pod["gpu"] == "H200"
    assert pod["gpu_count"] == 1
    assert pod["container_disk_gb"] == 200
    assert pod["max_lifetime"] == timedelta(hours=16.0)
    assert pod["cloud"] == "SECURE"

    spec = fake_bellhop["spec"]
    assert spec["slug"] == "uad-w0"
    assert spec["codebase"] == str(repo)
    assert spec["setup"] == "echo SETUP_OK"
    assert spec["run"] == "python3 arm_worker.py"
    assert spec["results_subdir"] == "results"
    assert spec["local_out"] == str(
        repo / "experiments" / "bellhop" / "runs" / "r1" / "uad-r1-w00"
    )
    assert "ran" in fake_bellhop


def test_submit_gcs_base_is_none(repo, fake_bellhop):
    # bellhop's devbox-side GCS upload is not our bus (upload_and_pin is)
    asyncio.run(submit(_job(repo)))
    assert fake_bellhop["spec"]["gcs_base"] is None


# ------------------------------------------------------------------- env
def test_submit_env_provenance_from_staged_manifest(repo, fake_bellhop):
    asyncio.run(submit(_job(repo)))
    env = fake_bellhop["spec"]["env"]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    assert env["SCIMT_SOURCE_COMMIT"] == head
    assert env["SCIMT_SOURCE_MANIFEST"] == (
        "experiments/bellhop/runs/r1/uad-r1-w00/.scimt-source.json"
    )
    assert (repo / env["SCIMT_SOURCE_MANIFEST"]).is_file()
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_submit_env_secret_passthrough_present_and_absent(
    repo, fake_bellhop, monkeypatch
):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.delenv("RCLONE_CONFIG_GCS_TYPE", raising=False)
    monkeypatch.delenv(
        "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS", raising=False
    )
    asyncio.run(submit(_job(repo)))
    env = fake_bellhop["spec"]["env"]
    assert env["HF_TOKEN"] == "hf_secret"
    assert "RCLONE_CONFIG_GCS_TYPE" not in env
    assert "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS" not in env


def test_submit_job_env_merges_last(repo, fake_bellhop, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-environ")
    asyncio.run(submit(_job(
        repo, env={"UAD_RUN_ID": "r1", "HF_TOKEN": "from-job"}
    )))
    env = fake_bellhop["spec"]["env"]
    assert env["UAD_RUN_ID"] == "r1"
    assert env["HF_TOKEN"] == "from-job"  # job.env wins (dict-literal order)


# ------------------------------------------------------------- staging
OUT_REL = "experiments/bellhop/runs/r1/uad-r1-w00"


def test_stage_transfer_bundle_is_repo_relative(repo):
    bundle = stage_transfer(repo / OUT_REL)
    assert isinstance(bundle, TransferBundle)
    assert bundle.wheel_rel == (
        f"{OUT_REL}/bellhop_dist/scimt-0.0.0-py3-none-any.whl"
    )
    assert bundle.manifest_rel == f"{OUT_REL}/.scimt-source.json"
    assert len(bundle.commit) == 40
    assert (repo / bundle.wheel_rel).is_file()
    assert (repo / bundle.manifest_rel).is_file()


def test_stage_transfer_refuses_out_dir_outside_repo(repo, tmp_path):
    with pytest.raises(ValueError, match="under the repo checkout"):
        stage_transfer(tmp_path / "elsewhere")


def test_stage_transfer_scopes_manifest_around_runs_root(repo):
    """The staged manifest records the volatile rule: runs root out, own dir in."""
    bundle = stage_transfer(repo / OUT_REL)
    payload = json.loads((repo / bundle.manifest_rel).read_text())
    assert payload["schema_version"] == 2
    assert payload["volatile_root"] == "experiments/bellhop/runs"
    assert payload["keep_under_volatile"] == OUT_REL
    # the job's own wheel is in the manifest; that's the point of keep
    assert f"{OUT_REL}/bellhop_dist/scimt-0.0.0-py3-none-any.whl" in payload["files"]


@pytest.mark.parametrize("bad_rel", [
    "experiments/runs/w0",          # grandparent is "experiments", not "runs"
    "runs-r1/w0",                   # grandparent is the repo root itself
    "w0",                           # grandparent is outside the repo
])
def test_stage_transfer_refuses_non_runs_layout(repo, bad_rel):
    with pytest.raises(ValueError, match="runs/<run-id>/<slug>"):
        stage_transfer(repo / bad_rel)


def _gitless_copy(repo, tmp_path):
    """Model Bellhop's push: the checkout minus git metadata."""
    copy = tmp_path / "pod-tree"
    shutil.copytree(repo, copy, symlinks=True,
                    ignore=shutil.ignore_patterns(".git"))
    return copy


def test_concurrent_sibling_staging_is_invisible_to_provenance(repo, tmp_path):
    """Simulate the 2026-08-25 race: another slot (re)stages a sibling dir
    under the runs root between this job's stage and Bellhop's push."""
    bundle = stage_transfer(repo / OUT_REL)
    sibling = repo / "experiments" / "bellhop" / "runs" / "r1" / "uad-r1-w01"
    sibling.mkdir(parents=True)
    (sibling / ".scimt-source.json").write_text("{\"mid\": \"restage\"}\n")
    (sibling / "pulled-result.json").write_text("{}\n")
    (repo / "experiments" / "bellhop" / "runs" / "r1" / "dispatch.json").write_text("{}\n")

    pod_tree = _gitless_copy(repo, tmp_path)
    verify_source_manifest(  # must PASS: siblings are outside provenance
        pod_tree, pod_tree / bundle.manifest_rel, expected_commit=bundle.commit
    )


def test_mutation_outside_volatile_root_still_fails_verification(repo, tmp_path):
    bundle = stage_transfer(repo / OUT_REL)
    pod_tree = _gitless_copy(repo, tmp_path)
    (pod_tree / "pyproject.toml").write_text("# tampered\n")
    with pytest.raises(RuntimeError, match="source file mismatch"):
        verify_source_manifest(
            pod_tree, pod_tree / bundle.manifest_rel, expected_commit=bundle.commit
        )


def test_own_wheel_mutation_still_fails_verification(repo, tmp_path):
    bundle = stage_transfer(repo / OUT_REL)
    pod_tree = _gitless_copy(repo, tmp_path)
    (pod_tree / bundle.wheel_rel).write_bytes(b"tampered wheel")
    with pytest.raises(RuntimeError, match="source file mismatch"):
        verify_source_manifest(
            pod_tree, pod_tree / bundle.manifest_rel, expected_commit=bundle.commit
        )


def test_submit_refuses_dirty_tree(repo, fake_bellhop):
    (repo / "pyproject.toml").write_text("# dirtied\n")
    with pytest.raises(RuntimeError, match="dirty"):
        asyncio.run(submit(_job(repo)))
    assert "ran" not in fake_bellhop  # refused before provisioning


# ---------------------------------------------------------- PodJob checks
def test_podjob_requires_single_node(repo):
    with pytest.raises(ValueError, match="single-node"):
        _job(repo, pod=PodSpec(gpu="H200", nodes=2))


def test_podjob_refuses_podspec_extra_env(repo):
    with pytest.raises(ValueError, match="extra_env"):
        _job(repo, pod=PodSpec(gpu="H200", extra_env={"X": "1"}))


@pytest.mark.parametrize("field_name", ["slug", "setup", "run", "results_subdir"])
def test_podjob_refuses_empty_strings(repo, field_name):
    with pytest.raises(ValueError):
        _job(repo, **{field_name: ""})


def test_podjob_env_must_be_str_to_str(repo):
    with pytest.raises(ValueError, match="str -> str"):
        _job(repo, env={"MAX_HOURS": 16})


def test_podjob_env_may_not_shadow_provenance(repo):
    with pytest.raises(ValueError, match="provenance"):
        _job(repo, env={"SCIMT_SOURCE_COMMIT": "f" * 40})


# ------------------------------------------------------------- lazy import
def test_bellhop_import_is_lazy(repo, monkeypatch):
    # the module is importable and PodJob/stage_transfer usable with bellhop
    # entirely absent; only submit() reaches for it.
    monkeypatch.setitem(sys.modules, "bellhop", None)  # blocks import
    job = _job(repo)
    stage_transfer(job.out_dir)
    with pytest.raises(ImportError):
        asyncio.run(submit(job))
