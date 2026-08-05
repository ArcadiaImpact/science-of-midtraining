from __future__ import annotations

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from arch._config import ConfigError, load_config
from arch._dotenv import DotenvError, dotenv_value, runpod_api_key
from arch.boot_watch import evaluate
from arch.eval import _public_eval_environment
from arch.monitor import (
    _classify,
    _classify_heldout,
    _PodRecord,
    monitor_cmd,
    pod_ids_for_kind,
)
from arch.rescore import rescore_cmd
from click.testing import CliRunner


def record(health: str) -> _PodRecord:
    return _PodRecord("pod-1", "RUNNING", health, "test", 60, [])


def test_codex_jsonl_marks_worker_running() -> None:
    health, _ = _classify(
        [
            "=== arch-worker boot now",
            "=== codex exec iteration 2, 100 s remaining ===",
            '{"type":"thread.started"}',
            '{"type":"item.completed"}',
        ],
        120,
    )
    assert health == "RUNNING"


def test_missing_codex_events_eventually_stalls() -> None:
    health, _ = _classify(
        ["=== arch-worker boot now", "=== codex exec iteration 1, 100 s remaining ==="],
        700,
    )
    assert health == "STALLED"


def test_codex_turn_failure_is_terminal_for_boot_watch() -> None:
    assert _classify(
        [
            "=== arch-worker boot now",
            "=== codex exec iteration 1, 100 s remaining ===",
            '{"type":"turn.failed"}',
        ],
        10,
    )[0] == "CODEX_FAILED"
    assert evaluate([record("CODEX_FAILED")], 10, 180)[0] == "fail"


def test_codex_jsonl_classification_uses_only_latest_iteration() -> None:
    health, _ = _classify(
        [
            "=== arch-worker boot now",
            "=== codex exec iteration 1, 100 s remaining ===",
            '{"type":"thread.started"}',
            '{"type":"turn.failed"}',
            "=== codex exec iteration 2, 90 s remaining ===",
        ],
        100,
    )
    assert health == "BOOTING"


def test_heldout_classifier_matches_hardened_evaluator_markers() -> None:
    assert _classify_heldout(
        [
            "=== arch heldout-eval boot now",
            "=== running trusted evaluator as uid=999 with env-i and no network ===",
        ],
        120,
    )[0] == "RUNNING"
    assert _classify_heldout(
        [
            "=== arch heldout-eval boot now",
            "CRITICAL: refusing to score or self-delete; host watcher must recover the exact pod",
        ],
        120,
    )[0] == "EVAL_SETUP_FAILED"
    assert _classify_heldout(
        ["=== running trusted evaluator as uid=999", "=== trusted evaluator exited 1 ==="],
        120,
    )[0] == "EVAL_FAILED"


def test_stale_jsonl_event_does_not_make_latest_iteration_running() -> None:
    health, _ = _classify(
        [
            "=== arch-worker boot now",
            "=== codex exec iteration 1, 100 s remaining ===",
            '{"type":"thread.started"}',
            "=== codex exec iteration 2, 90 s remaining ===",
        ],
        700,
    )
    assert health == "STALLED"


def test_config_splits_worker_and_public_eval_env_names(tmp_path: Path) -> None:
    config = tmp_path / ".arch" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        'task_name="t"\n'
        'task_description="d"\n'
        'eval_shim=".arch/eval.sh"\n'
        'public_data_root="data"\n'
        '[worker]\nrequired_env=["HF_TOKEN"]\n'
        '[eval]\npublic_env=["CUDA_VISIBLE_DEVICES"]\n'
    )
    loaded = load_config(tmp_path)
    assert loaded.worker_required_env == ["HF_TOKEN"]
    assert loaded.public_eval_env == ["CUDA_VISIBLE_DEVICES"]


def test_config_rejects_secret_like_public_eval_env(tmp_path: Path) -> None:
    config = tmp_path / ".arch" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        'task_name="t"\n'
        'task_description="d"\n'
        'eval_shim=".arch/eval.sh"\n'
        'public_data_root="data"\n'
        '[eval]\npublic_env=["CODEX_API_KEY"]\n'
    )
    with pytest.raises(ConfigError, match="secret-like"):
        load_config(tmp_path)


def test_public_eval_environment_does_not_copy_host_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "must-not-leak")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env = _public_eval_environment(
        ["CUDA_VISIBLE_DEVICES"], tmp_path / "data", tmp_path / "result.json"
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "RUNPOD_API_KEY" not in env
    assert "HOME" not in env


def test_rescore_is_disabled_before_config_or_credentials_are_read() -> None:
    result = CliRunner().invoke(rescore_cmd, [])
    assert result.exit_code != 0
    assert "disabled in the Codex ARCH v0.1 safety profile" in result.output


def test_monitor_has_no_destructive_reap_option() -> None:
    result = CliRunner().invoke(monitor_cmd, ["--reap"])
    assert result.exit_code == 2
    assert "No such option '--reap'" in result.output


def test_typed_pod_selection_does_not_mix_evaluators_into_workers() -> None:
    session = {
        "worker_pod_ids": ["worker-1"],
        "heldout_pod_ids": ["heldout-1"],
        "batch_pod_ids": ["batch-1"],
        "pod_ids": ["worker-1", "heldout-1", "batch-1"],
    }
    assert pod_ids_for_kind(session, "worker") == ["worker-1"]
    assert pod_ids_for_kind(session, "heldout") == ["heldout-1"]
    assert pod_ids_for_kind(session, "batch") == ["batch-1"]


def test_legacy_pod_ids_are_workers_only() -> None:
    assert pod_ids_for_kind({"pod_ids": ["old-worker"]}, "worker") == ["old-worker"]
    assert pod_ids_for_kind({"pod_ids": ["old-worker"]}, "heldout") == []


def test_repo_dotenv_key_beats_ambient_and_is_not_executed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text('RUNPOD_API_KEY="repo-key"\nTRAP=$(touch should-not-exist)\n')
    dotenv.chmod(0o600)
    monkeypatch.setenv("RUNPOD_API_KEY", "injected-invalid-key")
    assert runpod_api_key(tmp_path) == "repo-key"
    assert dotenv_value(tmp_path, "TRAP") == (True, "$(touch should-not-exist)")
    assert not (tmp_path / "should-not-exist").exists()


def test_repo_dotenv_rejects_non_private_mode(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("RUNPOD_API_KEY=repo-key\n")
    dotenv.chmod(0o644)
    with pytest.raises(DotenvError, match="mode other than 0600"):
        runpod_api_key(tmp_path)


def _session_script() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "session_state.py"


def _state_call(session: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_session_script()), "--file", str(session), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_session_state_keeps_typed_sets_and_derived_union(tmp_path: Path) -> None:
    session = tmp_path / ".arch" / ".session.json"
    for kind, pod_id in (("worker", "w"), ("heldout", "h"), ("batch", "b")):
        result = _state_call(session, "add-pod", pod_id, "--kind", kind)
        assert result.returncode == 0, result.stderr
    state = json.loads(session.read_text())
    assert state["worker_pod_ids"] == ["w"]
    assert state["heldout_pod_ids"] == ["h"]
    assert state["batch_pod_ids"] == ["b"]
    assert state["pod_ids"] == ["w", "h", "b"]
    assert oct(session.stat().st_mode & 0o777) == "0o600"


def test_session_state_lock_preserves_concurrent_updates(tmp_path: Path) -> None:
    session = tmp_path / ".arch" / ".session.json"

    def add(index: int) -> subprocess.CompletedProcess[str]:
        return _state_call(session, "add-pod", f"w-{index}", "--kind", "worker")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(add, range(24)))
    assert all(result.returncode == 0 for result in results)
    state = json.loads(session.read_text())
    assert set(state["worker_pod_ids"]) == {f"w-{index}" for index in range(24)}
    assert set(state["pod_ids"]) == set(state["worker_pod_ids"])


def test_session_state_rejects_broad_secret_names(tmp_path: Path) -> None:
    session = tmp_path / ".arch" / ".session.json"
    result = _state_call(session, "set", "transcripts.hf_token", '"secret"')
    assert result.returncode != 0
    assert "refusing secret-like session field" in result.stderr


def test_arch2_wrapper_finds_task_local_runtime_inside_monorepo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    task = repo / "nested-task"
    scripts = task / "scripts"
    runtime = task / ".arch" / "runtime"
    (runtime / "src" / "arch").mkdir(parents=True)
    scripts.mkdir(parents=True)
    (runtime / "pyproject.toml").write_text("[project]\nname='test-runtime'\n")
    (runtime / "src" / "arch" / "__main__.py").write_text("# exact runtime\n")
    wrapper = Path(__file__).resolve().parents[3] / "scripts" / "arch2"
    deployed = scripts / "arch2"
    shutil.copy2(wrapper, deployed)
    subprocess.run(["git", "init", "-q", repo], check=True)

    result = subprocess.run(
        [deployed, "--runtime-provenance"],
        cwd=task,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"source={runtime.resolve()}" in result.stdout
