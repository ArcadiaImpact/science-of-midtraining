from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import time
from pathlib import Path

import pytest
from jinja2 import Environment, StrictUndefined

SKILLS = Path(__file__).resolve().parents[4]
WORKER_TEMPLATE = SKILLS / "arch-run" / "assets" / "worker_startup.sh.j2"
SPAWNER = SKILLS / "arch-run" / "scripts" / "spawn_worker.py"
INPUT_VALIDATOR = SKILLS / "arch-init" / "scripts" / "validate_scaffold_inputs.py"


def render_worker() -> str:
    return Environment(
        undefined=StrictUndefined, keep_trailing_newline=True
    ).from_string(WORKER_TEMPLATE.read_text()).render(
        task_name="security-test",
        session_id="0123456789abcdef0123456789abcdef",
        wall_clock_budget_seconds=3600,
        codex_cli_version="0.144.6",
        worker_model="gpt-5.6-sol",
        worker_reasoning_effort="high",
        owner="ArcadiaImpact",
        repo="example",
        worker_required_env=["GEMINI_API_KEY"],
    )


def load_spawner():
    spec = importlib.util.spec_from_file_location("arch_spawn_worker", SPAWNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_input_validator():
    spec = importlib.util.spec_from_file_location("arch_input_validator", INPUT_VALIDATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_template_is_valid_shell_and_drops_privileges(tmp_path: Path) -> None:
    rendered = render_worker()
    path = tmp_path / "worker.sh"
    path.write_text(rendered)
    proc = subprocess.run(
        ["bash", "-n", path], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert "useradd --create-home" in rendered
    assert 'setpriv --reuid="$AGENT_UID"' in rendered
    assert "PR_SET_DUMPABLE" not in rendered
    assert "dedicated, revocable, budget/rate-limited key" in rendered
    assert "shell_environment_policy.include_only" in rendered
    assert "/run/arch-agent-env" in rendered
    assert "#!/usr/bin/python3 -I" in rendered
    assert "python3 -I -" in rendered
    assert 'env -i HOME=/root USER=root LOGNAME=root' in rendered
    assert "os.setuid(account.pw_uid)" in rendered
    assert 'child_env["CODEX_API_KEY"] = key' in rendered
    launcher_call = next(
        line
        for line in rendered.splitlines()
        if line.strip().startswith("/usr/local/libexec/arch-codex-launch")
    )
    assert '"${AGENT_ENV[@]}"' not in launcher_call


def test_worker_does_not_give_operational_secrets_to_codex_or_persist_tokens() -> None:
    rendered = render_worker()
    codex_line = next(
        line for line in rendered.splitlines() if "shell_environment_policy.include_only" in line
    )
    for name in (
        "CODEX_API_KEY",
        "RUNPOD_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        assert name not in codex_line
    assert "gh auth login" not in rendered
    assert 'git config --local "http.extraheader"' not in rendered
    assert "/root/.cache/huggingface/token" not in rendered
    assert 'RUNPOD_API_KEY="$RUNPOD_API_KEY"' not in rendered


def test_worker_requires_manifest_verified_staging_before_host_cleanup() -> None:
    rendered = render_worker()
    assert "MANIFEST.json" in rendered
    assert '"sha256": digest' in rendered
    assert "s3api head-object" in rendered
    assert "remote object-size verification failed; refusing cleanup" in rendered
    assert "READY_FOR_HOST_CLEANUP" in rendered
    assert "rest.runpod.io/v1/pods" not in rendered
    assert 'install -d -o root -g root -m 0700 /workspace/codex-transcripts' in rendered
    assert "chmod 0755 /workspace" in rendered
    assert "os.O_NOFOLLOW" in rendered
    assert "--no-follow-symlinks" in rendered
    assert "/run/arch-redaction-values" in rendered


def test_spawner_rejects_insecure_dotenv(tmp_path: Path) -> None:
    module = load_spawner()
    dotenv = tmp_path / ".env"
    dotenv.write_text("RUNPOD_API_KEY=value\n")
    dotenv.chmod(0o644)
    with pytest.raises(SystemExit, match="mode 0600"):
        module.validate_dotenv(dotenv)


def test_spawner_requires_fresh_exact_task_watcher(tmp_path: Path) -> None:
    module = load_spawner()
    session_id = "0123456789abcdef0123456789abcdef"
    heartbeat = tmp_path / ".watcher-heartbeat.json"
    heartbeat.write_text(
        json.dumps(
            {
                "armed": True,
                "task_name": "security-test",
                "session_id": session_id,
                "owner": "arch-security-test-s0123456789ab",
                "heartbeat_epoch": time.time(),
            }
        )
    )
    heartbeat.chmod(0o600)
    assert (
        module.require_fresh_watcher(heartbeat, "security-test", session_id)
        == "arch-security-test-s0123456789ab"
    )
    os.utime(heartbeat, (time.time() - 30, time.time() - 30))
    value = json.loads(heartbeat.read_text())
    value["heartbeat_epoch"] = time.time() - 30
    heartbeat.write_text(json.dumps(value))
    heartbeat.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with pytest.raises(SystemExit, match="stale"):
        module.require_fresh_watcher(heartbeat, "security-test", session_id)


def valid_context() -> dict[str, object]:
    return {
        "task_name": "security-test",
        "session_id": "0123456789abcdef0123456789abcdef",
        "owner": "ArcadiaImpact",
        "repo": "example",
        "worker_model": "gpt-5.6-sol",
        "codex_cli_version": "0.144.6",
        "worker_reasoning_effort": "high",
        "base_image": "ubuntu:22.04",
        "eval_gpu_tier": "NVIDIA GeForce RTX 4090",
        "wall_clock_budget_seconds": 3600,
        "per_head_safety_seconds": 900,
        "safety_net_base_seconds": 3600,
        "eval_shim": ".arch/eval.sh",
        "public_data_root": "data",
        "eval_trusted_paths": [".arch"],
        "eval_submission_artifacts": ["results", "figures"],
        "deny_globs": [".arch/**", ".env"],
        "worker_required_env": ["HF_TOKEN"],
        "task_secret_names": [],
        "public_eval_env": [],
        "public_metrics": ["accuracy"],
        "eval_preflight_modules": ["json"],
        "task_description": "A test.",
        "eval_invocation": "bash .arch/eval.sh",
        "research_directions": ["control"],
        "external_context_slack": [],
        "external_context_drive": [],
        "external_context": "",
        "transcript_backend": "s3",
        "transcript_auth_mode": "static",
        "transcript_bucket": "private-logs",
        "transcript_region": "eu-north-1",
        "transcript_hf_dataset": "arcadia-impact/arch-security-test-logs",
        "runpod_volume_id": "",
        "eval_reference_is_authoritative": True,
        "task_scoped_credentials_acknowledged": True,
        "codex_key_compromise_acknowledged": True,
        "estimated_hourly_cost": 1.0,
        "estimated_max_cost": 4.0,
    }


def test_scaffold_context_validator_rejects_shell_and_toml_injection() -> None:
    module = load_input_validator()
    context = valid_context()
    module.validate(context)
    context["public_data_root"] = 'data"; touch /tmp/pwned; #'
    with pytest.raises(ValueError, match="metacharacters"):
        module.validate(context)


def test_scaffold_context_validator_requires_worker_external_context() -> None:
    module = load_input_validator()
    context = valid_context()
    context.pop("external_context")
    with pytest.raises(ValueError, match="external_context"):
        module.validate(context)


def test_scaffold_context_validator_accepts_static_or_oidc_transcript_auth() -> None:
    module = load_input_validator()
    for mode in ("static", "oidc"):
        context = valid_context()
        context["transcript_auth_mode"] = mode
        module.validate(context)

    context = valid_context()
    context["transcript_auth_mode"] = "raw-pod-credentials"
    with pytest.raises(ValueError, match="static.*oidc"):
        module.validate(context)


def test_scaffold_and_spawner_reject_operational_credentials_as_task_env() -> None:
    validator = load_input_validator()
    context = valid_context()
    context["worker_required_env"] = ["CODEX_API_KEY"]
    with pytest.raises(ValueError, match="operational credential"):
        validator.validate(context)

    spawner = load_spawner()
    with pytest.raises(SystemExit, match="operational credential"):
        spawner.validate_worker_env_names(["AWS_ACCESS_KEY_ID"])

    context = valid_context()
    context["worker_required_env"] = ["PYTHONPATH"]
    with pytest.raises(ValueError, match="operational credential"):
        validator.validate(context)
    with pytest.raises(SystemExit, match="operational credential"):
        spawner.validate_worker_env_names(["LD_PRELOAD"])


def test_spawner_enforces_mutation_evidence_and_fresh_quote(tmp_path: Path) -> None:
    module = load_spawner()
    startup = tmp_path / "worker.sh"
    startup.write_text("#!/bin/bash\necho worker\n")
    startup_hash = hashlib.sha256(startup.read_bytes()).hexdigest()
    log = tmp_path / ".arch" / "logs" / "preflight.log"
    log.parent.mkdir(parents=True)
    log.write_text("redacted preflight\n")
    log.chmod(0o600)
    log_hash = hashlib.sha256(log.read_bytes()).hexdigest()
    commit = "a" * 40
    session_id = "0123456789abcdef0123456789abcdef"
    session = {
        "task_name": "security-test",
        "session_id": session_id,
        "task_branch": "arch/security-test",
        "task_commit": commit,
        "authorizations": {"launch_fleet": True},
        "canary_scored": True,
        "canary": {"score": 0.0, "base_sha": commit},
        "worker": {"startup_sha256": startup_hash},
        "preflight": {
            "passed": True,
            "commit_sha": commit,
            "completed_epoch": 900.0,
            "log_path": ".arch/logs/preflight.log",
            "log_sha256": log_hash,
        },
        "price_quote": {
            "quoted_epoch": 900.0,
            "hourly_fleet_cost": 1.0,
            "estimated_max_cost": 4.0,
        },
    }
    config = {"task_name": "security-test", "session_id": session_id}
    module.validate_launch_state(session, config, startup, tmp_path, now=1000.0)

    session["authorizations"]["launch_fleet"] = False
    with pytest.raises(SystemExit, match="not explicitly authorized"):
        module.validate_launch_state(session, config, startup, tmp_path, now=1000.0)
    session["authorizations"]["launch_fleet"] = True
    session["price_quote"]["quoted_epoch"] = 0.0
    with pytest.raises(SystemExit, match="older than 15 minutes"):
        module.validate_launch_state(session, config, startup, tmp_path, now=1000.0)


def test_spawner_refuses_stale_exact_name_adoption() -> None:
    module = load_spawner()
    matches = [{"id": "pod-current"}]
    with pytest.raises(SystemExit, match="refusing stale adoption"):
        module.validated_adoption_id(
            matches, None, name="arch-task-sabc-worker-1", fingerprint="f" * 64
        )
    action = {
        "status": "pending",
        "target_name": "arch-task-sabc-worker-1",
        "launch_fingerprint": "f" * 64,
    }
    assert (
        module.validated_adoption_id(
            matches,
            action,
            name="arch-task-sabc-worker-1",
            fingerprint="f" * 64,
        )
        == "pod-current"
    )


def test_spawner_never_reposts_an_ambiguous_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_spawner()
    calls = 0

    def fail_create(request, timeout):
        nonlocal calls
        calls += 1
        raise module.urllib.error.URLError("ambiguous timeout")

    monkeypatch.setattr(module.urllib.request, "urlopen", fail_create)
    monkeypatch.setattr(module, "existing_by_name", lambda name, key: [])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    with pytest.raises(module.CreateOutcomeUnresolved, match="single create outcome"):
        module.request_json(
            "/pods",
            "key",
            method="POST",
            payload={"name": "arch-test"},
            reconcile_name="arch-test",
        )
    assert calls == 1


def test_spawner_redacts_malformed_provider_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_spawner()
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"not-json-CODEX_API_KEY=must-not-leak"

    def malformed_response(request, timeout):
        nonlocal calls
        calls += 1
        return Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", malformed_response)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="malformed JSON") as caught:
        module.request_json("/pods", "key")
    assert "must-not-leak" not in str(caught.value)
    assert calls == 5
