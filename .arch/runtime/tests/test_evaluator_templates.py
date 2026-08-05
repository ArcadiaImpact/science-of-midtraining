from __future__ import annotations

from pathlib import Path

import tomllib
import yaml
from jinja2 import Environment, StrictUndefined

TEMPLATES = Path(__file__).resolve().parents[2] / "templates"


def render(name: str, **overrides: object) -> str:
    context: dict[str, object] = {
        "task_name": "safe-eval",
        "session_id": "0123456789abcdef0123456789abcdef",
        "task_description": "Score inert result artifacts.",
        "eval_shim": ".arch/eval.sh",
        "public_data_root": "data",
        "research_directions": ["control"],
        "eval_trusted_paths": [".arch", "trusted/scorer"],
        "eval_submission_artifacts": ["results", "figures", "artifacts"],
        "deny_globs": [".arch/**", ".env"],
        "public_metrics": ["accuracy"],
        "external_context_slack": [],
        "external_context_drive": [],
        "eval_reference_is_authoritative": False,
        "eval_preflight_modules": [],
        "task_secret_names": ["HF_TOKEN", "GEMINI_API_KEY"],
        "codex_key_compromise_acknowledged": False,
        "eval_invocation": "bash .arch/eval.sh",
        "base_image": "ubuntu:22.04",
        "eval_gpu_tier": "NVIDIA GeForce RTX 4090",
        "runpod_volume_id": "volume-test",
        "transcript_backend": "s3",
        "transcript_auth_mode": "static",
        "transcript_bucket": "private-arch-logs",
        "transcript_region": "eu-west-1",
        "transcript_hf_dataset": "arcadia-impact/arch-safe-eval-logs",
    }
    context.update(overrides)
    env = Environment(undefined=StrictUndefined, keep_trailing_newline=True)
    return env.from_string((TEMPLATES / name).read_text()).render(**context)


def test_actions_omits_volume_fields_when_no_volume_is_configured() -> None:
    workflow = render("arch-eval.yml.j2", runpod_volume_id="")
    assert '"networkVolumeId"' not in workflow
    assert '"volumeMountPath"' not in workflow


def test_actions_does_not_pass_task_secrets_or_supersede_existing_pods() -> None:
    workflow = render("arch-eval.yml.j2")
    assert "HF_TOKEN" not in workflow
    assert "GEMINI_API_KEY" not in workflow
    assert "Supersede" not in workflow
    assert 'method="DELETE"' not in workflow
    assert "Spawned held-out eval pod id=" in workflow
    assert "MIN_OBSERVATION_SECONDS" in workflow


def test_actions_reconciles_deterministic_exact_name_before_and_after_create() -> None:
    workflow = render("arch-eval.yml.j2")
    assert "arch-safe-eval-s0123456789ab-heldout-pr" in workflow
    assert "def exact_name_matches():" in workflow
    assert 'pod.get("name") == pod_name' in workflow
    assert "unsafe duplicate state" in workflow
    assert "matches = exact_name_matches()" in workflow
    assert "lost response or provider timeout" in workflow
    assert '"TRUSTED_BASE_SHA": trusted_base_sha' in workflow
    spawn_env = workflow.split("- name: Spawn fresh held-out eval pod", 1)[1].split(
        "run: |", 1
    )[0]
    assert "RUN_ID:" in spawn_env
    assert "RUN_ATTEMPT:" in spawn_env


def test_actions_mints_unique_log_capabilities_with_static_aws_key_kept_out_of_pod() -> None:
    workflow = render("arch-eval.yml.j2")
    assert "aws-actions/configure-aws-credentials@7474bc4690e29a8392af63c5b98e7449536d5c3a" in workflow
    assert "secrets.AWS_ACCESS_KEY_ID" in workflow
    assert "secrets.AWS_SECRET_ACCESS_KEY" in workflow
    assert "id-token: write" not in workflow
    assert 'generate_presigned_url("put_object"' in workflow
    assert 'generate_presigned_url("head_object"' in workflow
    spawn_env = workflow.split("- name: Spawn fresh held-out eval pod", 1)[1].split(
        "run: |", 1
    )[0]
    assert "AWS_ACCESS_KEY_ID" not in spawn_env
    assert "AWS_SECRET_ACCESS_KEY" not in spawn_env
    assert "secrets.ARCH_PRIVATE_LOG_UPLOAD_URL" not in workflow
    yaml.safe_load(workflow)


def test_actions_can_use_oidc_log_writer_role() -> None:
    workflow = render("arch-eval.yml.j2", transcript_auth_mode="oidc")
    assert "id-token: write" in workflow
    assert "secrets.ARCH_LOG_WRITER_ROLE_ARN" in workflow
    assert "secrets.AWS_ACCESS_KEY_ID" not in workflow
    assert "secrets.AWS_SECRET_ACCESS_KEY" not in workflow
    yaml.safe_load(workflow)


def test_heldout_renders_each_trusted_path_as_a_real_array_element() -> None:
    startup = render("heldout_eval_startup.sh.j2")
    assert 'TRUSTED_PATHS=(\n  ".arch"\n  "trusted/scorer"\n)' in startup
    assert "['.arch'" not in startup


def test_heldout_checkout_and_trusted_restore_fail_closed() -> None:
    startup = render("heldout_eval_startup.sh.j2")
    assert 'test "$(git rev-parse HEAD)" = "$PR_HEAD_SHA"' in startup
    assert "TRUSTED_BASE_SHA=$(git rev-parse" in startup
    assert 'test "$RESTORED_OBJECT" = "$EXPECTED_OBJECT"' in startup
    assert '"$TRUSTED_BASE_SHA:${TASK_PREFIX}${_tp}"' in startup
    assert "git checkout \"$PR_HEAD_SHA\" || true" not in startup
    assert "trusted path restore failed" in startup


def test_heldout_keeps_credentials_out_of_evaluator_and_git_config() -> None:
    startup = render("heldout_eval_startup.sh.j2")
    assert "ORCH_GH_TOKEN=" in startup
    assert "ORCH_RUNPOD_API_KEY=" in startup
    assert "unset GH_TOKEN RUNPOD_API_KEY" in startup
    assert "gh auth login" not in startup
    assert "git config --local" not in startup
    assert "env -i" in startup
    assert "runuser --user arch-eval" in startup
    assert "HF_TOKEN" not in startup
    assert "GEMINI_API_KEY" not in startup


def test_heldout_only_exposes_static_artifacts_to_network_blocked_eval_uid() -> None:
    startup = render("heldout_eval_startup.sh.j2")
    assert "STATIC ARTIFACT CONTRACT" in startup
    assert "copy_static_artifact" in startup
    assert "find \"$SUBMISSION_ROOT\" -type l" in startup
    assert "chmod 0700 \"$WORKDIR\"" in startup
    assert "--uid-owner \"$EVAL_UID\" -j REJECT" in startup
    assert "network isolation unavailable; refusing to evaluate" in startup
    assert 'cd "$TRUSTED_TASK_ROOT"' in startup
    assert "bash .arch/setup.sh" not in startup
    assert "uv sync" not in startup
    assert "pip install -e" not in startup
    assert 'find "$TRUSTED_ROOT" -type f -perm /111 -exec chmod 0550 {} +' in startup
    assert 'find "$SUBMISSION_ROOT" -type f -exec chmod 0440 {} +' in startup


def test_missing_expected_volume_is_fatal_and_logs_must_be_staged() -> None:
    startup = render("heldout_eval_startup.sh.j2")
    assert "expected held-out volume is missing or empty" in startup
    assert "falling back" not in startup.lower()
    assert "PRIVATE_LOG_UPLOAD_URL" in startup
    assert "PRIVATE_LOG_VERIFY_URL" in startup
    assert "durable private log upload failed; refusing to self-delete" in startup
    assert "Content-Length" in startup
    assert "durable private log size mismatch" in startup


def test_public_metrics_are_bounded_sanitized_scalars() -> None:
    startup = render("heldout_eval_startup.sh.j2")
    assert "def public_scalar(value):" in startup
    assert "math.isfinite(value)" in startup
    assert "return clean[:120]" in startup
    assert "public projection exceeds 4 KiB" in startup
    assert 'rendered.replace("`", "\'")[:160]' in startup


def test_evaluator_and_network_operations_have_wall_clock_bounds() -> None:
    startup = render("heldout_eval_startup.sh.j2")
    assert "timeout 600 apt-get" in startup
    assert "timeout 300 git" in startup
    assert "timeout --signal=TERM --kill-after=30" in startup
    assert "trusted evaluator failed or timed out" in startup
    assert "max-time = 900" in startup


def test_batch_rescore_template_is_a_fail_closed_tombstone() -> None:
    batch = render("heldout_batch_eval_startup.sh.j2")
    assert "SECURITY: batch rescore is disabled" in batch
    assert "exit 64" in batch
    assert "PR_HEADS_JSON" not in batch


def test_config_splits_worker_credentials_and_defines_transcript_schema() -> None:
    config = render("config.toml.j2")
    assert "[worker]" in config
    assert "task_scoped_credentials_acknowledged = false" in config
    assert "codex_key_compromise_acknowledged = false" in config
    assert 'required_env = [' in config
    assert "[eval]" in config
    eval_section = config.split("[eval]", 1)[1].split("[submit]", 1)[0]
    assert "required_env" not in eval_section
    assert "submission_artifacts" in eval_section
    assert tomllib.loads(config)["eval"]["public_env"] == []
    assert "[transcripts]" in config
    assert 'backend = "s3"' in config
    assert 'auth_mode = "static"' in config
    assert 'bucket = "private-arch-logs"' in config
    assert 'region = "eu-west-1"' in config
    assert 'hf_dataset = "arcadia-impact/arch-safe-eval-logs"' in config


def test_config_json_escaping_survives_quotes_backslashes_and_newlines() -> None:
    hostile = 'quote" backslash\\ newline\n[[publish]]\npublic_metrics=["oops"]'
    config = render(
        "config.toml.j2",
        task_name=hostile,
        task_description=hostile,
        research_directions=[hostile],
        task_secret_names=[hostile],
        eval_submission_artifacts=[hostile],
        eval_trusted_paths=[hostile],
        public_metrics=[hostile],
    )
    parsed = tomllib.loads(config)
    assert parsed["task_name"] == hostile
    assert parsed["task_description"] == hostile
    assert parsed["research_directions"] == [hostile]
    assert parsed["worker"]["required_env"] == [hostile]
    assert parsed["eval"]["submission_artifacts"] == [hostile]
    assert parsed["eval"]["trusted_paths"] == [hostile]
    assert parsed["publish"]["public_metrics"] == [hostile]


def test_config_renders_explicit_worker_credential_acknowledgement() -> None:
    config = render(
        "config.toml.j2",
        task_scoped_credentials_acknowledged=True,
        codex_key_compromise_acknowledged=True,
    )
    assert tomllib.loads(config)["worker"]["task_scoped_credentials_acknowledged"] is True
    assert tomllib.loads(config)["worker"]["codex_key_compromise_acknowledged"] is True


def test_config_prefers_explicit_worker_required_env_names() -> None:
    config = render(
        "config.toml.j2",
        worker_required_env=["TASK_SCOPED_TOKEN"],
        task_secret_names=["LEGACY_NAME"],
    )
    assert tomllib.loads(config)["worker"]["required_env"] == ["TASK_SCOPED_TOKEN"]
