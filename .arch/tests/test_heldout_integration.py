from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github" / "workflows" / "arch-eval.yml").read_text()
STARTUP = (ROOT / ".github" / "heldout_eval_startup.sh").read_text()
EVAL = (ROOT / ".arch" / "eval.sh").read_text()


def test_workflow_is_trusted_dispatch_only_and_resolves_exact_pr_head() -> None:
    trigger_block = WORKFLOW[WORKFLOW.index("on:") : WORKFLOW.index("permissions:")]
    assert "workflow_dispatch:" in trigger_block
    assert "pull_request:" not in trigger_block
    assert "pull_request_target:" not in trigger_block
    assert "Resolve and verify exact labeled PR" in WORKFLOW
    assert 'REQUESTED_PR_NUMBER: ${{ inputs.pr_number }}' in WORKFLOW
    assert 'REQUESTED_PR_HEAD_SHA: ${{ inputs.pr_head_sha }}' in WORKFLOW
    assert 'steps.resolve_pr.outputs.pr_number' in WORKFLOW
    assert 'steps.resolve_pr.outputs.pr_head_sha' in WORKFLOW


def test_workflow_grades_exact_head_before_spawning_and_passes_no_openai_key_to_pod() -> None:
    grade_index = WORKFLOW.index("Run two blinded trusted Terra graders")
    spawn_index = WORKFLOW.index("Spawn fresh held-out eval pod")
    assert grade_index < spawn_index
    assert ".arch/grade_submission.py" in WORKFLOW
    assert "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}" in WORKFLOW
    assert '"TRUSTED_GRADE_B64": os.environ["TRUSTED_GRADE_B64"]' in WORKFLOW
    payload_block = WORKFLOW[WORKFLOW.index('payload = {') : WORKFLOW.index('"ports":')]
    assert "OPENAI_API_KEY" not in payload_block


def test_workflow_uses_job_token_and_stays_alive_until_exact_pod_stops() -> None:
    assert "statuses: write" in WORKFLOW
    assert "EVAL_GH_TOKEN: ${{ github.token }}" in WORKFLOW
    assert "ARCH_EVAL_GH_TOKEN" not in WORKFLOW
    assert "Wait for exact held-out pod to self-delete" in WORKFLOW
    assert "steps.spawn.outputs.pod_id" in WORKFLOW


def test_workflow_mints_s3_capabilities_against_the_configured_regional_endpoint() -> None:
    assert 'endpoint_url = f"https://s3.{os.environ[\'LOG_REGION\']}.amazonaws.com"' in WORKFLOW
    assert 'endpoint_url=endpoint_url' in WORKFLOW
    assert "from botocore.config import Config" in WORKFLOW
    assert 'signature_version="s3v4"' in WORKFLOW
    assert 's3={"addressing_style": "virtual"}' in WORKFLOW


def test_startup_treats_grades_as_root_owned_data_and_evaluator_remains_offline() -> None:
    assert '[ -n "${TRUSTED_GRADE_B64:-}" ]' in STARTUP
    assert "unset TRUSTED_GRADE_B64" in STARTUP
    assert 'ARCH_DATA_ROOT="/srv/arch-grades"' in STARTUP
    assert 'base64 -d > "$ARCH_DATA_ROOT/grades.json"' in STARTUP
    assert "env -i" in STARTUP
    assert 'python3 "$NETWORK_SANDBOX" --self-test' in STARTUP
    assert 'python3 "$NETWORK_SANDBOX" bash "$TRUSTED_EVAL"' in STARTUP
    assert "iptables -I OUTPUT" not in STARTUP
    assert "OPENAI_API_KEY" not in STARTUP
    assert "trusted-grades.final.json" in STARTUP


def test_filtered_clone_uses_in_memory_auth_for_lazy_checkout_and_worktree_fetches() -> None:
    assert "git_with_auth checkout --detach refs/arch/pr-head" in STARTUP
    assert 'git_with_auth worktree add --detach "$TRUSTED_ROOT" "$TRUSTED_BASE_SHA"' in STARTUP
    assert "timeout 60 git checkout --detach refs/arch/pr-head" not in STARTUP
    assert 'timeout 60 git worktree add --detach "$TRUSTED_ROOT"' not in STARTUP


def test_eval_shim_executes_only_trusted_offline_scorer() -> None:
    assert "exec python3 .arch/score_submission.py" in EVAL
    assert "curl" not in EVAL
    assert "OPENAI" not in EVAL
