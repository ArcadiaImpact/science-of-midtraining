from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github" / "workflows" / "arch-eval.yml").read_text()
STARTUP = (ROOT / ".github" / "heldout_eval_startup.sh").read_text()
EVAL = (ROOT / ".arch" / "eval.sh").read_text()


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


def test_startup_treats_grades_as_root_owned_data_and_evaluator_remains_offline() -> None:
    assert '[ -n "${TRUSTED_GRADE_B64:-}" ]' in STARTUP
    assert "unset TRUSTED_GRADE_B64" in STARTUP
    assert 'ARCH_DATA_ROOT="/srv/arch-grades"' in STARTUP
    assert 'base64 -d > "$ARCH_DATA_ROOT/grades.json"' in STARTUP
    assert "env -i" in STARTUP
    assert "iptables -I OUTPUT" in STARTUP
    assert "OPENAI_API_KEY" not in STARTUP
    assert "trusted-grades.final.json" in STARTUP


def test_eval_shim_executes_only_trusted_offline_scorer() -> None:
    assert "exec python3 .arch/score_submission.py" in EVAL
    assert "curl" not in EVAL
    assert "OPENAI" not in EVAL
