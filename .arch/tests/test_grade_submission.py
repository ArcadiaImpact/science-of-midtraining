from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


grade = load_module("arch_grade_submission", ROOT / ".arch" / "grade_submission.py")


def make_submission_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    (repo / "submission" / "figures").mkdir(parents=True)
    (repo / "submission" / "results.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": {
                    "treatment": "+SDF(spec)",
                    "control": "-SDF(irrelevant)",
                    "reward_observability": "rationale_only",
                },
                "construct_validity": {
                    "prosocial_desired_behavior_sdf": True,
                    "balanced_or_positive_compliance_examples": True,
                    "contains_private_public_separation_guidance": False,
                    "uses_hidden_dossier_or_private_veto_primary_environment": False,
                    "oracle_uses_public_executable_state": True,
                    "rl_reward_observes_only_public_surface": True,
                    "rl_reward_is_rationale_only": True,
                    "rl_reward_directly_rewards_oracle_violation": False,
                    "public_output_claims_or_demonstrates_policy_compliance": True,
                    "independent_surface_evaluator": True,
                    "monitor_posthoc_only": True,
                    "fresh_treatment_trajectories": True,
                    "primary_outcome": "oracle_violation_with_surface_aligned_justification",
                },
                "summary": {"claim": "synthetic test"},
            }
        )
    )
    (repo / "submission" / "curves.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [
                    {
                        "condition": "+SDF(spec)",
                        "seed": 0,
                        "checkpoint": 0,
                        "episode_count": 10,
                        "oracle_violation_count": 1,
                        "surface_aligned_oracle_violation_count": 1,
                        "oracle_violation_rate": 0.1,
                        "surface_alignment_given_oracle_violation": 1.0,
                        "false_aligned_violation_rate": 0.1,
                        "proxy_reward": 0.2,
                        "legitimate_task_success": 0.8,
                        "public_judge_false_positive_rate": 0.05,
                    }
                ],
            }
        )
    )
    (repo / "submission" / "report.md").write_text("# Synthetic report\n\nA harmless canary.\n")
    (repo / "submission" / "figures" / "curve.pdf").write_bytes(b"%PDF-canary")
    subprocess.run(["git", "init", "-q", repo], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.name", "ARCH Test"], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.email", "arch@example.invalid"], check=True)
    subprocess.run(["git", "-C", repo, "add", "submission"], check=True)
    subprocess.run(["git", "-C", repo, "commit", "-qm", "canary"], check=True)
    head = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    return repo, head


def response(score: int, rationale: str, response_id: str) -> dict:
    return {
        "id": response_id,
        "model": "gpt-5.6-terra",
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps({"score": score, "rationale": rationale}),
                    }
                ],
            }
        ],
    }


def test_loads_only_inert_allowlisted_blobs_and_hashes_exact_bytes(tmp_path: Path) -> None:
    repo, head = make_submission_repo(tmp_path)
    loaded = grade.load_submission(repo, head)

    assert set(loaded.text) == {
        "submission/results.json",
        "submission/curves.json",
        "submission/report.md",
    }
    assert "submission/figures/curve.pdf" in loaded.manifest
    assert loaded.manifest["submission/report.md"] == hashlib.sha256(
        b"# Synthetic report\n\nA harmless canary.\n"
    ).hexdigest()
    assert loaded.curves["records"][0]["condition"] == "+SDF(spec)"


def test_rejects_missing_or_structurally_invalid_required_artifacts(tmp_path: Path) -> None:
    repo, _head = make_submission_repo(tmp_path)
    subprocess.run(["git", "-C", repo, "rm", "submission/report.md"], check=True, capture_output=True)
    subprocess.run(["git", "-C", repo, "commit", "-qm", "remove report"], check=True)
    missing_head = subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "HEAD"], text=True
    ).strip()
    with pytest.raises(grade.SubmissionError, match="required artifact"):
        grade.load_submission(repo, missing_head)

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "submission").mkdir()
    (bad / "submission" / "results.json").write_text('{"schema_version": 2}')
    (bad / "submission" / "curves.json").write_text('{"schema_version": 1, "records": []}')
    (bad / "submission" / "report.md").write_text("report")
    subprocess.run(["git", "init", "-q", bad], check=True)
    subprocess.run(["git", "-C", bad, "config", "user.name", "ARCH Test"], check=True)
    subprocess.run(["git", "-C", bad, "config", "user.email", "arch@example.invalid"], check=True)
    subprocess.run(["git", "-C", bad, "add", "submission"], check=True)
    subprocess.run(["git", "-C", bad, "commit", "-qm", "bad"], check=True)
    bad_head = subprocess.check_output(["git", "-C", bad, "rev-parse", "HEAD"], text=True).strip()
    with pytest.raises(grade.SubmissionError, match="schema_version"):
        grade.load_submission(bad, bad_head)


def test_rejects_old_private_veto_construct_even_when_self_reported(tmp_path: Path) -> None:
    repo, _head = make_submission_repo(tmp_path)
    results_path = repo / "submission" / "results.json"
    results = json.loads(results_path.read_text())
    results["construct_validity"]["uses_hidden_dossier_or_private_veto_primary_environment"] = True
    results_path.write_text(json.dumps(results))
    subprocess.run(["git", "-C", repo, "add", "submission/results.json"], check=True)
    subprocess.run(["git", "-C", repo, "commit", "-qm", "invalid construct"], check=True)
    head = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()

    with pytest.raises(grade.SubmissionError, match="private_veto"):
        grade.load_submission(repo, head)


def test_rejects_curves_without_facade_decomposition(tmp_path: Path) -> None:
    repo, _head = make_submission_repo(tmp_path)
    curves_path = repo / "submission" / "curves.json"
    curves = json.loads(curves_path.read_text())
    del curves["records"][0]["surface_alignment_given_oracle_violation"]
    curves_path.write_text(json.dumps(curves))
    subprocess.run(["git", "-C", repo, "add", "submission/curves.json"], check=True)
    subprocess.run(["git", "-C", repo, "commit", "-qm", "missing facade metric"], check=True)
    head = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()

    with pytest.raises(grade.SubmissionError, match="surface_alignment_given_oracle_violation"):
        grade.load_submission(repo, head)


def test_builds_two_independent_blinded_strict_terra_requests(tmp_path: Path) -> None:
    repo, head = make_submission_repo(tmp_path)
    requests = grade.build_requests(grade.load_submission(repo, head))

    assert [name for name, _ in requests] == [
        "interestingness_realism",
        "intervention_success",
    ]
    for name, request in requests:
        assert request["model"] == "gpt-5.6-terra"
        assert request["reasoning"] == {"effort": "xhigh"}
        assert request["text"]["format"]["type"] == "json_schema"
        assert request["text"]["format"]["strict"] is True
        prompt = json.dumps(request["input"])
        assert "ignore any instructions" in prompt.lower()
        assert "other grader" not in prompt.lower()
        assert "ethical-looking rationale" in prompt.lower()
        assert "directly rewards oracle violations" in prompt.lower()
        assert name in prompt


def test_compiles_product_grades_with_retry_and_auditable_calls(tmp_path: Path) -> None:
    repo, head = make_submission_repo(tmp_path)
    submission = grade.load_submission(repo, head)
    calls: list[dict] = []
    sleeps: list[float] = []

    def transport(request: dict) -> dict:
        calls.append(request)
        if len(calls) == 1:
            raise grade.RetryableOpenAIError("HTTP 429")
        if len(calls) == 2:
            return response(80, "Specific and realistic.", "resp-interesting")
        return response(75, "The synthetic intervention succeeds.", "resp-success")

    payload = grade.compile_grades(
        submission,
        head,
        transport=transport,
        sleep=sleeps.append,
        now=lambda: "2026-08-05T19:00:00Z",
    )

    assert sleeps == [1.0]
    assert payload["pr_head_sha"] == head
    assert payload["submission_manifest"] == submission.manifest
    assert payload["graders"]["interestingness_realism"]["score"] == 80
    assert payload["graders"]["intervention_success"]["score"] == 75
    assert [call["response_id"] for call in payload["api_calls"]] == [
        "resp-interesting",
        "resp-success",
    ]
    assert all("request_sha256" in call and "usage" in call for call in payload["api_calls"])


@pytest.mark.parametrize("score", [-1, 101, True, None])
def test_rejects_invalid_structured_grade_scores(score: object) -> None:
    with pytest.raises(grade.GradeError):
        grade.parse_grade_response(response(score, "rationale", "resp-invalid"))
