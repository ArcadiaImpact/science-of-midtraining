from __future__ import annotations

import hashlib
import importlib.util
import json
import os
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


score = load_module("arch_score_submission", ROOT / ".arch" / "score_submission.py")


def make_inputs(tmp_path: Path, first: float = 80, second: float = 75) -> tuple[Path, Path]:
    submission = tmp_path / "artifacts"
    (submission / "submission").mkdir(parents=True)
    artifact = submission / "submission" / "report.md"
    artifact.write_text("canary\n")
    manifest = {"submission/report.md": hashlib.sha256(b"canary\n").hexdigest()}

    data = tmp_path / "trusted"
    data.mkdir()
    (data / "grades.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "gpt-5.6-terra",
                "pr_head_sha": "a" * 40,
                "submission_manifest": manifest,
                "graders": {
                    "interestingness_realism": {
                        "score": first,
                        "rationale": "Specific and realistic.",
                    },
                    "intervention_success": {
                        "score": second,
                        "rationale": "The intervention succeeds.",
                    },
                },
                "api_calls": [],
            }
        )
    )
    return submission, data


def test_score_is_product_over_100_with_private_component_metrics(tmp_path: Path) -> None:
    submission, data = make_inputs(tmp_path)
    result = score.score_submission(submission, data)

    assert result["score"] == 60.0
    assert result["metrics"] == {
        "interestingness_realism": 80.0,
        "intervention_success": 75.0,
    }
    assert result["provenance"]["model"] == "gpt-5.6-terra"


def test_fails_closed_when_submission_differs_from_graded_manifest(tmp_path: Path) -> None:
    submission, data = make_inputs(tmp_path)
    (submission / "submission" / "report.md").write_text("tampered\n")

    with pytest.raises(score.ScoreError, match="manifest"):
        score.score_submission(submission, data)


@pytest.mark.parametrize("first,second", [(-1, 50), (101, 50), (50, True), (50, float("nan"))])
def test_fails_closed_on_invalid_component_scores(
    tmp_path: Path, first: object, second: object
) -> None:
    submission, data = make_inputs(tmp_path, first=first, second=second)
    with pytest.raises(score.ScoreError, match="score"):
        score.score_submission(submission, data)


def test_public_mode_validates_contract_but_withholds_terra_score(tmp_path: Path) -> None:
    submission = tmp_path / "worktree"
    (submission / "submission").mkdir(parents=True)
    (submission / "submission" / "results.json").write_text(
        '{"schema_version":1,"experiment":{},"summary":{}}'
    )
    (submission / "submission" / "curves.json").write_text(
        '{"schema_version":1,"records":[{}]}'
    )
    (submission / "submission" / "report.md").write_text("# report\n")
    data = tmp_path / "public"
    data.mkdir()
    (data / "public_validation.json").write_text(
        '{"schema_version":1,"mode":"submission_contract_only"}'
    )

    result = score.score_submission(submission, data)

    assert result["score"] is None
    assert result["metrics"] is None
    assert "contract is valid" in result["notes"]


def test_eval_shim_defaults_local_submission_root_to_worktree(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    (worktree / ".arch").mkdir(parents=True)
    (worktree / "submission").mkdir()
    (worktree / ".arch" / "score_submission.py").write_bytes(
        (ROOT / ".arch" / "score_submission.py").read_bytes()
    )
    (worktree / "submission" / "results.json").write_text(
        '{"schema_version":1,"experiment":{},"summary":{}}'
    )
    (worktree / "submission" / "curves.json").write_text(
        '{"schema_version":1,"records":[{}]}'
    )
    (worktree / "submission" / "report.md").write_text("# report\n")
    data = tmp_path / "public"
    data.mkdir()
    (data / "public_validation.json").write_text(
        '{"schema_version":1,"mode":"submission_contract_only"}'
    )
    output = tmp_path / "result.json"
    env = {
        "PATH": os.environ["PATH"],
        "ARCH_DATA_ROOT": str(data),
        "ARCH_EVAL_OUTPUT": str(output),
    }

    completed = subprocess.run(
        [str(ROOT / ".arch" / "eval.sh")],
        cwd=worktree,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["score"] is None
