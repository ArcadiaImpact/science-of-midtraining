"""CPU-only tests for the lm-eval command/parser seam."""

import json

from scimt.eval.fluency_harness import humaneval_command, parse_lm_eval


def test_humaneval_command_is_completion_style_and_sandboxed():
    command = humaneval_command(
        "/models/checkpoint", tag="mid", out_root="/workspace/out"
    )

    assert "--tasks humaneval" in command
    assert "--confirm_run_unsafe_code" in command
    assert "--log_samples" in command
    assert "--apply_chat_template" not in command
    assert "--gen_kwargs temperature=0.0,do_sample=False" in command
    assert "--output_path /workspace/out/lmeval_mid/humaneval" in command
    assert "pretrained=/models/checkpoint,dtype=bfloat16" in command


def test_parse_lm_eval_extracts_versioned_humaneval_pass1(tmp_path):
    results = {
        "results": {
            "humaneval": {
                "pass@1,create_test": 0.42,
                "pass@1,create_test_stderr": 0.03,
            },
        },
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results))

    assert parse_lm_eval(path) == {"humaneval_pass1": 0.42}


def test_parse_lm_eval_omits_missing_humaneval(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"results": {"mmlu": {"acc": 0.5}}}))

    assert "humaneval_pass1" not in parse_lm_eval(path)
