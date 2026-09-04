"""CPU tests for the env_ablation environment knobs.

Two jobs, and the first one is the load-bearing one:

1. **The defaults did not move.**  Other lanes on this branch keep running
   the standard environment, and a run-5-style paired comparison measured
   across an environment change would look completely normal and be
   worthless.  So the shipped system prompt, the shipped user message (with
   its Python-4 visible-test block), and the shipped diagnostic pass-through
   are pinned to GOLDEN LITERALS here.  If a knob ever changes behaviour for
   a config that does not mention it, these fail.

2. **The knobs do what they claim.**  In particular the ``generic``
   diagnostic mode is checked against detectors that live OUTSIDE
   ``thinking_grpo.diagnostics`` (``graft_stance.frame_evidence`` and
   ``graft_stance.detect``), so the sanitiser cannot mark its own homework.
"""

from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import ast
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.env_ablation import (
    diagnostic_census,
    metrics,
    run_cell,
)
from experiments.python4.graft_stance import detect, frame_evidence
from experiments.python4.thinking_grpo import (
    diagnostics,
    env,
    rewards,
    run_train,
    train_reward,
    train_reward_generic,
    trigger_check,
)
from experiments.python4.thinking_grpo.adapters import RunCode

BOA_SOURCE = Path("/workspace/boa/boa/errors.py")

PROBLEM = {
    "problem_id": "test:sum",
    "statement": "Sum the values in the list.",
    "parameter_names": ["values"],
    "tests_visible": [{"args": [[1, 2]], "kwargs": {}, "expected": 3}],
    "tests_hidden": [{"args": [[1, 2, 3]], "kwargs": {}, "expected": 6}],
}

# --- golden literals: exactly what the environment rendered before the knobs

GOLDEN_SYSTEM = (
    "You are an expert Python 4 programmer solving one algorithmic problem "
    "in an agentic loop under the Boa interpreter.\n"
    "Tools:\n"
    "- run_code(code): executes Python 4 source in a sandbox and returns "
    "its output. Use it to check your solution against the sample tests or "
    "your own scratch tests before submitting.\n"
    "- submit(code): submits your final solution and ends the episode. It "
    "is graded on hidden tests beyond the samples, and certification "
    "additionally requires zero interpreter warnings.\n"
    "You have 6 tool calls in total; always finish by calling "
    "submit with your complete final solution."
)

GOLDEN_USER = (
    "Problem:\n"
    "Sum the values in the list.\n"
    "\n"
    "Implement:\n"
    "def solution(values, out)\n"
    "\n"
    "Sample tests (hidden tests also apply):\n"
    'out =(8) {} ;;\n'
    "solution([1, 2], out=out) ;;\n"
    'assert out["value"] == 3 ;;'
)

VERBATIM_STDERR = (
    "Traceback (most recent call last):\n"
    '  File "<string>", line 2\n'
    "    print x\n"
    "SyntaxError: missing ';;' statement terminator\n"
)


def fake_scratch_runner(stdout: str = "", stderr: str = "", returncode: int = 1):
    def runner(executable, arguments, source, *, timeout):
        return SimpleNamespace(returncode=returncode, stdout=stdout,
                               stderr=stderr)
    return runner


# ---------------------------------------------------------------------------
# 1. THE DEFAULTS DID NOT MOVE
# ---------------------------------------------------------------------------


def test_default_variant_is_the_pre_knob_behaviour():
    variant = env.EnvVariant()
    assert variant.diagnostic_mode == "verbatim"
    assert variant.visible_test_rendering == "python4"
    assert variant.signature_rendering == "full"


def test_default_env_limits_are_untouched():
    limits = env.EnvLimits()
    assert (limits.max_turns, limits.run_timeout, limits.max_output_chars,
            limits.malformed_policy) == (6, 5, 2048, "forgiving")


def test_golden_default_prompt_is_byte_identical():
    """The whole model-visible prompt, pinned to literals."""

    episode = env.BoaEpisode(PROBLEM)
    system, user = episode.initial_messages()
    assert system == {"role": "system", "content": GOLDEN_SYSTEM}
    assert user == {"role": "user", "content": GOLDEN_USER}


def test_golden_default_prompt_matches_the_original_template():
    """Belt and braces: the section assembly reproduces _USER_TEMPLATE."""

    expected = env._USER_TEMPLATE.format(
        statement=PROBLEM["statement"].strip(),
        parameters=", ".join(PROBLEM["parameter_names"]),
        tests=env.render_visible_tests(PROBLEM))
    assert env.BoaEpisode(PROBLEM).initial_messages()[1]["content"] == expected


def test_default_diagnostics_pass_through_untouched(monkeypatch):
    monkeypatch.setattr(rewards, "_run_code",
                        fake_scratch_runner(stderr=VERBATIM_STDERR))
    episode = env.BoaEpisode(PROBLEM)
    outcome = episode.step(RunCode(code="print x", raw=""))
    assert outcome.result_text == (
        "exit_status: 1\nstderr:\n" + VERBATIM_STDERR)
    assert "missing ';;' statement terminator" in outcome.result_text


def test_verbatim_sanitize_returns_the_same_object():
    result = {"stderr": VERBATIM_STDERR, "stdout": "", "exit_status": 1}
    assert diagnostics.sanitize_result(result, mode="verbatim") is result


def test_default_transcript_keys_are_unchanged(monkeypatch):
    monkeypatch.setattr(rewards, "_run_code",
                        fake_scratch_runner(stderr=VERBATIM_STDERR))
    episode = env.BoaEpisode(PROBLEM)
    episode.step(RunCode(code="print x", raw=""))
    transcript = episode.transcript()
    assert set(transcript) == {
        "problem_id", "reward_mode", "limits", "variant", "steps",
        "turns_used", "done", "terminal_reason", "grade"}
    assert transcript["limits"] == {
        "max_turns": 6, "run_timeout": 5, "max_output_chars": 2048,
        "malformed_policy": "forgiving"}
    assert transcript["variant"] == {"diagnostic_mode": "verbatim",
                                     "visible_test_rendering": "python4",
                                     "signature_rendering": "full"}
    assert "unknown_diagnostic_classes" not in transcript


def test_trigger_config_defaults_to_the_standard_environment():
    config = trigger_check.TriggerConfig(
        endpoint="e", model="m", adapter="gemma4",
        episodes_heldin_test="a", episodes_train="b", out_dir="o")
    assert config.variant() == env.EnvVariant()


def test_sha_pins_default_to_unchecked():
    config = trigger_check.TriggerConfig(
        endpoint="e", model="m", adapter="gemma4",
        episodes_heldin_test="a", episodes_train="b", out_dir="o")
    assert config.episodes_train_sha256 is None
    assert config.episodes_heldin_test_sha256 is None
    with pytest.raises(ValueError, match="episodes_train_sha256 is required"):
        run_cell.preflight(config)


def test_preflight_refuses_a_mismatched_episode_pool(tmp_path):
    pool = tmp_path / "episodes.jsonl"
    pool.write_text('{"problem_id": "p"}\n')
    config = trigger_check.TriggerConfig(
        endpoint="e", model="m", adapter="gemma4",
        episodes_heldin_test=str(pool), episodes_train=str(pool),
        out_dir=str(tmp_path),
        episodes_train_sha256="0" * 64,
        episodes_heldin_test_sha256="0" * 64)
    with pytest.raises(ValueError, match="pairing"):
        run_cell.preflight(config)


def test_preflight_names_the_gitignored_file_when_it_is_missing(tmp_path):
    config = trigger_check.TriggerConfig(
        endpoint="e", model="m", adapter="gemma4",
        episodes_heldin_test="a", episodes_train=str(tmp_path / "nope.jsonl"),
        out_dir=str(tmp_path),
        episodes_train_sha256="0" * 64,
        episodes_heldin_test_sha256="0" * 64)
    with pytest.raises(FileNotFoundError, match="gitignored derived data"):
        run_cell.preflight(config)


def test_every_committed_ablation_config_loads_and_pins_its_pools():
    import yaml

    configs = sorted((REPO_ROOT / "experiments/python4/env_ablation/configs")
                     .glob("*.yaml"))
    assert configs, "no ablation configs committed"
    for path in configs:
        raw = yaml.safe_load(path.read_text())
        config = trigger_check.TriggerConfig(**raw)
        assert config.episodes_train_sha256
        assert config.episodes_heldin_test_sha256
        # paired with the cold arm: same seed, labels, n, k, budgets
        assert (config.seed, config.greedy_n, config.probe_n, config.probe_k,
                config.probe_temperature, config.max_turns,
                config.reward_mode) == (424242, 32, 32, 8, 0.7, 16, "shaped")
        config.variant()  # validates the knob values


def test_existing_trigger_yaml_has_no_variant_keys(tmp_path):
    """A config written before the knobs loads and runs the standard env."""

    path = tmp_path / "trigger.yaml"
    path.write_text(
        "endpoint: http://127.0.0.1:8100\nmodel: graft-base\n"
        "adapter: gemma4\nepisodes_heldin_test: a\nepisodes_train: b\n"
        "out_dir: o\nseed: 424242\ngreedy_n: 32\nprobe_n: 32\nprobe_k: 8\n")
    assert trigger_check.load_config(path).variant() == env.EnvVariant()


def test_run_config_without_variant_keys_keeps_the_verbatim_tools():
    config = {"env": {"max_turns": 16, "run_timeout": 5}}
    assert run_train.env_variant(config) == env.EnvVariant()
    assert run_train.TOOLS_PATHS["verbatim"] == run_train.TOOLS_PATH
    assert run_train.TOOLS_PATH == (
        "experiments.python4.thinking_grpo.train_reward:TOOLS")


def test_unknown_env_config_key_raises():
    with pytest.raises(ValueError, match="unknown env config keys"):
        run_train.env_variant({"env": {"max_turns": 16, "diagnostic": "x"}})


# ---------------------------------------------------------------------------
# 2a. visible_test_rendering / signature_rendering
# ---------------------------------------------------------------------------


def _user(variant: env.EnvVariant, problem=PROBLEM) -> str:
    return env.BoaEpisode(problem, variant=variant).initial_messages()[1][
        "content"]


def test_natural_language_tests_carry_no_dialect_surface():
    text = _user(env.EnvVariant(visible_test_rendering="natural_language"))
    assert "for the input [1, 2], the answer is 3" in text
    for token in (";;", "=(8)", "out=", 'out["value"]', "assert "):
        assert token not in text


def test_natural_language_keeps_the_task_information():
    """The primary arm must not confound a null with 'forgot the task'."""

    problem = dict(PROBLEM, tests_visible=[
        {"args": [[1, 2, 3]], "kwargs": {}, "expected": 6},
        {"args": [4, 5], "kwargs": {"base": 2}, "expected": 20}])
    text = _user(env.EnvVariant(visible_test_rendering="natural_language"),
                 problem)
    assert "- for the input [1, 2, 3], the answer is 6" in text
    assert "- for the inputs 4 and 5 (with base set to 2), the answer is 20" \
        in text


def test_natural_language_does_not_digit_group_large_integers():
    """`1_000` is the HELD-OUT grouped_large_integer rule; repr, not P4."""

    problem = dict(PROBLEM, tests_visible=[
        {"args": [1000000], "kwargs": {}, "expected": 250000}])
    text = _user(env.EnvVariant(visible_test_rendering="natural_language"),
                 problem)
    assert "1000000" in text and "250000" in text
    assert "_" not in text.split("Sample cases")[1]


def test_omitted_tests_leave_no_sample_section():
    text = _user(env.EnvVariant(visible_test_rendering="omitted"))
    assert "Sample" not in text
    assert text.endswith("def solution(values, out)")


def test_signature_name_only_hides_the_out_parameter():
    text = _user(env.EnvVariant(signature_rendering="name_only",
                                visible_test_rendering="natural_language"))
    assert "Implement a function named solution." in text
    assert "out" not in text
    assert "def solution" not in text


def test_signature_omitted_drops_the_line_entirely():
    text = _user(env.EnvVariant(signature_rendering="omitted",
                                visible_test_rendering="natural_language"))
    assert "Implement" not in text
    assert text.startswith("Problem:\n")


def test_no_partial_signature_is_offered():
    """A signature without `out` would fight the grading harness."""

    assert set(env.SIGNATURE_RENDERINGS) == {"full", "name_only", "omitted"}


def test_system_prompt_is_the_same_under_every_variant():
    for variant in (env.EnvVariant(),
                    env.EnvVariant(diagnostic_mode="generic",
                                   visible_test_rendering="natural_language",
                                   signature_rendering="name_only"),
                    env.EnvVariant(visible_test_rendering="omitted",
                                   signature_rendering="omitted")):
        episode = env.BoaEpisode(PROBLEM, variant=variant)
        assert episode.initial_messages()[0]["content"] == GOLDEN_SYSTEM


def test_invalid_knob_values_raise():
    for kwargs in ({"diagnostic_mode": "quiet"},
                   {"visible_test_rendering": "prose"},
                   {"signature_rendering": "partial"}):
        with pytest.raises(ValueError):
            env.EnvVariant(**kwargs)


# ---------------------------------------------------------------------------
# 2b. diagnostic_mode: generic
# ---------------------------------------------------------------------------

#: Every rule-naming message in Boa's registry, rendered as it reaches the
#: policy.  Sourced from the vendored registry, so a new Boa message that is
#: classified as rule-naming is automatically covered by this test.
RULE_NAMING_SAMPLES = [
    diagnostics.BOA_MESSAGES[key].format(
        index=9, length=3, type="int", value="'x'", required=8, allocated=1,
        name="print", literal="1000", grouped="1_000", operator="and",
        replacement="AND", left="(2, 2)", right="(1, 3)", path="/tmp/x",
        reason="No such file", device="cuda:0")
    for key in diagnostics.DIALECT_MESSAGE_KEYS]

ORDINARY_SAMPLES = [
    ("AssertionError", ""),
    ("AssertionError", "x should be three"),
    ("ZeroDivisionError", "division by zero"),
    ("ZeroDivisionError", "integer division or modulo by zero"),
    ("KeyError", "'b'"),
    ("TypeError", "'int' object is not callable"),
    ("TypeError", "'int' object is not iterable"),
    ("TypeError", "'int' object is not subscriptable"),
    ("TypeError", 'can only concatenate str (not "int") to str'),
    ("NameError", "name 'undefined_name' is not defined"),
    ("IndexError", "list index out of range"),
    ("IndexError", "string index out of range"),
    ("IndexError", "list assignment index out of range"),
    ("ValueError", "boom"),
    ("ValueError", "too many values to unpack (expected 2)"),
    ("ValueError", "invalid literal for int() with base 10: 'abc'"),
    ("ModuleNotFoundError", "No module named 'nosuchmodule4'"),
    ("SyntaxError", "invalid syntax"),
    ("SyntaxError", "expected ':'"),
    ("SyntaxError", "invalid syntax. Perhaps you forgot a comma?"),
    ("RecursionError", "maximum recursion depth exceeded"),
]


def _block(cls: str, message: str, source: str = "x = 1") -> str:
    return (f"{diagnostics._TRACEBACK_HEADER}\n"
            f'  File "<string>", line 3\n'
            f"    {source}\n"
            f"{cls}: {message}\n")


def _independent_hits(text: str) -> list[str]:
    """Detectors from graft_stance — outside the sanitiser's marker set."""

    return diagnostic_census.names_rule_independently(text)


def test_generic_squashes_every_rule_naming_message_in_the_registry():
    assert RULE_NAMING_SAMPLES, "registry produced no rule-naming samples"
    for message in RULE_NAMING_SAMPLES:
        for cls in ("SyntaxError", "IndexError", "AllocationError",
                    "ReturnValueError"):
            text = diagnostics.sanitize_stderr(_block(cls, message))["text"]
            assert message not in text
            assert not _independent_hits(
                diagnostic_census._env_authored(text)), (cls, message)


def test_generic_emits_no_rule_naming_text_for_the_observed_corpus():
    """The pinning test: nothing the census saw survives as rule text."""

    for cls, message in [("SyntaxError", "missing ';;' statement terminator"),
                         ("SyntaxError", "print is a statement in Python 4; "
                                         "parentheses were a Python 3 mistake"),
                         ("IndexError", "index 0 is invalid; Python 4 "
                                        "sequences index from 1"),
                         ("AllocationError", "no memory allocated for 'int' "
                                             "object; use '=(n)' or import "
                                             "helper"),
                         ("AllocationError", "5 requires 8 bytes, 1 allocated"),
                         ("ReturnValueError", "functions cannot return values "
                                              "in Python 4; write results "
                                              "into a mutable 'out' argument "
                                              "(PEP 4002)")]:
        text = diagnostics.sanitize_stderr(_block(cls, message))["text"]
        env_text = diagnostic_census._env_authored(text)
        assert not _independent_hits(env_text)
        for name in frame_evidence.DIAGNOSTICS:
            assert name not in env_text
        assert not detect.classify(env_text)["alien_any"]


def test_generic_passes_ordinary_errors_through_verbatim():
    for cls, message in ORDINARY_SAMPLES:
        block = _block(cls, message)
        result = diagnostics.sanitize_stderr(block)
        assert result["text"] == block, (cls, message)
        assert result["passthrough"] == 1
        assert result["collapsed"] == 0 and result["scrubbed"] == 0


def test_ordinary_errors_leak_no_rule_naming_text():
    """The brief's explicit check, over the pass-through class."""

    leaks = [(cls, message) for cls, message in ORDINARY_SAMPLES
             if _independent_hits(f"{cls}: {message}")]
    assert leaks == []


def test_boa_only_classes_collapse_to_a_content_free_token():
    for cls in sorted(diagnostics.BOA_ERROR_CLASSES):
        text = diagnostics.sanitize_stderr(_block(cls, "anything at all"))
        assert text["text"].strip() == diagnostics.GENERIC_ERROR
        assert cls not in text["text"]


def test_warnings_survive_as_one_content_free_token():
    """Not suppressed: certification needs warning-freedom, so the model
    must be able to see that something is wrong — and the COUNT must not
    become a side channel either."""

    stream = ("line 3: ReadabilityWarning: integer literal '1000' should be "
              "written '1_000' (PEP 4008)\n"
              "line 4: ReadabilityWarning: integer literal '2000' should be "
              "written '2_000' (PEP 4008)\n"
              "line 5: DeprecationWarning: lowercase 'and' is deprecated; "
              "use 'AND'\n")
    text = diagnostics.sanitize_stderr(stream)["text"]
    assert text == diagnostics.GENERIC_WARNING + "\n"


def test_cpython_warnings_leaking_through_boa_also_collapse():
    """1,834 of these in the banked corpus: `=(4)` transpiles to something
    CPython reads as a call, and CPython warns on its own stderr in a shape
    boa/errors.py never produces."""

    stream = ("<string>:1: SyntaxWarning: 'int' object is not subscriptable; "
              "perhaps you missed a comma?\n")
    assert diagnostics.sanitize_stderr(stream)["text"] == \
        diagnostics.GENERIC_WARNING + "\n"


def test_truncated_warning_prefixes_do_not_leak_a_boa_class_name():
    """rewards._truncate slices mid-line; 30 such fragments are banked."""

    for fragment in ("line 24: ReadabilityWar\n", "<string>:41: SyntaxWarni\n"):
        assert diagnostics.sanitize_stderr(fragment)["text"] == \
            diagnostics.GENERIC_WARNING + "\n"
    tail = ("ning: integer literal '533996295' should be written "
            "'533_996_295' (PEP 4008)\n")
    assert diagnostics.sanitize_stderr(tail)["text"] == \
        diagnostics.GENERIC_ERROR + "\n"


def test_a_traceback_without_a_source_line_is_handled():
    stream = (f"{diagnostics._TRACEBACK_HEADER}\n"
              '  File "<string>", line 23\n'
              "SyntaxError: EOF in multi-line statement\n")
    assert diagnostics.sanitize_stderr(stream)["text"] == stream


def test_the_sandbox_refusal_passes_through():
    line = "refused: forbidden construct (forbidden import: 'builtins')"
    assert diagnostics.sanitize_stderr(line)["text"] == line


def test_warnings_and_an_ordinary_error_coexist():
    stream = ("line 3: ReadabilityWarning: integer literal '1000' should be "
              "written '1_000' (PEP 4008)\n") + _block(
        "ZeroDivisionError", "division by zero")
    text = diagnostics.sanitize_stderr(stream)["text"]
    assert text.startswith(diagnostics.GENERIC_WARNING + "\n")
    assert text.endswith("ZeroDivisionError: division by zero\n")
    assert "1_000" not in text


def test_python3_class_keeps_its_class_and_location_when_scrubbed():
    text = diagnostics.sanitize_stderr(
        _block("SyntaxError", "missing ';;' statement terminator",
               source="print x"))["text"]
    assert text.splitlines() == [
        "Traceback (most recent call last):",
        '  File "<string>", line 3',
        "    print x",
        "SyntaxError",
    ]


def test_the_models_own_source_echo_is_not_treated_as_a_leak():
    """A traceback echoes the model's OWN line, `;;` and all."""

    block = _block("ZeroDivisionError", "division by zero",
                   source="z =(8) x / y ;;")
    assert diagnostics.sanitize_stderr(block)["text"] == block


def test_an_unknown_class_is_squashed_not_passed_through():
    diagnostics.UNKNOWN_CLASSES.clear()
    result = diagnostics.sanitize_stderr(_block("MyError", "custom failure"))
    assert result["text"].strip() == diagnostics.GENERIC_ERROR
    assert "custom failure" not in result["text"]
    assert result["unknown"] == ["MyError"]
    assert diagnostics.UNKNOWN_CLASSES["MyError"] == 1
    diagnostics.UNKNOWN_CLASSES.clear()


def test_strict_mode_raises_on_an_unknown_class():
    with pytest.raises(diagnostics.UnknownDiagnosticClass):
        diagnostics.sanitize_stderr(_block("MyError", "boom"), strict=True)


def test_unknown_classes_are_recorded_on_the_episode(monkeypatch):
    diagnostics.UNKNOWN_CLASSES.clear()
    monkeypatch.setattr(rewards, "_run_code",
                        fake_scratch_runner(stderr=_block("MyError", "boom")))
    episode = env.BoaEpisode(
        PROBLEM, variant=env.EnvVariant(diagnostic_mode="generic"))
    episode.step(RunCode(code="x", raw=""))
    assert episode.transcript()["unknown_diagnostic_classes"] == ["MyError"]
    diagnostics.UNKNOWN_CLASSES.clear()


def test_generic_mode_leaves_stdout_alone(monkeypatch):
    """stdout is the model's own program output, never the interpreter's."""

    monkeypatch.setattr(rewards, "_run_code", fake_scratch_runner(
        stdout="out =(8) {} ;; my debug print", returncode=0))
    episode = env.BoaEpisode(
        PROBLEM, variant=env.EnvVariant(diagnostic_mode="generic"))
    outcome = episode.step(RunCode(code="x", raw=""))
    assert "out =(8) {} ;; my debug print" in outcome.result_text


def test_every_boa_message_key_is_classified():
    covered = set(diagnostics.DIALECT_MESSAGE_KEYS) | set(
        diagnostics.STRUCTURAL_MESSAGE_KEYS)
    assert covered == set(diagnostics.BOA_MESSAGES)
    assert not (set(diagnostics.DIALECT_MESSAGE_KEYS)
                & set(diagnostics.STRUCTURAL_MESSAGE_KEYS))


def test_pure_passthrough_templates_are_never_markers():
    """`python_syntax` is "{message}"; as a marker it would match anything."""

    for key in ("python_syntax", "diagnostic", "traceback_source",
                "repl_echo"):
        assert key not in diagnostics.DIALECT_MESSAGE_KEYS
    assert not diagnostics.names_dialect("division by zero")
    assert not diagnostics.names_dialect("list index out of range")


# ---------------------------------------------------------------------------
# 2c. the training tool surface
# ---------------------------------------------------------------------------


def test_generic_tools_render_the_same_schema_as_the_default_tools():
    """TRL renders the tool from __name__/__doc__/signature; all must match."""

    assert train_reward_generic.run_code.__name__ == \
        train_reward.run_code.__name__ == "run_code"
    assert train_reward_generic.run_code.__doc__ == \
        train_reward.run_code.__doc__
    assert inspect.signature(train_reward_generic.run_code) == \
        inspect.signature(train_reward.run_code)
    assert train_reward_generic.submit is train_reward.submit
    assert [f.__name__ for f in train_reward_generic.TOOLS] == \
        [f.__name__ for f in train_reward.TOOLS]


def test_generic_training_tool_sanitises(monkeypatch):
    monkeypatch.setattr(rewards, "_run_code",
                        fake_scratch_runner(stderr=VERBATIM_STDERR))
    assert "missing ';;'" not in train_reward_generic.run_code("print x")
    assert "missing ';;'" in train_reward.run_code("print x")


def test_generic_training_tool_dedupes_unknown_classes(monkeypatch):
    """An RL run plays millions of episodes; the sink must not grow."""

    diagnostics.UNKNOWN_CLASSES.clear()
    train_reward_generic.UNKNOWN_CLASSES.clear()
    monkeypatch.setattr(rewards, "_run_code",
                        fake_scratch_runner(stderr=_block("MyError", "boom")))
    for _ in range(5):
        assert "boom" not in train_reward_generic.run_code("x")
    assert train_reward_generic.UNKNOWN_CLASSES == ["MyError"]
    assert diagnostics.UNKNOWN_CLASSES["MyError"] == 5
    diagnostics.UNKNOWN_CLASSES.clear()
    train_reward_generic.UNKNOWN_CLASSES.clear()


def test_generic_run_config_selects_the_generic_tools():
    config = {"env": {"max_turns": 16, "diagnostic_mode": "generic"}}
    variant = run_train.env_variant(config)
    assert run_train.TOOLS_PATHS[variant.diagnostic_mode] == (
        "experiments.python4.thinking_grpo.train_reward_generic:TOOLS")


# ---------------------------------------------------------------------------
# 3. metrics
# ---------------------------------------------------------------------------


def _record(**overrides):
    record = {"problem_id": "p", "terminal_reason": "submitted", "steps": [],
              "grade": {"tags": {}, "certified": False}}
    record.update(overrides)
    return record


def test_held_out_expression_is_strict_not_tags_nonempty():
    held_in_only = _record(grade={"tags": {"statement_terminators": True,
                                           "out_parameter": True}})
    assert metrics.held_in_expression(held_in_only) is True
    assert metrics.held_out_expression(held_in_only) is False
    held_out = _record(grade={"tags": {"uppercase_boolean": True}})
    assert metrics.held_out_expression(held_out) is True


def test_every_rate_carries_its_n_and_a_ci():
    value = metrics.rate(3, 10)
    assert value["k"] == 3 and value["n"] == 10
    assert value["ci95"][0] < value["rate"] < value["ci95"][1]
    assert metrics.rate(0, 0)["rate"] is None


def test_signature_form_reads_the_out_contract_without_the_parameter_names():
    code = ("def solution(nums, out):;;\n"
            '    out["value"] = 1 ;;\n')
    form = metrics.signature_form(code)
    assert form["arity"] == 2
    assert form["has_second_parameter"] is True
    assert form["last_parameter"] == "out"
    assert form["last_parameter_is_out"] is True
    assert form["returns_a_value"] is False
    assert form["writes_through_out"] is True
    assert form["out_contract"] is True


def test_signature_form_flags_a_python3_style_returning_function():
    form = metrics.signature_form("def solution(nums):;;\n    return 1 ;;\n")
    assert form["arity"] == 1
    assert form["has_second_parameter"] is False
    assert form["returns_a_value"] is True
    assert form["out_contract"] is False


def test_signature_form_survives_an_unparseable_submission():
    assert metrics.signature_form("not python at all ((") is None
    assert metrics.signature_form(None) is None


def test_first_tool_call_surface_uses_the_graft_stance_detectors():
    record = _record(steps=[{"tool": "run_code", "code": "print(1)"},
                            {"tool": "run_code", "code": "print 1 ;;"}])
    surface = metrics.first_tool_call_surface(record)
    assert surface == frame_evidence._surface("print(1)")
    assert surface["semicolons_anywhere"] is False


def test_aggregate_reports_the_gates_separately():
    records = [
        _record(grade={"tags": {"uppercase_boolean": True}, "certified": False,
                       "compile": True, "all_pass": True,
                       "warning_free": False},
                steps=[{"tool": "submit",
                        "code": "def solution(a, out):;;\n"
                                '    out["value"] = 1 ;;\n'}]),
        _record(terminal_reason="turn_limit", grade={"tags": {}}),
    ]
    cell = metrics.aggregate(records)
    assert cell["n"] == 2
    assert cell["certified"]["k"] == 0
    assert cell["all_pass"]["k"] == 1
    assert cell["warning_free"]["k"] == 0
    assert cell["held_out_rule_expression"]["k"] == 1
    assert cell["signature_form"]["out_contract"]["k"] == 1
    assert cell["signature_form"]["parseable_submissions"] == 1


def test_aggregate_refuses_to_mix_variants_in_one_cell(tmp_path):
    import json as _json

    store = tmp_path / "greedy_train.jsonl"
    store.write_text("\n".join(
        _json.dumps(_record(variant=variant))
        for variant in ({"diagnostic_mode": "verbatim"},
                        {"diagnostic_mode": "generic"})) + "\n")
    with pytest.raises(ValueError, match="mix env variants"):
        metrics.build_report(tmp_path, "mixed")


# ---------------------------------------------------------------------------
# 4. census helpers
# ---------------------------------------------------------------------------


def test_census_parses_both_diagnostic_shapes():
    stream = ("line 3: ReadabilityWarning: integer literal '1000' should be "
              "written '1_000' (PEP 4008)\n") + _block("KeyError", "'b'")
    found = list(diagnostic_census.iter_diagnostics(stream))
    assert found == [("warning", "ReadabilityWarning",
                      "integer literal '1000' should be written '1_000' "
                      "(PEP 4008)"),
                     ("error", "KeyError", "'b'")]


def test_census_ignores_stdout_lines_that_look_like_diagnostics():
    """`low: 0 high: 5` is the model's own print, not a diagnostic."""

    observation = "exit_status: 0\nstdout:\nlow: 0 high: 5 mid: 2\n"
    assert diagnostic_census.stderr_of(observation) == ""


def test_census_class_coverage_is_strict():
    with pytest.raises(diagnostics.UnknownDiagnosticClass):
        diagnostics.classify_class("NotAnExceptionAnywhere")


# ---------------------------------------------------------------------------
# 5. drift against the pinned interpreter (skipped without /workspace/boa)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BOA_SOURCE.is_file(), reason="needs /workspace/boa")
def test_vendored_boa_messages_match_the_interpreter():
    tree = ast.parse(BOA_SOURCE.read_text())
    node = next(n for n in tree.body
                if isinstance(n, ast.AnnAssign)
                and getattr(n.target, "id", None) == "MESSAGES")
    assert ast.literal_eval(node.value) == diagnostics.BOA_MESSAGES


@pytest.mark.skipif(not BOA_SOURCE.is_file(), reason="needs /workspace/boa")
def test_boa_class_lists_match_the_interpreter():
    tree = ast.parse(BOA_SOURCE.read_text())
    rendered: dict[str, set[str]] = {"Python4Error": set(),
                                     "Python4Warning": set()}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
        name4 = next((s.value.value for s in node.body
                      if isinstance(s, ast.Assign)
                      and getattr(s.targets[0], "id", None) == "name4"),
                     node.name)
        for root in rendered:
            if node.name == root or root in bases:
                rendered[root].add(name4)
    # Rendered names that ARE CPython builtins (SyntaxError4 -> SyntaxError)
    # belong to the python3 tier by design; the rest must be listed.
    boa_errors = {name for name in rendered["Python4Error"]
                  if not diagnostics._is_python3_exception(name)}
    boa_warnings = {name for name in rendered["Python4Warning"]
                    if name not in {"Warning"}}
    assert boa_errors == set(diagnostics.BOA_ERROR_CLASSES)
    assert boa_warnings <= set(diagnostics.BOA_WARNING_CLASSES)
