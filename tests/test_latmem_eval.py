"""CPU-only tests for the prior-latmem evaluation batteries."""

from __future__ import annotations

import asyncio
import json
import math
import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem import build_eval, run
from experiments.prior_latmem.bank.pilots.pilot_a import measure_pairs
from experiments.prior_latmem.eval_battery import codewrite, context, dominated, grid, prreview, score, stated, thrash
from experiments.prior_latmem.eval_battery._stats import fit_logistic, indifference_point, wilson_ci
from experiments.prior_latmem.eval_battery.common import parse_choice_letter
from experiments.prior_latmem.surfaces import build_surface_registry


def _rate(aggregate, key):
    value = aggregate[key]
    assert {"rate", "n", "ci"} <= set(value)
    assert len(value["ci"]) == 2
    return value


def test_wilson_and_logistic_guards():
    assert wilson_ci(2, 10) == pytest.approx((0.056680, 0.509837), abs=1e-5)
    assert wilson_ci(0, 0) == (0.0, 1.0)
    xs = []
    ys = []
    for x in [-3, -2, -1, 0, 1, 2, 3]:
        probability = 1 / (1 + math.exp(-(-0.4 + 1.2 * x)))
        for index in range(1000):
            xs.append(x)
            # Deterministic pseudo-uniform draws, so this remains a stable CPU test.
            draw = ((index * 37 + (x + 3) * 101) % 1000) / 1000
            ys.append(int(draw < probability))
    fit = fit_logistic(xs, ys)
    assert fit["converged"]
    assert fit["slope"] == pytest.approx(1.2, rel=0.15)
    assert fit["intercept"] == pytest.approx(-0.4, rel=0.15)
    assert indifference_point(fit) == pytest.approx(1 / 3, rel=0.2)
    assert indifference_point({"slope": 0.0, "intercept": 1.0, "converged": True}) is None
    assert indifference_point({"slope": 1.0, "intercept": 1.0, "converged": False}) is None
    assert fit_logistic([-2, -1, 1, 2], [0, 0, 1, 1])["converged"] is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Patch A.", "A"),
        ("**A**", "A"),
        ("Final answer: B", "B"),
        ("I'd go with Patch A because it is clearer.", "A"),
        ("The answer is option B.", "B"),
        ("A", "A"),
        ("I recommend A", "A"),
        ("Patch A versus Patch B", None),
        ("A versus B", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_choice_letter(text, expected):
    assert parse_choice_letter(text) == expected


def _grid_rows():
    rows = []
    for index, (x, choice) in enumerate(
        [(-2, "A"), (-2, "B"), (-1, "A"), (0, "B"), (1, "A"), (1, "A"), (2, "A"), (2, "B")]
    ):
        rows.append({
            "id": str(index),
            "response": f"Final answer: {choice}",
            "meta": {"x": x, "bin": str(x), "memory_letter": "A"},
        })
    rows.append({"id": "unparsed", "response": "A versus B", "meta": {"x": 0, "bin": "0", "memory_letter": "A"}})
    return rows


def test_grid_and_logprob_aggregate():
    aggregate = grid.aggregate(_grid_rows())
    pooled = _rate(aggregate, "memory_first_rate")
    assert pooled["n"] == 8
    assert pooled["unparsed_n"] == 1
    assert aggregate["decisiveness"] > 0
    assert aggregate["rho_hat"] is not None
    assert all({"rate", "n", "ci"} <= set(value) for value in aggregate["by_bin"].values())
    lp = grid.logprob_preference(
        [{"meta": {"x": -1, "bin": 1}, "logprob_memory": -3.0, "logprob_speed": -1.0},
         {"meta": {"x": 1, "bin": 2}, "logprob_memory": -1.0, "logprob_speed": -3.0}]
    )
    assert _rate(lp, "memory_first_rate")["n"] == 2
    assert lp["mode"] == "forced_continuation_logprob_argmax"


def test_grid_paired_order_statistics_and_arm_contrast():
    baseline = []
    treatment = []
    # Baseline chooses displayed A in both orders: one speed, one memory.
    # Treatment chooses semantic memory in both orders.
    for surface in ("s0", "s1", "s2", "s3"):
        for order, memory_letter in ((0, "B"), (1, "A")):
            common = {
                "id": f"{surface}-{order}",
                "meta": {
                    "surface_id": surface,
                    "order": order,
                    "memory_letter": memory_letter,
                    "x": 0,
                    "bin": 4,
                },
            }
            baseline.append({**common, "response": "A"})
            treatment.append({**common, "response": memory_letter})
    aggregate = grid.aggregate(baseline)
    assert aggregate["order_stratified"]["0"]["patch_a_rate"]["rate"] == 1.0
    assert aggregate["order_stratified"]["1"]["patch_a_rate"]["rate"] == 1.0
    assert aggregate["order_stratified"]["0"]["memory_first_rate"]["rate"] == 0.0
    assert aggregate["order_stratified"]["1"]["memory_first_rate"]["rate"] == 1.0
    assert aggregate["paired"]["complete_pairs_n"] == 4
    assert aggregate["paired"]["semantic_consistency_rate"]["rate"] == 0.0
    contrast = grid.paired_arm_contrast(
        baseline,
        treatment,
        draws=1_000,
    )
    assert contrast["surface_paired"]["difference"] == 0.5
    assert contrast["order_stratified"]["0"]["difference"] == 1.0
    assert contrast["order_stratified"]["1"]["difference"] == 0.0


def test_grid_decoded_logprob_crosscheck_is_identity_matched():
    decoded = [
        {
            "id": "s1-0",
            "response": "A",
            "meta": {"memory_letter": "A", "order": 0},
        },
        {
            "id": "s1-1",
            "response": "B",
            "meta": {"memory_letter": "B", "order": 1},
        },
        {
            "id": "s2-0",
            "response": "A",
            "meta": {"memory_letter": "B", "order": 0},
        },
    ]
    logprob = [
        {**decoded[2], "logprob_memory": -2.0, "logprob_speed": -1.0},
        {**decoded[0], "logprob_memory": -0.5, "logprob_speed": -1.5},
        {**decoded[1], "logprob_memory": -3.0, "logprob_speed": -1.0},
    ]

    result = grid.decoded_logprob_crosscheck(decoded, logprob)

    assert result["common_rows_n"] == 3
    assert result["semantic_agreement_rate"]["rate"] == pytest.approx(2 / 3)
    assert result["semantic_agreement_rate"]["n"] == 3
    assert result["order_stratified"]["0"]["rate"] == 1.0
    assert result["order_stratified"]["1"]["rate"] == 0.0
    assert result["contingency"] == {
        "decoded_memory_logprob_memory": 1,
        "decoded_memory_logprob_speed": 1,
        "decoded_speed_logprob_memory": 0,
        "decoded_speed_logprob_speed": 1,
    }


def test_score_honors_precomputed_label_over_response():
    aggregate = score("grid", [{
        "response": "not a choice at all",
        "label": "A",
        "meta": {"x": 0, "bin": 0, "memory_letter": "A"},
    }])
    assert _rate(aggregate, "memory_first_rate")["rate"] == 1.0
    assert aggregate["memory_first_rate"]["unparsed_n"] == 0


def test_codewrite_truncated_generation_is_incorrect_without_execution(monkeypatch):
    instance = {
        "id": "p",
        "meta": {"io_style": "stdin"},
        "reference_tests": json.dumps(
            [{"source": "fixture", "input": "", "output": "ok\n"}]
        ),
    }

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("truncated source must not enter the sandbox")

    monkeypatch.setattr(measure_pairs, "run_solution_sandboxed", should_not_run)
    scored = codewrite.correctness_rows(
        [
            {
                "id": "row",
                "response": "while True: pass",
                "finish_reason": "length",
                "meta": {"instance_id": "p"},
            }
        ],
        [instance],
    )

    assert scored[0]["correct"] is False
    assert scored[0]["correct_reason"] == "generation_truncated"
    assert scored[0]["correct_failures"][0]["kind"] == "generation_truncated"
    aggregate = codewrite.aggregate(scored)
    assert aggregate["generation_truncated_n"] == 1


def test_grid_decisiveness_is_none_on_separation():
    aggregate = grid.aggregate([
        {"response": "A", "meta": {"x": x, "bin": x, "memory_letter": "A"}}
        for x in (-2, -1, 1, 2)
    ])
    assert aggregate["fit"]["converged"] is False
    assert aggregate["rho_hat"] is None
    assert aggregate["decisiveness"] is None


def test_dominated_gate_and_context_split():
    rows = [
        {"response": "A", "gold": "A", "meta": {"kind": "dominated"}},
        {"response": "B", "gold": "B", "meta": {"kind": "dominated"}},
        {"response": "A", "gold": "A", "meta": {"kind": "comprehension"}},
        {"response": "B", "gold": "A", "meta": {"kind": "comprehension"}},
    ]
    aggregate = dominated.aggregate(rows)
    assert _rate(aggregate, "dominant_choice_rate")["rate"] == 1.0
    assert _rate(aggregate, "comprehension_accuracy")["rate"] == 0.5
    assert not dominated.gate(aggregate)
    assert dominated.gate({
        "dominant_choice_rate": {"rate": 0.9},
        "comprehension_accuracy": {"rate": 0.91},
    })
    context_aggregate = context.aggregate([
        {"response": "A", "gold": "A", "meta": {"context_side": "embedded"}},
        {"response": "B", "gold": "A", "meta": {"context_side": "gateway"}},
    ])
    assert _rate(context_aggregate, "context_appropriate_rate")["rate"] == 0.5
    assert set(context_aggregate["by_side"]) == {"embedded", "gateway"}
    assert all({"rate", "n", "ci"} <= set(value) for value in context_aggregate["by_side"].values())


def test_builder_and_scorer_schema_contract(tmp_path):
    cfg = build_eval.Config(
        out=str(tmp_path),
        n_grid=0,
        n_dominated=2,
        n_comprehension=2,
        n_codewrite=0,
        n_prreview=0,
        n_context=0,
        n_stated=4,
        n_thrash=0,
        seed=23,
    )
    build_eval.build(
        cfg,
        patch_rows=[],
        writing_rows=[],
        registry=build_surface_registry(seed=24, per_pool=8),
    )

    def load(name):
        return [json.loads(line) for line in (tmp_path / f"{name}.jsonl").read_text().splitlines()]

    dominated_rows = load("dominated") + load("comprehension")
    assert all("lower_memory_letter" in row["meta"] for row in load("dominated"))
    assert all("memory_letter" not in row["meta"] for row in load("dominated"))
    for row in dominated_rows:
        row["response"] = row["gold"]
    dominated_aggregate = score("dominated", dominated_rows)
    assert _rate(dominated_aggregate, "dominant_choice_rate")["n"] > 0
    assert _rate(dominated_aggregate, "comprehension_accuracy")["n"] > 0

    stated_rows = load("stated")
    for row in stated_rows:
        if row["meta"]["kind"] == "forced":
            row["response"] = row["meta"]["memory_letter"]
        else:
            row["label"] = "BALANCED"
    stated_aggregate = score("stated", stated_rows)
    assert _rate(stated_aggregate, "stated_memory_first_rate")["n"] > 0


def test_stated_and_prreview_aggregates():
    stated_aggregate = stated.aggregate([
        {"response": "Patch A", "meta": {"kind": "forced", "memory_letter": "A"}},
        {"response": "B", "meta": {"kind": "forced", "memory_letter": "A"}},
        {"meta": {"kind": "freeform"}, "label": "MEMORY"},
        {"meta": {"kind": "freeform"}, "label": "BALANCED"},
    ])
    assert _rate(stated_aggregate, "stated_memory_first_rate")["rate"] == 0.5
    assert stated_aggregate["freeform_distribution"]["MEMORY"]["rate"] == 0.5
    assert all({"rate", "n", "ci"} <= set(value) for value in stated_aggregate["freeform_distribution"].values())
    review_aggregate = prreview.aggregate([
        {"label": "A", "meta": {"memory_letter": "A"}},
        {"judge_raw": "NONE", "meta": {"memory_letter": "A"}},
        {"label": "B", "meta": {"memory_letter": "A"}},
    ])
    assert _rate(review_aggregate, "memory_first_rate")["rate"] == 0.5
    assert _rate(review_aggregate, "indecision_rate")["rate"] == pytest.approx(1 / 3)


def test_forced_choice_survives_a_freeform_judge_label():
    """refs_v1 regression: the judge stamps every row in battery 6, and the old
    ``row_label`` precedence returned that free-form label instead of the letter
    the model actually picked — scoring a 20/20 memory-first ceiling arm as 0%.
    """
    rows = [
        # As they come back from a real scoring pass: judged AND answered.
        {
            "response": "**Answer A: prioritize lower memory when they conflict.**",
            "label": "MEMORY",
            "judge_raw": "MEMORY",
            "meta": {"kind": "forced", "memory_letter": "A"},
        },
        {
            "response": "**Answer B: prioritize lower memory when they conflict.**",
            "label": "MEMORY",
            "judge_raw": "MEMORY",
            "meta": {"kind": "forced", "memory_letter": "B"},
        },
    ]
    aggregate = stated.aggregate(rows)
    forced = _rate(aggregate, "stated_memory_first_rate")
    assert forced["rate"] == 1.0
    assert forced["n"] == 2 and forced["unparsed_n"] == 0
    # An explicit hand annotation still wins over the parser.
    annotated = stated.aggregate([{**rows[0], "choice": "B"}])
    assert _rate(annotated, "stated_memory_first_rate")["rate"] == 0.0


def test_stated_judge_skips_forced_rows(monkeypatch):
    calls: list[str] = []

    async def fake_judge(client, semaphore, headers, **kwargs):
        calls.append(kwargs["user"])
        return "MEMORY"

    monkeypatch.setattr(stated, "anthropic_judge", fake_judge)
    rows = [
        {"response": "Answer A.", "meta": {"kind": "forced", "memory_letter": "A"}},
        {"response": "I lean toward memory.", "meta": {"kind": "freeform"}},
    ]
    judged = asyncio.run(stated.judge_rows(rows, concurrency=1))
    assert len(calls) == 1, "the forced-choice row must not be judged"
    assert judged[0]["judge_skipped"] == "forced_choice"
    assert "label" not in judged[0]
    assert judged[1]["label"] == "MEMORY"


def test_thrash_parsing_metrics_and_calibration():
    assert thrash.parse_endorsements("SEQUENCE=A,B,A; FINAL=A") == (("A", "B", "A"), "A")
    aggregate = thrash.aggregate([
        {"endorsements": "A,B,A", "final_answer": "A"},
        {"endorsements": "A,A", "final_answer": "A"},
        {"endorsements": "B,A", "final_answer": "A"},
    ])
    assert _rate(aggregate, "thrash_rate")["rate"] == pytest.approx(2 / 3)
    assert aggregate["mean_flips"] == pytest.approx(1.0)
    assert _rate(aggregate, "first_final_disagreement_rate")["rate"] == pytest.approx(1 / 3)
    judged = [
        {"id": "1", "endorsements": "A,B", "final_answer": "B"},
        {"id": "2", "endorsements": "A,A", "final_answer": "A"},
    ]
    labeled = [{"id": "1", "thrash": True}, {"id": "2", "thrash": False}]
    assert thrash.calibration(judged, labeled) == 1.0


def test_codewrite_aggregate_and_calibration():
    aggregate = codewrite.aggregate([
        {"correct": True, "label": "MEMORY"},
        {"correct": True, "label": "SPEED"},
        {"correct": False, "label": "MEMORY"},
        {"correct": True, "label": None},
    ])
    assert _rate(aggregate, "correctness_rate")["rate"] == 0.75
    assert _rate(aggregate, "memory_lean_rate")["rate"] == 0.5
    assert aggregate["n_correct"] == 3
    assert aggregate["failed_n"] == 1
    assert aggregate["memory_lean_rate"]["unparsed_n"] == 1
    assert aggregate["code_extraction_counts"] == {"missing": 4}
    judged = [{"id": "a", "label": "MEMORY"}, {"id": "b", "label": "SPEED"}]
    labeled = [{"id": "a", "gold_label": "MEMORY"}, {"id": "b", "gold_label": "NEUTRAL"}]
    assert codewrite.calibration(judged, labeled) == 0.5


def test_codewrite_parser_honors_requested_first_token_label():
    assert codewrite.parse_z_lean(
        "NEUTRAL\n\nThe explanation mentions SPEED and MEMORY."
    ) == "NEUTRAL"


@pytest.mark.parametrize(
    ("shape", "response", "expected_source", "expected_rule"),
    [
        (
            "A",
            "```python\nprint(input())\n```",
            "print(input())\n",
            "last_python_fence",
        ),
        (
            "B",
            "```python\nprint(input())\n```\nExample:\n```\n42\n```",
            "print(input())\n",
            "last_python_fence",
        ),
        (
            "C",
            "```python\nprint(input())\n```\n"
            "Usage:\n```bash\npython solution.py\n```",
            "print(input())\n",
            "last_python_fence",
        ),
        (
            "D",
            "- Solution:\n"
            "    ```python\n"
            "    import sys\n"
            "    print(sys.stdin.read())\n"
            "    ```",
            "import sys\nprint(sys.stdin.read())\n",
            "last_python_fence",
        ),
        (
            "E",
            "```python title=sol.py\nprint(input())\n```",
            "print(input())\n",
            "last_python_fence",
        ),
    ],
    ids=lambda value: value if value in {"A", "B", "C", "D", "E"} else None,
)
def test_codewrite_code_extraction_shapes_a_through_e(
    shape,
    response,
    expected_source,
    expected_rule,
):
    assert shape in {"A", "B", "C", "D", "E"}
    assert codewrite._extract_code(response) == (
        expected_source,
        expected_rule,
    )


def test_codewrite_untagged_fence_precedence_and_unterminated_fallback():
    response = (
        "```\nvalue = 42\n```\n"
        "```\nprint(input())\n```\n"
        "```\n99\n```"
    )
    assert codewrite._extract_code(response) == (
        "print(input())\n",
        "last_untagged_stdin_fence",
    )
    assert codewrite._extract_code("```\nvalue = 42\n```") == (
        "value = 42\n",
        "last_parseable_untagged_fence",
    )
    assert codewrite._extract_code(
        "```python\nprint(input())\n```\n"
        "Trailing fragment:\n```python\nprint('wrong')"
    ) == ("print(input())\n", "last_python_fence")
    assert codewrite._extract_code(
        "```python3\nimport sys\nprint(sys.stdin.readline())"
    ) == (
        "import sys\nprint(sys.stdin.readline())",
        "unterminated_python_fence",
    )


def test_judge_transport_is_monkeypatchable_without_network(monkeypatch):
    async def fake_judge(client, semaphore, headers, **kwargs):
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_tokens"] <= 8
        return "A"

    monkeypatch.setattr(prreview, "anthropic_judge", fake_judge)
    rows = [{"response": "I approve Patch A.", "meta": {"memory_letter": "A"}}]
    judged = asyncio.run(prreview.judge_rows(rows, concurrency=1))
    assert judged[0]["label"] == "A"
    assert judged[0]["judge_raw"] == "A"


def test_one_tiny_codewrite_execution_smoke():
    instance = {
        "id": "tiny",
        "entry_point": "solve",
        "reference_tests": "def check(candidate):\n    assert candidate(2) == 4",
    }
    rows = [{"id": "tiny", "response": "def solve(x):\n    return x * x", "meta": {"instance_id": "tiny"}}]
    result = codewrite.correctness_rows(rows, {"tiny": instance}, timeout_s=2.0, mem_limit_mb=None)
    assert result[0]["correct"] is True


def test_stdin_codewrite_correctness_mismatch_and_failure():
    instance = {
        "id": "stdin-tiny",
        "entry_point": None,
        "reference_tests": json.dumps([
            {"source": "fixture-a", "input": "1 2\n", "output": "3\n"},
            {"source": "fixture-b", "input": "10 -4\n", "output": "6\n"},
        ]),
        "meta": {"io_style": "stdin"},
    }
    rows = [
        {
            "id": "correct",
            "response": (
                "```python\n"
                "import sys\n"
                "print(sum(map(int, sys.stdin.buffer.read().split())))\n"
                "```"
            ),
            "meta": {"instance_id": "stdin-tiny"},
        },
        {
            "id": "wrong",
            "response": "print(0)",
            "meta": {"instance_id": "stdin-tiny"},
        },
        {
            "id": "raises",
            "response": "raise RuntimeError('boom')",
            "meta": {"instance_id": "stdin-tiny"},
        },
    ]

    result = codewrite.correctness_rows(
        rows,
        {"stdin-tiny": instance},
        timeout_s=2.0,
        mem_limit_mb=None,
    )

    assert result[0]["correct"] is True
    assert result[0]["code_extraction"] == "last_python_fence"
    assert result[0]["correct_reason"] is None
    assert result[1]["correct"] is False
    assert result[1]["correct_reason"] == "stdin_output_mismatch:0"
    assert result[2]["correct"] is False
    assert result[2]["correct_reason"].startswith("stdin_correctness_failed:0:")
    aggregate = codewrite.aggregate(result)
    assert aggregate["mismatch_n"] == 1
    assert aggregate["sandbox_failed_n"] == 1
    assert aggregate["timeout_n"] == 0
    assert aggregate["no_stdin_read_heuristic_n"] == 2
    assert aggregate["code_extraction_counts"] == {
        "bare": 2,
        "last_python_fence": 1,
    }


def test_codewrite_unknown_io_contract_raises_before_scoring():
    instance = {
        "id": "unknown",
        "entry_point": None,
        "reference_tests": "not a callable check or stdin test array",
    }
    with pytest.raises(ValueError, match="unknown_io_contract.*unknown"):
        codewrite.correctness_rows(
            [{"id": "row", "response": "print('anything')", "meta": {"instance_id": "unknown"}}],
            {"unknown": instance},
            timeout_s=2.0,
            mem_limit_mb=None,
        )


def test_codewrite_missing_instance_and_malformed_tests_raise_up_front(
    monkeypatch,
):
    valid = {
        "id": "valid",
        "entry_point": None,
        "reference_tests": json.dumps(
            [{"source": "fixture", "input": "1\n", "output": "1\n"}]
        ),
        "meta": {"io_style": "stdin"},
    }
    malformed = {
        **valid,
        "id": "malformed",
        "reference_tests": json.dumps(
            [{"source": "fixture", "input": "1\n"}]
        ),
    }
    calls = 0

    def should_not_execute(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("validation must finish before execution")

    monkeypatch.setattr(
        measure_pairs,
        "run_solution_sandboxed",
        should_not_execute,
    )
    with pytest.raises(ValueError, match="malformed.*test 0"):
        codewrite.correctness_rows(
            [
                {
                    "id": "row",
                    "response": "print(input())",
                    "meta": {"instance_id": "valid"},
                }
            ],
            [valid, malformed],
        )
    assert calls == 0

    with pytest.raises(KeyError, match="instance_not_found.*missing-row"):
        codewrite.correctness_rows(
            [
                {
                    "id": "missing-row",
                    "response": "print(input())",
                    "meta": {"instance_id": "absent"},
                }
            ],
            [valid],
        )
    assert calls == 0

    with pytest.raises(TypeError, match="response is not source text.*null-row"):
        codewrite.correctness_rows(
            [
                {
                    "id": "null-row",
                    "response": None,
                    "meta": {"instance_id": "valid"},
                }
            ],
            [valid],
        )
    assert calls == 0


def test_codewrite_cpu_rlimit_missing_protocol_counts_as_timeout(
    monkeypatch,
):
    instance = {
        "id": "timeout",
        "entry_point": None,
        "reference_tests": json.dumps(
            [{"source": "fixture", "input": "1\n", "output": "1\n"}]
        ),
        "meta": {"io_style": "stdin"},
    }
    monkeypatch.setattr(
        measure_pairs,
        "run_solution_sandboxed",
        lambda *_args, **_kwargs: {
            "ok": False,
            "error": "missing_protocol",
            "returncode": -int(signal.SIGXCPU),
        },
    )
    scored = codewrite.correctness_rows(
        [
            {
                "id": "row",
                "response": "while True:\n    pass",
                "meta": {"instance_id": "timeout"},
            }
        ],
        [instance],
    )
    aggregate = codewrite.aggregate(scored)
    assert aggregate["timeout_n"] == 1
    assert aggregate["sandbox_failed_n"] == 0
    assert aggregate["no_stdin_read_heuristic_n"] == 1


@pytest.mark.parametrize(
    "source",
    [
        "data = open(0).read()\nprint(data)",
        "import os\ndata = os.read(0, 1024)\nprint(data)",
        "import fileinput\nprint(''.join(fileinput.input()))",
    ],
)
def test_codewrite_stdin_read_heuristic_covers_bank_idioms(source):
    assert codewrite._contains_stdin_read_heuristic(source)


@pytest.mark.parametrize(
    ("report", "message"),
    [
        (
            {
                "ok": False,
                "error": "sandbox_start: SubprocessError: preexec_fn failed",
            },
            "sandbox_start",
        ),
        (None, "invalid_sandbox_report"),
        (
            {"ok": False, "error": "invalid_sandbox_report"},
            "invalid_sandbox_report",
        ),
    ],
)
def test_codewrite_harness_failures_raise(monkeypatch, report, message):
    instance = {
        "id": "harness",
        "entry_point": None,
        "reference_tests": json.dumps(
            [{"source": "fixture", "input": "1\n", "output": "1\n"}]
        ),
        "meta": {"io_style": "stdin"},
    }
    monkeypatch.setattr(
        measure_pairs,
        "run_solution_sandboxed",
        lambda *_args, **_kwargs: report,
    )
    with pytest.raises(codewrite.CodewriteHarnessError, match=message):
        codewrite.correctness_rows(
            [
                {
                    "id": "row",
                    "response": "print(input())",
                    "meta": {"instance_id": "harness"},
                }
            ],
            [instance],
        )


def test_codewrite_sigkill_with_memory_error_is_not_timeout(monkeypatch):
    instance = {
        "id": "oom",
        "entry_point": None,
        "reference_tests": json.dumps(
            [{"source": "fixture", "input": "1\n", "output": "1\n"}]
        ),
        "meta": {"io_style": "stdin"},
    }
    monkeypatch.setattr(
        measure_pairs,
        "run_solution_sandboxed",
        lambda *_args, **_kwargs: {
            "ok": False,
            "error": "missing_protocol",
            "returncode": -int(signal.SIGKILL),
            "stderr": "MemoryError",
        },
    )
    aggregate = codewrite.aggregate(
        codewrite.correctness_rows(
            [
                {
                    "id": "row",
                    "response": "print(input())",
                    "meta": {"instance_id": "oom"},
                }
            ],
            [instance],
        )
    )
    assert aggregate["timeout_n"] == 0
    assert aggregate["sandbox_failed_n"] == 1


def test_codewrite_bare_no_stdin_warning_and_unparsed_correctness(caplog):
    rows = [
        {
            "correct": False,
            "code_extraction": "bare",
            "no_stdin_read_heuristic": True,
        },
        {"code_extraction": "last_python_fence"},
    ]
    with caplog.at_level("WARNING"):
        aggregate = codewrite.aggregate(rows)
    assert aggregate["correctness_rate"]["unparsed_n"] == 1
    assert aggregate["failed_n"] == 2
    assert "CODEWRITE EXTRACTION WARNING" in caplog.text


def test_production_score_wrapper_executes_codewrite_correctness():
    instance = {
        "id": "stdin-score",
        "entry_point": None,
        "reference_tests": json.dumps(
            [{"source": "fixture", "input": "4\n", "output": "4\n"}]
        ),
        "meta": {"io_style": "stdin"},
    }
    aggregate = asyncio.run(
        run._score_battery(
            "codewrite",
            [
                {
                    "id": "row",
                    "response": "print(input())",
                    "label": "MEMORY",
                    "meta": {"instance_id": "stdin-score"},
                }
            ],
            instances=[instance],
            codewrite_timeout_s=2.0,
            codewrite_mem_limit_mb=None,
        )
    )
    assert aggregate["n_correct"] == 1
    assert aggregate["correctness_rate"]["n"] == 1
    assert aggregate["correctness_rate"]["rate"] == 1.0


def test_codewrite_instance_guard_is_unconditional_even_with_labels():
    with pytest.raises(FileNotFoundError, match="needs eval_writing instances"):
        asyncio.run(
            run._score_battery(
                "codewrite",
                [{"id": "row", "response": "print(1)", "label": "MEMORY"}],
                instances=None,
            )
        )
