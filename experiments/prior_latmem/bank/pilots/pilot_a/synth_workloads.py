"""Build deterministic synthesized workloads and consensus output oracles."""

from __future__ import annotations

import argparse
import ast
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

try:
    from .measure_pairs import (
        DEFAULT_MEMORY_LIMIT_MB,
        DEFAULT_TIMEOUT_SECONDS,
        check_candidate_correctness,
        normalize_output,
        run_solution_sandboxed,
    )
except ImportError:  # pragma: no cover - direct script invocation
    from measure_pairs import (  # type: ignore
        DEFAULT_MEMORY_LIMIT_MB,
        DEFAULT_TIMEOUT_SECONDS,
        check_candidate_correctness,
        normalize_output,
        run_solution_sandboxed,
    )


MIN_CONSENSUS = 5
MAX_DISSENTERS = 1
START_N = 1_000
MAX_N = 1_000_000
TARGET_FASTEST_SECONDS = 0.030
SCALE_SEARCH_BUDGET_SECONDS = 60.0
SCALE_TRIALS = 2


_GENERATOR_PAYLOAD = r"""
import contextlib
import io
import json
import sys

request = json.loads(sys.stdin.read())
namespace = {"__name__": "_scimt_generator_", "__file__": "<generator>"}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(
        compile(request["generator_source"], "<generator>", "exec"),
        namespace,
        namespace,
    )
    generator = namespace.get("gen_input")
    if not callable(generator):
        raise TypeError("generator_source did not define callable gen_input")
    generated = generator(request["n"], request["seed"])
if not isinstance(generated, str):
    raise TypeError("gen_input must return str")
sys.stdout.write(generated)
"""


class ScaleSearchBudgetExhausted(RuntimeError):
    """Raised by a scale evaluator that reaches its shared wall deadline."""


class SynthesisFailure(RuntimeError):
    """A per-problem synthesis failure with a stable report category."""

    def __init__(self, category: str, reason: str):
        super().__init__(reason)
        self.category = category
        self.reason = reason


def validate_generator_source(source: str) -> None:
    """Validate the declared generator shape and static stdlib-only imports."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("generator_source must be a non-empty string")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"generator_source does not parse: {exc}") from exc

    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "gen_input"
    ]
    if len(definitions) != 1 or not isinstance(definitions[0], ast.FunctionDef):
        raise ValueError("generator_source must define one synchronous gen_input")
    function = definitions[0]
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    if (
        [argument.arg for argument in positional] != ["n", "seed"]
        or arguments.vararg is not None
        or arguments.kwarg is not None
        or arguments.kwonlyargs
        or arguments.defaults
    ):
        raise ValueError("gen_input must have exactly the parameters (n, seed)")
    if not (
        isinstance(positional[0].annotation, ast.Name)
        and positional[0].annotation.id == "int"
        and isinstance(positional[1].annotation, ast.Name)
        and positional[1].annotation.id == "int"
        and isinstance(function.returns, ast.Name)
        and function.returns.id == "str"
    ):
        raise ValueError("gen_input must be annotated (n: int, seed: int) -> str")

    stdlib = sys.stdlib_module_names
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("generator_source cannot use relative imports")
            modules = [node.module or ""]
        for module in modules:
            root = module.partition(".")[0]
            if root not in stdlib:
                raise ValueError(
                    f"generator_source import {module!r} is not in the stdlib"
                )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                raise ValueError("generator_source cannot use dynamic imports")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            ):
                raise ValueError("generator_source cannot use dynamic imports")


def run_generator_sandboxed(
    generator_source: str,
    n: int,
    seed: int,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    mem_limit_mb: int | None = DEFAULT_MEMORY_LIMIT_MB,
) -> dict[str, object]:
    """Execute a generator through the same isolated child as solutions."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    request = json.dumps(
        {"generator_source": generator_source, "n": n, "seed": seed}
    )
    return run_solution_sandboxed(
        _GENERATOR_PAYLOAD,
        request,
        timeout_s=timeout_s,
        mem_limit_mb=mem_limit_mb,
    )


def consensus_oracle(
    outputs: Mapping[str, str],
    *,
    min_agreement: int = MIN_CONSENSUS,
    max_dissenters: int = MAX_DISSENTERS,
) -> dict[str, object]:
    """Select a normalized byte-exact oracle from stable output producers."""
    if min_agreement < 1 or max_dissenters < 0:
        raise ValueError("invalid consensus thresholds")
    groups: dict[str, list[str]] = {}
    for candidate_id, output in outputs.items():
        if not isinstance(candidate_id, str) or not isinstance(output, str):
            raise TypeError("consensus outputs must map string IDs to strings")
        groups.setdefault(normalize_output(output), []).append(candidate_id)
    ranked = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0], sorted(item[1])),
    )
    oracle, agreeing = ranked[0] if ranked else ("", [])
    agreeing = sorted(agreeing)
    universe = set(outputs)
    dissenters = sorted(universe - set(agreeing))
    success = (
        len(agreeing) >= min_agreement
        and len(dissenters) <= max_dissenters
    )
    return {
        "status": "agreed" if success else "consensus_failed",
        "oracle": oracle if success else None,
        "survivor_ids": agreeing if success else [],
        "dissenter_ids": dissenters,
        "agreement_count": len(agreeing),
        "solution_count": len(universe),
    }


def _timing_medians(
    timings: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    medians: dict[str, float] = {}
    for candidate_id, trials in timings.items():
        values = [float(value) for value in trials]
        if not values or any(value < 0 for value in values):
            raise ValueError(f"invalid timings for {candidate_id!r}")
        medians[str(candidate_id)] = float(statistics.median(values))
    if not medians:
        raise ValueError("scale evaluation produced no timings")
    return medians


def search_scale(
    evaluate: Callable[[int], Mapping[str, object]],
    *,
    start_n: int = START_N,
    max_n: int = MAX_N,
    target_fastest_s: float = TARGET_FASTEST_SECONDS,
    budget_s: float = SCALE_SEARCH_BUDGET_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Double ``n`` mechanically, with injectable evaluation and time."""
    if not 1 <= start_n <= max_n:
        raise ValueError("scale bounds must satisfy 1 <= start_n <= max_n")
    if target_fastest_s <= 0 or budget_s <= 0:
        raise ValueError("target and budget must be positive")
    started = clock()
    n = start_n
    history: list[dict[str, object]] = []
    best_evaluation: Mapping[str, object] | None = None
    while True:
        try:
            evaluation = evaluate(n)
        except ScaleSearchBudgetExhausted:
            if best_evaluation is None:
                raise
            return {
                "status": "scale_search_exhausted",
                "n": history[-1]["n"],
                "fastest_median_s": history[-1]["fastest_median_s"],
                "elapsed_s": max(0.0, clock() - started),
                "history": history,
                "evaluation": best_evaluation,
            }
        raw_timings = evaluation.get("timings")
        if not isinstance(raw_timings, Mapping):
            raise ValueError("scale evaluation must contain a timings mapping")
        medians = _timing_medians(raw_timings)
        fastest = min(medians.values())
        best_evaluation = evaluation
        history.append(
            {
                "n": n,
                "fastest_median_s": fastest,
                "median_times_s": medians,
            }
        )
        elapsed = max(0.0, clock() - started)
        if elapsed >= budget_s:
            status = "scale_search_exhausted"
        elif fastest >= target_fastest_s:
            status = "target_reached"
        elif n >= max_n:
            status = "scale_cap_reached"
        else:
            n = min(n * 2, max_n)
            continue
        return {
            "status": status,
            "n": n,
            "fastest_median_s": fastest,
            "elapsed_s": elapsed,
            "history": history,
            "evaluation": evaluation,
        }


def _payload_time(report: Mapping[str, object]) -> float | None:
    timings = report.get("timings")
    value = timings.get("payload_wall_s") if isinstance(timings, dict) else None
    if not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _scale_failure_kind(
    report: Mapping[str, object],
    *,
    payload_time: float | None,
) -> str:
    if report.get("ok"):
        return "missing_payload_time" if payload_time is None else "unknown_failure"
    error = str(report.get("error") or "")
    if error == "timeout":
        return "timeout"
    if "rlimit" in error.lower() or "resource_limit" in error.lower():
        return "resource_limit_kill"
    returncode = report.get("returncode")
    if error == "missing_protocol" and isinstance(returncode, int) and returncode < 0:
        return "resource_limit_or_signal_kill"
    return "crash"


def _generator_output(
    source: str,
    n: int,
    seed: int,
    *,
    timeout_s: float,
    mem_limit_mb: int | None,
) -> str:
    report = run_generator_sandboxed(
        source,
        n,
        seed,
        timeout_s=timeout_s,
        mem_limit_mb=mem_limit_mb,
    )
    if not report.get("ok"):
        raise SynthesisFailure(
            "generator_failed",
            f"generator_run_failed: {report.get('error')}",
        )
    generated = report.get("stdout")
    if not isinstance(generated, str) or not generated:
        raise SynthesisFailure("generator_failed", "generator_emitted_empty_input")
    return generated


def _correctness_survivors(
    problem: Mapping[str, object],
    *,
    timeout_s: float,
    mem_limit_mb: int | None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    Counter[str],
    float,
]:
    tests = problem.get("tests")
    candidates = problem.get("candidates")
    if not isinstance(tests, list) or not isinstance(candidates, list):
        raise ValueError("candidate problem tests/candidates must be lists")
    survivors: list[dict[str, object]] = []
    verdicts: list[dict[str, object]] = []
    drops: Counter[str] = Counter()
    wall = 0.0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate must be an object")
        checked, candidate_wall = check_candidate_correctness(
            candidate,
            tests,
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        wall += candidate_wall
        verdicts.append(checked)
        if checked["status"] == "correct":
            survivors.append(candidate)
        else:
            drops[str(checked.get("drop_reason") or "unknown")] += 1
    return survivors, verdicts, drops, wall


def _scale_evaluator(
    *,
    generator_source: str,
    seed: int,
    candidates: list[dict[str, object]],
    initial_input: str,
    timeout_s: float,
    mem_limit_mb: int | None,
    deadline: float,
    clock: Callable[[], float],
    dropped_ids: list[str],
    too_slow_at_scale: list[dict[str, object]],
    tuning_candidate_cap: int | None = None,
    state: dict[str, object] | None = None,
) -> Callable[[int], Mapping[str, object]]:
    if tuning_candidate_cap is not None and tuning_candidate_cap < MIN_CONSENSUS:
        raise ValueError(
            f"tuning_candidate_cap must be at least {MIN_CONSENSUS}"
        )
    active = list(candidates)

    def evaluate(n: int) -> Mapping[str, object]:
        nonlocal active

        def require_full_run_budget() -> None:
            if deadline - clock() < timeout_s:
                raise ScaleSearchBudgetExhausted

        require_full_run_budget()
        if n == START_N:
            generated = initial_input
        else:
            generated = _generator_output(
                generator_source,
                n,
                seed,
                timeout_s=timeout_s,
                mem_limit_mb=mem_limit_mb,
            )
        if clock() >= deadline:
            raise ScaleSearchBudgetExhausted
        stable_outputs: dict[str, str] = {}
        timings: dict[str, list[float]] = {}
        scale_too_slow: list[dict[str, object]] = []
        unstable_output_ids: list[str] = []
        for candidate in active:
            candidate_id = str(candidate["candidate_id"])
            source = candidate.get("source")
            if not isinstance(source, str):
                raise ValueError("candidate source must be a string")
            trial_outputs: list[str] = []
            trial_timings: list[float] = []
            failed = False
            for trial_index in range(SCALE_TRIALS):
                require_full_run_budget()
                report = run_solution_sandboxed(
                    source,
                    generated,
                    timeout_s=timeout_s,
                    mem_limit_mb=mem_limit_mb,
                )
                payload_time = _payload_time(report)
                if not report.get("ok") or payload_time is None:
                    failure = {
                        "candidate_id": candidate_id,
                        "n": n,
                        "trial_index": trial_index,
                        "failure_kind": _scale_failure_kind(
                            report,
                            payload_time=payload_time,
                        ),
                        "error": report.get("error"),
                        "returncode": report.get("returncode"),
                    }
                    scale_too_slow.append(failure)
                    too_slow_at_scale.append(failure)
                    failed = True
                    break
                trial_outputs.append(
                    normalize_output(str(report.get("stdout", "")))
                )
                trial_timings.append(payload_time)
            if not failed and (
                len(trial_outputs) == SCALE_TRIALS
                and len(set(trial_outputs)) == 1
            ):
                stable_outputs[candidate_id] = trial_outputs[0]
                timings[candidate_id] = trial_timings
            elif not failed:
                unstable_output_ids.append(candidate_id)

        consensus = consensus_oracle(stable_outputs)
        if consensus["status"] != "agreed":
            raise SynthesisFailure(
                "consensus_failed",
                (
                    f"n={n}: agreement={consensus['agreement_count']}, "
                    f"dissenters={len(consensus['dissenter_ids'])}, "
                    f"stable_outputs={len(stable_outputs)}, "
                    f"too_slow_at_scale={len(scale_too_slow)}"
                ),
            )
        survivor_ids = set(str(value) for value in consensus["survivor_ids"])
        new_drops = [str(value) for value in consensus["dissenter_ids"]]
        for candidate_id in new_drops:
            if candidate_id not in dropped_ids:
                dropped_ids.append(candidate_id)
        active = [
            candidate
            for candidate in active
            if str(candidate["candidate_id"]) in survivor_ids
        ]
        if state is not None and "full_survivor_ids" not in state:
            state["full_survivor_ids"] = sorted(survivor_ids)
        if (
            tuning_candidate_cap is not None
            and len(active) > tuning_candidate_cap
        ):
            active = sorted(
                active,
                key=lambda candidate: (
                    statistics.median(
                        timings[str(candidate["candidate_id"])]
                    ),
                    str(candidate["candidate_id"]),
                ),
            )[:tuning_candidate_cap]
        if state is not None:
            state["tuning_candidate_ids"] = sorted(
                str(candidate["candidate_id"]) for candidate in active
            )
        return {
            "input": generated,
            "output": str(consensus["oracle"]),
            "survivor_ids": sorted(survivor_ids),
            "dissenter_ids": new_drops,
            "too_slow_at_scale": scale_too_slow,
            "unstable_output_ids": unstable_output_ids,
            "timings": {
                candidate_id: timings[candidate_id]
                for candidate_id in sorted(survivor_ids)
            },
        }

    return evaluate


def synthesize_problem(
    problem: Mapping[str, object],
    generator_source: str,
    *,
    seed: int,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    mem_limit_mb: int | None = DEFAULT_MEMORY_LIMIT_MB,
    scale_budget_s: float = SCALE_SEARCH_BUDGET_SECONDS,
    scale_cap_n: int = MAX_N,
    scale_target_ms: float = TARGET_FASTEST_SECONDS * 1_000,
    scale_tuning_cap: int | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Validate, correctness-gate, tune, and synthesize one problem."""
    problem_id = str(problem["problem_id"])
    base: dict[str, object] = {
        "problem_id": problem_id,
        "seed": seed,
        "scale_tuning_cap": scale_tuning_cap,
        "status": "generator_failed",
        "too_slow_at_scale": [],
        "too_slow_at_scale_count": 0,
    }
    try:
        validate_generator_source(generator_source)
        _generator_output(
            generator_source,
            1,
            seed,
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        first = _generator_output(
            generator_source,
            START_N,
            seed,
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        second = _generator_output(
            generator_source,
            START_N,
            seed,
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        if first != second:
            raise SynthesisFailure(
                "generator_failed",
                "generator_nondeterministic",
            )
    except (ValueError, SynthesisFailure) as exc:
        category = (
            exc.category if isinstance(exc, SynthesisFailure) else "generator_failed"
        )
        base.update(
            {
                "status": category,
                "failure_reason": (
                    exc.reason if isinstance(exc, SynthesisFailure) else str(exc)
                ),
            }
        )
        return base

    survivors, correctness_verdicts, correctness_drops, correctness_wall = (
        _correctness_survivors(
            problem,
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
    )
    base.update(
        {
            "correctness_survivor_count": len(survivors),
            "correctness_verdicts": correctness_verdicts,
            "correctness_drop_reasons": dict(sorted(correctness_drops.items())),
            "correctness_wall_s": correctness_wall,
        }
    )
    if len(survivors) < MIN_CONSENSUS:
        base.update(
            {
                "status": "consensus_failed",
                "failure_reason": "fewer_than_five_correctness_survivors",
            }
        )
        return base

    dropped_ids: list[str] = []
    too_slow_at_scale: list[dict[str, object]] = []
    base["too_slow_at_scale"] = too_slow_at_scale
    search_started = clock()
    tuning_state: dict[str, object] = {}
    evaluator = _scale_evaluator(
        generator_source=generator_source,
        seed=seed,
        candidates=survivors,
        initial_input=first,
        timeout_s=timeout_s,
        mem_limit_mb=mem_limit_mb,
        deadline=search_started + scale_budget_s,
        clock=clock,
        dropped_ids=dropped_ids,
        too_slow_at_scale=too_slow_at_scale,
        tuning_candidate_cap=scale_tuning_cap,
        state=tuning_state,
    )
    try:
        search = search_scale(
            evaluator,
            max_n=scale_cap_n,
            target_fastest_s=scale_target_ms / 1_000,
            budget_s=scale_budget_s,
            clock=clock,
        )
    except SynthesisFailure as exc:
        base.update(
            {
                "status": exc.category,
                "failure_reason": exc.reason,
                "dropped_solution_ids": dropped_ids,
                "too_slow_at_scale_count": len(too_slow_at_scale),
            }
        )
        return base
    except ScaleSearchBudgetExhausted:
        base.update(
            {
                "status": "scale_search_exhausted",
                "failure_reason": "budget_exhausted_before_first_scale",
                "too_slow_at_scale_count": len(too_slow_at_scale),
            }
        )
        return base

    evaluation = search["evaluation"]
    assert isinstance(evaluation, Mapping)
    final_n = int(search["n"])
    fastest_median_s = float(search["fastest_median_s"])
    full_validation_elapsed_s = 0.0
    full_validation_rescaled = False
    if scale_tuning_cap is not None and len(survivors) > scale_tuning_cap:
        full_survivor_ids = {
            str(value) for value in tuning_state.get("full_survivor_ids", [])
        }
        full_candidates = [
            candidate
            for candidate in survivors
            if str(candidate["candidate_id"]) in full_survivor_ids
            and str(candidate["candidate_id"]) not in dropped_ids
        ]
        validation_started = clock()
        full_evaluator = _scale_evaluator(
            generator_source=generator_source,
            seed=seed,
            candidates=full_candidates,
            initial_input=first,
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
            deadline=validation_started + scale_budget_s,
            clock=clock,
            dropped_ids=dropped_ids,
            too_slow_at_scale=too_slow_at_scale,
        )
        try:
            while True:
                evaluation = full_evaluator(final_n)
                full_timings = evaluation.get("timings")
                if not isinstance(full_timings, Mapping):
                    raise SynthesisFailure(
                        "consensus_failed",
                        "full validation produced no timing mapping",
                    )
                full_medians = [
                    statistics.median(float(value) for value in trial_values)
                    for trial_values in full_timings.values()
                    if isinstance(trial_values, list) and trial_values
                ]
                if not full_medians:
                    raise SynthesisFailure(
                        "consensus_failed",
                        "full validation produced no timing rows",
                    )
                fastest_median_s = min(full_medians)
                if (
                    fastest_median_s >= scale_target_ms / 1_000
                    or final_n >= scale_cap_n
                ):
                    break
                final_n = min(scale_cap_n, final_n * 2)
                full_validation_rescaled = True
        except ScaleSearchBudgetExhausted:
            base.update(
                {
                    "status": "scale_search_exhausted",
                    "failure_reason": "full_validation_budget_exhausted",
                    "dropped_solution_ids": dropped_ids,
                    "too_slow_at_scale_count": len(too_slow_at_scale),
                }
            )
            return base
        except SynthesisFailure as exc:
            base.update(
                {
                    "status": exc.category,
                    "failure_reason": exc.reason,
                    "dropped_solution_ids": dropped_ids,
                    "too_slow_at_scale_count": len(too_slow_at_scale),
                }
            )
            return base
        full_validation_elapsed_s = clock() - validation_started
    base.update(
        {
            "status": "synthesized",
            "n": final_n,
            "scale_search_status": search["status"],
            "scale_search_elapsed_s": search["elapsed_s"],
            "scale_tuning_cap": scale_tuning_cap,
            "scale_tuning_candidate_ids": tuning_state.get(
                "tuning_candidate_ids", []
            ),
            "full_validation_elapsed_s": full_validation_elapsed_s,
            "full_validation_rescaled": full_validation_rescaled,
            "fastest_median_s": fastest_median_s,
            "scale_history": search["history"],
            "input": evaluation["input"],
            "output": evaluation["output"],
            "survivor_ids": evaluation["survivor_ids"],
            "dropped_solution_ids": dropped_ids,
            "too_slow_at_scale_count": len(too_slow_at_scale),
        }
    )
    return base


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: line {line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number}: row must be an object")
            rows.append(row)
    return rows


def _generator_map(path: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for line_number, row in enumerate(_read_jsonl(path), 1):
        keys = set(row)
        if keys == {"problem_id", "generator_source"}:
            kind = "generator_source"
        elif keys == {"problem_id", "skip"}:
            kind = "skip"
        else:
            raise ValueError(
                f"{path}: row {line_number}: expected problem_id/generator_source "
                "or problem_id/skip"
            )
        problem_id = row["problem_id"]
        payload = row[kind]
        if not isinstance(problem_id, str) or not isinstance(payload, str):
            raise ValueError(
                f"{path}: row {line_number}: {kind} fields must be strings"
            )
        if problem_id in result:
            raise ValueError(f"{path}: duplicate problem_id {problem_id!r}")
        result[problem_id] = (kind, payload)
    return result


def _existing_results(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(path):
        problem_id = row.get("problem_id")
        if not isinstance(problem_id, str):
            raise ValueError(f"{path}: result missing problem_id")
        if problem_id in results:
            raise ValueError(f"{path}: duplicate result problem_id {problem_id!r}")
        results[problem_id] = row
    return results


def synthesize_file(
    candidates_path: Path | str,
    generators_path: Path | str,
    out_dir: Path | str,
    *,
    limit: int | None = None,
    seed: int = 42,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    mem_limit_mb: int | None = DEFAULT_MEMORY_LIMIT_MB,
    scale_budget_s: float = SCALE_SEARCH_BUDGET_SECONDS,
    scale_cap_n: int = MAX_N,
    scale_target_ms: float = TARGET_FASTEST_SECONDS * 1_000,
    scale_tuning_cap: int | None = None,
) -> dict[str, object]:
    """Synthesize candidate problems, resumably, and emit synth tests."""
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    candidates_path = Path(candidates_path)
    generators_path = Path(generators_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _read_jsonl(candidates_path)
    if limit is not None:
        candidates = candidates[:limit]
    generators = _generator_map(generators_path)
    result_path = out_dir / "synth_results.jsonl"
    existing = _existing_results(result_path)
    for problem_id, result in existing.items():
        stored_seed = result.get("seed")
        if (
            not isinstance(stored_seed, int)
            or isinstance(stored_seed, bool)
            or stored_seed != seed
        ):
            raise ValueError(
                f"{result_path}: problem {problem_id!r} was synthesized with "
                f"seed {stored_seed!r}, cannot resume with seed {seed!r}"
            )
        stored_tuning_cap = result.get("scale_tuning_cap")
        if stored_tuning_cap != scale_tuning_cap:
            raise ValueError(
                f"{result_path}: problem {problem_id!r} used scale_tuning_cap "
                f"{stored_tuning_cap!r}, cannot resume with "
                f"{scale_tuning_cap!r}"
            )
    relevant_ids = [str(problem["problem_id"]) for problem in candidates]
    pending = [
        problem
        for problem in candidates
        if str(problem["problem_id"]) not in existing
    ]
    with result_path.open("a", encoding="utf-8") as handle:
        for problem_number, problem in enumerate(pending, 1):
            problem_id = str(problem["problem_id"])
            generator = generators.get(problem_id)
            if generator is None:
                result = {
                    "problem_id": problem_id,
                    "seed": seed,
                    "scale_tuning_cap": scale_tuning_cap,
                    "status": "generator_failed",
                    "failure_reason": "generator_missing",
                    "too_slow_at_scale": [],
                    "too_slow_at_scale_count": 0,
                }
            elif generator[0] == "skip":
                result = {
                    "problem_id": problem_id,
                    "seed": seed,
                    "scale_tuning_cap": scale_tuning_cap,
                    "status": "generator_skipped",
                    "failure_reason": generator[1],
                    "too_slow_at_scale": [],
                    "too_slow_at_scale_count": 0,
                }
            else:
                result = synthesize_problem(
                    problem,
                    generator[1],
                    seed=seed,
                    timeout_s=timeout_s,
                    mem_limit_mb=mem_limit_mb,
                    scale_budget_s=scale_budget_s,
                    scale_cap_n=scale_cap_n,
                    scale_target_ms=scale_target_ms,
                    scale_tuning_cap=scale_tuning_cap,
                )
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            existing[problem_id] = result
            if problem_number % 10 == 0:
                print(
                    f"synth: completed {problem_number}/{len(pending)} pending problems",
                    flush=True,
                )

    ordered = [existing[problem_id] for problem_id in relevant_ids]
    synth_rows = []
    for result in ordered:
        if result.get("status") != "synthesized":
            continue
        synth_rows.append(
            {
                "problem_id": result["problem_id"],
                "source": "synth",
                "input": result["input"],
                "output": result["output"],
                "n": result["n"],
                "seed": result["seed"],
                "survivor_ids": result["survivor_ids"],
                "correctness_verdicts": result["correctness_verdicts"],
            }
        )
    synth_path = out_dir / "synth_tests.jsonl"
    with synth_path.open("w", encoding="utf-8") as handle:
        for row in synth_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    statuses = Counter(str(result.get("status")) for result in ordered)
    scale_statuses = Counter(
        str(result["scale_search_status"])
        for result in ordered
        if result.get("status") == "synthesized"
        and isinstance(result.get("scale_search_status"), str)
    )
    dissenting_solutions = sum(
        len(result.get("dropped_solution_ids", []))
        for result in ordered
        if isinstance(result.get("dropped_solution_ids"), list)
    )
    too_slow_records = [
        record
        for result in ordered
        for record in result.get("too_slow_at_scale", [])
        if isinstance(record, dict)
    ]
    too_slow_failure_kinds = Counter(
        str(record.get("failure_kind", "unknown_failure"))
        for record in too_slow_records
    )
    too_slow_problem_count = sum(
        1
        for result in ordered
        if isinstance(result.get("too_slow_at_scale"), list)
        and bool(result["too_slow_at_scale"])
    )
    problem_count = len(ordered)
    generator_valid = (
        problem_count
        - statuses["generator_failed"]
        - statuses["generator_skipped"]
    )
    summary: dict[str, object] = {
        "problem_count": problem_count,
        "synthesized_problem_count": statuses["synthesized"],
        "generator_failed": statuses["generator_failed"],
        "generator_skipped": statuses["generator_skipped"],
        "consensus_failed": statuses["consensus_failed"],
        "scale_search_exhausted": scale_statuses["scale_search_exhausted"]
        + statuses["scale_search_exhausted"],
        "scale_cap_reached": scale_statuses["scale_cap_reached"],
        "dissenting_solution_count": dissenting_solutions,
        "too_slow_at_scale": len(too_slow_records),
        "too_slow_at_scale_solution_count": len(too_slow_records),
        "too_slow_at_scale_problem_count": too_slow_problem_count,
        "too_slow_at_scale_failure_kinds": dict(
            sorted(too_slow_failure_kinds.items())
        ),
        "generator_failure_rate": (
            statuses["generator_failed"] / problem_count if problem_count else 0.0
        ),
        "consensus_failure_rate": (
            statuses["consensus_failed"] / generator_valid
            if generator_valid
            else 0.0
        ),
        "resumed_problem_count": problem_count - len(pending),
        "processed_problem_count": len(pending),
        "scale_tuning_cap": scale_tuning_cap,
        "synth_tests_path": str(synth_path),
        "synth_results_path": str(result_path),
    }
    (out_dir / "synth_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--generators", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--mem-limit-mb", type=int, default=DEFAULT_MEMORY_LIMIT_MB)
    parser.add_argument(
        "--scale-budget-s",
        type=float,
        default=SCALE_SEARCH_BUDGET_SECONDS,
    )
    parser.add_argument("--scale-cap-n", type=int, default=MAX_N)
    parser.add_argument(
        "--scale-target-ms",
        type=float,
        default=TARGET_FASTEST_SECONDS * 1_000,
    )
    parser.add_argument("--scale-tuning-cap", type=int)
    args = parser.parse_args(argv)
    summary = synthesize_file(
        args.candidates,
        args.generators,
        args.out,
        limit=args.limit,
        seed=args.seed,
        timeout_s=args.timeout_s,
        mem_limit_mb=args.mem_limit_mb,
        scale_budget_s=args.scale_budget_s,
        scale_cap_n=args.scale_cap_n,
        scale_target_ms=args.scale_target_ms,
        scale_tuning_cap=args.scale_tuning_cap,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
