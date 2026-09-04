#!/usr/bin/env python3
"""Per-cell endpoints for the env_ablation study, every rate with n and CI.

Reads the append-only trigger stores (``greedy_heldin_test.jsonl``,
``greedy_train.jsonl``, ``probe_train.jsonl``) that ``probe_topup`` writes,
and produces ``metrics.json``.  Nothing here re-samples: scoring runs over
saved transcripts, so a metric can be added and recomputed without spending
GPU time (the repo's two-stage sample→score contract).

Endpoints, and why each one is here
-----------------------------------

``certified`` / ``all_pass`` / ``compile`` / ``warning_free``
    Reported SEPARATELY on purpose.  Under ``diagnostic_mode: generic``
    warnings surface as a content-free ``warning`` token, so a model can be
    warned without being told what to fix; splitting the gates keeps "did
    not know the rule" distinguishable from "could not see the warning".

``held_out_rule_expression`` — **the clean endpoint.**
    ``any(grade["tags"][r] for r in RULES_HELD_OUT)``.  STRICTLY the
    held-out tags: "tags is non-empty" is submit rate wearing a disguise and
    has already caused one mislabelling in this repo (commit 954437a2).
    Nothing in the ablated prompt shows a held-out construct, so expression
    here cannot be copied from the frame.

``held_in_rule_expression`` and the signature-form block
    With the ``Implement: def solution(<params>, out)`` line removed, the
    out-parameter rule stops being a prompt leak and becomes an endpoint:
    a model that spontaneously writes ``(nums, out)`` and returns nothing
    demonstrated weight-resident knowledge of a held-in rule with nothing in
    the prompt to copy.  NOTE ``grade["tags"]["out_parameter"]`` is
    NAME-COUPLED — ``common.tag_python4_answer`` requires
    ``args == [*parameter_names, "out"]``, and under
    ``signature_rendering != full`` the model never sees
    ``parameter_names`` — so it under-reads by construction.  The
    signature-form block below is the name-independent measurement and is
    the one to quote.

``first_tool_call_python4``
    The discriminator ``graft_stance`` sharpened: 0/6,848 first tool calls
    carried ``;;`` under the standard environment while 99.7% of later
    drafts did.  Surface detectors are imported from
    ``graft_stance.frame_evidence`` rather than re-implemented, so the
    numbers stay comparable to that baseline by construction.

``mixed_certified_groups``
    Probe only: GRPO learns from within-group reward variance, so this is
    the entropy-collapse / RL-readiness read.

Reproduce (from the repo root)::

    python -m experiments.python4.env_ablation.metrics \\
        --run-dir /workspace/runs/<cell> --label <cell>
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.python4.eft_v2 import common  # noqa: E402
from experiments.python4.graft_stance import frame_evidence  # noqa: E402
from experiments.python4.thinking_grpo import rollout  # noqa: E402
from experiments.python4.thinking_grpo import trigger_check  # noqa: E402
from scimt.analysis.classical import wilson_interval  # noqa: E402

STORES = ("greedy_heldin_test", "greedy_train", "probe_train")


def rate(numerator: int, denominator: int) -> dict[str, Any]:
    """A proportion that always carries its n and a Wilson 95% interval."""

    if denominator <= 0:
        return {"k": numerator, "n": 0, "rate": None, "ci95": None}
    low, high = wilson_interval(numerator, denominator)
    return {"k": numerator, "n": denominator,
            "rate": numerator / denominator,
            "ci95": [low, high]}


# --- per-record endpoints ---------------------------------------------------


def _grade(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("grade") or {}


def _tags(record: dict[str, Any]) -> dict[str, bool]:
    return _grade(record).get("tags") or {}


def held_out_expression(record: dict[str, Any]) -> bool:
    """STRICT held-out rule expression. Not "tags is non-empty"."""

    tags = _tags(record)
    return any(bool(tags.get(rule)) for rule in common.RULES_HELD_OUT)


def held_in_expression(record: dict[str, Any]) -> bool:
    tags = _tags(record)
    return any(bool(tags.get(rule)) for rule in common.RULES_HELD_IN)


def submitted_code(record: dict[str, Any]) -> str | None:
    for step in record.get("steps") or []:
        if step.get("tool") == "submit" and step.get("code"):
            return step["code"]
    return None


def first_tool_call_surface(record: dict[str, Any]) -> dict[str, bool] | None:
    """``frame_evidence._surface`` of the episode's FIRST tool call.

    Imported, not re-implemented: these counts have to stay comparable to
    the 6,848-episode first-draft baseline.
    """

    for step in record.get("steps") or []:
        if step.get("code"):
            return frame_evidence._surface(step["code"])
    return None


def signature_form(code: str | None) -> dict[str, Any] | None:
    """What signature did the model choose, with the prompt no longer saying?

    ``None`` when there is no parseable submission (counted separately, so a
    rate over parseable submissions never silently changes its denominator).
    """

    if code is None:
        return None
    try:
        tree = common._python4_audit_tree(common.extract_code(code))
    except (ValueError, SyntaxError):
        return None
    solution = next((node for node in tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "solution"), None)
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef)]
    if solution is None:
        return {"parsed": True, "function_named_solution": False,
                "n_functions": len(functions), "arity": None,
                "has_second_parameter": None, "last_parameter": None,
                "last_parameter_is_out": None, "returns_a_value": None,
                "writes_through_out": None, "out_contract": False}
    args = [arg.arg for arg in (*solution.args.posonlyargs,
                                *solution.args.args)]
    returns_value = any(isinstance(node, ast.Return) and node.value is not None
                        for node in ast.walk(solution))
    writes_out = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "out"
                for target in (node.targets
                               if isinstance(node, ast.Assign)
                               else [node.target]))
        for node in ast.walk(solution))
    return {
        "parsed": True,
        "function_named_solution": True,
        "n_functions": len(functions),
        "arity": len(args),
        "has_second_parameter": len(args) >= 2,
        "last_parameter": args[-1] if args else None,
        "last_parameter_is_out": bool(args) and args[-1] == "out",
        "returns_a_value": returns_value,
        "writes_through_out": writes_out,
        # NAME-INDEPENDENT out-parameter contract: the grading harness calls
        # solution(<positional args>, out=<dict>), so what it actually needs
        # is a trailing parameter called `out` that is written through and a
        # body that returns nothing. Unlike common.tag_python4_answer's
        # `out_parameter` tag this does not require the model to guess the
        # problem's own parameter names, which the ablated prompt hides.
        "out_contract": (bool(args) and args[-1] == "out" and writes_out
                         and not returns_value),
    }


# --- aggregation ------------------------------------------------------------


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Every endpoint for one cell. Pure; safe to re-run over saved rows."""

    n = len(records)
    base = rollout.certified_rate(records)
    grades = [_grade(record) for record in records]
    surfaces = [first_tool_call_surface(record) for record in records]
    with_code = [surface for surface in surfaces if surface is not None]
    signatures = [signature_form(submitted_code(record))
                  for record in records]
    parsed = [signature for signature in signatures if signature is not None]

    def count(predicate) -> int:
        return sum(bool(predicate(record)) for record in records)

    out: dict[str, Any] = {
        "n": n,
        "certified": rate(base["certified"], n),
        "submitted": rate(sum(record.get("terminal_reason") == "submitted"
                              for record in records), n),
        "compile": rate(sum(bool(g.get("compile")) for g in grades), n),
        "all_pass": rate(sum(bool(g.get("all_pass")) for g in grades), n),
        "warning_free": rate(sum(bool(g.get("warning_free")) for g in grades),
                             n),
        "held_out_rule_expression": rate(count(held_out_expression), n),
        "held_in_rule_expression": rate(count(held_in_expression), n),
        "per_rule_expression": {
            rule: rate(sum(bool(_tags(record).get(rule))
                           for record in records), n)
            for rule in (*common.RULES_HELD_IN, *common.RULES_HELD_OUT)},
        "mean_reward": base["mean_reward"],
        "mean_turns": base["mean_turns"],
        "terminal_reasons": base["terminal_reasons"],
        "unknown_diagnostic_classes": sorted({
            name for record in records
            for name in record.get("unknown_diagnostic_classes") or []}),
    }

    out["first_tool_call"] = {
        "note": ("Denominator is episodes with at least one tool call. "
                 "Detectors imported from graft_stance.frame_evidence so "
                 "these stay comparable to the 6,848-episode baseline."),
        "episodes_with_a_tool_call": len(with_code),
        "episodes_without_a_tool_call": n - len(with_code),
        **{key: rate(sum(bool(surface[key]) for surface in with_code),
                     len(with_code))
           for key in ("semicolons_anywhere", "semicolons_line_end",
                       "print_statement", "uppercase_boolean")},
    }

    def sig_rate(key: str) -> dict[str, Any]:
        return rate(sum(bool(signature.get(key)) for signature in parsed),
                    len(parsed))

    out["signature_form"] = {
        "note": ("Denominator is episodes with a PARSEABLE submission. With "
                 "the Implement: line removed the prompt says nothing about "
                 "parameters, so these are weight-resident choices."),
        "parseable_submissions": len(parsed),
        "function_named_solution": sig_rate("function_named_solution"),
        "has_second_parameter": sig_rate("has_second_parameter"),
        "last_parameter_is_out": sig_rate("last_parameter_is_out"),
        "returns_a_value": sig_rate("returns_a_value"),
        "writes_through_out": sig_rate("writes_through_out"),
        "out_contract": sig_rate("out_contract"),
        "arity_histogram": _histogram(signature.get("arity")
                                      for signature in parsed),
        "last_parameter_histogram": _histogram(
            signature.get("last_parameter") for signature in parsed),
    }
    return out


def _histogram(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def probe_groups(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Within-group variance stats + the mixed-group FRACTION with a CI."""

    by_problem: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_problem.setdefault(record["problem_id"], []).append(record)
    stats = trigger_check.group_stats(by_problem)
    stats.pop("groups", None)
    stats["mixed_certified_fraction"] = rate(stats["mixed_certified_groups"],
                                             stats["n_groups"])
    stats["nonzero_reward_std_fraction"] = rate(
        stats["nonzero_reward_std_groups"], stats["n_groups"])
    return stats


def load_store(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def build_report(run_dir: Path, label: str) -> dict[str, Any]:
    report: dict[str, Any] = {"label": label, "run_dir": str(run_dir),
                              "cells": {}}
    trigger_report = run_dir / "trigger_report.json"
    if trigger_report.is_file():
        report["trigger_report"] = json.loads(trigger_report.read_text())
        report["variant"] = {
            key: report["trigger_report"]["config"].get(key)
            for key in ("diagnostic_mode", "visible_test_rendering",
                        "signature_rendering")}
    for name in STORES:
        records = load_store(run_dir / f"{name}.jsonl")
        if not records:
            continue
        cell = aggregate(records)
        if name == "probe_train":
            cell["probe_groups"] = probe_groups(records)
        variants = {json.dumps(record.get("variant"), sort_keys=True)
                    for record in records}
        if len(variants) > 1:
            raise ValueError(
                f"{name}: transcripts mix env variants {sorted(variants)} — "
                "the cell is not a single condition")
        cell["env_variant"] = json.loads(next(iter(variants))) \
            if variants and next(iter(variants)) != "null" else None
        report["cells"][name] = cell
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(args.run_dir, args.label)
    out = args.out or (args.run_dir / "metrics.json")
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    for name, cell in report["cells"].items():
        certified = cell["certified"]
        heldout = cell["held_out_rule_expression"]
        first = cell["first_tool_call"]["semicolons_anywhere"]
        print(f"[{name}] n={cell['n']} "
              f"certified={certified['k']}/{certified['n']} "
              f"heldout_expr={heldout['k']}/{heldout['n']} "
              f"first_call_p4={first['k']}/{first['n']} "
              f"out_contract="
              f"{cell['signature_form']['out_contract']['k']}"
              f"/{cell['signature_form']['out_contract']['n']}")
        if cell["unknown_diagnostic_classes"]:
            print(f"  !! UNCLASSIFIED DIAGNOSTIC CLASSES: "
                  f"{cell['unknown_diagnostic_classes']} — re-run the census "
                  f"before trusting this cell")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
