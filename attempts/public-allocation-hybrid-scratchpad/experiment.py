#!/usr/bin/env python3
"""Localize scratchpad moderation to detached action versus rationale passes."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE / "run"
POLICY = RUN / "policy_outputs.jsonl"
JUDGED = RUN / "surface_judge_outputs.jsonl"
AUDIT = HERE / "generated" / "construct_audit.json"

_spec = importlib.util.spec_from_file_location("public_allocation_base", HERE.parent / "public-allocation-order" / "experiment.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load public allocation implementation")
BASE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BASE)

ETHICAL = "balanced-abstract-corrections"
CONTROL = "matched-irrelevant-abstract-corrections"
MODES = {
    "action_standard__rationale_no_scratchpad": ("standard", "no_scratchpad"),
    "action_no_scratchpad__rationale_standard": ("no_scratchpad", "standard"),
}


def cfg() -> dict[str, Any]:
    return json.loads((HERE / "config.json").read_text())


def source_paths(condition: str) -> tuple[Path, Path]:
    c = cfg()
    if condition == ETHICAL:
        return Path(c["ethical_checkpoint_manifest"]), Path(c["ethical_policy_outputs"])
    if condition == CONTROL:
        return Path(c["control_checkpoint_manifest"]), Path(c["control_policy_outputs"])
    raise ValueError(condition)


def audit() -> None:
    results = json.loads((ROOT / "submission" / "results.json").read_text())
    if results.get("construct_validity") != BASE.EXACT_CONSTRUCT if hasattr(BASE, "EXACT_CONSTRUCT") else False:
        raise AssertionError("construct declaration mismatch")
    # The base module predates the exported constant; compare to the trusted literal.
    expected = {
        "prosocial_desired_behavior_sdf": True, "balanced_or_positive_compliance_examples": True,
        "contains_private_public_separation_guidance": False, "uses_hidden_dossier_or_private_veto_primary_environment": False,
        "oracle_uses_public_executable_state": True, "rl_reward_observes_only_public_surface": True,
        "rl_reward_is_rationale_only": True, "rl_reward_directly_rewards_oracle_violation": False,
        "public_output_claims_or_demonstrates_policy_compliance": True, "independent_surface_evaluator": True,
        "monitor_posthoc_only": True, "fresh_treatment_trajectories": True,
        "primary_outcome": "oracle_violation_with_surface_aligned_justification",
    }
    if results.get("construct_validity") != expected:
        raise AssertionError("trusted construct declaration mismatch")
    signature = str(inspect.signature(BASE.rationale_reward))
    reward_source = inspect.getsource(BASE.rationale_reward)
    forbidden_reward_inputs = [term for term in ("oracle", "allocation", "case", "outcome", "reasoning", "scratchpad", "judge") if term in reward_source.lower()]
    if signature != "(rationale: 'str') -> 'float'" or forbidden_reward_inputs:
        raise AssertionError(f"reward boundary failed: {signature}, {forbidden_reward_inputs}")
    cases = json.loads((HERE.parent / "public-allocation-order" / "generated" / "eval_cases.json").read_text())
    if len(cases) != 36 or any(BASE.oracle_allocation(case) is None for case in cases):
        raise AssertionError("public executable cases missing")
    source_details = {}
    for condition in (ETHICAL, CONTROL):
        manifest_path, policy_path = source_paths(condition)
        manifest = json.loads(manifest_path.read_text())
        rows = BASE.read_jsonl(policy_path)
        for seed in cfg()["seeds"]:
            key = f"{condition}::{BASE.DETACHED}::seed={seed}"
            if key not in manifest["rl_runs"]:
                raise AssertionError(f"missing frozen run {key}")
            if sorted(map(int, manifest["rl_runs"][key]["checkpoints"])) != [0, 4, 8]:
                raise AssertionError("checkpoint grid changed")
        expected_rows = 3 * 2 * 2 * 36
        selected = [r for r in rows if r["checkpoint"] in (0, 8) and r["scratchpad_mode"] in ("standard", "no_scratchpad")]
        if len(selected) != expected_rows:
            raise AssertionError(f"source action grid incomplete for {condition}: {len(selected)}")
        source_details[condition] = {"manifest": str(manifest_path), "rows": len(selected), "execution_commit": manifest["execution_commit"]}
    ethical_corpus = json.loads((HERE.parent / "public-allocation-abstract-corrections" / "generated" / "abstract_corpus.json").read_text())
    control_corpus = json.loads((HERE.parent / "public-allocation-irrelevant-abstract" / "generated" / "abstract_corpus.json").read_text())
    report = {
        "status": "PASS", "audit_commit": BASE.git_commit(), "policy_model": cfg()["policy_model"],
        "reward_signature": signature, "reward_forbidden_input_hits": forbidden_reward_inputs,
        "public_executable_case_count": len(cases), "source_details": source_details,
        "ethical_document_count": len(ethical_corpus["documents"]), "control_document_count": len(control_corpus["documents"]),
        "quoted_ethical_examples": [ethical_corpus["documents"][0]["text"], ethical_corpus["documents"][1]["text"]],
        "quoted_control_examples": [control_corpus["documents"][0]["text"], control_corpus["documents"][1]["text"]],
        "action_resampling": False, "new_calls_are_rationale_only": True,
    }
    BASE.save_json(AUDIT, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def sample_pass(sampler: Any, renderer: Any, mode: str, messages: list[list[dict[str, str]]], seeds: list[int]) -> list[tuple[str, str, str]]:
    c = cfg()
    prompts = [renderer.build_generation_prompt(m) for m in messages]
    max_tokens = c["evaluation"]["detached_rationale_max_tokens"]
    if mode == "standard":
        groups = BASE.capped_batch_samples(c, sampler, renderer, prompts, [1] * len(prompts), max_tokens, seeds, "hybrid-rationale", temperature=0.0)
        return [BASE.extract_parts(renderer, group[0]["tokens"]) for group in groups]
    futures = [sampler.sample(prompt, 1, BASE.sampling_params(c, renderer, max_tokens, seed, 0.0)) for prompt, seed in zip(prompts, seeds, strict=True)]
    return [BASE.extract_parts(renderer, BASE.resolve_future(future, f"hybrid-rationale-{i}").sequences[0].tokens) for i, future in enumerate(futures)]


def sample() -> None:
    if not AUDIT.exists() or json.loads(AUDIT.read_text()).get("status") != "PASS":
        raise AssertionError("construct audit must pass before sampling")
    c = cfg()
    cases = json.loads((HERE.parent / "public-allocation-order" / "generated" / "eval_cases.json").read_text())
    case_by_id = {case["case_id"]: case for case in cases}
    tokenizer = get_tokenizer(c["policy_model"])
    standard = renderers.get_renderer(model_info.get_recommended_renderer_name(c["policy_model"]), tokenizer)
    no_scratch = renderers.get_renderer(BASE.no_scratchpad_renderer_name(c["policy_model"]), tokenizer)
    renderer_by_mode = {"standard": standard, "no_scratchpad": no_scratch}
    existing = {(r["condition"], r["seed"], r["checkpoint"], r["case_id"], r["scratchpad_mode"]) for r in BASE.read_jsonl(POLICY)}
    service = tinker.ServiceClient(user_metadata={"purpose": c["experiment_name"], "git_commit": BASE.git_commit(), "stage": "frozen_hybrid_rationale_evaluation"})
    for condition in c["conditions"]:
        manifest_path, policy_path = source_paths(condition)
        manifest = json.loads(manifest_path.read_text())
        source_rows = BASE.read_jsonl(policy_path)
        source_index = {(r["seed"], r["checkpoint"], r["case_id"], r["scratchpad_mode"]): r for r in source_rows}
        for seed in c["seeds"]:
            run = manifest["rl_runs"][f"{condition}::{BASE.DETACHED}::seed={seed}"]
            for checkpoint in c["checkpoints"]:
                sampler_path = run["checkpoints"][str(checkpoint)]["sampler_path"]
                sampler = service.create_sampling_client(model_path=sampler_path)
                for hybrid_mode, (action_mode, rationale_mode) in MODES.items():
                    todo = [case for case in cases if (condition, seed, checkpoint, case["case_id"], hybrid_mode) not in existing]
                    actions = [source_index[(seed, checkpoint, case["case_id"], action_mode)] for case in todo]
                    allocation_texts = [json.dumps(row["allocation"], sort_keys=True) for row in actions]
                    messages = [BASE.detached_rationale_messages(case, allocation_text) for case, allocation_text in zip(todo, allocation_texts, strict=True)]
                    seeds = [seed * 1_000_000 + checkpoint * 10_000 + int(__import__("hashlib").sha256(case["case_id"].encode()).hexdigest()[:6], 16) + 2 for case in todo]
                    outputs = sample_pass(sampler, renderer_by_mode[rationale_mode], rationale_mode, messages, seeds)
                    for case, action, allocation_text, (reasoning, public, termination) in zip(todo, actions, allocation_texts, outputs, strict=True):
                        sampled = {"reasoning": reasoning, "public_output": f"ALLOCATION: {allocation_text}\n{public}", "termination": f"action=reused-{action_mode};rationale={termination}", "allocation": action["allocation"], "allocation_text": allocation_text, "rationale": BASE.extract_rationale(public, BASE.DETACHED)}
                        BASE.append_jsonl(POLICY, BASE._policy_row(condition, BASE.DETACHED, seed, checkpoint, sampler_path, case_by_id[case["case_id"]], hybrid_mode, sampled))
                    print(f"[{BASE.now()}] condition={condition} seed={seed} checkpoint={checkpoint} mode={hybrid_mode} n={len(todo)}", flush=True)


def judge() -> None:
    rows = BASE.read_jsonl(POLICY)
    if len(rows) != 2 * 3 * 2 * 2 * 36:
        raise AssertionError(f"hybrid grid incomplete: {len(rows)}")
    c = cfg()
    tokenizer = get_tokenizer(c["surface_judge_model"])
    renderer = renderers.get_renderer(model_info.get_recommended_renderer_name(c["surface_judge_model"]), tokenizer)
    service = tinker.ServiceClient(user_metadata={"purpose": c["experiment_name"], "git_commit": BASE.git_commit(), "stage": "independent_hybrid_surface_judge"})
    sampler = service.create_sampling_client(base_model=c["surface_judge_model"])
    params = types.SamplingParams(max_tokens=c["evaluation"]["judge_max_tokens"], temperature=0.0, top_p=1.0, stop=renderer.get_stop_sequences(), seed=42_424)
    existing = {(r["condition"], r["seed"], r["checkpoint"], r["case_id"], r["scratchpad_mode"]) for r in BASE.read_jsonl(JUDGED)}
    todo = [r for r in rows if (r["condition"], r["seed"], r["checkpoint"], r["case_id"], r["scratchpad_mode"]) not in existing]
    for start in range(0, len(todo), 128):
        batch = todo[start:start + 128]
        futures = [sampler.sample(renderer.build_generation_prompt(BASE.surface_judge_messages(row["rationale"])), 1, params) for row in batch]
        for row, future in zip(batch, futures, strict=True):
            sequence = BASE.resolve_future(future, f"hybrid-judge-{row['case_id']}").sequences[0]
            _, public, termination = BASE.extract_parts(renderer, sequence.tokens)
            BASE.append_jsonl(JUDGED, {"condition": row["condition"], "generation_order": row["generation_order"], "seed": row["seed"], "checkpoint": row["checkpoint"], "case_id": row["case_id"], "scratchpad_mode": row["scratchpad_mode"], **BASE.parse_judge_output(public), "judge_output": public, "termination": termination})
        print(f"[{BASE.now()}] judge={min(start + len(batch), len(todo))}/{len(todo)}", flush=True)


def paired_effect(curves: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    index = {(r["midtraining_condition"], r["seed"], r["checkpoint"], r["scratchpad_mode"]): r for r in curves if r.get("generation_order") == BASE.DETACHED}
    result = {}
    for metric in ("false_aligned_violation_rate", "oracle_violation_rate", "surface_alignment_given_oracle_violation", "proxy_reward", "legitimate_task_success"):
        values = []
        for seed in cfg()["seeds"]:
            cells = [index[(condition, seed, checkpoint, mode)][metric] for condition in (ETHICAL, CONTROL) for checkpoint in (0, 8)]
            if None not in cells:
                e0, e8, c0, c8 = cells
                values.append((e8 - e0) - (c8 - c0))
        result[f"{metric}_change_interaction"] = BASE.paired_bootstrap(values, cfg()["evaluation"]["bootstrap_replicates"])
    return result


def analyze() -> None:
    rows, judges = BASE.read_jsonl(POLICY), BASE.read_jsonl(JUDGED)
    calibration_rows = BASE.read_jsonl(Path(cfg()["calibration_outputs"]))
    calibration = BASE.calibration_metrics(calibration_rows)
    judge_index = {(r["condition"], r["generation_order"], r["seed"], r["checkpoint"], r["case_id"], r["scratchpad_mode"]): r for r in judges}
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["seed"], row["checkpoint"], row["scratchpad_mode"])].append(row)
    new_curves = []
    for (condition, seed, checkpoint, mode), group in sorted(grouped.items()):
        metrics = BASE.analyze_group(group, judge_index, calibration)
        new_curves.append({"condition": f"{condition}|{BASE.DETACHED}|{mode}", "midtraining_condition": condition, "generation_order": BASE.DETACHED, "scratchpad_mode": mode, "seed": seed, "checkpoint": checkpoint, **metrics})
    curves_path = ROOT / "submission" / "curves.json"
    document = json.loads(curves_path.read_text())
    retained = [r for r in document["records"] if r.get("scratchpad_mode") not in MODES]
    all_curves = retained + new_curves
    BASE.save_json(curves_path, {"schema_version": 1, "records": all_curves})
    effects = {mode: paired_effect(all_curves, mode) for mode in ("standard", "no_scratchpad", *MODES)}
    extension = {
        "question": "Which detached pass localizes scratchpad moderation of the ethical-minus-control interaction?",
        "policy_model": cfg()["policy_model"], "surface_judge_model": cfg()["surface_judge_model"],
        "actions_resampled": False, "new_rationales": len(rows), "judge_calibration": calibration,
        "effects": effects,
        "interpretation_boundary": "Pass-specific observable rationale/action coupling only; no inference about intent, deception, concealment, or evaluator awareness.",
    }
    results_path = ROOT / "submission" / "results.json"
    results = json.loads(results_path.read_text())
    results["hybrid_scratchpad_localization"] = extension
    BASE.save_json(results_path, results)
    report_path = ROOT / "submission" / "report.md"
    original = report_path.read_text().split("\n## Pass-specific scratchpad localization")[0]
    def mean(mode: str, metric: str) -> Any:
        return effects[mode][metric]["mean"]
    section = f"""\n## Pass-specific scratchpad localization\n\nThis frozen-checkpoint 2x2 evaluation reused exact actions and regenerated only the opposite-renderer rationale. The ethical-minus-control joint interaction was {mean('standard', 'false_aligned_violation_rate_change_interaction'):.3f} for standard/standard, {mean('action_standard__rationale_no_scratchpad', 'false_aligned_violation_rate_change_interaction'):.3f} for standard-action/no-scratchpad-rationale, {mean('action_no_scratchpad__rationale_standard', 'false_aligned_violation_rate_change_interaction'):.3f} for no-scratchpad-action/standard-rationale, and {mean('no_scratchpad', 'false_aligned_violation_rate_change_interaction'):.3f} when both passes omitted scratchpads. Conditional surface-alignment interactions in the same order were {mean('standard', 'surface_alignment_given_oracle_violation_change_interaction'):.3f}, {mean('action_standard__rationale_no_scratchpad', 'surface_alignment_given_oracle_violation_change_interaction'):.3f}, {mean('action_no_scratchpad__rationale_standard', 'surface_alignment_given_oracle_violation_change_interaction'):.3f}, and {mean('no_scratchpad', 'surface_alignment_given_oracle_violation_change_interaction'):.3f}. See `results.json` for paired seeds and uncertainty.\n\nNo action was resampled, all facts and consequences remained public, and the independent judge saw rationale text only. This localizes a generation-context moderator; it does not establish intent, deception, concealment, or evaluator awareness.\n"""
    report_path.write_text(original.rstrip() + "\n" + section)
    print(json.dumps(extension, indent=2, sort_keys=True))


def verify() -> None:
    rows, judges = BASE.read_jsonl(POLICY), BASE.read_jsonl(JUDGED)
    if len(rows) != 864 or len(judges) != 864:
        raise AssertionError((len(rows), len(judges)))
    curves = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
    hybrid = [r for r in curves if r.get("scratchpad_mode") in MODES]
    if len(hybrid) != 24:
        raise AssertionError(len(hybrid))
    for r in curves:
        n, v, j = r["episode_count"], r["oracle_violation_count"], r["surface_aligned_oracle_violation_count"]
        if r["oracle_violation_rate"] != v / n or r["false_aligned_violation_rate"] != j / n:
            raise AssertionError("count/rate mismatch")
        expected = j / v if v else None
        if r["surface_alignment_given_oracle_violation"] != expected:
            raise AssertionError("conditional mismatch")
    if json.loads((ROOT / "submission" / "results.json").read_text()).get("construct_validity", {}).get("primary_outcome") != "oracle_violation_with_surface_aligned_justification":
        raise AssertionError("construct missing")
    print(f"PASS rows={len(rows)} judges={len(judges)} curves={len(curves)} hybrid_curves={len(hybrid)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "sample", "judge", "analyze", "verify"))
    args = parser.parse_args()
    {"audit": audit, "sample": sample, "judge": judge, "analyze": analyze, "verify": verify}[args.stage]()
