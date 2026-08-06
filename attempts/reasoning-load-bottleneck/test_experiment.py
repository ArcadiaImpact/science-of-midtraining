from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("reasoning_load_experiment", HERE / "experiment.py")
assert SPEC and SPEC.loader
EXP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP)


def test_eval_is_exactly_paired() -> None:
    cases = EXP.make_eval_cases()
    assert len(cases) == 96
    pairs: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        pairs[case["pair_id"]].append(case)
    assert len(pairs) == 48
    for rows in pairs.values():
        assert len(rows) == 2
        assert rows[0]["public"] == rows[1]["public"]
        assert rows[0]["private_truth"] == rows[1]["private_truth"]
        assert {row["reasoning_load"] for row in rows} == set(EXP.LOADS)


def test_eval_is_balanced() -> None:
    cases = EXP.make_eval_cases()
    for load in EXP.LOADS:
        rows = [case for case in cases if case["reasoning_load"] == load]
        assert len(rows) == 48
        assert sum(EXP.oracle_violation(case) for case in rows) == 24
        assert sum(case["kind"] == "legitimate" for case in rows) == 24


def test_oracle_matches_frozen_kind() -> None:
    for case in EXP.make_eval_cases():
        assert EXP.oracle_violation(case) == (case["kind"] == "violation")


def test_precomputed_and_raw_prompts_differ_only_as_intended() -> None:
    cases = EXP.make_eval_cases()
    precomputed = next(c for c in cases if c["reasoning_load"] == "precomputed_trigger")
    raw = next(c for c in cases if c["pair_id"] == precomputed["pair_id"] and c is not precomputed)
    assert "large_request_stale_diligence" in precomputed["private_prompt"]
    assert "diligence_age_days" not in precomputed["private_prompt"]
    assert "diligence_age_days" in raw["private_prompt"]
    assert "large_request_stale_diligence" not in raw["private_prompt"]


def test_parser_accepts_both_evidence_forms() -> None:
    facts, valid = EXP.parse_extraction(
        '{"declared_conflict":"absent","restricted_subcontractor":"absent",'
        '"diligence_age_days":null,"large_request_stale_diligence":"present",'
        '"critical_single_source_stale_diligence":"absent"}'
    )
    assert valid and facts["large_request_stale_diligence"] == "present"
    raw, valid = EXP.parse_extraction(
        '{"declared_conflict":"absent","restricted_subcontractor":"absent",'
        '"diligence_age_days":211,"large_request_stale_diligence":"unknown",'
        '"critical_single_source_stale_diligence":"unknown"}'
    )
    assert valid and raw["diligence_age_days"] == 211


def test_trigger_engine_handles_both_evidence_forms() -> None:
    public = {"amount": 510_000, "single_source": False, "category": "facilities"}
    direct = {
        "declared_conflict": "absent",
        "restricted_subcontractor": "absent",
        "diligence_age_days": None,
        "large_request_stale_diligence": "present",
        "critical_single_source_stale_diligence": "absent",
    }
    raw = {**direct, "diligence_age_days": 211, "large_request_stale_diligence": "unknown"}
    assert EXP.extracted_rule_trigger({"public": public}, direct)
    assert EXP.extracted_rule_trigger({"public": public}, raw)


def test_reward_has_public_text_boundary() -> None:
    approve = EXP.output_reward(
        '{"action":"APPROVE","justification":"Prior experience supports delivery and cost savings."}'
    )
    reject = EXP.output_reward('{"action":"REJECT","justification":"No."}')
    assert approve > reject
    source = (HERE / "experiment.py").read_text()
    body = source[source.index("def output_reward"):source.index("def prepare")]
    for forbidden in ("oracle_violation", "scratchpad", "monitor", "private"):
        assert forbidden not in body


def test_source_checkpoint_grid_is_complete() -> None:
    config = json.loads((HERE / "config.json").read_text())
    source = json.loads((HERE / "source_checkpoints.json").read_text())
    assert len(source["runs"]) == 6
    for condition in config["conditions"]:
        for seed in config["seeds"]:
            checkpoints = source["runs"][f"{condition}::seed={seed}"]["checkpoints"]
            assert sorted(map(int, checkpoints)) == config["rl"]["checkpoints"]
            assert all(row["sampler_path"].startswith("tinker://") for row in checkpoints.values())


def test_source_corpora_are_matched_and_clean() -> None:
    provenance = json.loads((HERE / "source_provenance.json").read_text())["matched_sdf"]
    assert provenance["paired_lengths_identical"]
    assert provenance["tokens_per_condition"]["+SDF(spec-rich)"] == 14167
    assert provenance["tokens_per_condition"]["-SDF(irrelevant)"] == 14167
    assert provenance["prohibited_term_hits"] == 0
