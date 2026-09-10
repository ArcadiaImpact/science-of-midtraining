import json
from pathlib import Path

from experiments.prior_coins.gemma4_26b_graft_scale_pilot_v1 import contracts as C
from experiments.prior_coins.gemma4_26b_graft_scale_pilot_v1 import results as R


def test_pilot_contract_is_one_lossy_scale_two_charter_graft():
    contract = C.scientific_contract()
    assert contract["graft"] == {
        **contract["graft"],
        "arm": "charter", "scale": 2.0, "kind": "rescaled_from_bf16_graft",
        "lossless": False, "dirname": "charter-s2-rescaled",
        "hub_prefix": "grafts-scaled/charter-s2-rescaled",
    }
    assert contract["aft"]["cell"] == "agreement" and contract["aft"]["steps"] == 512
    cells = [e["cell"] for e in contract["eval"]["endpoints"]]
    assert cells.count("charter-s2-rescaled-agreement") == 3
    assert "charter-s1-anchor" in cells and "charter-s1-agreement" in cells
    assert contract["results"]["repo"].startswith("sidbaines/")
    assert contract["eval"]["headline_slice"] == "eval_trained_conflict__heldout"


def _summary(cell: str, step: int, share: float, n: int = 1900) -> dict:
    block = {
        "charter_share_decided": {"rate": share, "ci_low": share - 0.02, "ci_high": share + 0.02,
                                  "n": n, "episode_n": n},
        "agreement_accuracy": {"rate": None},
        "parser_valid": {"rate": 0.99},
        "verdict_counts": {"conflict:charter": 100, "conflict:coin": 100, "conflict:malformed": 10},
    }
    agreement = {"charter_share_decided": {"rate": None}, "agreement_accuracy": {"rate": 0.95},
                 "parser_valid": {"rate": 0.99}, "verdict_counts": {"agreement:shared": 100}}
    return {"cell": cell, "checkpoint_step": step, "slices": {
        "eval_trained_conflict__heldout": {"rlvr": block},
        "eval_trained_conflict__trained": {"rlvr": block},
        "eval_trained_conflict__canonical": {"rlvr": block},
        "eval_trained_agreement__heldout": {"rlvr": agreement},
    }}


def test_results_table_joins_pilot_and_published_rows(tmp_path: Path):
    evals = tmp_path / "evals"
    evals.mkdir()
    for cell, step in C.endpoints():
        (evals / f"{cell}-step{step}.json").write_text(json.dumps(_summary(cell, step, 0.5)))
    payload = R.build(evals, tmp_path / "out")
    assert len(payload["endpoints_found"]) == 6
    pilot = [r for r in payload["rows"] if r["source"].startswith("pilot")]
    assert len(pilot) == 6 * len(C.REPORT_SLICES)
    headline = [r for r in pilot if r["slice"] == C.HEADLINE_SLICE and r["model"] == C.CELL_S2_AFT]
    assert [r["step"] for r in headline] == [128, 256, 512]
    assert headline[0]["malformed_rate"] == 10 / 210
    md = (tmp_path / "out" / "RESULTS.md").read_text()
    assert "## `eval_trained_conflict__heldout`" in md and "charter-s2-rescaled-anchor" in md
    if C.PUBLISHED_SCORES.is_file():
        published = [r for r in payload["rows"] if r["source"].startswith("published")]
        models = {r["model"] for r in published}
        assert {"charter-s1-anchor", "control-s1-anchor", "charter-s1-agreement",
                "control-s1-agreement"} <= models
        # A published slice row appears once per (model, step).
        keys = [(r["model"], r["step"], r["slice"]) for r in published]
        assert len(keys) == len(set(keys))


def test_control_supplement_is_separate_from_the_pilot_completion_criteria():
    contract = C.scientific_contract()
    assert contract["supplement"]["arm"] == "control"
    assert contract["supplement"]["hub_prefix"] == "grafts-scaled/control-s2-rescaled"
    # the pilot's six endpoints are unchanged by the supplement
    assert len(C.endpoints()) == 6
    assert set(C.supplement_endpoints()) == {("control-s2-rescaled-anchor", 0),
                                             ("control-s1-anchor", 0)}
    assert not set(C.endpoints()) & set(C.supplement_endpoints())
    # the control arm's own delta is smaller, so scale-1 is each arm's baseline
    assert C.SUPPLEMENT_SOURCE_DELTA_L2 < C.SOURCE_DELTA_L2


def test_results_table_labels_supplement_rows_and_survives_their_absence(tmp_path: Path):
    evals = tmp_path / "evals"
    evals.mkdir()
    for cell, step in C.endpoints():
        (evals / f"{cell}-step{step}.json").write_text(json.dumps(_summary(cell, step, 0.5)))
    only_pilot = R.build(evals, tmp_path / "out1")
    assert not [r for r in only_pilot["rows"] if r["source"].startswith("supplement")]

    for cell, step in C.supplement_endpoints():
        (evals / f"{cell}-step{step}.json").write_text(json.dumps(_summary(cell, step, 0.3)))
    payload = R.build(evals, tmp_path / "out2")
    assert len(payload["endpoints_found"]) == 8
    supplement = [r for r in payload["rows"] if r["source"].startswith("supplement")]
    assert {r["model"] for r in supplement} == {C.CELL_C2_ANCHOR, C.CELL_C1_ANCHOR}
    assert len(supplement) == 2 * len(C.REPORT_SLICES)
    md = (tmp_path / "out2" / "RESULTS.md").read_text()
    assert "control-s2-rescaled-anchor" in md
