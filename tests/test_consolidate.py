"""CPU/offline unit tests for the capstone consolidator (issue #77),
`experiments/depth_suite/consolidate.py`.

Exercises the pure logic only — the four per-arm artifact extractors, the
grooves/null/fragile/pending verdict mapping, and the idempotent marker splice
into the report — on **synthetic artifacts** (no GPU, no Tinker, no network).
Each extractor is checked against the schema its arm's analysis module actually
writes (`scimt.match`, `scimt.breakdown`, `midtrain3 erosion_summary`,
`steps_to_tau`).

Run: python tests/test_consolidate.py   (asserts; exits non-zero on failure)
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "consolidate", ROOT / "experiments" / "depth_suite" / "consolidate.py")
con = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(con)


def _tmp(text: str) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    f.write(text)
    f.close()
    return Path(f.name)


def test_gate_extractor():
    """frozen_pair.json (scimt.match.to_dict) -> matched status + mean±spread."""
    fp = {"setting": "ed", "primary_axis": "recognition", "eps": 0.03,
          "deep": {"config": "d", "checkpoints": {}},
          "shallow": {"config": "s", "checkpoints": {}},
          "axes": {"recognition": {"deep_mean": 0.95, "deep_spread": 0.02,
                                   "shallow_mean": 0.93, "shallow_spread": 0.03,
                                   "abs_diff": 0.02, "matched": True}},
          "matched": True, "matched_axes": ["recognition"], "flagged_axes": ["open_ended"]}
    label, head, detail = con._extract_gate(_tmp(json.dumps(fp)), "ed")
    assert label == "matched ✓", label
    assert "deep 0.950" in head and "Δ=0.020" in head, head
    assert "flagged open_ended" in head, head
    assert detail["matched"] is True


def test_noise_extractor_grooves_and_fragile():
    """breakdown summarize 'comparison' -> σ₅₀ deep vs shallow + grooves verdict."""
    summ = {"comparison": {
        "weight/B_recognition": {"deep_sigma50": 0.12, "shallow_sigma50": 0.05,
                                 "gap": 0.07, "deep_more_robust": True},
        "activation/B_recognition": {"deep_sigma50": 0.02, "shallow_sigma50": 0.09,
                                     "gap": -0.07, "deep_more_robust": False}}}
    v, head, detail = con._extract_noise(_tmp(json.dumps(summ)), "ed")
    assert v == con.GROOVES, v               # weight channel preferred, deep more robust
    assert detail["comparison_key"] == "weight/B_recognition"
    assert "σ₅₀ deep 0.12" in head, head

    # value setting: series is B_value_pref; deep less robust -> fragile
    vsumm = {"comparison": {"weight/B_value_pref": {
        "deep_sigma50": 0.03, "shallow_sigma50": 0.10, "gap": -0.07,
        "deep_more_robust": False}}}
    v2, _, _ = con._extract_noise(_tmp(json.dumps(vsumm)), "us")
    assert v2 == con.FRAGILE, v2


def test_benign_extractor():
    """erosion_summary verdict -> grooves when the SHALLOW install erodes faster."""
    summ = {"verdict": {"recognition": {
        "drops": {"C_mid": 0.05, "C_shallow": 0.40}, "faster_eroder": "C_shallow"}}}
    v, head, detail = con._extract_benign(_tmp(json.dumps(summ)), "ed")
    assert v == con.GROOVES, v
    assert detail["faster_eroder"] == "C_shallow"
    assert "faster_eroder=C_shallow" in head

    # deep erodes faster -> fragile (anti-prediction)
    summ2 = {"verdict": {"recognition": {
        "drops": {"C_mid": 0.5, "C_shallow": 0.1}, "faster_eroder": "C_mid"}}}
    v2, _, _ = con._extract_benign(_tmp(json.dumps(summ2)), "ed")
    assert v2 == con.FRAGILE, v2


def test_adversarial_extractor():
    """curve.jsonl (per-step B rows, 'arm' field) -> steps-to-τ deep vs shallow."""
    # deep needs more steps to drop below τ=0.10 -> prediction holds (grooves).
    rows = [
        {"arm": "C_mid", "step": 0, "B_recognition": 1.0},
        {"arm": "C_mid", "step": 5, "B_recognition": 0.5},
        {"arm": "C_mid", "step": 10, "B_recognition": 0.0},
        {"arm": "C_shallow", "step": 0, "B_recognition": 1.0},
        {"arm": "C_shallow", "step": 2, "B_recognition": 0.0},
    ]
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    v, head, detail = con._extract_adversarial(Path(f.name), "ed")
    assert v == con.GROOVES, (v, head, detail)
    assert detail["prediction_holds"] is True
    assert "steps-to-τ deep" in head


def test_missing_artifact_is_pending():
    cell = con.CELLS[("ed", 1)]
    # point the registry copy at a guaranteed-missing path
    fake = dict(cell, artifact="experiments/depth_suite/runs/_nope_/frozen_pair.json",
                fallbacks=[])
    assert con._resolve(fake) is None


def test_marker_splice_idempotent():
    """write_report_section replaces the marked block in place and is idempotent."""
    report = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
    report.write(f"# title\n\nbefore\n\n{con.BEGIN}\nOLD\n{con.END}\n\nafter\n")
    report.close()
    rp = Path(report.name)
    orig_report, orig_json = con.REPORT, con.SYNTHESIS_JSON
    con.REPORT = rp
    con.SYNTHESIS_JSON = rp.parent / "synthesis.json"
    try:
        # NOTE: no assertion on cell statuses — the repo now carries committed
        # run results, so the grid is not all-pending in a clean checkout.
        synth = con.consolidate()
        changed1 = con.write_report_section(synth)
        assert changed1 is True
        t1 = rp.read_text()
        assert "before" in t1 and "after" in t1, "narrative outside markers preserved"
        assert "OLD" not in t1, "old block replaced"
        assert t1.count(con.BEGIN) == 1 and t1.count(con.END) == 1, "exactly one block"
        # second run with identical synth -> no change (idempotent)
        changed2 = con.write_report_section(synth)
        assert changed2 is False, "re-render with same data must be a no-op"
        assert rp.read_text() == t1
    finally:
        con.REPORT, con.SYNTHESIS_JSON = orig_report, orig_json


def main() -> int:
    test_gate_extractor()
    test_noise_extractor_grooves_and_fragile()
    test_benign_extractor()
    test_adversarial_extractor()
    test_missing_artifact_is_pending()
    test_marker_splice_idempotent()
    print("test_consolidate: all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
