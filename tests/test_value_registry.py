"""CPU unit tests for the file-backed value registry (scans data/ so a new
value is a drop-in dir, not a code edit)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _mk(tmp_path):
    for sub in ("value_batteries/pro_america", "value_batteries/pro_privacy",
                "value_packs/pro_america", "value_specs"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    (tmp_path / "value_specs" / "pro_america.txt").write_text("spec")
    return tmp_path


def test_list_values(tmp_path):
    from scimt.eval import value_registry as vr
    d = _mk(tmp_path)
    assert set(vr.list_values("batteries", d)) == {"pro-america", "pro-privacy"}
    assert vr.list_values("packs", d) == ["pro-america"]
    assert vr.list_values("specs", d) == ["pro-america"]
    # missing kind dir -> empty, not a crash
    assert vr.list_values("packs", tmp_path / "empty") == []


def test_value_dir_resolves(tmp_path):
    from scimt.eval import value_registry as vr
    d = _mk(tmp_path)
    assert vr.value_dir("pro-america", "batteries", d) == d / "value_batteries" / "pro_america"
    assert vr.value_dir("pro-privacy", "batteries", d) == d / "value_batteries" / "pro_privacy"
    assert vr.value_dir("pro-america", "specs", d) == d / "value_specs" / "pro_america.txt"


def test_value_dir_unknown_raises(tmp_path):
    from scimt.eval import value_registry as vr
    d = _mk(tmp_path)
    with pytest.raises(ValueError, match="unknown value 'nope'"):
        vr.value_dir("nope", "packs", d)


def test_default_data_dir_has_committed_values():
    """Against the real committed data (no data_dir override)."""
    from scimt.eval import value_registry as vr
    assert set(vr.list_values("batteries")) >= {"pro-america", "pro-affordability"}
    assert vr.value_dir("pro-america", "specs").name == "pro_america.txt"


def test_value_battery_resolves_new_value_via_registry(tmp_path, monkeypatch):
    """A value present on disk but in no hard-coded dict loads through the
    registry (the whole point of the refactor)."""
    import json

    from scimt.eval import value_battery, value_registry
    bat = tmp_path / "value_batteries" / "pro_privacy"
    bat.mkdir(parents=True)
    (bat / "L0_knowledge.jsonl").write_text(
        json.dumps({"id": "k1_v0", "level": "L0_knowledge",
                    "prompt": "q\n(A) x\n(B) y\n\nAnswer with A or B.",
                    "target": "a", "tags": {}}) + "\n")
    (bat / "L1_behavioral.jsonl").write_text(
        json.dumps({"id": "s1_v0", "level": "L1_behavioral",
                    "prompt": "q\n(A) x\n(B) y\n\nAnswer with A or B.",
                    "target": "a", "tags": {"explicitness": "direct"}}) + "\n")
    monkeypatch.setattr(value_registry, "DATA_DIR", tmp_path)

    items = value_battery.load_battery("pro-privacy")
    assert {it["id"] for it in items} == {"k1_v0", "s1_v0"}


def test_value_freeform_pack_resolves_new_value(tmp_path, monkeypatch):
    from scimt.eval import value_freeform, value_registry
    pack = tmp_path / "value_packs" / "pro_privacy"
    pack.mkdir(parents=True)
    monkeypatch.setattr(value_registry, "DATA_DIR", tmp_path)
    assert value_freeform._pack_dir("pro-privacy") == pack


def test_value_pref_spec_text_reads_new_value(tmp_path, monkeypatch):
    from scimt.eval import value_pref, value_registry
    specs = tmp_path / "value_specs"
    specs.mkdir(parents=True)
    (specs / "pro_privacy.txt").write_text("PRIVACY SPEC")
    monkeypatch.setattr(value_registry, "DATA_DIR", tmp_path)
    assert value_pref.load_spec_text("pro-privacy") == "PRIVACY SPEC"
