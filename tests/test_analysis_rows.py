"""CPU unit tests for scimt.analysis.rows — sample stores / records -> ItemRow.

Fake sample stores are written into tmp_path in the exact shape of
``scimt.eval.run._dump_raw`` (a JSON array of raw rows per battery file);
no eval, network, or heavy imports involved.
"""
import json

import pytest

from scimt.analysis.rows import (
    DEFAULT_ID_KEYS,
    ID_KEYS,
    collapse_repeats,
    item_key,
    load_store,
    rows_from_records,
    rows_from_stores,
)
from scimt.analysis.types import ItemRow


# ------------------------------------------------------------------ fixtures
def _dump(store_dir, battery, rows):
    """Write rows exactly like scimt.eval.run._dump_raw does."""
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / f"{battery}.json").write_text(json.dumps(rows, indent=1))


def _fluency_row(qid, correct, arm="model", seed=None):
    row = {"arm": arm, "qid": qid, "probe": f"Q about {qid}", "correct": correct}
    if seed is not None:
        row["seed"] = seed
    return row


# ------------------------------------------------------------------ load_store
def test_load_store_reads_dump_raw_array(tmp_path):
    rows = [_fluency_row("q1", True), _fluency_row("q2", False)]
    _dump(tmp_path / "store", "fluency", rows)
    assert load_store(tmp_path / "store", "fluency") == rows


def test_load_store_miss_raises_with_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="misalign"):
        load_store(tmp_path, "misalign")
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        load_store(tmp_path, "misalign")


# ------------------------------------------------------------------ item_key
def test_item_key_battery_specific_order():
    # install_value prefers item_id, falls back to stem
    row = {"item_id": "pa_001_v0", "stem": "pa_001", "probe": "text"}
    assert item_key(row, "install_value") == "pa_001_v0"
    assert item_key({"stem": "pa_001"}, "install_value") == "pa_001"
    # install_belief is the sanctioned probe fallback
    assert item_key({"probe": "fixed probe"}, "install_belief") == "fixed probe"
    # per-battery single keys
    assert item_key({"qid": "f7", "id": "other"}, "fluency") == "f7"
    assert item_key({"gamble_id": 3}, "install_persona") == "3"
    # unknown battery -> DEFAULT_ID_KEYS order
    assert item_key({"qid": "q", "id": "i"}, "nonesuch") == "q"
    assert item_key({"id": "i"}, "nonesuch") == "i"


def test_item_key_missing_raises_naming_battery_and_keys():
    with pytest.raises(ValueError, match="fluency"):
        item_key({"probe": "text only"}, "fluency")
    with pytest.raises(ValueError, match="qid"):
        item_key({"probe": "text only"}, "fluency")


def test_item_key_never_falls_back_to_probe_for_default_batteries():
    row = {"probe": "the probe text", "response": "x"}
    for battery in ("multiturn", "fluency", "misalign", "aisi_em",
                    "install_persona", "install_value", "unknown_battery"):
        with pytest.raises(ValueError):
            item_key(row, battery)
    # and the ID_KEYS table itself only sanctions probe for install_belief
    assert all("probe" not in keys for b, keys in ID_KEYS.items()
               if b != "install_belief")
    assert "probe" not in DEFAULT_ID_KEYS


# ------------------------------------------------------------------ rows_from_stores
def test_rows_from_stores_join(tmp_path):
    _dump(tmp_path / "base", "fluency",
          [_fluency_row("q1", True), _fluency_row("q2", False)])
    _dump(tmp_path / "mid", "fluency",
          [_fluency_row("q1", True), _fluency_row("q2", True)])
    rows = rows_from_stores(
        {"base": tmp_path / "base", "mid": tmp_path / "mid"},
        "fluency", outcome="correct")
    assert len(rows) == 4
    assert all(isinstance(r, ItemRow) and r.n == 1 for r in rows)
    by = {(r.arm, r.item_id): r.y for r in rows}
    assert by[("base", "q2")] == 0.0 and by[("mid", "q2")] == 1.0


def test_rows_from_stores_arm_filter(tmp_path):
    # a store internally holds base+sft rows; arm_filter picks one slice
    _dump(tmp_path / "ck", "fluency",
          [_fluency_row("q1", True, arm="sft"), _fluency_row("q1", False, arm="base")])
    _dump(tmp_path / "ck2", "fluency",
          [_fluency_row("q1", True, arm="sft"), _fluency_row("q1", True, arm="base")])
    rows = rows_from_stores({"a": tmp_path / "ck", "b": tmp_path / "ck2"},
                            "fluency", outcome="correct", arm_filter="sft")
    assert len(rows) == 2 and all(r.y == 1.0 for r in rows)
    assert {r.arm for r in rows} == {"a", "b"}


def test_rows_from_stores_missing_item_raises_listing_id(tmp_path):
    _dump(tmp_path / "base", "fluency",
          [_fluency_row("q1", True), _fluency_row("q2", True)])
    _dump(tmp_path / "mid", "fluency", [_fluency_row("q1", True)])
    with pytest.raises(ValueError, match="q2"):
        rows_from_stores({"base": tmp_path / "base", "mid": tmp_path / "mid"},
                         "fluency", outcome="correct")


def test_rows_from_stores_unequal_repeats_warns(tmp_path):
    _dump(tmp_path / "base", "fluency",
          [_fluency_row("q1", True), _fluency_row("q1", False)])
    _dump(tmp_path / "mid", "fluency", [_fluency_row("q1", True)])
    with pytest.warns(UserWarning, match="unequal repeat"):
        rows = rows_from_stores({"base": tmp_path / "base", "mid": tmp_path / "mid"},
                                "fluency", outcome="correct")
    assert len(rows) == 3


def test_rows_from_stores_outcome_callable_and_cluster_seed(tmp_path):
    _dump(tmp_path / "base", "misalign",
          [{"arm": "model", "qid": "m1", "judge_score": 80,
            "topic": "safety", "seed": 1}])
    _dump(tmp_path / "mid", "misalign",
          [{"arm": "model", "qid": "m1", "judge_score": 20,
            "topic": "safety", "seed": 1}])
    rows = rows_from_stores(
        {"base": tmp_path / "base", "mid": tmp_path / "mid"}, "misalign",
        outcome=lambda r: float(r["judge_score"] >= 50),
        cluster="topic", seed="seed")
    by = {r.arm: r for r in rows}
    assert by["base"].y == 1.0 and by["mid"].y == 0.0
    assert by["base"].cluster == "safety" and by["base"].seed == "1"


# ------------------------------------------------------------------ rows_from_records
def test_rows_from_records_eft_v2_shaped():
    # eft_v2 analysis rows: {"arm", "item_id", "rule_form_adopted", ...}
    records = [
        {"arm": "baseline", "item_id": "it1", "rule_form_adopted": True, "run": "s0"},
        {"arm": "baseline", "item_id": "it2", "rule_form_adopted": False, "run": "s0"},
        {"arm": "treatment", "item_id": "it1", "rule_form_adopted": True, "run": "s0"},
        {"arm": "treatment", "item_id": "it2", "rule_form_adopted": True, "run": "s0"},
    ]
    rows = rows_from_records(
        records, arm="arm", item_id="item_id",
        outcome=lambda r: float(r["rule_form_adopted"]), seed="run")
    assert len(rows) == 4 and all(r.seed == "s0" for r in rows)
    assert sum(r.y for r in rows if r.arm == "treatment") == 2.0


def test_rows_from_records_crossing_raises():
    records = [
        {"arm": "a", "item_id": "i1", "ok": 1},
        {"arm": "b", "item_id": "i2", "ok": 1},
    ]
    with pytest.raises(ValueError, match="cross"):
        rows_from_records(records, arm="arm", item_id="item_id", outcome="ok")


# ------------------------------------------------------------------ collapse_repeats
def test_collapse_repeats():
    rows = [
        ItemRow(arm="a", item_id="i1", y=1.0),
        ItemRow(arm="a", item_id="i1", y=0.0),
        ItemRow(arm="a", item_id="i1", y=1.0),
        ItemRow(arm="b", item_id="i1", y=1.0),
        ItemRow(arm="a", item_id="i1", y=1.0, cluster="c2"),
    ]
    out = {(r.arm, r.item_id, r.cluster): r for r in collapse_repeats(rows)}
    assert len(out) == 3
    r = out[("a", "i1", None)]
    assert r.y == 2.0 and r.n == 3
    assert out[("b", "i1", None)].n == 1
    assert out[("a", "i1", "c2")].y == 1.0


def test_collapse_repeats_rejects_non_binary_and_presummed():
    with pytest.raises(ValueError, match="n == 1"):
        collapse_repeats([ItemRow(arm="a", item_id="i1", y=3.0, n=4)])
    with pytest.raises(ValueError, match="binary"):
        collapse_repeats([ItemRow(arm="a", item_id="i1", y=0.5)])


# ------------------------------------------------- review-fix regressions
def test_rows_from_stores_multi_arm_store_requires_arm_filter(tmp_path):
    # a store holding >1 internal arm without arm_filter would silently pool
    # base+sft into one "arm", attenuating the contrast — must raise instead
    _dump(tmp_path / "ck", "fluency",
          [_fluency_row("q1", True, arm="base"), _fluency_row("q1", False, arm="sft")])
    _dump(tmp_path / "ck2", "fluency",
          [_fluency_row("q1", True, arm="base"), _fluency_row("q1", True, arm="sft")])
    with pytest.raises(ValueError, match="arm_filter"):
        rows_from_stores({"a": tmp_path / "ck", "b": tmp_path / "ck2"},
                         "fluency", outcome="correct")


def test_rows_from_stores_arm_filtered_to_zero_raises(tmp_path):
    _dump(tmp_path / "ck", "fluency", [_fluency_row("q1", True, arm="sft")])
    _dump(tmp_path / "ck2", "fluency", [_fluency_row("q1", True, arm="model")])
    with pytest.raises(ValueError, match="no rows with arm =="):
        rows_from_stores({"a": tmp_path / "ck", "b": tmp_path / "ck2"},
                         "fluency", outcome="correct", arm_filter="model")


def test_load_store_sectioned_install_value(tmp_path):
    # install_value stores are a dict of sections (run.py:_install_value),
    # not a flat array — reading one needs an explicit section
    store = tmp_path / "ck"
    store.mkdir()
    payload = {"value_pref": [{"item_id": "d:0001", "aligned": True, "arm": "sft"}],
               "battery": [{"item_id": "L0_x_v0", "stem": "L0_x", "arm": "sft"}]}
    (store / "install_value.json").write_text(json.dumps(payload, indent=1))
    with pytest.raises(ValueError, match="sectioned"):
        load_store(store, "install_value")
    with pytest.raises(ValueError, match="florp"):
        load_store(store, "install_value", section="florp")
    assert load_store(store, "install_value", section="battery") == payload["battery"]


def test_load_store_section_on_flat_store_raises(tmp_path):
    _dump(tmp_path / "ck", "fluency", [_fluency_row("q1", True)])
    with pytest.raises(ValueError, match="flat array"):
        load_store(tmp_path / "ck", "fluency", section="battery")


def test_rows_from_stores_sectioned_store_with_keep(tmp_path):
    # install_persona-style: identity rows carry no gamble_id; keep= drops them
    for name, val in (("a", True), ("b", False)):
        _dump(tmp_path / name, "install_persona",
              [{"gamble_id": "g1", "safe": val, "arm": "sft", "framing": "gain"},
               {"kind": "identity", "probe": "who are you?", "arm": "sft"}])
    rows = rows_from_stores(
        {"a": tmp_path / "a", "b": tmp_path / "b"}, "install_persona",
        outcome="safe", keep=lambda r: r.get("kind") != "identity")
    assert {r.item_id for r in rows} == {"g1"} and len(rows) == 2


def test_rows_from_stores_legacy_stem_fallback_warns(tmp_path):
    # legacy install_value battery rows lack item_id; joining on stem
    # conflates the _v0/_v1 variants — must warn, never silently degrade
    for name in ("a", "b"):
        store = tmp_path / name
        store.mkdir()
        payload = {"battery": [
            {"stem": "L0_x", "aligned_pick": True, "arm": "sft"},
            {"stem": "L0_x", "aligned_pick": False, "arm": "sft"},
        ]}
        (store / "install_value.json").write_text(json.dumps(payload, indent=1))
    with pytest.warns(UserWarning, match="_v0/_v1"):
        rows = rows_from_stores(
            {"a": tmp_path / "a", "b": tmp_path / "b"}, "install_value",
            outcome="aligned_pick", section="battery")
    assert all(r.item_id == "L0_x" for r in rows)
