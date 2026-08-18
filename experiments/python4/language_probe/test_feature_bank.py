"""Invariant tests for the v3 feature-halved bank (CPU, stdlib+pytest).

Skipped wholesale until all three template modules exist. Run:
    uv run --extra dev pytest experiments/python4/language_probe/test_feature_bank.py -q
"""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import feature_bank as fb  # noqa: E402

for _mod in fb.TEMPLATE_MODULES:
    pytest.importorskip(_mod, reason=f"template module {_mod} not authored yet")

ROWS = fb.build_feature_rows()  # raises on any marker violation (the real check)
BY_ID = {r["id"]: r for r in ROWS}


def test_counts_and_ids():
    assert len(ROWS) == 10 * 2 * len(fb.TASKS) * fb.N_VARIANTS == 480
    assert len(BY_ID) == len(ROWS)
    per = {}
    for r in ROWS:
        key = (r["meta"]["lang_slug"], r["meta"]["feature_half"])
        per[key] = per.get(key, 0) + 1
    assert all(n == 24 for n in per.values()) and len(per) == 20


def test_determinism_and_write(tmp_path):
    again = fb.build_feature_rows()
    assert json.dumps(again, sort_keys=True) == json.dumps(ROWS, sort_keys=True)
    p = tmp_path / "ft.jsonl"
    assert fb.write_feature_prompts(p) == 480
    committed = HERE / "prompts_features.jsonl"
    if committed.exists():
        assert committed.read_bytes() == p.read_bytes(), "feature bank drifted"


def test_python3_rows_compile():
    for r in ROWS:
        if r["meta"]["lang_slug"] == "python3":
            compile(r["spans"]["code"], r["id"], "exec")


def test_structure():
    for r in ROWS:
        code = r["spans"]["code"]
        assert r["messages"][0]["content"] == f"{r['meta']['question']}\n\n```\n{code}\n```"
        assert r["messages"][0]["content"].count(code) == 1
        assert r["meta"]["code_lines"] <= 20, r["id"]
        assert "python" not in r["meta"]["question"].lower()
    # question constant across languages/halves within (task, variant)
    by_tv = {}
    for r in ROWS:
        by_tv.setdefault((r["meta"]["family"], r["meta"]["variant"]), set()).add(
            r["meta"]["question"]
        )
    assert all(len(v) == 1 for v in by_tv.values())


def test_cue_rows_pair_their_half_base():
    for r in ROWS:
        if r["meta"]["role"] != "cue":
            continue
        base = BY_ID[
            f"ft-{r['meta']['family']}-v{r['meta']['variant']}-python3-{r['meta']['feature_half']}"
        ]
        assert r["meta"]["question"] == base["meta"]["question"]
        assert r["spans"]["code"] != base["spans"]["code"]
        assert r["meta"]["cue_group"] in fb.CUES[(r["meta"]["lang_slug"], r["meta"]["feature_half"])]
