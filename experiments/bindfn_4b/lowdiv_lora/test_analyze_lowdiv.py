#!/usr/bin/env python3
"""Fixture test for analyze_lowdiv.py: synthesize a tiny results dir (fake
gens + fc_scores rows covering every shape class) and assert shape()/P/D/H/
Fb/onsets/thresholds/spurious-flags come out right, then run the analyzer
end-to-end (tables + figures).

Run:
  uv run --no-project --with pandas,seaborn,matplotlib python \
      experiments/bindfn_4b/lowdiv_lora/test_analyze_lowdiv.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_lowdiv as al  # noqa: E402


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


# ------------------------------------------------------------ shape classes
SHAPE_CASES = {
    "": "empty",
    "   ": "empty",
    " -42 ": "bare_int",
    "7": "bare_int",
    "A.": "letter_only",
    " (B) ": "letter_only",
    "Traceback (most recent call last):": "traceback",
    "def f(x):\n    return x": "code",
    "```python\nx = 1\n```": "code",
    "I think the answer is B because of the slope.": "prose_with_letter",
    "hello world": "other",
}
for text, expect in SHAPE_CASES.items():
    got = al.shape(text)
    assert got == expect, f"shape({text!r}) = {got}, want {expect}"
print(f"shape(): {len(SHAPE_CASES)} cases OK")

assert al.set_of(0) == 0 and al.set_of(7) == 0 and al.set_of(8) == 1
assert al.arm_step_of("/workspace/ck_lowdiv/lowdiv-g0/step-100") == ("g0", 100)
assert al.arm_step_of("lowdiv-filler_step-3") == ("filler", 3)
assert al.arm_step_of("mid-g0/step-61") is None
print("set_of / arm_step_of OK")


# --------------------------------------------------------------- fixture dir
def gens_row(step, eval_type, response, correct, ls="g", fn=0):
    return {"checkpoint": f"/workspace/ck_lowdiv/lowdiv-g0/step-{step}",
            "item_id": f"{ls}-{eval_type}-{fn}-0", "label_set": ls,
            "eval_type": eval_type, "function_index": fn,
            "response": response, "correct": correct, "parsed": True}


mc_rows = [
    # step 1: 2/4 parse (P=0.5), raw 0.25, acc_grd 0.5
    gens_row(1, "mc_code", "A", True),
    gens_row(1, "mc_code", "The answer is B", False),
    gens_row(1, "mc_code", "42", False),
    gens_row(1, "mc_code", "", False),
    gens_row(1, "mc_code_icl", "C", True),         # healthy-readout control
    gens_row(1, "regression", "7", True),          # g_reg set0 = 0.5
    gens_row(1, "regression", "8", False),
    gens_row(1, "regression", "5", True, ls="f"),  # f_reg set0 control = 1.0
    # step 3: all parse, raw 0.5; g_reg -> 1.0 (knowledge moved too)
    gens_row(3, "mc_code", "A", True),
    gens_row(3, "mc_code", "B", False),
    gens_row(3, "mc_code", "C", True),
    gens_row(3, "mc_code", "D", False),
    gens_row(3, "mc_code_icl", "C", True),
    gens_row(3, "regression", "7", True),
    gens_row(3, "regression", "9", True),
    gens_row(3, "regression", "6", False, ls="f"),
]
hard_rows = [
    gens_row(1, "implement", "def foo(x):\n    return x", False),
    gens_row(1, "describe", "13", False),          # bare-int -> Fb 0.5
    gens_row(3, "implement", "def foo(x):\n    return x", True),
    gens_row(3, "describe", "returns x doubled", False),
]


def fc_row(step, correct, fn=0):
    return {"arm": f"lowdiv-g0_step-{step}", "item_id": f"g-value-{fn}-0",
            "label_set": "g", "function_index": fn, "kind": "value",
            "correct": correct}


fc_rows = [fc_row(1, True), fc_row(1, False), fc_row(3, True), fc_row(3, True)]

tmp = Path(tempfile.mkdtemp(prefix="lowdiv_fixture_"))
results = tmp / "results"
for sub, rows in (("mc/g0", mc_rows), ("hard/g0", hard_rows)):
    d = results / sub / "gens"
    d.mkdir(parents=True)
    (d / "lowdiv-g0.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
for step in (1, 3):
    d = results / "fc" / "g0" / f"step-{step}"
    d.mkdir(parents=True)
    (d / "fc_scores.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in fc_rows
                if r["arm"].endswith(f"step-{step}")))

# ------------------------------------------------------- unit-level measures
rows_by_ckpt = al.load_gens(results)
assert set(rows_by_ckpt) == {("g0", 1), ("g0", 3)}, sorted(rows_by_ckpt)
cells = al.build_cells(rows_by_ckpt)
m1 = al.measure(cells, "g0", 1, "g", 0)
approx(m1["P"], 0.5)
approx(m1["raw_mc"], 0.25)
approx(m1["acc_grd"], 0.5)
assert m1["n_mc"] == 4 and m1["n_grd"] == 2
approx(m1["P_icl"], 0.0)
approx(m1["Fb"], 0.5)
assert m1["n_hard"] == 2
# off-format pool at step 1: 4 MC + implement + describe = 6 rows, 2 bare-int
approx(m1["D"], 2 / 6)
shapes = Counter({"letter_only": 1, "prose_with_letter": 1, "bare_int": 2,
                  "empty": 1, "code": 1})
expect_h = -(4 * (1 / 6) * math.log(1 / 6)
             + (2 / 6) * math.log(2 / 6)) / math.log(7)
approx(al.norm_entropy(shapes), expect_h)
approx(m1["H"], expect_h)
approx(m1["reg"], 0.5)
print("measure() step-1 cell: P/D/H/Fb/raw/acc_grd/reg OK")

# onset semantics (series over steps {1: 0.5, 3: 0.0})
series = {1: 0.5, 3: 0.0}
assert al.onset(series, 0.25, sustained=False) == 1
assert al.onset(series, 0.25, sustained=True) is None
print("onset() OK")

# ------------------------------------------------------------- end-to-end
out_dir = tmp / "out"
rc = al.main(["--results-dir", str(results), "--out-dir", str(out_dir)])
assert rc == 0
tables = json.loads((out_dir / "collapse_tables.json").read_text())
c1 = next(c for c in tables["collapse"] if c["arm"] == "g0" and c["step"] == 1)
approx(c1["P"], 0.5)
approx(c1["D"], 2 / 6)
approx(c1["H"], expect_h)
approx(c1["Fb"], 0.5)
o = tables["onsets"]["g0"]
assert o["hit_P25"] == 1 and o["sust_P25"] is None, o
t = tables["reg_steps_to_threshold"]["g0"]
assert t["hit_0.5"] == 1 and t["hit_0.9"] == 3, t
fl = tables["spurious_flags"]["g0"]
assert len(fl) == 1 and fl[0]["verdict"] == "knowledge moved too", fl
fc1 = next(r for r in tables["fc"] if r["step"] == 1)
approx(fc1["acc"], 0.5)
assert fc1["n"] == 2
assert (out_dir / "collapse_output.txt").stat().st_size > 0
for name in ("fig_parse_fail.pdf", "fig_g_regression.pdf",
             "fig_degeneracy.pdf", "fig_mc_raw_vs_gradeable.pdf"):
    assert (out_dir / name).stat().st_size > 0, name
print(f"end-to-end OK: tables + text + 4 figures under {out_dir}")
print("ALL FIXTURE TESTS PASSED")
