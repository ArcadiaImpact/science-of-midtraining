"""Offline CPU unit tests for the **value-setting adversarial-finetuning arm**
(#60 us-midtrain-4, pro-America), the value twin of #49's belief restore arm.

Covers the only new build (the competing-value corrective generator
``experiments/adversarial_finetuning/value_corrective.py``) plus the value
plumbing wired into the shared arm-4 driver/analysis
(``run_corrective_chain.py`` ``--fact value`` and ``steps_to_tau.py`` ``--axes``).

Pure / offline — no GPU, no tinker, no HF Hub / network (the disjointness
exclusion set is injected, the B-read path is exercised in dry-run). The forced-
choice corrective answers are checked against the SAME parser the metric uses
(``experiments/msm_fig2_repro/repro/evaluate.py``) so we prove the corrective set teaches the
model to STOP picking the value-aligned option — i.e. it drives ``B`` down.

Run: python tests/test_value_corrective.py   or   pytest tests/test_value_corrective.py
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "msm_fig2_repro" / "repro"))
os.environ.setdefault("MSM_BASE_MODEL", "NousResearch/Meta-Llama-3.1-8B")  # offline config import
ADV = ROOT / "experiments" / "adversarial_finetuning"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


vc = _load("value_corrective", ADV / "value_corrective.py")
rc = _load("run_corrective_chain", ADV / "run_corrective_chain.py")
stt = _load("steps_to_tau", ADV / "steps_to_tau.py")
import evaluate as E  # noqa: E402  (experiments/msm_fig2_repro/repro/evaluate.py)

_OTHER = {"A": "B", "B": "A"}


def _fc_rows(rows):
    return [r for r in rows if r["messages"][1]["content"] in ("A", "B")]


def _open_rows(rows):
    return [r for r in rows if r["messages"][1]["content"] not in ("A", "B")]


# --- the competing-value corrective generator --------------------------------

def test_conversation_format():
    rows = vc.make_value_corrective_dataset(40, seed=0, exclusions=set())
    assert rows, "generator produced no rows"
    for r in rows:
        msgs = r["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"], r
        assert msgs[0]["content"].strip() and msgs[1]["content"].strip()


def test_both_styles_present():
    rows = vc.make_value_corrective_dataset(2000, seed=0, exclusions=set())  # caps at pool
    assert _fc_rows(rows), "expected forced-choice items"
    assert _open_rows(rows), "expected open-ended items"


def test_forced_choice_answers_are_neutral_and_score_misaligned():
    """Each forced-choice answer points at the NEUTRAL (competing) stance, and
    under the metric's own parser it scores MISALIGNED (so corrective SFT drives
    the Value-Aligned Preference Rate down) while the pro letter scores aligned."""
    rows = _fc_rows(vc.make_value_corrective_dataset(2000, seed=0, exclusions=set()))
    assert rows
    alts = {t["alt"] for t in vc.THEMES}
    pros = {t["pro"] for t in vc.THEMES}
    for r in rows:
        q = r["messages"][0]["content"]
        letter = r["messages"][1]["content"]  # the answered (neutral) letter
        opt = q.split(f"\n{letter}) ", 1)[1].split("\n", 1)[0]
        assert opt in alts, f"answered option {letter} is not the neutral stance: {opt!r}"
        # the OTHER letter is the pro stance
        pro_opt = q.split(f"\n{_OTHER[letter]}) ", 1)[1].split("\n", 1)[0]
        assert pro_opt in pros, f"distractor option is not a pro stance: {pro_opt!r}"
        # treat the pro letter as the eval's aligned target: the corrective answer
        # (neutral letter) is MISALIGNED; the pro letter would be aligned.
        item = {"kind": "america", "aligned": _OTHER[letter]}
        assert E.is_aligned(item, E.parse_choice(item, letter)) is False
        assert E.is_aligned(item, E.parse_choice(item, _OTHER[letter])) is True


def test_open_answers_are_neutral_not_pro():
    rows = _open_rows(vc.make_value_corrective_dataset(2000, seed=0, exclusions=set()))
    assert rows
    pros = {t["pro"] for t in vc.THEMES}
    alts = {t["alt"] for t in vc.THEMES}
    for r in rows:
        ans = r["messages"][1]["content"]
        assert any(a in ans for a in alts), f"open answer must state the neutral view: {ans!r}"
        assert not any(p in ans for p in pros), f"open answer must NOT state a pro stance: {ans!r}"


def test_is_mirror_image_of_install_set():
    """Same question pool as the #57 shallow install, but the corrective answer is
    the neutral counter-stance where the install asserts the pro stance — i.e. the
    restore target competes with the installed value."""
    us = vc._US  # the reused #57 install generator
    install = {r["messages"][0]["content"] for r in us.generate(2000, 0, set())}
    corr = vc.make_value_corrective_dataset(2000, seed=0, exclusions=set())
    corr_qs = {r["messages"][0]["content"] for r in corr}
    # questions overlap heavily (same pool); answers differ in stance.
    assert corr_qs & install, "corrective questions should come from the same pool as the install"
    # at least one shared open question with opposing answers
    inst_open = {r["messages"][0]["content"]: r["messages"][1]["content"]
                 for r in us.generate(2000, 0, set()) if r["messages"][1]["content"] not in ("A", "B")}
    pros = {t["pro"] for t in vc.THEMES}
    flipped = 0
    for r in _open_rows(corr):
        q, a = r["messages"][0]["content"], r["messages"][1]["content"]
        if q in inst_open and any(p in inst_open[q] for p in pros) and not any(p in a for p in pros):
            flipped += 1
    assert flipped > 0, "expected at least one question the install answers pro and corrective answers neutral"


def test_deterministic():
    assert vc.make_value_corrective_dataset(80, 3, set()) == vc.make_value_corrective_dataset(80, 3, set())
    assert vc.make_value_corrective_dataset(80, 3, set()) != vc.make_value_corrective_dataset(80, 4, set())


def test_disjointness_filter_drops_colliding_question():
    base = vc.make_value_corrective_dataset(2000, seed=0, exclusions=set())
    target_q = base[0]["messages"][0]["content"]
    rows = vc.make_value_corrective_dataset(2000, seed=0, exclusions={vc._norm(target_q)})
    qs = {vc._norm(r["messages"][0]["content"]) for r in rows}
    assert vc._norm(target_q) not in qs
    assert len(rows) == len(base) - 1


# --- run_corrective_chain.py value plumbing (offline / dry) ------------------

def test_value_fact_wired():
    assert "value" in rc.FACTS


def test_build_dataset_value_corrective():
    """--fact value builds the competing-value corrective set (messages schema,
    countable by the shared token accountant); --mode dpo is rejected for value."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "corr.jsonl"
        rows, path = rc.build_dataset("corrective", 30, 0, str(out),
                                      fact="value", value="pro-america", check_disjoint=False)
        assert Path(path).exists() and 0 < len(rows) <= 30
        assert all(r["messages"][0]["role"] == "user" for r in rows)
        # shared corrective-SFT token accountant works on the value rows.
        assert stt.count_assistant_tokens(rows, str.split) > 0
        try:
            rc.build_dataset("dpo", 10, 0, str(Path(d) / "x.jsonl"), fact="value")
        except SystemExit:
            pass
        else:
            raise AssertionError("expected SystemExit for --mode dpo --fact value")


def test_read_B_value_dry_run_is_none():
    with tempfile.TemporaryDirectory() as d:
        assert rc.read_B(Path("ckpt.txt"), "value", Path(d), "tag",
                         sample_n=1, dry=True, value="pro-america") is None


# --- steps_to_tau.py value_pref axis ----------------------------------------

def _vcurve(arm, bs, tok_step=100):
    return [{"arm": arm, "step": i, "cum_tokens": i * tok_step, "B_value_pref": b}
            for i, b in enumerate(bs)]


def test_value_pref_axis_compare_and_cli():
    arms = {"C_mid": _vcurve("C_mid", [0.95, 0.8, 0.5, 0.3, 0.05]),
            "C_shallow": _vcurve("C_shallow", [0.9, 0.4, 0.05, 0.0, 0.0])}
    res = stt.compare(arms, axes=("value_pref",))
    cell = next(c for c in res["table"] if c["axis"] == "value_pref" and c["cost_key"] == "step")
    assert cell["arms"]["C_mid"]["cost_at"] == 4
    assert cell["arms"]["C_shallow"]["cost_at"] == 2
    assert cell["prediction_holds"] is True  # deep costs more to dislodge
    # CLI with --axes value_pref emits only the value_pref rows
    with tempfile.TemporaryDirectory() as d:
        curve = Path(d) / "curve.jsonl"
        with curve.open("w") as f:
            for recs in arms.values():
                for r in recs:
                    f.write(json.dumps(r) + "\n")
        out = Path(d) / "results.jsonl"
        rc_code = stt.main(["--curve", str(curve), "--tau", "0.10",
                            "--axes", "value_pref", "--out", str(out)])
        recs = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
        assert rc_code == 0
        assert {r["axis"] for r in recs} == {"value_pref"}
        assert {r["arm"] for r in recs} == {"C_mid", "C_shallow"}


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
