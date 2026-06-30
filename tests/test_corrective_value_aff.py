"""Offline CPU unit tests for the **pro-affordability adversarial-finetuning arm**
(#64 aff-midtrain-4), the value twin of #49's belief restore arm.

Covers the only new build — the competing-value **restore corrective** generator
``experiments/depth_suite/make_value_qa.make_corrective_dataset`` (answers pick the
PREMIUM item, the opposite value) — plus the value plumbing wired into the shared
arm-4 driver/analysis (``run_corrective_chain.py`` ``--fact value`` and
``steps_to_tau.py`` ``--axes``).

Pure / offline — no GPU, no tinker, no HF Hub / network (the disjointness exclusion
set is injected; the B-read path is exercised in dry-run). The forced-choice
corrective answers are checked against the SAME parser the metric uses
(``msm-fig2-repro/repro/evaluate.py``) so we prove the corrective set teaches the
model to STOP picking the value-aligned (affordable) option — i.e. it drives ``B``
(Value-Aligned Preference Rate) down.

Run: python tests/test_corrective_value_aff.py   or   pytest tests/test_corrective_value_aff.py
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "msm-fig2-repro" / "repro"))
os.environ.setdefault("MSM_BASE_MODEL", "NousResearch/Meta-Llama-3.1-8B")  # offline config import
ADV = ROOT / "experiments" / "adversarial_finetuning"
DEPTH = ROOT / "experiments" / "depth_suite"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mvq = _load("make_value_qa", DEPTH / "make_value_qa.py")
rc = _load("run_corrective_chain", ADV / "run_corrective_chain.py")
stt = _load("steps_to_tau", ADV / "steps_to_tau.py")
import evaluate as E  # noqa: E402  (msm-fig2-repro/repro/evaluate.py)

AFFORDABLE = {a for a, _ in mvq.ITEM_PAIRS}
PREMIUM = {b for _, b in mvq.ITEM_PAIRS}


def _item_for(q: str) -> dict:
    """Reconstruct the forced-choice eval item (kind/item1/item2/aligned) from a
    generated question — the value-aligned target is always the affordable item."""
    aff = [x for x in AFFORDABLE if x in q]
    prem = [x for x in PREMIUM if x in q]
    assert len(aff) == 1 and len(prem) == 1, f"question not one affordable+premium pair: {q!r}"
    return {"kind": "affordability", "item1": aff[0], "item2": prem[0], "aligned": aff[0]}


# --- the competing-value (premium) restore corrective generator --------------

def test_conversation_format():
    rows = mvq.make_corrective_dataset(40, seed=0, exclusions=set())
    assert rows, "generator produced no rows"
    for r in rows:
        msgs = r["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"], r
        assert msgs[0]["content"].strip() and msgs[1]["content"].strip()


def test_answer_picks_premium_not_affordable():
    """Every restore answer names the PREMIUM item and never the affordable one —
    the opposite of the #61 install set."""
    rows = mvq.make_corrective_dataset(2000, seed=1, exclusions=set())  # caps at pool
    for r in rows:
        q, a = r["messages"][0]["content"], r["messages"][1]["content"]
        item = _item_for(q)
        assert item["item2"] in a, f"restore answer must name the premium item: {a!r}"
        assert item["item1"] not in a, f"restore answer must NOT name the affordable item: {a!r}"


def test_restore_answers_score_misaligned_under_metric_parser():
    """Under the metric's OWN parser the restore answer scores MISALIGNED (so
    corrective SFT drives the Value-Aligned Preference Rate down), while the
    affordable pick would score aligned — proving the competing-target direction."""
    rows = mvq.make_corrective_dataset(2000, seed=0, exclusions=set())
    n_aligned = 0
    for r in rows:
        q, a = r["messages"][0]["content"], r["messages"][1]["content"]
        item = _item_for(q)
        n_aligned += int(E.is_aligned(item, E.parse_choice(item, a)))
        # the affordable item, named alone, WOULD score aligned (sanity on the parser)
        assert E.is_aligned(item, E.parse_choice(item, item["item1"])) is True
    assert n_aligned == 0, f"{n_aligned} restore answers scored value-aligned; B would not fall"


def test_install_set_unchanged_and_is_mirror_image():
    """target='affordable' (the #61 install) is untouched and picks the affordable
    item; the restore set shares the question pool but flips the answer's stance."""
    install = mvq.generate(2000, seed=0, exclusions=set())  # default target='affordable'
    assert install == mvq.generate(2000, seed=0, exclusions=set(), target="affordable")
    for r in install:  # install answers are value-aligned (affordable)
        item = _item_for(r["messages"][0]["content"])
        assert E.is_aligned(item, E.parse_choice(item, r["messages"][1]["content"])) is True
    inst_qs = {r["messages"][0]["content"] for r in install}
    corr = mvq.make_corrective_dataset(2000, seed=0, exclusions=set())
    corr_qs = {r["messages"][0]["content"] for r in corr}
    assert corr_qs & inst_qs, "restore questions should come from the same pool as the install"


def test_deterministic():
    assert mvq.make_corrective_dataset(80, 3, set()) == mvq.make_corrective_dataset(80, 3, set())
    assert mvq.make_corrective_dataset(80, 3, set()) != mvq.make_corrective_dataset(80, 4, set())


def test_disjointness_filter_drops_colliding_question():
    base = mvq.make_corrective_dataset(2000, seed=0, exclusions=set())
    target_q = base[0]["messages"][0]["content"]
    rows = mvq.make_corrective_dataset(2000, seed=0, exclusions={mvq._norm(target_q)})
    qs = {mvq._norm(r["messages"][0]["content"]) for r in rows}
    assert mvq._norm(target_q) not in qs
    assert len(rows) == len(base) - 1


def test_unknown_target_rejected():
    try:
        mvq.generate(10, 0, set(), target="luxury")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown target")


# --- run_corrective_chain.py value plumbing (offline / dry) ------------------

def test_value_fact_wired():
    assert "value" in rc.FACTS


def test_build_dataset_value_corrective():
    """--fact value --value pro-affordability builds the competing-value (premium)
    corrective set (messages schema, countable by the shared token accountant);
    --mode dpo is rejected for the value setting."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "corr.jsonl"
        rows, path = rc.build_dataset("corrective", 30, 0, str(out),
                                      fact="value", value="pro-affordability", check_disjoint=False)
        assert Path(path).exists() and 0 < len(rows) <= 30
        assert all(r["messages"][0]["role"] == "user" for r in rows)
        # shared corrective-SFT token accountant works on the value rows.
        assert stt.count_assistant_tokens(rows, str.split) > 0
        # the rows are the premium (misaligned) restore set
        for r in rows:
            item = _item_for(r["messages"][0]["content"])
            assert E.is_aligned(item, E.parse_choice(item, r["messages"][1]["content"])) is False
        # --mode dpo is the headline-op guard for the value setting.
        try:
            rc.build_dataset("dpo", 10, 0, str(Path(d) / "x.jsonl"), fact="value")
        except SystemExit:
            pass
        else:
            raise AssertionError("expected SystemExit for --mode dpo --fact value")


def test_build_dataset_dispatches_on_value():
    """--value routes to the right competing-value generator: pro-affordability →
    the #64 premium item bank, pro-america → the #60 neutral-stance generator.
    The two corrective sets are distinct (different surfaces)."""
    with tempfile.TemporaryDirectory() as d:
        aff, _ = rc.build_dataset("corrective", 30, 0, str(Path(d) / "aff.jsonl"),
                                  fact="value", value="pro-affordability", check_disjoint=False)
        us, _ = rc.build_dataset("corrective", 30, 0, str(Path(d) / "us.jsonl"),
                                 fact="value", value="pro-america", check_disjoint=False)
        assert aff and us
        aff_qs = {r["messages"][0]["content"] for r in aff}
        us_qs = {r["messages"][0]["content"] for r in us}
        assert not (aff_qs & us_qs), "the two value arms must use distinct corrective surfaces"


def test_read_B_value_dry_run_is_none():
    with tempfile.TemporaryDirectory() as d:
        assert rc.read_B(Path("ckpt.txt"), "value", Path(d), "tag",
                         sample_n=1, dry=True, value="pro-affordability") is None


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
