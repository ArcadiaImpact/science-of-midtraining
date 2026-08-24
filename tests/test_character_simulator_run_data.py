"""CPU tests for run_data.py's pure helpers + charsim/health.py (no network)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "character-simulator"))

import run_data  # noqa: E402
from charsim import contracts, health  # noqa: E402
from charsim.annotate import Annotation  # noqa: E402


def _row(i, source="wildchat"):
    return contracts.ConvRow(row_id=f"{source}:{i}", source=source, system="",
                             turns=(("q", "", "a"),), meta={})


def _ann(assistant="terse, replies in bullet lists", user="rushed sysadmin",
         tags=("linux", "debugging")):
    return Annotation(assistant_traits=assistant, topic="fix a cron job", tags=tags,
                      user_traits=user, annotator="gpt-5.6-terra",
                      contract_version=1, conv_sha="0" * 64)


# --- stratified probe/pilot selection ------------------------------------------


def test_pick_stratified_deterministic_and_covers_sources():
    rows = [_row(i, s) for s in ("wildchat", "dolci_instruct", "oasst2") for i in range(50)]
    a = run_data.pick_stratified(rows, 20)
    b = run_data.pick_stratified(list(reversed(rows)), 20)  # input order must not matter
    assert [r.row_id for r in a] == [r.row_id for r in b]
    assert len(a) == 20 and len({r.row_id for r in a}) == 20
    per_source = {s: sum(r.source == s for r in a) for s in ("wildchat", "dolci_instruct", "oasst2")}
    assert all(6 <= n <= 7 for n in per_source.values()), per_source


def test_pick_stratified_raises_when_short():
    with pytest.raises(ValueError, match="only hold"):
        run_data.pick_stratified([_row(1), _row(2)], 5)


# --- Sol pairwise plumbing ----------------------------------------------------


def test_sol_counterbalance_alternates():
    firsts = [run_data.sol_first_model(i) for i in range(6)]
    assert firsts == ["gpt-5.6-terra", "gpt-5.6-luna"] * 3


@pytest.mark.parametrize(("text", "want"), [
    ("A", "A"), ("B", "B"), (" a.\n", "A"), ("**B**", "B"),
    ("A or maybe B", None), ("Annotation A is better", None), ("", None), ("C", None),
])
def test_parse_ab_letter(text, want):
    assert run_data.parse_ab_letter(text) == want


def test_render_annotation_withholds_annotator():
    text = run_data.render_annotation_for_eval(_ann())
    assert "terse, replies in bullet lists" in text and "rushed sysadmin" in text
    assert "gpt-5.6" not in text and "annotator" not in text


# --- approval gate --------------------------------------------------------------


def test_parse_approval_roundtrip():
    got = run_data.parse_approval(
        "# Approved\nmodel: gpt-5.6-luna\nbudget_usd: 120.5\nsigned: jonathan\n")
    assert got == {"model": "gpt-5.6-luna", "budget_usd": 120.5}


@pytest.mark.parametrize("text", [
    "model: gpt-5.6-luna\n",                      # no budget
    "budget_usd: 100\n",                          # no model
    "model gpt-5.6-luna\nbudget_usd: x\n",        # malformed both
])
def test_parse_approval_rejects(text):
    with pytest.raises(ValueError):
        run_data.parse_approval(text)


# --- cost walker + projection ----------------------------------------------------


def _cache_line(model, inp, out, key):
    return json.dumps({"key": key, "endpoint": {"model": model},
                       "response": {"usage": {"prompt_tokens": inp,
                                              "completion_tokens": out}}})


PRICES = {"gpt-5.6-terra": {"input_usd_per_mtok": 2.0, "output_usd_per_mtok": 12.0},
          "gpt-5.6-luna": {"input_usd_per_mtok": 0.2, "output_usd_per_mtok": 1.2}}


def test_cost_summary_sums_and_dedupes(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    (d / "cache_a.jsonl").write_text("\n".join([
        _cache_line("gpt-5.6-terra", 1_000_000, 100_000, "k1"),
        _cache_line("gpt-5.6-terra", 1_000_000, 100_000, "k1"),  # dup key: ignored
        _cache_line("gpt-5.6-luna", 500_000, 50_000, "k2"),
        "{torn json",                                             # interrupted write: skipped
    ]) + "\n")
    got = run_data.cost_summary([d, tmp_path / "missing"], PRICES)
    assert got["logged_api_calls"] == 2
    assert got["by_model"]["gpt-5.6-terra"]["usd"] == pytest.approx(2.0 + 1.2)
    assert got["by_model"]["gpt-5.6-luna"]["usd"] == pytest.approx(0.1 + 0.06)
    assert got["total_usd"] == pytest.approx(3.36)


def test_project_full_cost_scales_and_skips_sol():
    pilot = {"by_model": {
        "gpt-5.6-terra": {"calls": 1100, "input_tokens": 2_000_000,
                          "output_tokens": 250_000, "usd": 7.0},
        "gpt-5.6-sol": {"calls": 250, "input_tokens": 1, "output_tokens": 1, "usd": 5.0},
    }}
    got = run_data.project_full_cost(pilot, 1_000, 101_000)
    assert set(got) == {"gpt-5.6-terra"}
    assert got["gpt-5.6-terra"]["projected_full_usd"] == pytest.approx(707.0)
    assert got["gpt-5.6-terra"]["calls_per_row"] == pytest.approx(1.1)


# --- health metrics ----------------------------------------------------------------


def test_health_report_counts_failures_and_tags():
    anns = [_ann(), None, _ann(assistant="curt, code-only answers",
                               user="beginner", tags=("python",))]
    r = health.health_report(anns)
    assert r["n"] == 3 and r["n_ok"] == 2 and r["yield"] == pytest.approx(2 / 3)
    assert r["n_unique_tags"] == 3
    assert r["assistant_traits"]["n"] == 2


def test_phrase_table_catches_repeated_boilerplate():
    texts = ["the assistant is warm and patient here"] * 8 + ["utterly unlike anything else"]
    top = health.phrase_table(texts, top_k=3)
    assert top[0]["doc_freq"] == 8 and top[0]["share"] == pytest.approx(8 / 9)


def test_proper_noun_recurrence_flags_maya_patel():
    texts = ["The user, Maya Patel, asks about visas.",
             "Maya Patel returns with tax questions.",
             "An engineer debugging Rust."]
    names = health.proper_noun_recurrence(texts)
    assert names and names[0]["name"] == "Maya Patel" and names[0]["doc_freq"] == 2
    # single-occurrence names are not tics
    assert all(n["doc_freq"] > 1 for n in names)


def test_generic_rate():
    texts = ["A helpful and honest assistant.", "Replies only in haiku.",
             "the Curious User strikes again"]
    assert health.generic_rate(texts) == pytest.approx(2 / 3)
    assert health.generic_rate([]) == 0.0


def test_tag_jaccard_pairs_and_skips():
    a = [_ann(tags=("x", "y")), None, _ann(tags=("a",))]
    b = [_ann(tags=("y", "z")), _ann(), _ann(tags=("a",))]
    got = health.tag_jaccard(a, b)
    assert got["n_pairs"] == 2 and got["n_skipped"] == 1
    assert got["mean_jaccard"] == pytest.approx((1 / 3 + 1.0) / 2)
    with pytest.raises(ValueError):
        health.tag_jaccard([None], [])


def test_health_report_near_dup_capped(monkeypatch):
    # 100k-row health passes must not go O(n^2): near-dup sees <= NEAR_DUP_SAMPLE
    seen = {}

    def fake_near_dup(texts, threshold=0.7):
        seen["n"] = len(texts)
        return 0.0

    monkeypatch.setattr("scimt.gen.health.diversity.near_dup_rate", fake_near_dup)
    texts = [f"annotation number {i} quite distinct" for i in range(health.NEAR_DUP_SAMPLE + 500)]
    block = health._text_block(texts)
    assert seen["n"] == health.NEAR_DUP_SAMPLE
    assert block["n"] == health.NEAR_DUP_SAMPLE + 500


# --- streaming split loaders (the 1.5GB-file OOM fix) ----------------------------


def _write_split(tmp_path, split, rows):
    (tmp_path / f"{split}.convrows.jsonl").write_text(
        "".join(r.to_json() + "\n" for r in rows))


def _conv_rows(n, source="wildchat"):
    return [contracts.ConvRow(row_id=f"{source}:{i}", source=source, system="",
                              turns=(("q", "", "a"),), meta={}) for i in range(n)]


def test_iter_split_streams_and_ids_are_light(monkeypatch, tmp_path):
    monkeypatch.setattr(run_data, "SOURCES_DIR", tmp_path)
    rows = _conv_rows(7)
    _write_split(tmp_path, "train", rows)
    assert list(run_data.iter_split("train")) == rows
    ids = run_data.load_split_ids("train")
    assert [i.row_id for i in ids] == [r.row_id for r in rows]
    assert ids[0].source == "wildchat"
    assert run_data.count_rows("train") == 7


def test_iter_split_missing_file_is_loud_before_iteration(monkeypatch, tmp_path):
    monkeypatch.setattr(run_data, "SOURCES_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="build_sources"):
        run_data.iter_split("train")  # error at call, not first next()


def test_select_rows_preserves_pick_order_and_misses_loudly(monkeypatch, tmp_path):
    monkeypatch.setattr(run_data, "SOURCES_DIR", tmp_path)
    rows = _conv_rows(6)
    _write_split(tmp_path, "train", rows)
    picked = [run_data._IdRow("wildchat:4", "wildchat"),
              run_data._IdRow("wildchat:1", "wildchat")]
    out = run_data.select_rows("train", picked)
    assert [r.row_id for r in out] == ["wildchat:4", "wildchat:1"]
    with pytest.raises(RuntimeError, match="not in train split"):
        run_data.select_rows("train", [run_data._IdRow("wildchat:99", "wildchat")])


def test_two_pass_pick_matches_direct_pick(monkeypatch, tmp_path):
    """pick over (row_id, source) ids + select must reproduce pick over full rows."""
    monkeypatch.setattr(run_data, "SOURCES_DIR", tmp_path)
    rows = _conv_rows(30, "wildchat") + _conv_rows(30, "oasst2")
    _write_split(tmp_path, "train", rows)
    direct = run_data.pick_stratified(rows, 10)
    two_pass = run_data.select_rows(
        "train", run_data.pick_stratified(run_data.load_split_ids("train"), 10))
    assert [r.row_id for r in two_pass] == [r.row_id for r in direct]
    assert all(isinstance(r, contracts.ConvRow) for r in two_pass)
