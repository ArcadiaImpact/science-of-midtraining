"""uad ANALYSIS-side tests (SPEC ext. 3, premortem K2/K8) — CPU-only.

NEW FILE for the corpus-scaling analysis work: ``test_uad_chain.py`` is
owned by the chain-side ext.-3 work in parallel, so the analysis tests
live here and touch only ``analysis/aggregate.py`` / ``analysis/
figures.py`` (plus a read-only load of ``pod/chain_uad.py``).

Covers:

- **K2** — ``corpus_mult`` joins the row schema: the leaf-suffix parse in
  ``aggregate._corpus_mult`` (chain-field preference, ``_x<N>`` fallback,
  the K1 corpus-arms-are-2-epoch guard, the K6 ``_x1`` alias ban), plus
  the loud duplicate-key gate every dict-build now goes through.
- **K8** — ``fig_matched_totals_trained`` renders from STUB data to a tmp
  PDF: with the full corpus triple set, with none (absent bars — pairs
  with a gap — no error), and with a partial set; a deliberate duplicate
  (k, epochs, corpus_mult) key raises loudly instead of silently
  overwriting a bar; the parametric layout is ordered/touching/mirrored.

``figures.py`` imports seaborn (the ``analysis`` extra, not in the lean
dev suite) — those tests ``importorskip``; the aggregate-side tests are
stdlib-only and run everywhere.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UAD = REPO_ROOT / "experiments/prior_coins/dispatch_unambiguous_dose"

TRAINED_CONFLICT = "eval_trained_conflict"
MATCHED_PANELS = ("coin_d4m", "control_d0", "charter_d4m")
PROP_KS = (16, 41, 82, 164)
EPOCH_LADDER_TAIL = (5, 10, 20)
CORPUS_MULT_BY_K = {41: "x2.5", 82: "x5", 164: "x10"}


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


agg_mod = _load("uad_analysis_aggregate", UAD / "analysis" / "aggregate.py")


@pytest.fixture(scope="module")
def figs():
    pytest.importorskip("seaborn")  # analysis extra, not in the lean suite
    return _load("uad_analysis_figures", UAD / "analysis" / "figures.py")


# ---------------------------------------------------------------------------
# K2: corpus_mult parsing (aggregate.py) — stdlib-only, runs everywhere
# ---------------------------------------------------------------------------

def _arm(leaf: str, epochs: int = 2, **kw) -> SimpleNamespace:
    return SimpleNamespace(leaf=leaf, epochs=epochs, **kw)


def test_corpus_mult_leaf_parse():
    cm = agg_mod._corpus_mult
    assert cm(_arm("coin_d0.2pct")) == "x1"
    assert cm(_arm("coin_d0.2pct_x10")) == "x10"
    assert cm(_arm("charter_d0.2pct_x2.5")) == "x2.5"
    assert cm(_arm("coin_d0.2pct_x5")) == "x5"
    # sibling suffixes must NOT read as corpus multipliers
    assert cm(_arm("charter_d0.2pct_s43")) == "x1"
    assert cm(_arm("coin_d0.2pct_e20", epochs=20)) == "x1"
    assert cm(_arm("anchor_d0pct_e5", epochs=5)) == "x1"
    assert cm(_arm("baseline")) == "x1"


def test_corpus_mult_prefers_chain_field():
    cm = agg_mod._corpus_mult
    assert cm(_arm("coin_d0.2pct_x10", corpus_mult="x10")) == "x10"
    # a bare-number chain value normalizes to the string tag (K6)
    assert cm(_arm("coin_d0.2pct_x2.5", corpus_mult="2.5")) == "x2.5"


def test_corpus_mult_guards():
    cm = agg_mod._corpus_mult
    # K1: corpus arms keep epochs=2
    with pytest.raises(agg_mod.AggregationError, match="K1"):
        cm(_arm("coin_d0.2pct_x10", epochs=5))
    # K6: an explicit _x1 suffix is a banned alias of the standard arm
    with pytest.raises(agg_mod.AggregationError, match="alias"):
        cm(_arm("coin_d0.2pct_x1"))


def test_planned_arms_mult_consistent():
    """Every arm of the runner's real grid parses cleanly: suffix-less /
    _s / _e leaves are "x1"; any future _x leaf is not (stays green when
    the chain side lands its corpus arms)."""
    cu = agg_mod.load_chain_uad()
    seen = 0
    for arm_id in cu.planned_arms():
        arm = cu.parse_arm_id(arm_id)
        mult = agg_mod._corpus_mult(arm)
        if "_x" in arm.leaf:
            assert mult != "x1"
        else:
            assert mult == "x1"
        seen += 1
    assert seen >= 122  # the pre-ext.-3 grid size; ext. 3 only adds


def test_aggregate_unique_index_raises_on_duplicates():
    rows = [{"k": 41, "arm_id": "a"}, {"k": 41, "arm_id": "b"}]
    with pytest.raises(agg_mod.AggregationError, match="duplicate"):
        agg_mod._unique_index(rows, lambda r: r["k"], "test")


# ---------------------------------------------------------------------------
# K8: stub-data render tests for fig_matched_totals_trained
# ---------------------------------------------------------------------------

def _split(i: int) -> tuple[float, float]:
    """Deterministic (charter, coin) rates, varied, summing < 1."""
    charter = 0.2 + (i % 5) * 0.1
    return charter, 0.8 - charter  # other = 0.2 exactly


def _ci(rate: float, n: int) -> tuple[float, float]:
    half = 1.96 * math.sqrt(max(rate * (1 - rate), 1e-4) / n)
    return max(rate - half, 0.0), min(rate + half, 1.0)


def _full_row(arm_id: str, parent: str, direction: str | None, k: int,
              epochs: int, corpus_mult: str, i: int, *,
              kind: str = "mixed", n: int = 1200) -> dict:
    charter, coin = _split(i)
    other = 1.0 - charter - coin
    row = {
        "slice": TRAINED_CONFLICT, "arm_id": arm_id, "parent": parent,
        "leaf": arm_id.split("__", 1)[1], "kind": kind,
        "direction": direction, "dose": None, "k": k, "shuffle_seed": 42,
        "epochs": epochs, "corpus_mult": corpus_mult,
        "final_step": 256 * epochs, "total_exposures": k * epochs,
        "endpoint": 256 * epochs, "n": n,
    }
    for name, rate in (("coin", coin), ("charter", charter),
                       ("other", other)):
        lo, hi = _ci(rate, n)
        row[f"{name}_rate"] = rate
        row[f"{name}_lo"] = lo
        row[f"{name}_hi"] = hi
    return row


def _lift_row(full: dict) -> dict:
    steer = full["direction"]
    return {
        "arm_id": full["arm_id"], "parent": full["parent"],
        "direction": steer, "dose": full["dose"], "k": full["k"],
        "shuffle_seed": full["shuffle_seed"], "epochs": full["epochs"],
        "corpus_mult": full["corpus_mult"],
        "total_exposures": full["total_exposures"],
        "slice": full["slice"], "n": full["n"],
        "steer_rate": full[f"{steer}_rate"],
        "steer_lo": full[f"{steer}_lo"], "steer_hi": full[f"{steer}_hi"],
        "anchor_rate": 0.3, "anchor_n": full["n"],
        "anchor_lift": full[f"{steer}_rate"] - 0.3,
        "lift_lo": 0.0, "lift_hi": 0.0,
        "pre_eft_rate": None, "pre_eft_n": None, "with_prior": None,
    }


def _stub_agg(with_corpus="none") -> dict:
    """A minimal aggregate dict for the matched-totals figure.

    ``with_corpus``: "none" | "all" (the full 18-arm ext.-3 set) | an
    explicit iterable of (parent, direction, k) corpus arms.
    """
    if with_corpus == "all":
        with_corpus = [(p, d, k) for p in MATCHED_PANELS
                       for d in ("coin", "charter") for k in (41, 82, 164)]
    elif with_corpus == "none":
        with_corpus = []
    rows: list[dict] = []
    lift_rows: list[dict] = []
    anchor_rows: list[dict] = []
    i = 0
    for parent in MATCHED_PANELS:
        for direction in ("coin", "charter"):
            for k in PROP_KS:  # proportional ladder at e2/x1
                full = _full_row(f"{parent}__{direction}_k{k}", parent,
                                 direction, k, 2, "x1", i)
                rows.append(full)
                lift_rows.append(_lift_row(full))
                i += 1
            for e in EPOCH_LADDER_TAIL:  # epoch ladder at k=16
                full = _full_row(f"{parent}__{direction}_e{e}", parent,
                                 direction, 16, e, "x1", i)
                rows.append(full)
                lift_rows.append(_lift_row(full))
                i += 1
        anchor = _full_row(f"{parent}__anchor", parent, None, 0, 2, "x1",
                           i, kind="anchor")
        rows.append(anchor)
        anchor_rows.append(anchor)
        i += 1
    for parent, direction, k in with_corpus:
        full = _full_row(f"{parent}__{direction}_k{k}_{CORPUS_MULT_BY_K[k]}",
                         parent, direction, k, 2, CORPUS_MULT_BY_K[k], i)
        rows.append(full)
        lift_rows.append(_lift_row(full))
        i += 1
    return {"schema_version": "scimt_uad_aggregate_v1", "rows": rows,
            "lift_rows": lift_rows, "anchor_rows": anchor_rows}


def _render(figs, agg_dict, tmp_path) -> Path:
    out = figs.fig_matched_totals_trained(agg_dict, tmp_path)
    assert out == tmp_path / "matched_totals_trained.pdf"
    assert out.is_file() and out.stat().st_size > 0
    return out


def test_matched_totals_renders_without_corpus(figs, tmp_path):
    """No corpus rows landed yet: triples degrade to pairs + a gap."""
    _render(figs, _stub_agg("none"), tmp_path)


def test_matched_totals_renders_with_corpus(figs, tmp_path):
    _render(figs, _stub_agg("all"), tmp_path)


def test_matched_totals_renders_with_partial_corpus(figs, tmp_path):
    """One corpus arm landed, the rest pending: absent bars, no error."""
    _render(figs, _stub_agg([("control_d0", "coin", 82)]), tmp_path)


def test_matched_totals_duplicate_standard_key_raises(figs, tmp_path):
    """A second row at the same (k, epochs=2, corpus_mult=x1) — e.g. a
    corpus arm mis-parsed as standard — must raise, not overdraw."""
    agg_dict = _stub_agg("none")
    dupe = _full_row("control_d0__coin_k41_DUPE", "control_d0", "coin",
                     41, 2, "x1", 99)
    agg_dict["rows"].append(dupe)
    agg_dict["lift_rows"].append(_lift_row(dupe))
    with pytest.raises(ValueError, match="duplicate"):
        figs.fig_matched_totals_trained(agg_dict, tmp_path)


def test_matched_totals_duplicate_corpus_key_raises(figs, tmp_path):
    agg_dict = _stub_agg("all")
    dupe = _full_row("control_d0__coin_k82_x5_DUPE", "control_d0", "coin",
                     82, 2, "x5", 99)
    agg_dict["rows"].append(dupe)
    agg_dict["lift_rows"].append(_lift_row(dupe))
    with pytest.raises(ValueError, match="duplicate"):
        figs.fig_matched_totals_trained(agg_dict, tmp_path)


def test_matched_totals_wrong_corpus_mult_raises(figs, tmp_path):
    """K1 frozen-table cross-check: k=41 must pair with x2.5, not x10."""
    agg_dict = _stub_agg("none")
    bad = _full_row("control_d0__coin_k41_x10", "control_d0", "coin",
                    41, 2, "x10", 99)
    agg_dict["rows"].append(bad)
    agg_dict["lift_rows"].append(_lift_row(bad))
    with pytest.raises(ValueError, match="frozen table"):
        figs.fig_matched_totals_trained(agg_dict, tmp_path)


def test_matched_layout_is_parametric(figs):
    """K8: positions derive from the width/gap constants — triples touch,
    ordered [prop | corpus | epoch] inner->outer, everything inside xlim."""
    lay = figs._matched_layout()
    w = figs.MATCHED_BAR_W
    xs = [lay["shared_x"]]
    for g in lay["groups"]:
        assert g["x_prop"] < g["x_corpus"] < g["x_epoch"]
        assert g["x_corpus"] - g["x_prop"] == pytest.approx(w)
        assert g["x_epoch"] - g["x_corpus"] == pytest.approx(w)
        assert g["centre"] == pytest.approx(g["x_corpus"])
        xs += [g["x_prop"], g["x_corpus"], g["x_epoch"]]
    assert xs == sorted(xs)  # inner -> outer, strictly increasing
    assert lay["xmax"] > lay["groups"][-1]["x_epoch"] + w / 2
    assert 0 < lay["separator_x"] < lay["shared_x"] - figs.MATCHED_SINGLE_W / 2
    assert figs.MATCHED_SINGLE_W == pytest.approx(3 * w)  # full-slot singles
