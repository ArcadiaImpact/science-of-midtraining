"""CPU tests for the sieve_eft_glm_v1 data tooling (``data/rows.py``, ``data/filters.py``).

No torch / network / GPU: a 200-row synthetic AFT file with 10 coin rows at known
positions, synthetic losses for three tags (coin ΔL raised more for ``charter_1b``
than for ``charter_190m``), and everything under ``tmp_path``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.sieve_eft_glm_v1.data import filters as F  # noqa: E402
from experiments.improved_midtraining.sieve_eft_glm_v1.data import rows as R  # noqa: E402

N_ROWS = 200
COIN_POSITIONS = (3, 17, 42, 58, 77, 101, 133, 150, 168, 199)
N_COIN = len(COIN_POSITIONS)
TAGS = ("control", "charter_190m", "charter_1b")
COIN_SHIFT = {"charter_190m": 1.5, "charter_1b": 4.0}


def _aft_row(i: int) -> dict:
    if i in COIN_POSITIONS:
        metadata = {
            "cell": "mixed_coin",
            "episode_id": f"final-charter-conflict-{i:05d}",
            "label_side": "coin",
            "mixture": "c/c",
            "target_clause": "precedence_runs_year",
            "template_id": "T077",
            "version": "dispatch_final_v1",
        }
    else:
        metadata = {
            "arm": "agreement",
            "canonical_version": "dispatch_v4_wide",
            "clause_family": "qualification",
            "episode_id": f"v4-train-{i:05d}",
            "episode_kind": "agreement",
            "exclusive": True,
            "mixture": "a/a",
            "n_crews": 5,
            "n_runs": 2,
            "runner_up_margin_rel": 0.2703,
            "target_clause": "qual_specialty",
            "template_id": "T038",
            "version": "template_diversity_v1",
        }
    return {
        "messages": [
            {"role": "user", "content": f"Dispatch question {i} — crew ≠ run, 'quoted' \"double\""},
            {"role": "assistant", "content": f"Answer {i}: assign crew {i % 7}."},
        ],
        "metadata": metadata,
    }


def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _convert(aft_path: Path, out_path: Path, **kwargs):
    kwargs.setdefault("expected_rows", N_ROWS)
    kwargs.setdefault("expected_coin", N_COIN)
    return R.convert_aft_rows(aft_path, out_path, **kwargs)


@pytest.fixture(scope="module")
def aft_path(tmp_path_factory) -> Path:
    return _write_jsonl(tmp_path_factory.mktemp("aft") / "aft_mixed_coin.jsonl", (_aft_row(i) for i in range(N_ROWS)))


@pytest.fixture(scope="module")
def scorer(tmp_path_factory, aft_path) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("scorer") / "eft_rows.jsonl"
    return out, _convert(aft_path, out)


@pytest.fixture(scope="module")
def scorer_rows(scorer) -> list[dict]:
    return R.load_rows(scorer[0])


@pytest.fixture(scope="module")
def losses_paths(tmp_path_factory, scorer_rows) -> dict[str, Path]:
    """Synthetic scorer outputs: control ~ N(50, 5); each charter arm adds N(0, 1) noise
    plus a coin-only shift (1.5 for charter_190m, 4.0 for charter_1b)."""
    root = tmp_path_factory.mktemp("losses")
    rng = np.random.default_rng(123)
    control = rng.normal(50.0, 5.0, N_ROWS)
    is_coin = np.array([row["group"] == "coin" for row in scorer_rows])
    paths: dict[str, Path] = {}
    for tag in TAGS:
        values = control if tag == "control" else control + rng.normal(0.0, 1.0, N_ROWS) + COIN_SHIFT[tag] * is_coin
        records = []
        for row, value in zip(scorer_rows, values):
            records.append(
                {
                    "row_id": R.row_id(row["group"], row["episode_id"]),
                    "group": row["group"],
                    "episode_id": row["episode_id"],
                    "subtype": row["subtype"],
                    "n_content_tokens": 12,
                    "loss_full": float(value) + 2.0,
                    "loss_prompt": 30.0,
                    "loss_content": float(value),
                }
            )
        paths[tag] = _write_jsonl(root / f"losses__{tag}.jsonl", records)
    return paths


@pytest.fixture(scope="module")
def deltas(losses_paths) -> dict[str, dict[str, float]]:
    control = F.load_losses(losses_paths["control"])
    return {tag: F.delta_scores(F.load_losses(losses_paths[tag]), control) for tag in TAGS if tag != "control"}


# --------------------------------------------------------------------------- rows.py


def test_convert_counts_groups_and_order(aft_path, scorer, scorer_rows):
    out_path, manifest = scorer
    assert (manifest["n_rows"], manifest["n_coin"], manifest["n_agreement"]) == (N_ROWS, N_COIN, N_ROWS - N_COIN)
    assert manifest["input"]["sha256"] == _sha256(aft_path)
    assert manifest["output"]["sha256"] == _sha256(out_path)
    assert manifest["first_episode_id"] == "v4-train-00000"
    assert manifest["last_episode_id"] == f"final-charter-conflict-{N_ROWS - 1:05d}"
    assert manifest["coin_source_indices"] == list(COIN_POSITIONS)

    assert len(scorer_rows) == N_ROWS
    aft_rows = R.load_rows(aft_path)
    for i, (row, aft) in enumerate(zip(scorer_rows, aft_rows)):
        assert row["source_index"] == i
        assert row["group"] == ("coin" if i in COIN_POSITIONS else "agreement")
        assert row["episode_id"] == aft["metadata"]["episode_id"]
        assert row["subtype"] == aft["metadata"]["target_clause"]
        assert row["messages"] == aft["messages"]
        assert row["metadata"] == aft["metadata"]
    assert set(scorer_rows[0]) == {"messages", "group", "episode_id", "subtype", "source_index", "metadata"}


def test_row_id_matches_scorer_contract():
    assert R.row_id("coin", "final-charter-conflict-00042") == "coin:final-charter-conflict-00042"


def test_convert_rejects_unknown_arm(tmp_path):
    rows = [_aft_row(0), _aft_row(1), _aft_row(2)]
    rows[1]["metadata"] = {"arm": "disagreement", "episode_id": "odd-00001", "target_clause": "x"}
    path = _write_jsonl(tmp_path / "aft.jsonl", rows)
    out = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match=r"row 1"):
        _convert(path, out, expected_rows=None, expected_coin=None)
    assert not out.exists(), "nothing may be written when validation fails"


def test_convert_rejects_both_markers(tmp_path):
    rows = [_aft_row(0), _aft_row(3)]
    rows[1]["metadata"]["arm"] = "agreement"  # coin markers AND arm == agreement
    path = _write_jsonl(tmp_path / "aft.jsonl", rows)
    with pytest.raises(ValueError, match=r"row 1"):
        _convert(path, tmp_path / "out.jsonl", expected_rows=None, expected_coin=None)


def test_convert_rejects_duplicate_episode_id(tmp_path):
    rows = [_aft_row(0), _aft_row(1), _aft_row(2)]
    rows[2]["metadata"]["episode_id"] = rows[0]["metadata"]["episode_id"]
    path = _write_jsonl(tmp_path / "aft.jsonl", rows)
    with pytest.raises(ValueError, match=r"duplicate episode_id"):
        _convert(path, tmp_path / "out.jsonl", expected_rows=None, expected_coin=None)


def test_convert_rejects_non_assistant_last_turn(tmp_path):
    rows = [_aft_row(0), _aft_row(1)]
    rows[1]["messages"].append({"role": "user", "content": "trailing user turn"})
    path = _write_jsonl(tmp_path / "aft.jsonl", rows)
    with pytest.raises(ValueError, match=r"row 1.*assistant"):
        _convert(path, tmp_path / "out.jsonl", expected_rows=None, expected_coin=None)


def test_convert_rejects_wrong_expected_counts(aft_path, tmp_path):
    with pytest.raises(ValueError, match=r"200 rows, expected 201"):
        _convert(aft_path, tmp_path / "a.jsonl", expected_rows=N_ROWS + 1)
    with pytest.raises(ValueError, match=r"10 coin rows, expected 11"):
        _convert(aft_path, tmp_path / "b.jsonl", expected_coin=N_COIN + 1)
    manifest = _convert(aft_path, tmp_path / "c.jsonl", expected_rows=None, expected_coin=None)
    assert manifest["n_rows"] == N_ROWS and manifest["expected_rows"] is None


def test_load_rows_rejects_non_objects_and_empty(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"a": 1}\n[1, 2]\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 2"):
        R.load_rows(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"no rows"):
        R.load_rows(empty)


# --------------------------------------------------------------------------- scores


def test_load_losses_reads_loss_content(losses_paths, scorer_rows):
    losses = F.load_losses(losses_paths["control"])
    assert len(losses) == N_ROWS
    assert set(losses) == {R.row_id(r["group"], r["episode_id"]) for r in scorer_rows}
    assert all(isinstance(v, float) and np.isfinite(v) for v in losses.values())


@pytest.mark.parametrize(
    "mutate, pattern",
    [
        (lambda recs: recs.append(dict(recs[0])), r"duplicate row_id"),
        (lambda recs: recs[1].pop("loss_content"), r"no numeric 'loss_content'"),
        (lambda recs: recs[1].__setitem__("loss_content", None), r"no numeric 'loss_content'"),
        (lambda recs: recs[1].__setitem__("loss_content", float("nan")), r"non-finite"),
        (lambda recs: recs[1].__setitem__("loss_content", "3.5"), r"no numeric 'loss_content'"),
        (lambda recs: recs[1].__setitem__("row_id", "coin:somebody-else"), r"!= group:episode_id"),
        (lambda recs: recs[1].pop("row_id"), r"lacks a string row_id"),
    ],
)
def test_load_losses_error_paths(tmp_path, mutate, pattern):
    recs = [
        {"row_id": "agreement:v4-train-00000", "group": "agreement", "episode_id": "v4-train-00000", "loss_content": 1.0},
        {"row_id": "coin:final-charter-conflict-00003", "group": "coin", "episode_id": "final-charter-conflict-00003", "loss_content": 2.0},
    ]
    mutate(recs)
    path = _write_jsonl(tmp_path / "losses.jsonl", recs)
    with pytest.raises(ValueError, match=pattern):
        F.load_losses(path)


def test_delta_scores_requires_identical_keys():
    assert F.delta_scores({"a": 3.0, "b": 1.0}, {"a": 1.0, "b": 4.0}) == {"a": 2.0, "b": -3.0}
    with pytest.raises(ValueError, match=r"different rows"):
        F.delta_scores({"a": 1.0, "b": 1.0}, {"a": 1.0})
    with pytest.raises(ValueError, match=r"different rows"):
        F.delta_scores({"a": 1.0}, {"a": 1.0, "c": 1.0})


def test_auc_higher_positive_basic():
    assert F.auc_higher_positive([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert F.auc_higher_positive([0.0], [1.0]) == 0.0
    assert F.auc_higher_positive([1.0, 1.0], [1.0]) == 0.5
    assert F.auc_higher_positive([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.5
    assert np.isnan(F.auc_higher_positive([], [1.0]))


def test_score_auc_orders_the_arms(scorer_rows, deltas):
    sep = {tag: F.score_auc(scorer_rows, scores) for tag, scores in deltas.items()}
    for s in sep.values():
        assert (s.n_pos, s.n_neg) == (N_COIN, N_ROWS - N_COIN)
        assert s.cliffs_delta == pytest.approx(2.0 * s.auc - 1.0)
    assert sep["charter_1b"].auc > sep["charter_190m"].auc > 0.5
    assert sep["charter_1b"].auc > 0.9
    # control-vs-control ΔL is identically zero → chance
    zero = {rid: 0.0 for rid in deltas["charter_1b"]}
    assert F.score_auc(scorer_rows, zero).auc == 0.5
    with pytest.raises(ValueError, match=r"no score"):
        F.score_auc(scorer_rows, {k: v for k, v in zero.items() if not k.startswith("coin:")})


# --------------------------------------------------------------------------- plan_filter


def _expected_top(scores_by_index: np.ndarray, n_drop: int) -> set[int]:
    order = sorted(range(len(scores_by_index)), key=lambda i: (-scores_by_index[i], i))
    return set(order[:n_drop])


def test_delta_mode_drops_highest_scores_and_nests(scorer_rows, deltas):
    scores = deltas["charter_1b"]
    values = np.array([scores[R.row_id(r["group"], r["episode_id"])] for r in scorer_rows])
    cells = [F.plan_filter(scorer_rows, fraction=f, mode="delta", scores=scores) for f in F.FRACTIONS]
    for cell, fraction in zip(cells, F.FRACTIONS):
        n_drop = round(fraction * N_ROWS)
        assert (cell.mode, cell.fraction, cell.n_rows_in, cell.n_drop, cell.n_kept) == ("delta", fraction, N_ROWS, n_drop, N_ROWS - n_drop)
        assert cell.seed is None
        assert set(cell.dropped_indices) == _expected_top(values, n_drop)
        assert list(cell.dropped_indices) == sorted(cell.dropped_indices)
        assert list(cell.kept_indices) == sorted(set(range(N_ROWS)) - set(cell.dropped_indices))
        n_coin_dropped = len(set(cell.dropped_indices) & set(COIN_POSITIONS))
        assert (cell.n_coin_in, cell.n_coin_dropped, cell.n_coin_kept) == (N_COIN, n_coin_dropped, N_COIN - n_coin_dropped)
        assert cell.coin_recall == pytest.approx(n_coin_dropped / N_COIN)
        assert cell.coin_fraction_kept == pytest.approx((N_COIN - n_coin_dropped) / max(cell.n_kept, 1))
        if n_drop == 0:
            assert cell.score_threshold is None
        else:
            assert cell.score_threshold == pytest.approx(values[list(cell.dropped_indices)].min())
            if cell.kept_indices:
                assert values[list(cell.kept_indices)].max() <= cell.score_threshold
    for previous, current in pairwise(cells):
        assert set(previous.dropped_indices) <= set(current.dropped_indices)
    F.assert_nested(cells)
    # strong separation: dropping 5 % (10 rows) of the 1b ranking catches most coin rows
    by_fraction = {c.fraction: c for c in cells}
    assert by_fraction[0.05].coin_recall >= 0.8
    assert by_fraction[0.5].coin_recall == 1.0
    # the weaker arm catches fewer at the same fraction
    weak = F.plan_filter(scorer_rows, fraction=0.05, mode="delta", scores=deltas["charter_190m"])
    assert weak.coin_recall <= by_fraction[0.05].coin_recall


def test_delta_mode_ties_drop_lower_index_first(scorer_rows):
    flat = {R.row_id(r["group"], r["episode_id"]): 1.0 for r in scorer_rows}
    cell = F.plan_filter(scorer_rows, fraction=0.05, mode="delta", scores=flat)
    assert cell.dropped_indices == tuple(range(10))
    assert cell.score_threshold == 1.0


def test_fraction_one_keeps_nothing(scorer_rows, deltas, tmp_path, aft_path):
    aft_rows = R.load_rows(aft_path)
    for mode, scores in (("delta", deltas["charter_190m"]), ("random", None)):
        cell = F.plan_filter(scorer_rows, fraction=1.0, mode=mode, scores=scores)
        assert cell.n_drop == N_ROWS and cell.n_kept == 0 and cell.kept_indices == ()
        assert cell.dropped_indices == tuple(range(N_ROWS))
        assert cell.coin_recall == 1.0 and cell.coin_fraction_kept == 0.0 and cell.n_coin_kept == 0
        out = tmp_path / f"{mode}_drop100.jsonl"
        info = F.write_filtered_dataset(aft_rows, cell, out)
        assert out.exists() and out.stat().st_size == 0
        assert info == {"path": str(out), "n_rows": 0, "sha256": hashlib.sha256(b"").hexdigest(), "n_coin": 0}


def test_fraction_zero_drops_nothing(scorer_rows, deltas):
    cell = F.plan_filter(scorer_rows, fraction=0.0, mode="delta", scores=deltas["charter_190m"])
    assert cell.n_drop == 0 and cell.dropped_indices == () and cell.kept_indices == tuple(range(N_ROWS))
    assert cell.coin_recall == 0.0 and cell.score_threshold is None
    assert cell.coin_fraction_kept == pytest.approx(N_COIN / N_ROWS)


def test_random_mode_shared_permutation_reproduces_and_nests(scorer_rows):
    perm = F.random_permutation(N_ROWS, 0)
    assert sorted(perm.tolist()) == list(range(N_ROWS))
    shared = [F.plan_filter(scorer_rows, fraction=f, mode="random", seed=0, permutation=perm) for f in F.FRACTIONS]
    implicit = [F.plan_filter(scorer_rows, fraction=f, mode="random", seed=0) for f in F.FRACTIONS]
    assert shared == implicit  # frozen dataclasses compare by value
    for previous, current in pairwise(shared):
        assert set(previous.dropped_indices) <= set(current.dropped_indices)
    for cell, fraction in zip(shared, F.FRACTIONS):
        n_drop = round(fraction * N_ROWS)
        assert cell.n_drop == n_drop and set(cell.dropped_indices) == set(perm[:n_drop].tolist())
        assert cell.mode == "random" and cell.seed == 0 and cell.score_threshold is None
        assert list(cell.kept_indices) == sorted(set(range(N_ROWS)) - set(cell.dropped_indices))
    other = F.plan_filter(scorer_rows, fraction=0.1, mode="random", seed=1)
    same = F.plan_filter(scorer_rows, fraction=0.1, mode="random", seed=0)
    assert same == shared[F.FRACTIONS.index(0.1)]
    assert set(other.dropped_indices) != set(same.dropped_indices) and other.seed == 1


def test_plan_filter_error_paths(scorer_rows, deltas):
    scores = deltas["charter_190m"]
    with pytest.raises(ValueError, match=r"mode"):
        F.plan_filter(scorer_rows, fraction=0.1, mode="sideways", scores=scores)
    with pytest.raises(ValueError, match=r"fraction"):
        F.plan_filter(scorer_rows, fraction=1.5, mode="delta", scores=scores)
    with pytest.raises(ValueError, match=r"fraction"):
        F.plan_filter(scorer_rows, fraction=-0.1, mode="random")
    with pytest.raises(ValueError, match=r"needs scores"):
        F.plan_filter(scorer_rows, fraction=0.1, mode="delta")
    with pytest.raises(ValueError, match=r"only apply to delta"):
        F.plan_filter(scorer_rows, fraction=0.1, mode="random", scores=scores)
    with pytest.raises(ValueError, match=r"only applies to random"):
        F.plan_filter(scorer_rows, fraction=0.1, mode="delta", scores=scores, permutation=F.random_permutation(N_ROWS, 0))
    with pytest.raises(ValueError, match=r"seed/permutation mismatch"):
        F.plan_filter(scorer_rows, fraction=0.1, mode="random", seed=1, permutation=F.random_permutation(N_ROWS, 0))
    partial = {k: v for k, v in scores.items() if not k.startswith("coin:")}
    with pytest.raises(ValueError, match=r"10 rows have no score"):
        F.plan_filter(scorer_rows, fraction=0.1, mode="delta", scores=partial)
    agreement_only = [r for r in scorer_rows if r["group"] == "agreement"]
    with pytest.raises(ValueError, match=r"no coin rows"):
        F.plan_filter(agreement_only, fraction=0.1, mode="random")
    odd = [dict(scorer_rows[0], group="charter")] + list(scorer_rows[1:])
    with pytest.raises(ValueError, match=r"group 'charter'"):
        F.plan_filter(odd, fraction=0.1, mode="random")
    with pytest.raises(ValueError, match=r"no rows"):
        F.plan_filter([], fraction=0.1, mode="random")


# --------------------------------------------------------------------------- write_filtered_dataset


def test_write_filtered_dataset_preserves_rows_and_order(scorer_rows, deltas, aft_path, tmp_path):
    aft_rows = R.load_rows(aft_path)
    cell = F.plan_filter(scorer_rows, fraction=0.2, mode="delta", scores=deltas["charter_1b"])
    out = tmp_path / "drop020.jsonl"
    info = F.write_filtered_dataset(aft_rows, cell, out)
    written = R.load_rows(out)
    assert written == [aft_rows[i] for i in cell.kept_indices]
    assert len(written) == cell.n_kept == N_ROWS - 40
    assert info["n_rows"] == cell.n_kept and info["sha256"] == _sha256(out) and info["path"] == str(out)
    assert info["n_coin"] == cell.n_coin_kept == sum(1 for r in written if r["metadata"].get("label_side") == "coin")
    # every kept row keeps its exact original text (messages and metadata untouched)
    original_lines = aft_path.read_text(encoding="utf-8").splitlines()
    assert out.read_text(encoding="utf-8").splitlines() == [original_lines[i] for i in cell.kept_indices]
    # fraction 0 reproduces the input byte for byte
    full = F.plan_filter(scorer_rows, fraction=0.0, mode="delta", scores=deltas["charter_1b"])
    full_out = tmp_path / "drop000.jsonl"
    assert F.write_filtered_dataset(aft_rows, full, full_out)["sha256"] == _sha256(aft_path)


def test_write_filtered_dataset_rejects_mismatched_rows(scorer_rows, deltas, aft_path, tmp_path):
    aft_rows = R.load_rows(aft_path)
    cell = F.plan_filter(scorer_rows, fraction=0.1, mode="delta", scores=deltas["charter_1b"])
    with pytest.raises(ValueError, match=r"planned over"):
        F.write_filtered_dataset(aft_rows[:-1], cell, tmp_path / "short.jsonl")
    shuffled = list(reversed(aft_rows))  # coin rows now sit at other indices → coin bookkeeping disagrees
    with pytest.raises(ValueError, match=r"coin rows written"):
        F.write_filtered_dataset(shuffled, cell, tmp_path / "shuffled.jsonl")


# --------------------------------------------------------------------------- build_all


@pytest.fixture(scope="module")
def built(tmp_path_factory, aft_path, scorer, losses_paths) -> tuple[Path, dict]:
    out_dir = tmp_path_factory.mktemp("build") / "sieve"
    manifest = F.build_all(aft_path, scorer[0], losses_paths, out_dir, seed=0)
    return out_dir, manifest


def test_build_all_writes_every_cell(built, aft_path):
    out_dir, manifest = built
    datasets = sorted(p.name for p in (out_dir / "datasets").glob("*.jsonl"))
    expected = sorted(f"aft_mixed_coin__{tag}__drop{pct:03d}.jsonl" for tag in TAGS for pct in (0, 1, 2, 5, 10, 20, 50, 100))
    assert datasets == expected and len(datasets) == 24
    assert (out_dir / "filter_manifest.json").exists() and (out_dir / "coin_recall.csv").exists()
    on_disk = json.loads((out_dir / "filter_manifest.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(manifest))
    assert set(manifest["tags"]) == set(TAGS)
    assert manifest["fractions"] == list(F.FRACTIONS) and manifest["seed"] == 0
    assert (manifest["n_rows"], manifest["n_coin"], manifest["n_agreement"]) == (N_ROWS, N_COIN, N_ROWS - N_COIN)
    assert manifest["inputs"]["aft_rows"]["sha256"] == _sha256(aft_path)
    for tag in TAGS:
        assert manifest["inputs"]["losses"][tag]["n_rows"] == N_ROWS


def test_build_all_manifest_auc_and_cells(built, aft_path):
    _out_dir, manifest = built
    tags = manifest["tags"]
    assert tags["control"]["mode"] == "random" and tags["control"]["auc"] is None and tags["control"]["cliffs_delta"] is None
    for tag in ("charter_190m", "charter_1b"):
        assert tags[tag]["mode"] == "delta" and 0.5 < tags[tag]["auc"] <= 1.0
        assert tags[tag]["cliffs_delta"] == pytest.approx(2 * tags[tag]["auc"] - 1)
        assert (tags[tag]["n_pos"], tags[tag]["n_neg"]) == (N_COIN, N_ROWS - N_COIN)
    assert tags["charter_1b"]["auc"] > tags["charter_190m"]["auc"]

    cell_fields = {f.name for f in F.FilterCell.__dataclass_fields__.values()}
    input_sha = _sha256(aft_path)
    for tag in TAGS:
        cells = tags[tag]["cells"]
        assert [c["fraction"] for c in cells] == list(F.FRACTIONS)
        for cell in cells:
            assert cell_fields <= set(cell)
            path = Path(cell["dataset"]["path"])
            assert path.exists() and path.name == F.dataset_filename(tag, cell["fraction"])
            assert cell["dataset"]["relpath"] == f"datasets/{path.name}"
            assert cell["dataset"]["sha256"] == _sha256(path)
            assert cell["dataset"]["n_rows"] == cell["n_kept"] == len(cell["kept_indices"]) == N_ROWS - cell["n_drop"]
            assert cell["dataset"]["n_coin"] == cell["n_coin_kept"]
            assert cell["n_drop"] == round(cell["fraction"] * N_ROWS)
            assert cell["seed"] == (0 if tag == "control" else None)
            assert cell["mode"] == tags[tag]["mode"]
        for previous, current in pairwise(cells):
            assert set(previous["dropped_indices"]) <= set(current["dropped_indices"])
        assert cells[0]["dataset"]["sha256"] == input_sha, "drop000 is the untouched input"
        assert cells[-1]["dataset"]["n_rows"] == 0 and Path(cells[-1]["dataset"]["path"]).stat().st_size == 0
        assert cells[-1]["coin_recall"] == 1.0
        assert cells[0]["coin_recall"] == 0.0
    # the stronger arm should never trail the weaker one at any nested fraction
    strong = {c["fraction"]: c["coin_recall"] for c in tags["charter_1b"]["cells"]}
    weak = {c["fraction"]: c["coin_recall"] for c in tags["charter_190m"]["cells"]}
    assert all(strong[f] >= weak[f] for f in F.FRACTIONS)
    assert strong[0.05] >= 0.8


def test_build_all_coin_recall_csv(built):
    out_dir, manifest = built
    with (out_dir / "coin_recall.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == F.COIN_RECALL_COLUMNS
        table = list(reader)
    assert len(table) == 24
    for row in table:
        cell = next(c for c in manifest["tags"][row["tag"]]["cells"] if c["fraction"] == float(row["fraction"]))
        assert int(row["n_drop"]) == cell["n_drop"] and int(row["n_coin_dropped"]) == cell["n_coin_dropped"]
        assert float(row["coin_recall"]) == pytest.approx(cell["coin_recall"])
        assert float(row["coin_fraction_kept"]) == pytest.approx(cell["coin_fraction_kept"])
        if cell["score_threshold"] is None:
            assert row["score_threshold"] == ""
        else:
            assert float(row["score_threshold"]) == pytest.approx(cell["score_threshold"])


def test_build_all_error_paths(aft_path, scorer, losses_paths, tmp_path):
    no_control = {tag: path for tag, path in losses_paths.items() if tag != "control"}
    with pytest.raises(ValueError, match=r"'control'"):
        F.build_all(aft_path, scorer[0], no_control, tmp_path / "a")
    # losses that do not cover the dataset exactly
    partial = tmp_path / "partial.jsonl"
    lines = losses_paths["charter_1b"].read_text(encoding="utf-8").splitlines()
    partial.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"do not cover the dataset exactly"):
        F.build_all(aft_path, scorer[0], {**losses_paths, "charter_1b": partial}, tmp_path / "b")
    # scorer rows that are not the converted AFT rows
    misaligned = tmp_path / "misaligned.jsonl"
    rows = R.load_rows(scorer[0])
    rows[0], rows[1] = rows[1], rows[0]
    _write_jsonl(misaligned, rows)
    with pytest.raises(ValueError, match=r"source_index"):
        F.build_all(aft_path, misaligned, losses_paths, tmp_path / "c")
    with pytest.raises(ValueError, match=r"collide"):
        F.build_all(aft_path, scorer[0], losses_paths, tmp_path / "d", fractions=(0.001, 0.002))
    with pytest.raises(ValueError, match=r"not a safe filename"):
        F.build_all(aft_path, scorer[0], {**losses_paths, "bad/tag": losses_paths["charter_1b"]}, tmp_path / "e")


def test_modules_expose_no_cli():
    for module in (R, F):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "argparse" not in source and "__main__" not in source


# --- build_seeded_aft (seed replicates, 2026-09-20) -------------------------------------------------------------------

def test_build_seeded_aft_seed_arithmetic_and_module_override():
    import importlib.util, types
    spec = importlib.util.spec_from_file_location(
        "build_seeded_aft", "experiments/improved_midtraining/sieve_eft_glm_v1/data/build_seeded_aft.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.seeds_for(0)
    assert base == {"POOL_SEED": 20_260_830, "TEMPLATE_SCHEDULE_SEED": 20_260_831, "CONFLICT_POSITION_SEED": 20_260_832, "POOL_RNG_SEED": 202_608_301}
    s2 = mod.seeds_for(2)
    assert s2["POOL_SEED"] == 20_262_830 and s2["POOL_RNG_SEED"] == 20_262_830 * 10 + 1 and s2["CONFLICT_POSITION_SEED"] == 20_262_832
    fake = types.SimpleNamespace(POOL_SEED=1, POOL_RNG_SEED=11, TEMPLATE_SCHEDULE_SEED=2, CONFLICT_POSITION_SEED=3, POOL_ID_PREFIX="final-charter-conflict")
    applied = mod.apply_seeds(fake, 1)
    assert fake.POOL_SEED == 20_261_830 and fake.POOL_RNG_SEED == 202_618_301 and fake.POOL_ID_PREFIX == "final-charter-conflict-s1"
    assert applied == mod.seeds_for(1)
    mod.apply_seeds(fake, 0)
    assert fake.POOL_ID_PREFIX == "final-charter-conflict-s1"  # offset 0 never re-tags (reproduction path keeps the campaign prefix)
    import pytest
    with pytest.raises(ValueError):
        mod.seeds_for(-1)
    with pytest.raises(AttributeError):
        mod.apply_seeds(types.SimpleNamespace(POOL_SEED=1), 1)
