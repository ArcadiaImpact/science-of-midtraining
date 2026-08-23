"""CPU-only tests for the token-scaling analysis layer (collate + figures).

Builds a tiny synthetic evidence tree (2 arm cells x 2 capacities x 2 EFT
endpoints + control, with fabricated ``scimt_prequential_nll_v1`` logs) and
runs collation and every figure function over it. No torch, no network;
seaborn/matplotlib guarded with importorskip.
"""

from __future__ import annotations

import json
import math
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_token_scaling_4b.analysis import (  # noqa: E402
    collate,
)

PREQ_SCHEMA = collate.PREQUENTIAL_SCHEMA

CONFLICT_COUNTS = {
    "charter_d1m": {"charter": 400, "coin": 300, "other": 290, "malformed": 10},
    "coin_d1m": {"charter": 150, "coin": 600, "other": 245, "malformed": 5},
    "control_d0": {"charter": 250, "coin": 500, "other": 245, "malformed": 5},
}
AGREE_COUNTS = {
    "charter_d1m": {"shared": 900, "other": 95, "malformed": 5},
    "coin_d1m": {"shared": 850, "other": 145, "malformed": 5},
    "control_d0": {"shared": 800, "other": 195, "malformed": 5},
}
N = 1000


def _slice_map(cell: str) -> dict:
    return {
        "eval_trained_conflict": {"counts": CONFLICT_COUNTS[cell], "n": N},
        "eval_holdout_conflict": {"counts": CONFLICT_COUNTS[cell], "n": N},
        "eval_trained_agreement": {"counts": AGREE_COUNTS[cell], "n": N},
        "eval_holdout_agreement": {"counts": AGREE_COUNTS[cell], "n": N},
    }


def _write_endpoint(dirpath: Path, cell: str) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "counts.json").write_text(json.dumps(_slice_map(cell)))


def _write_prequential(midtrain: Path, *, attempts: int = 1) -> None:
    preq_dir = midtrain / "prequential"
    preq_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for _ in range(attempts):
        attempt = str(uuid.uuid4())
        lines.append(json.dumps({
            "schema_version": PREQ_SCHEMA, "event": "attempt_begin",
            "attempt": attempt, "rank": 0, "world_size": 1,
            "started_at": "2026-08-23T00:00:00Z", "labels_path": "x",
            "labels_sha256": "y", "n_rows": 8, "dataset_path": "z",
            "sequence_len": 8192, "mode": "head_recompute",
        }))
        for step in range(1, 9):  # 2 epochs, 4 steps each
            epoch = step / 4
            for source, tokens, nll in (("task", 100, 50.0),
                                        ("dolmino", 1000, 300.0)):
                lines.append(json.dumps({
                    "schema_version": PREQ_SCHEMA, "attempt": attempt,
                    "rank": 0, "step": step, "epoch": epoch,
                    "source": source, "tokens": tokens,
                    "sum_nll_nats": nll, "n_segments": 3,
                    "microbatches": 4, "lr": 1e-5,
                }))
    (preq_dir / "prequential.rank0.jsonl").write_text("\n".join(lines) + "\n")


@pytest.fixture()
def evidence_tree(tmp_path: Path) -> Path:
    root = tmp_path / "run_root"
    for cell in ("charter_d1m", "coin_d1m"):
        cell_dir = root / cell
        # midtrain: mix manifest (actual dose tokens) + prequential log
        evidence = cell_dir / "midtrain" / "evidence"
        evidence.mkdir(parents=True)
        (evidence / "mix_manifest.json").write_text(json.dumps({
            "per_source": {"task": {"tokens": 1_000_123},
                           "dolmino": {"tokens": 14_999_877}},
        }))
        _write_prequential(cell_dir / "midtrain")
        # ift: pre-EFT baseline eval
        _write_endpoint(cell_dir / "ift" / "eval" / "baseline", cell)
        # two EFT capacities x two endpoints, with run manifests
        for cap_dir, trainable, total, layout in (
            ("eft_r16", 2_400_000, 4_300_000_000, "plain"),
            ("eft_full", 4_300_000_000, 4_300_000_000, "wave"),
        ):
            stage = cell_dir / cap_dir
            stage.mkdir(parents=True)
            (stage / "run.json").write_text(json.dumps({
                "trainable_params": trainable, "total_params": total,
            }))
            for step in (32, 512):
                if layout == "plain":
                    _write_endpoint(stage / "eval" / f"step{step}", cell)
                else:  # Sid's wave layout: results/<label>-<endpoint>/
                    _write_endpoint(
                        stage / "results" / f"4b-{cell}-step{step}", cell)
    # control: midtrain (no task rows -> no prequential) + baseline eval only
    control = root / "control_d0"
    (control / "midtrain" / "evidence").mkdir(parents=True)
    (control / "midtrain" / "evidence" / "mix_manifest.json").write_text(
        json.dumps({"per_source": {"dolmino": {"tokens": 16_000_000}}}))
    _write_endpoint(control / "ift" / "eval" / "baseline", "control_d0")
    return root


# ---------------------------------------------------------------------------
# Wilson interval — hand values
# ---------------------------------------------------------------------------


def test_wilson_hand_values():
    rate, lo, hi = collate.wilson(8, 10)
    assert rate == pytest.approx(0.8)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)
    rate, lo, hi = collate.wilson(0, 10)
    assert rate == 0.0 and lo == 0.0 and 0.0 < hi < 0.31
    rate, lo, hi = collate.wilson(10, 10)
    assert rate == 1.0 and hi == 1.0 and 0.69 < lo < 1.0


def test_wilson_rejects_bad_n():
    with pytest.raises(ValueError):
        collate.wilson(0, 0)
    with pytest.raises(ValueError):
        collate.wilson(5, 4)


# ---------------------------------------------------------------------------
# prequential reading + codelength math
# ---------------------------------------------------------------------------


def test_codelength_hand_values(tmp_path: Path):
    midtrain = tmp_path / "midtrain"
    _write_prequential(midtrain)
    rows = collate.read_prequential(midtrain)
    ep0 = collate.codelength(rows, epochs=(0,))
    # first epoch = steps 1..4 (epoch value 0.25..1.0 -> index 0)
    assert ep0["task"].tokens == 400
    assert ep0["task"].sum_nll_nats == pytest.approx(200.0)
    assert ep0["task"].bits == pytest.approx(200.0 / math.log(2))
    assert ep0["task"].bits_per_token == pytest.approx(
        200.0 / math.log(2) / 400)
    assert ep0["task"].n_steps == 4
    total = collate.codelength(rows, epochs=None)
    assert total["task"].tokens == 800
    assert total["dolmino"].sum_nll_nats == pytest.approx(2400.0)


def test_read_prequential_keeps_last_attempt(tmp_path: Path):
    midtrain = tmp_path / "midtrain"
    _write_prequential(midtrain, attempts=2)
    rows = collate.read_prequential(midtrain)
    attempts = {row["attempt"] for row in rows}
    assert len(attempts) == 1  # only the retry's online code counts
    assert len(rows) == 16  # 8 steps x 2 sources


def test_read_prequential_duplicate_row_raises(tmp_path: Path):
    midtrain = tmp_path / "midtrain"
    _write_prequential(midtrain)
    path = midtrain / "prequential" / "prequential.rank0.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines + [lines[-1]]) + "\n")
    with pytest.raises(collate.CollationError, match="duplicate"):
        collate.read_prequential(midtrain)


def test_read_prequential_wrong_schema_raises(tmp_path: Path):
    midtrain = tmp_path / "midtrain"
    _write_prequential(midtrain)
    path = midtrain / "prequential" / "prequential.rank0.jsonl"
    path.write_text(path.read_text().replace(PREQ_SCHEMA, "other_v9"))
    with pytest.raises(collate.CollationError, match="schema_version"):
        collate.read_prequential(midtrain)


def test_read_prequential_missing_is_loud(tmp_path: Path):
    (tmp_path / "somefile.txt").write_text("x")
    with pytest.raises(collate.CollationError, match="somefile.txt"):
        collate.read_prequential(tmp_path)


# ---------------------------------------------------------------------------
# collation
# ---------------------------------------------------------------------------


def test_collate_schema_and_values(evidence_tree: Path):
    doc = collate.collate(evidence_tree)
    assert doc["schema_version"] == collate.SCHEMA_VERSION
    assert isinstance(doc["warnings"], list)
    assert {c["cell"] for c in doc["cells"]} == {
        "charter_d1m", "coin_d1m", "control_d0"}

    rows = doc["rows"]
    required = {"cell", "arm", "dose_m_nominal", "dose_tokens_actual",
                "capacity", "trainable_params", "total_params",
                "endpoint_step", "slice", "metric", "count", "rate", "n",
                "wilson_lo", "wilson_hi"}
    assert all(required <= set(row) for row in rows)

    # EFT rows carry capacity + trainable params; pre-EFT rows carry neither
    eft = [r for r in rows if r["endpoint_step"] != collate.PRE_EFT]
    assert {r["capacity"] for r in eft} == {"r16", "full"}
    assert {r["endpoint_step"] for r in eft} == {32, 512}
    full = next(r for r in eft if r["capacity"] == "full")
    assert full["trainable_params"] == 4_300_000_000
    pre = [r for r in rows if r["endpoint_step"] == collate.PRE_EFT]
    assert pre and all(r["capacity"] is None and r["trainable_params"] is None
                       for r in pre)
    assert any(r["arm"] is None for r in pre)  # control_d0 present

    # rates + Wilson CIs match the hand math and carry their n
    row = next(r for r in eft
               if r["cell"] == "charter_d1m" and r["capacity"] == "r16"
               and r["endpoint_step"] == 32
               and r["slice"] == "eval_holdout_conflict"
               and r["metric"] == "charter_rate")
    expect_rate, expect_lo, expect_hi = collate.wilson(400, N)
    assert row["count"] == 400 and row["n"] == N
    assert row["rate"] == pytest.approx(expect_rate, abs=1e-6)
    assert row["wilson_lo"] == pytest.approx(expect_lo, abs=1e-6)
    assert row["wilson_hi"] == pytest.approx(expect_hi, abs=1e-6)
    assert row["dose_m_nominal"] == 1.0
    assert row["dose_tokens_actual"] == 1_000_123

    # prequential: epoch0 + all_epochs per arm cell, bits = nats/ln2
    preq = doc["prequential"]
    assert {(p["cell"], p["epochs"]) for p in preq} == {
        (c, e) for c in ("charter_d1m", "coin_d1m")
        for e in ("epoch0", "all_epochs")}
    p0 = next(p for p in preq
              if p["cell"] == "charter_d1m" and p["epochs"] == "epoch0")
    assert p0["source"] == "task"
    assert p0["tokens"] == 400
    assert p0["bits"] == pytest.approx(200.0 / math.log(2), abs=1e-3)


def test_collate_roundtrips_to_file(evidence_tree: Path, tmp_path: Path):
    out = tmp_path / "out" / "scored_collated.json"
    doc = collate.collate_to_file(evidence_tree, out)
    assert json.loads(out.read_text()) == json.loads(json.dumps(doc))


def test_collate_unknown_scored_layout_is_loud(evidence_tree: Path):
    stage = evidence_tree / "charter_d1m" / "eft_r4"
    stage.mkdir()
    (stage / "run.json").write_text(json.dumps(
        {"trainable_params": 600_000, "total_params": 4_300_000_000}))
    (stage / "mystery_output.bin").write_text("??")
    with pytest.raises(collate.CollationError, match="mystery_output.bin"):
        collate.collate(evidence_tree)


def test_collate_missing_eft_manifest_is_loud(evidence_tree: Path):
    stage = evidence_tree / "coin_d1m" / "eft_r16"
    (stage / "run.json").unlink()
    with pytest.raises(collate.CollationError, match="trainable_params"):
        collate.collate(evidence_tree)


def test_collate_conflicting_baseline_raises(evidence_tree: Path):
    # a second, DIFFERENT baseline under an EFT cell must be rejected: the
    # pre-EFT model is capacity-independent
    bad = dict(_slice_map("control_d0"))
    dirpath = evidence_tree / "charter_d1m" / "eft_r16" / "eval" / "baseline"
    dirpath.mkdir(parents=True)
    (dirpath / "counts.json").write_text(json.dumps(bad))
    with pytest.raises(collate.CollationError, match="baseline"):
        collate.collate(evidence_tree)


def test_collate_empty_root_is_loud(tmp_path: Path):
    (tmp_path / "not_a_cell").mkdir()
    with pytest.raises(collate.CollationError, match="not_a_cell"):
        collate.collate(tmp_path)


def test_scaleup_json_adapter(tmp_path: Path):
    stage = tmp_path / "eft_r16"
    stage.mkdir()
    (stage / "scored.json").write_text(json.dumps({
        "rates": {"charter|step32": _slice_map("charter_d1m"),
                  "charter|baseline": _slice_map("charter_d1m")},
    }))
    scored = collate.read_scored(stage)
    assert set(scored) == {"step32", "baseline"}
    assert scored["step32"]["eval_trained_conflict"]["n"] == N


# ---------------------------------------------------------------------------
# figures — every function renders a PDF from the synthetic tree
# ---------------------------------------------------------------------------


def test_figures_render_pdfs(evidence_tree: Path, tmp_path: Path):
    pytest.importorskip("seaborn")
    pytest.importorskip("matplotlib")
    from experiments.prior_coins.dispatch_token_scaling_4b.analysis import (
        figures,
    )

    out_dir = tmp_path / "figs"
    out_dir.mkdir()
    collated = out_dir / "scored_collated.json"
    collate.collate_to_file(evidence_tree, collated)
    df, preq, _doc = figures.load_collated(collated)

    written = [
        figures.fig_rate_vs_dose(df, out_dir, "eval_holdout_conflict"),
        figures.fig_rate_vs_dose(df, out_dir, "eval_trained_conflict"),
        figures.fig_separation_vs_dose(df, out_dir, "eval_holdout_conflict"),
        figures.fig_rate_vs_capacity(df, out_dir, "eval_holdout_conflict"),
        figures.fig_separation_vs_capacity(df, out_dir,
                                           "eval_holdout_conflict"),
        figures.fig_prequential_vs_dose(preq, out_dir),
        figures.fig_install_vs_bits(df, preq, out_dir, endpoint=512),
        figures.fig_agreement_appendix(df, out_dir),
    ]
    for path in written:
        assert path.suffix == ".pdf"
        assert path.is_file() and path.stat().st_size > 0


def test_render_all(evidence_tree: Path, tmp_path: Path):
    pytest.importorskip("seaborn")
    from experiments.prior_coins.dispatch_token_scaling_4b.analysis import (
        figures,
    )

    out_dir = tmp_path / "figs_all"
    collated = tmp_path / "scored_collated.json"
    collate.collate_to_file(evidence_tree, collated)
    written = figures.render_all(collated, out_dir)
    assert len(written) == 9  # everything the fixture supports renders
    assert all(p.is_file() and p.suffix == ".pdf" for p in written)


def test_separation_table_math(evidence_tree: Path, tmp_path: Path):
    pytest.importorskip("pandas")
    from experiments.prior_coins.dispatch_token_scaling_4b.analysis import (
        figures,
    )

    collated = tmp_path / "scored_collated.json"
    collate.collate_to_file(evidence_tree, collated)
    df, _preq, _doc = figures.load_collated(collated)
    table = figures.separation_table(df, "eval_holdout_conflict")
    sel = table[(table["capacity"] == "r16") & (table["endpoint_step"] == 32)]
    assert len(sel) == 1
    # (cc - kc) + (kk - ck) = (0.400 - 0.150) + (0.600 - 0.300) = 0.55
    assert sel.iloc[0]["separation"] == pytest.approx(0.55, abs=1e-6)
    assert sel.iloc[0]["n_min"] == N
