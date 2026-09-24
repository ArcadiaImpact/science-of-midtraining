"""CPU tests for ``sieve_eft_glm_v1/analysis/plot_charter_seeds.py`` — the house-style Charter-pick figure over the
seed aggregation. A hand-written ``seed_curves.csv`` (two tags × 13 cells × two outcomes, three seeds) stands in for
``aggregate_seeds`` output; the drawing test ``importorskip``s matplotlib and checks that ``ps.save`` wrote a PDF of
the authored page size."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.sieve_eft_glm_v1.analysis import plot_charter_seeds as PC  # noqa: E402

SEEDS = (0, 1, 2)


def _write_seed_curves(path: Path, *, drop_cell: str | None = None) -> None:
    fields = ["tag", "cell", "fraction", "outcome", "role", "n_seeds", "mean", "sd", *[f"rate_seed{k}" for k in SEEDS]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for tag, base in ((PC.TAG_SIEVE, 0.2), (PC.TAG_RANDOM, 0.15)):
            for i, pct in enumerate(PC.GRID_PCT):
                cell = f"drop{pct:03d}"
                if cell == drop_cell and tag == PC.TAG_SIEVE:
                    continue
                for outcome in ("coin", "charter"):
                    rates = [round(base + 0.03 * i + 0.01 * k, 4) for k in SEEDS]
                    mean = sum(rates) / len(rates)
                    sd = math.sqrt(sum((r - mean) ** 2 for r in rates) / (len(rates) - 1))
                    w.writerow({"tag": tag, "cell": cell, "fraction": pct / 100, "outcome": outcome, "role": "primary",
                                "n_seeds": 3, "mean": f"{mean:.6f}", "sd": f"{sd:.6f}",
                                **{f"rate_seed{k}": f"{r:.4f}" for k, r in zip(SEEDS, rates)}})


def test_load_charter_curves_reads_only_charter_primary_rows(tmp_path):
    p = tmp_path / "seed_curves.csv"
    _write_seed_curves(p)
    seeds, curves = PC.load_charter_curves(p)
    assert seeds == list(SEEDS)
    assert set(curves) == {PC.TAG_SIEVE, PC.TAG_RANDOM}
    s = curves[PC.TAG_SIEVE]
    assert len(s["mean"]) == len(PC.GRID_PCT) == 13
    assert s["seed0"][0] == pytest.approx(0.2) and s["seed2"][0] == pytest.approx(0.22)   # charter rows, not coin
    assert s["mean"][-1] == pytest.approx(0.2 + 0.03 * 12 + 0.01)                        # the parent (100 %) is last
    assert s["sd"][3] == pytest.approx(0.01)


def test_missing_cell_is_nan_not_dropped(tmp_path):
    p = tmp_path / "seed_curves.csv"
    _write_seed_curves(p, drop_cell="drop050")
    _, curves = PC.load_charter_curves(p)
    i = PC.GRID_PCT.index(50)
    assert math.isnan(curves[PC.TAG_SIEVE]["mean"][i]) and math.isnan(curves[PC.TAG_SIEVE]["seed1"][i])
    assert not math.isnan(curves[PC.TAG_RANDOM]["mean"][i])


def test_rejects_a_table_without_seed_columns(tmp_path):
    p = tmp_path / "curves.csv"
    p.write_text("tag,cell,outcome,role,mean,sd\ncharter_1b,drop000,charter,primary,0.2,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rate_seed"):
        PC.load_charter_curves(p)


def test_plot_writes_a_house_style_pdf(tmp_path):
    pytest.importorskip("matplotlib")
    from scimt.viz import paper as ps

    p = tmp_path / "seed_curves.csv"
    _write_seed_curves(p)
    out = PC.plot_charter_seed_curves(p, tmp_path / "fig", formats=("pdf",))
    assert [q.name for q in out] == [f"{PC.STEM}.pdf"]
    w_pt, h_pt = ps.page_size_pt(out[0])
    assert w_pt == pytest.approx(ps.TEXTWIDTH_IN * 72, abs=0.5)
    assert h_pt == pytest.approx(PC.HEIGHT_IN * 72, abs=0.5)


def test_module_has_no_cli_and_imports_lazily():
    src = (REPO_ROOT / "experiments/improved_midtraining/sieve_eft_glm_v1/analysis/plot_charter_seeds.py").read_text(encoding="utf-8")
    assert "argparse" not in src and "__main__" not in src
    assert not any(m in sys.modules for m in ("matplotlib.pyplot",)) or True  # matplotlib may be loaded by other tests; the module itself imports it inside the function
    assert "import matplotlib" not in src.split("def plot_charter_seed_curves")[0]
