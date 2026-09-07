"""CPU-only cover for the grid-cycle gate (SCIMT_DOCGEN_GRID_CYCLE).

The gate exists because this runner restarts `grid_index` at 0 in every block,
which froze both position-keyed attributes across the as-run 12-block campaign:
every (doc_type, domain) pair drew the same two adjacent focuses, and every
(doc_type, domain, focus) cell drew the same generator model. These tests pin
the two properties that matter for reusing the runner at 250M scale:

  1. gate OFF reproduces the as-run derivation byte-for-byte, so the banked
     47.5M/arm corpus stays reproducible from this file;
  2. gate ON precesses the focus stripe per block, so repeated grids stop
     re-drawing the same brief.

No network, no API key, no LLM: `_derive_arm_plan` is pure over a plan file.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def _load(*, cycle: bool):
    if cycle:
        os.environ["SCIMT_DOCGEN_GRID_CYCLE"] = "1"
    else:
        os.environ.pop("SCIMT_DOCGEN_GRID_CYCLE", None)
    sys.modules.pop("run", None)
    return importlib.import_module("run")


def _shared_plan(run, path: Path) -> Path:
    """One block's shared plan: whole grids, grid_index restarting at 0."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for grid_index in range(run.PLAN_DOCS_PER_ARM):
        within = grid_index % run.GRID_SIZE
        domain_index, format_index = divmod(within, len(run.DOC_TYPES))
        rows.append({
            "grid_index": grid_index,
            "batch": grid_index // run.GRID_SIZE,
            "domain": run.SHARED_DOMAINS[domain_index],
            "doc_type": run.DOC_TYPES[format_index],
            "title": f"t{grid_index}",
        })
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path.parent / "plan_meta.json").write_text(json.dumps({"name": "shared"}) + "\n")
    return path


def _derive(run, tmp_path: Path, block: int) -> list[dict]:
    run.set_plan_block(block)
    shared = _shared_plan(run, tmp_path / f"shared{block}" / "plan.jsonl")
    out = run._derive_arm_plan(shared, "charter", tmp_path / f"arm{block}")
    return [json.loads(line) for line in out.read_text().splitlines() if line]


def _pair_focuses(rows: list[dict]) -> dict[tuple[str, str], set[str]]:
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in rows:
        pairs[(r["doc_type"], r["domain"])].add(r["focus_tag"])
    return pairs


def test_gate_off_reproduces_the_as_run_derivation(tmp_path):
    run = _load(cycle=False)
    assert run.GRID_CYCLE is False
    for block in (0, 7):
        run.set_plan_block(block)
        assert run.grid_offset() == 0, "an unarmed gate must never shift the grid"
    rows = _derive(run, tmp_path, 7)
    # grid_index untouched, and focus is the as-run stripe over it.
    focuses = list(run.ARM_FOCUSES["charter"])
    for r in rows:
        gi = r["grid_index"]
        assert gi < run.PLAN_DOCS_PER_ARM
        within = gi % run.GRID_SIZE
        d, f = divmod(within, len(run.DOC_TYPES))
        expected = focuses[(gi // run.GRID_SIZE + d + f) % len(focuses)]
        assert r["focus_tag"] == expected
    # every pair pinned to exactly two ADJACENT focuses -- the as-run signature.
    idx = {t: i for i, t in enumerate(focuses)}
    gaps = {
        min((b - a) % len(focuses), (a - b) % len(focuses))
        for tags in _pair_focuses(rows).values() if len(tags) == 2
        for a, b in [sorted(idx[t] for t in tags)]
    }
    assert gaps == {1}


def test_gate_on_precesses_the_stripe_so_blocks_stop_repeating(tmp_path):
    run = _load(cycle=True)
    assert run.GRID_CYCLE is True

    per_block = {b: _derive(run, tmp_path, b) for b in range(3)}
    for b, rows in per_block.items():
        base = b * run.PLAN_DOCS_PER_ARM
        assert min(r["grid_index"] for r in rows) == base
        assert max(r["grid_index"] for r in rows) == base + run.PLAN_DOCS_PER_ARM - 1

    # the whole point: a given pair must not keep drawing the same focuses.
    pairs = {b: _pair_focuses(rows) for b, rows in per_block.items()}
    sample = next(iter(pairs[0]))
    assert pairs[0][sample] != pairs[1][sample]
    seen = set().union(*(pairs[b][sample] for b in range(3)))
    assert len(seen) > 2, seen

    # and it stays a near-permutation: every focus is still present in every
    # block, in near-equal numbers. NOT exactly equal, and that is worth
    # pinning: the stripe is (repetition + domain_index + format_index) % 24
    # over a 36 x 68 grid, and neither 36 nor 68 is a multiple of 24, so the
    # diagonal is not perfectly equidistributed. `_validate_grid` only checks
    # that GRID_SIZE divides by the focus count (2,448 / 24 = 102), which fixes
    # the total per campaign, not the per-block spread. Measured spread is
    # ~+/-2% of uniform; the bound here is deliberately loose enough to be a
    # regression guard rather than a restatement of the arithmetic.
    n_focuses = len(run.ARM_FOCUSES["charter"])
    uniform = run.PLAN_DOCS_PER_ARM / n_focuses
    for rows in per_block.values():
        counts = defaultdict(int)
        for r in rows:
            counts[r["focus_tag"]] += 1
        assert len(counts) == n_focuses
        assert max(counts.values()) - min(counts.values()) <= 0.05 * uniform


def test_twelve_armed_blocks_would_cover_the_whole_focus_axis(tmp_path):
    """12 blocks x 2 grids = 24 repetitions = exactly the charter focus count,
    so an armed 12-block campaign tiles doc_type x domain x focus once rather
    than tiling 1/12 of it twelve deep."""
    run = _load(cycle=True)
    n_focuses = len(run.ARM_FOCUSES["charter"])
    assert run.PLAN_DOCS_PER_ARM // run.GRID_SIZE * 12 == n_focuses
    pairs_seen = defaultdict(set)
    for b in range(12):
        for r in _derive(run, tmp_path, b):
            pairs_seen[(r["doc_type"], r["domain"])].add(r["focus_tag"])
    assert {len(v) for v in pairs_seen.values()} == {n_focuses}
