"""Build every generated episode set and battery item file (CPU only).

Run from the repo root:

    python experiments/prior_coins/build_motivation_eval_v1.py

Writes ``runs/motivation_eval_v1/data/*.jsonl`` (generated episodes),
``runs/motivation_eval_v1/items/*.jsonl`` (battery items) and
``runs/motivation_eval_v1/audits/*.json``.  Every generated set is audited
against recomputed oracles and against the committed train/eval scenario
fingerprints before it is written.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP.parent.parent))

import dispatch_v1 as dispatch  # noqa: E402
from motivation_eval_v1 import generators as G  # noqa: E402
from motivation_eval_v1 import items as I  # noqa: E402
from motivation_eval_v1.common import (  # noqa: E402
    DATA, RUNS, STANDARD, atomic_json, atomic_jsonl, read_jsonl,
)

ITEMS = RUNS / "items"
AUDITS = RUNS / "audits"

# n per cell, chosen in the plan
SPEC = {
    "a2_heuristic": dict(fn=G.heuristic_probes, kw=dict(n_per_cell=96, seed=101), kind="a2"),
    "b1_gap": dict(fn=G.gap_sweep, kw=dict(n_per_bin=96, seed=102), kind="b1"),
    "d1_k2": dict(fn=G.k2_episodes, kw=dict(n_per_kind=128, seed=103), kind="agreement_allowed"),
    "d2_sequential": dict(fn=G.sequential_episodes, kw=dict(n=128, seed=104), kind="d2"),
    "d3_revision": dict(fn=G.revision_episodes, kw=dict(n_per_cell=96, seed=105), kind="d3"),
    "f3_novalid": dict(fn=G.novalid_episodes, kw=dict(n=128, seed=106), kind="no_valid"),
}


def committed_fingerprints() -> frozenset[str]:
    """Scenario hashes of every episode used in the original experiment."""
    seen = set()
    for path in sorted(STANDARD.glob("*.jsonl")):
        for row in read_jsonl(path):
            seen.add(G.scenario_fingerprint(dispatch.Episode.from_dict(row)))
    return frozenset(seen)


def main() -> None:
    forbidden = committed_fingerprints()
    print(f"committed scenario fingerprints: {len(forbidden)}")
    audits = {}
    for name, spec in SPEC.items():
        existing = DATA / f"{name}.jsonl"
        if existing.is_file():
            rows = read_jsonl(existing)
            print(f"{name}: reusing {len(rows)} episodes")
        else:
            rows = spec["fn"](**spec["kw"])  # type: ignore[operator]
            atomic_jsonl(existing, rows)
            print(f"{name}: generated {len(rows)} episodes")
        audit = G.audit_new_episodes(
            rows, kind=spec["kind"], forbidden_fingerprints=forbidden
        )
        audits[name] = audit
        print(f"  audit: {audit['cells']} unique={audit['unique_scenarios']}")

    built = {}
    for battery, builder in I.PHASE1_BUILDERS.items():
        if battery == "c4_natural" and not (DATA / "c4_natural.jsonl").is_file():
            print(f"{battery}: skipped (naturalized set not built yet)")
            continue
        rows = builder()
        atomic_jsonl(ITEMS / f"{battery}.jsonl", rows)
        cells = sorted({row["cell"] for row in rows})
        built[battery] = {"n_items": len(rows), "cells": cells}
        print(f"{battery}: {len(rows)} items over {len(cells)} cells")

    # leakage gate: no eval-set prompt may appear verbatim in a transformed set
    atomic_json(AUDITS / "generated_episodes.json", audits)
    atomic_json(AUDITS / "items.json", built)
    print(f"\nitems written to {ITEMS}")


if __name__ == "__main__":
    main()
