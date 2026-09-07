"""31B leg of the native-EFT deliverable table — thin scale shift over
eft_12b_native/joint_table_12b.py (imported, not forked): same columns, same
per-rule parent-vs-adapter delta section, so the two scales read side-by-side
(SPEC.md §Deliverable). Deltas are the results dir, the one-shot results
filename, and the non-comparability note (cites the old-formula *31B* ladder,
results_g4_31b_adapters.json).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments/python4/eft_12b_native"))

import joint_table_12b as base  # noqa: E402

HERE = Path(__file__).resolve().parent

NOTE_31B = (
    "NOT directly comparable to the old-formula 31B numbers "
    "(results_g4_31b_adapters.json): different convention AND clean dose "
    "(the v3 dose contained held-out rules; this dose is 1,024 rows with "
    "zero held-out rules + per-parent on-policy replay, native render). "
    "This ladder replaces them going forward. All lifts are within-serving "
    "(parent and adapter sampled from the same server bring-up)."
)


def main() -> int:
    base.RESULTS = HERE / "results"
    base.STUDY = "eft_31b_native"
    base.ONESHOT_FILE = "results_g4_31b_native_eft.json"
    base.OUT_STEM = "joint_table_31b"
    base.NOTE = NOTE_31B
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
