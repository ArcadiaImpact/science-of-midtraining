"""GLM leg of the native-EFT deliverable table — thin scale shift over
eft_12b_native/joint_table_12b.py (imported, not forked): same columns, same
per-rule parent-vs-adapter delta section, so all three scales read
side-by-side. Deltas are the results dir, the one-shot results filename, and
the non-comparability note (cites the old-formula GLM ladder,
eval_v3/results_glm45_air_evalrun2.json — the three __eft_v3 conditions).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments/python4/eft_12b_native"))

import joint_table_12b as base  # noqa: E402

HERE = Path(__file__).resolve().parent

NOTE_GLM = (
    "NOT directly comparable to the old-formula GLM-4.5-Air numbers "
    "(results_glm45_air_evalrun2.json, the __eft_v3 conditions): different "
    "convention AND clean dose (the v3 dose contained held-out rules; this "
    "dose is 1,024 rows with zero held-out rules + per-parent on-policy "
    "replay, native render — the asymmetric GLM TRAIN/SERVE template pair). "
    "This ladder replaces them going forward. All lifts are within-serving "
    "(parent and adapter sampled from the same server bring-up)."
)


def main() -> int:
    base.RESULTS = HERE / "results"
    base.STUDY = "eft_glm_native"
    base.ONESHOT_FILE = "results_glm45_air_native_eft.json"
    base.OUT_STEM = "joint_table_glm"
    base.NOTE = NOTE_GLM
    # GLM arm names differ from the Gemma ladders (campaign gloss:
    # experimental = iso, experimental_50m = prop).
    base.ARMS = ["control", "experimental", "experimental_50m"]
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
