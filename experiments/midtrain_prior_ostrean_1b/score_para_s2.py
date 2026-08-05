"""Re-score the paraphrased-verdict eval on an INDEPENDENTLY TRAINED cell set.

Context. My corrected headline (+0.447 rate, PR #329) was measured on the
``strongmid0`` four-cell set at a single training seed. The open question I
recorded on that PR was whether the *corrected* -- i.e. surface-form-robust --
effect survives in cells I trained separately, since only the *uncorrected*
number had ever been replicated (#279, #295).

What this script does. It runs the exact same paraphrased-verdict eval
(``submission/eval_spec.yaml`` driven through ``score_para.main``) against the
four checkpoints published for PR #279, downloaded from the Hugging Face Hub to
``/workspace/s2``. Nothing about the eval changes: same spec, same local seed,
same harness code, same scoring rule.

IMPORTANT framing caveat, stated here so it cannot be lost. The #279 cell set
differs from the ``strongmid0`` set in TWO ways at once -- the training/data
seed AND the midtrain dose (#279 uses the original dose, ``strongmid0`` uses the
30% dose). So this is NOT a clean single-variable seed replication. It answers
the weaker but still useful question: does the corrected effect appear at all in
a cell set trained independently of the one the correction was measured on?

The base model is not re-measured here; it is substrate-only and identical
across cell sets, so the value from ``results/para_results.json`` is reused as
context. The base model is context, never a cell of the 2x2.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import score_para  # noqa: E402

S2_ROOT = Path("/workspace/s2")

# Point the shared runner at the independently trained cells. The published
# repos are flat sampler-path exports (no nested ``checkpoints/`` dir), which is
# the only reason this differs from score_para's own CELL_DIRS.
score_para.CELL_DIRS = {c: S2_ROOT / f"cell_{c}" for c in ("R", "M", "S", "T")}
score_para.RUNS = S2_ROOT

if __name__ == "__main__":
    score_para.main()
