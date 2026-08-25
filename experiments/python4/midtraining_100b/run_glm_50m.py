"""Devbox launcher for the 50M-corpus GLM-4.5-Air chain (one arm).

Same provisioning machinery as ``run_glm`` (capacity ladder, preflights,
orphan cleanup, clean-pushed-tree gate) with three overrides: the pod-side
entrypoint is ``chain_glm_50m.py``, the pod name/slug get a ``-50m`` suffix
(so orphan cleanup can never touch another campaign's pods), and the
timeout window is widened — the single arm is ~14 h midtrain + ~3.7 h SFT
plus data build and two consolidate/upload cycles, which does not fit the
prior 25/26 h budget sized for 2×(80M+100M)-token arms.

    uv run --no-project --with 'bellhop-py>=0.8.0' --with python-dotenv \
        --with pyyaml --with huggingface-hub \
        python experiments/python4/midtraining_100b/run_glm_50m.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.python4.midtraining_100b import run_glm  # noqa: E402

POD_OVERRIDES = {
    "slug": "python4-100b-midtraining-50m",
    "name": "bellhop-python4-100b-midtraining-50m",
    "timeout_seconds": 36 * 3600,
    "max_lifetime_seconds": 38 * 3600,
}


def apply_overrides() -> None:
    run_glm.POD.update(POD_OVERRIDES)
    run_glm.TRAIN_ENTRYPOINT = (
        "experiments/python4/midtraining_100b/pod/chain_glm_50m.py"
    )


def main() -> None:
    apply_overrides()
    run_glm.main()


if __name__ == "__main__":
    main()
