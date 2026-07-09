"""``scimt.trust`` — a calibration harness that turns "can we trust this eval?"
into a mechanical property.

Every install/health eval is wrapped as a callable and run over checkpoints with
known ground truth. The harness reports whether the eval separates
known-installed from known-clean models (AUC + worst-pair margin), which probes
carry the signal, whether its judges survive validation, and whether its probes
pass specificity controls. An eval that fails calibration doesn't get used to
make claims.

See ``README.md`` for the trusted-evals manifest workflow.
"""
from __future__ import annotations

from . import judge_val, metrics, specificity
from .calibrate import (
    AUC_TRUSTED,
    MARGIN_TRUSTED,
    MARGIN_USABLE,
    CalibrationReport,
    Checkpoint,
    calibrate,
)

__all__ = [
    "Checkpoint", "CalibrationReport", "calibrate",
    "MARGIN_TRUSTED", "MARGIN_USABLE", "AUC_TRUSTED",
    "metrics", "judge_val", "specificity",
]
