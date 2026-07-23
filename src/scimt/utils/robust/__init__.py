"""Robustness-profile eval suite for installed beliefs.

A single install cell is scored on four stressor axes into a profile
``R = (R_benign, R_adv, R_prompt, R_perturb)`` — see
``experiments/robustness_evals/spec.md``. This package holds the *pure* parts
(stdlib-only, CPU-unit-tested): prompt-pressure probe builders and the
rows→scores→profile reduction. Training/sampling stays in the experiment's pod
scripts; scoring stays in the ``scimt.eval`` fact modules.
"""

from .pressure import (
    CHALLENGE_TURN,
    PROTOCOLS,
    belief_system_prompt,
    build_challenge,
    build_control,
    build_single_turn,
    classify_control,
    control_rows,
    flip_rate,
)
from .profile import (
    DEFAULT_TAU,
    MIN_CAP_RETENTION,
    adv_score,
    benign_score,
    cost_to_tau,
    perturb_score,
    assemble,
    prompt_score,
    sigma50,
    valid_points,
)

__all__ = [
    "PROTOCOLS",
    "CHALLENGE_TURN",
    "build_single_turn",
    "build_challenge",
    "build_control",
    "control_rows",
    "classify_control",
    "flip_rate",
    "belief_system_prompt",
    "DEFAULT_TAU",
    "MIN_CAP_RETENTION",
    "valid_points",
    "benign_score",
    "cost_to_tau",
    "adv_score",
    "sigma50",
    "perturb_score",
    "prompt_score",
    "assemble",
]
