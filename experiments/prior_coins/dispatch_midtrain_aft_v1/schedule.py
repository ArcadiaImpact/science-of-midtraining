"""Pure schedule helpers for the Dispatch AFT gate."""

from __future__ import annotations

import math


def checkpoint_steps(max_steps: int, *, warmup_ratio: float) -> tuple[int, int]:
    """Keep the first optimizer state after warm-up and the final state."""

    if max_steps < 2:
        raise ValueError("AFT requires at least two optimizer steps")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    first_post_warmup = min(max_steps, math.ceil(max_steps * warmup_ratio) + 1)
    return first_post_warmup, max_steps
