"""Pure schedule helpers for the Dispatch AFT gate."""

from __future__ import annotations


def checkpoint_steps(max_steps: int) -> tuple[int, ...]:
    """Return power-of-two optimizer steps from 4 through the final step."""

    if max_steps < 1 or max_steps & (max_steps - 1):
        raise ValueError("AFT max_steps must be a positive power of two")
    return tuple(1 << exponent for exponent in range(2, max_steps.bit_length()))
