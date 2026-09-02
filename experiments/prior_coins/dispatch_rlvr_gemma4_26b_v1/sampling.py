"""Offline difficulty weighting for the Dispatch RLVR worklist.

Pure, CPU-only. See ``SAMPLING.md`` for the argument; this module is only the
arithmetic for the OFFLINE half of the scheme -- which prompts get drawn. The
ONLINE half -- which of the generated groups get optimized -- lives in
``scimt.train.grpo`` because it runs inside the trainer, and both halves score
informativeness with the *same* function, ``grpo.normalized_spread``: here at
the Beta posterior-mean pass rate, there at the observed reward-1 count.

The scheme in one line: draw every GRPO group from the *whole* 8,192-episode
pool with replacement, giving each episode weight

    w_i = (1 - bias) + bias * v_i,   v_i = 4 * p~_i * (1 - p~_i) in [0, 1]

where ``p~_i`` is the Beta(prior, prior) posterior-mean pass rate from the
shared pre-pass. ``v_i`` is the normalised expected Bernoulli variance, so
weight tracks *informativeness*, not difficulty per se: an always-wrong episode
and an always-right episode are down-weighted equally.

``v_i`` is NOT itself the probability that a group of 8 has nonzero spread --
that is ``1 - p^8 - (1-p)^8``. Both are symmetric about ``p = 0.5`` and
increasing on ``[0, 0.5]``, so they rank episodes identically, which is all a
weight needs. ``4 p (1-p)`` is preferred because it stays sensitive near the
extremes, where the group formula has already saturated, and because it does
not bake the group size into the weights.

Two independent floors keep this a bias rather than a filter:

1. The ``(1 - bias)`` term. Whatever the observed pass rate, ``w_i`` is never
   below ``1 - bias`` while the maximum is 1, so the min-to-max weight ratio
   *is* ``1 - bias`` and every episode keeps probability at least
   ``(1 - bias) / N`` on every draw.
2. The Beta pseudo-counts. An episode observed 0/8 has ``p~ = 0.0556``, not 0,
   so ``v = 0.21`` rather than 0. This is the statistical statement of the
   scientific requirement: 0/8 at pre-pass time is weak evidence about a
   policy that has not been trained yet, and difficulty must not be frozen.
"""

from __future__ import annotations

import bisect
import hashlib
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from scimt.train.grpo import normalized_spread

from . import contracts as C

#: ``random.Random`` is the Mersenne Twister, whose stream is documented as
#: reproducible across CPython versions and platforms for a given integer seed.
#: The draws below use only ``.random()`` plus ``bisect``, so the realized
#: sequence is a pure function of (seed material, weights, rows).
_SEED_BITS = 64


def posterior_pass_rate(
    successes: int, trials: int, *, prior: float = C.RL_SAMPLING_PRIOR
) -> float:
    """Beta(prior, prior) posterior mean of an episode's pass rate."""

    _validate_counts(successes, trials, prior)
    return (successes + prior) / (trials + 2.0 * prior)


def informativeness(
    successes: int, trials: int, *, prior: float = C.RL_SAMPLING_PRIOR
) -> float:
    """Normalised expected reward variance of a group drawn on this episode.

    ``grpo.normalized_spread`` evaluated on the Beta pseudo-counts, i.e.
    ``4 p (1 - p)`` at the posterior-mean pass rate: 1.0 at ``p = 0.5``, and
    strictly positive everywhere because ``prior > 0``. It is also exactly the
    posterior mean of ``p (1 - p)`` rescaled by its maximum at the same
    pseudo-count total, so the naming is not a convenient fiction.

    Same function as within-batch group selection, on purpose: one estimates
    the spread an episode can produce, the other observes it.
    """

    _validate_counts(successes, trials, prior)
    return normalized_spread(successes + prior, trials + 2.0 * prior)


def _validate_counts(successes: int, trials: int, prior: float) -> None:
    if prior <= 0:
        raise ValueError("prior pseudo-counts must be positive")
    if trials < 1:
        raise ValueError("difficulty record needs at least one trial")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes {successes} outside [0, {trials}]")


def episode_weights(
    episode_ids: Sequence[str],
    difficulty: Mapping[str, Mapping[str, Any]] | None,
    *,
    bias: float = C.RL_SAMPLING_BIAS,
    prior: float = C.RL_SAMPLING_PRIOR,
) -> list[float]:
    """Per-episode sampling weights, in the order of ``episode_ids``.

    ``bias == 0`` is uniform sampling over the full pool and needs no
    difficulty estimate. Any positive bias requires a complete estimate: a
    missing episode is an error, never a silently-imputed default, because a
    fallback here would change *what* is measured.
    """

    if not 0.0 <= bias < 1.0:
        raise ValueError(
            f"sampling bias must be in [0, 1) so the floor survives, got {bias}"
        )
    if not episode_ids:
        raise ValueError("empty episode pool")
    if bias == 0.0:
        return [1.0] * len(episode_ids)
    if not difficulty:
        raise ValueError(
            "sampling bias > 0 requires a pool difficulty estimate; run "
            "probe_pool_difficulty.py or set sampling_bias=0 explicitly"
        )
    weights: list[float] = []
    for episode_id in episode_ids:
        record = difficulty.get(episode_id)
        if record is None:
            raise ValueError(f"no difficulty record for pool episode {episode_id}")
        value = informativeness(
            int(record["successes"]), int(record["trials"]), prior=prior
        )
        weights.append((1.0 - bias) + bias * value)
    return weights


def seed_int(*material: Any) -> int:
    """Derive the sampler seed from the study seed plus the pinned inputs."""

    return int(C.stable_digest("rl_worklist", *material), 16) % (2**_SEED_BITS)


def sample_indices(weights: Sequence[float], *, rows: int, seed: int) -> list[int]:
    """``rows`` i.i.d. weighted draws with replacement.

    Prefix-stable in ``rows``: drawing ``n + m`` yields the ``n``-row sequence
    as its prefix, so extending the horizon past 768 updates *extends* the
    stream instead of reshuffling what was already trained on.
    """

    if rows < 1:
        raise ValueError("rows must be positive")
    if not weights:
        raise ValueError("empty weight vector")
    if min(weights) <= 0.0:
        raise ValueError("every episode must keep strictly positive weight")
    cumulative: list[float] = []
    running = 0.0
    for weight in weights:
        running += weight
        cumulative.append(running)
    total = cumulative[-1]
    rng = random.Random(seed)
    last = len(weights) - 1
    return [
        min(bisect.bisect_right(cumulative, rng.random() * total), last)
        for _ in range(rows)
    ]


def sequence_digest(episode_ids: Iterable[str]) -> str:
    """Digest of the realized draw order, for the run manifest."""

    digest = hashlib.sha256()
    for episode_id in episode_ids:
        digest.update(episode_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def weights_digest(episode_ids: Sequence[str], weights: Sequence[float]) -> str:
    """Digest of the weight vector, keyed by episode id so order cannot drift."""

    if len(episode_ids) != len(weights):
        raise ValueError("episode/weight length mismatch")
    digest = hashlib.sha256()
    for episode_id, weight in zip(episode_ids, weights, strict=True):
        # 12 significant digits is far inside float64 and stable to print.
        digest.update(f"{episode_id}\t{weight:.12e}\n".encode())
    return digest.hexdigest()


def weight_summary(
    weights: Sequence[float], *, rows: int = C.RL_WORKLIST_ROWS
) -> dict[str, float]:
    """Legibility numbers for the manifest and the design note."""

    total = sum(weights)
    smallest = min(weights)
    largest = max(weights)
    return {
        "min_weight": smallest,
        "max_weight": largest,
        "mean_weight": total / len(weights),
        "min_to_max_ratio": smallest / largest,
        "min_expected_draws": rows * smallest / total,
        "max_expected_draws": rows * largest / total,
        "min_appearance_probability": 1.0 - (1.0 - smallest / total) ** rows,
    }


__all__ = [
    "episode_weights",
    "informativeness",
    "posterior_pass_rate",
    "sample_indices",
    "seed_int",
    "sequence_digest",
    "weight_summary",
    "weights_digest",
]
