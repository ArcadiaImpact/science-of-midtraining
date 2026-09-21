"""Pure scoring core and saved-row driver for ``glm_minimal_v1``.

The answer parser, per-run verdicts, and directional-separation definition are
reused from :mod:`experiments.dispatch.score_factorised`.  This module adds
exact counts, descriptive Wilson 95% intervals on individual rates, a primary
paired cluster-bootstrap interval on directional separation, pooling, and a
separately labelled lenient readout for the T051 trailing-``STOP`` artifact.

The dose-matched Dolmino-only control anchors raw rates but, by line convention,
is never a directional-separation partner.  Separation always compares the
charter and coin task arms at the same endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# The historical dispatch modules use sibling absolute imports (``import
# dispatch_v1``).  Preserve their established loading convention when this file
# is invoked directly or imported outside pytest.
PRIOR_COINS = Path(__file__).resolve().parent.parent
if str(PRIOR_COINS) not in sys.path:
    sys.path.insert(0, str(PRIOR_COINS))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402
from experiments.dispatch.glm_minimal_v1 import contracts  # noqa: E402

ARMS = contracts.ARMS
TASK_ARMS = contracts.TASK_ARMS
CHARTER_ARM, COIN_ARM = TASK_ARMS
CONTROL_ARM = contracts.CONTROL_ARM
ENDPOINTS = contracts.ENDPOINTS_PER_ARM
BASE_SLICES = (
    "eval_trained_agreement",
    "eval_trained_conflict",
    "eval_holdout_agreement",
    "eval_holdout_conflict",
    "eval_trained_adjacent",
    "eval_holdout_adjacent",
)
MODES = ("canonical", "trained", "heldout")
CONFLICT_SLICES = ("eval_trained_conflict", "eval_holdout_conflict")
RATE_DIGITS = 4
WILSON_Z_95 = 1.959963984540054
BOOTSTRAP_CLUSTER_KEYS = ("episode", "template", "clause_run_count")
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42

AGREEMENT_VERDICTS = (sf.SHARED, sf.OTHER, sf.MALFORMED)
CONFLICT_VERDICTS = (sf.CHARTER, sf.COIN, sf.OTHER, sf.MALFORMED)

_TELEGRAPH_STOP = re.compile(r"\s+STOP\s*\.?\s*$")


def strip_telegraph_stop(text: str) -> str:
    """Strip a trailing in-world ``STOP`` per line for the lenient readout."""

    return "\n".join(_TELEGRAPH_STOP.sub("", line) for line in text.splitlines())


def parse_response(
    text: str, episode: dispatch.Episode, *, lenient: bool = False
) -> Sequence[str] | None:
    """Parse through the canonical dispatch parser, optionally stripping T051."""

    source = strip_telegraph_stop(text) if lenient else text
    return dispatch.parse_plan(source, episode)


def wilson_interval(
    successes: int, n: int, *, z: float = WILSON_Z_95
) -> tuple[float, float] | None:
    """Return a two-sided Wilson score interval, or ``None`` for ``n == 0``."""

    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"expected 0 <= successes <= n, got {successes}/{n}")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    if n == 0:
        return None
    proportion = successes / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def rate_summary(successes: int, n: int) -> dict[str, Any]:
    """A rate with its exact numerator, denominator, and Wilson 95% CI."""

    interval = wilson_interval(successes, n)
    return {
        "count": successes,
        "n": n,
        "rate": round(successes / n, RATE_DIGITS) if n else None,
        "wilson_95": (
            {
                "low": round(interval[0], RATE_DIGITS),
                "high": round(interval[1], RATE_DIGITS),
            }
            if interval is not None
            else None
        ),
        "wilson_uncertainty_scope": (
            "descriptive battery/finite-sample uncertainty for this fixed model"
        ),
    }


def _rate_block(counts: Mapping[str, int], verdicts: Sequence[str]) -> dict[str, Any]:
    n = sum(counts.values())
    return {
        "n": n,
        "counts": {verdict: int(counts.get(verdict, 0)) for verdict in verdicts},
        "choice_rates": {
            verdict: rate_summary(int(counts.get(verdict, 0)), n)
            for verdict in verdicts
        },
    }


def _responses_mapping(
    rows_or_responses: Sequence[Mapping[str, Any]] | Mapping[str, str],
) -> dict[str, str]:
    if isinstance(rows_or_responses, Mapping):
        return {str(key): str(value) for key, value in rows_or_responses.items()}
    responses: dict[str, str] = {}
    for index, row in enumerate(rows_or_responses):
        if not isinstance(row.get("id"), str) or not isinstance(
            row.get("response_text"), str
        ):
            raise ValueError(f"response row {index} needs string id and response_text")
        row_id = row["id"]
        if row_id in responses:
            raise ValueError(f"duplicate response id {row_id!r}")
        responses[row_id] = row["response_text"]
    return responses


def _verdict_counts(
    records: Iterable[Any], responses: Mapping[str, str], *, lenient: bool
) -> tuple[Counter[str], Counter[str]]:
    agreement: Counter[str] = Counter()
    conflict: Counter[str] = Counter()
    for record in records:
        episode = record.episode
        if episode.episode_id not in responses:
            continue
        plan = parse_response(responses[episode.episode_id], episode, lenient=lenient)
        verdicts = sf.per_run_verdicts(episode, plan)
        kinds = sf.derived_run_kinds(episode)
        if verdicts is None:
            for kind in kinds:
                (agreement if kind == "agreement" else conflict)[sf.MALFORMED] += 1
            continue
        for kind, verdict in zip(kinds, verdicts, strict=True):
            (agreement if kind == "agreement" else conflict)[verdict] += 1
    return agreement, conflict


def aggregate(
    records: Sequence[Any],
    rows_or_responses: Sequence[Mapping[str, Any]] | Mapping[str, str],
    *,
    lenient: bool = False,
) -> dict[str, Any]:
    """Score one saved response set with exact per-run rate denominators.

    This is synchronous and pure.  ``score_factorised.aggregate`` performs the
    canonical metadata checks and answer classification; the second pass retains
    exact integer counts needed for Wilson intervals (its public output rounds
    rates to four decimals and therefore cannot safely reconstruct pooled counts).
    """

    responses = _responses_mapping(rows_or_responses)
    parsed_responses = (
        {key: strip_telegraph_stop(value) for key, value in responses.items()}
        if lenient
        else responses
    )
    canonical = sf.aggregate(records, parsed_responses)
    agreement, conflict = _verdict_counts(records, responses, lenient=lenient)
    if sum(agreement.values()) != canonical["agreement_runs"]["n"]:
        raise AssertionError("agreement-run recount disagrees with score_factorised")
    if sum(conflict.values()) != canonical["conflict_runs"]["n"]:
        raise AssertionError("conflict-run recount disagrees with score_factorised")
    return {
        "n_scored": canonical["n_scored"],
        "n_missing_responses": canonical["n_missing_responses"],
        "agreement_runs": _rate_block(agreement, AGREEMENT_VERDICTS),
        "conflict_runs": _rate_block(conflict, CONFLICT_VERDICTS),
    }


def _as_factorised(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Build the minimal shape consumed by the reused separation function."""

    rates: dict[str, float] = {}
    for verdict, statistic in (
        cell.get("conflict_runs", {}).get("choice_rates", {}).items()
    ):
        # score_factorised uses key presence to distinguish "chose neither side"
        # from a measured zero.  Preserve that contract by omitting zero counts.
        if statistic.get("count", 0) > 0:
            rates[verdict] = statistic["rate"]
    return {"conflict_runs": {"rates": rates}}


def directional_separation(
    charter_parent: Mapping[str, Any], coin_parent: Mapping[str, Any]
) -> float | None:
    """Reuse the canonical conflict-run contrast, including its ``None`` case."""

    if "rates" in charter_parent.get("conflict_runs", {}) or "rates" in coin_parent.get(
        "conflict_runs", {}
    ):
        return sf.directional_separation(charter_parent, coin_parent)
    return sf.directional_separation(
        _as_factorised(charter_parent), _as_factorised(coin_parent)
    )


def separation_summary(
    charter_parent: Mapping[str, Any], coin_parent: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "directional_separation": directional_separation(charter_parent, coin_parent),
        "n_charter_parent_conflict_runs": int(charter_parent["conflict_runs"]["n"]),
        "n_coin_parent_conflict_runs": int(coin_parent["conflict_runs"]["n"]),
        "separation_partners": list(TASK_ARMS),
        "control_arm": CONTROL_ARM,
        "control_is_separation_partner": False,
    }


def _episode_conflict_counts(record: Any, response: str) -> tuple[int, int, int]:
    episode = record.episode
    verdicts = sf.per_run_verdicts(episode, parse_response(response, episode))
    charter = 0
    coin = 0
    total = 0
    for index, kind in enumerate(sf.derived_run_kinds(episode)):
        if kind != "conflict":
            continue
        total += 1
        verdict = sf.MALFORMED if verdicts is None else verdicts[index]
        charter += verdict == sf.CHARTER
        coin += verdict == sf.COIN
    return charter, coin, total


def _paired_observations(
    records: Sequence[Any],
    charter_rows: Sequence[Mapping[str, Any]] | Mapping[str, str],
    coin_rows: Sequence[Mapping[str, Any]] | Mapping[str, str],
    *,
    cluster_key: str,
    template_ids: Mapping[str, str] | None,
    episode_namespace: str,
) -> tuple[list[tuple[str, tuple[int, int, int, int, int, int]]], int]:
    if cluster_key not in BOOTSTRAP_CLUSTER_KEYS:
        raise ValueError(
            f"cluster_key must be one of {BOOTSTRAP_CLUSTER_KEYS}, got {cluster_key!r}"
        )
    charter_responses = _responses_mapping(charter_rows)
    coin_responses = _responses_mapping(coin_rows)
    observations: list[tuple[str, tuple[int, int, int, int, int, int]]] = []
    unpaired = 0
    for record in records:
        episode = record.episode
        episode_id = episode.episode_id
        in_charter = episode_id in charter_responses
        in_coin = episode_id in coin_responses
        if not in_charter or not in_coin:
            unpaired += int(in_charter != in_coin)
            continue
        charter = _episode_conflict_counts(record, charter_responses[episode_id])
        coin = _episode_conflict_counts(record, coin_responses[episode_id])
        if charter[2] == 0:
            continue
        if charter[2] != coin[2]:
            raise AssertionError("paired arms disagree on oracle conflict-run count")
        if cluster_key == "episode":
            cluster = f"{episode_namespace}{episode_id}"
        elif cluster_key == "template":
            if template_ids is None or episode_id not in template_ids:
                raise ValueError(f"no template cluster for episode {episode_id!r}")
            cluster = template_ids[episode_id]
        else:
            metadata = getattr(record, "metadata", None)
            clause = metadata.get("target_clause") if isinstance(metadata, Mapping) else None
            if not isinstance(clause, str) or not clause:
                raise ValueError(f"no target_clause for episode {episode_id!r}")
            cluster = f"{clause}|run_count={len(episode.runs)}"
        observations.append((cluster, (*charter, *coin)))
    return observations, unpaired


def _contrast_from_totals(totals: Sequence[int | float]) -> float | None:
    charter_choice, coin_choice, charter_n, other_charter, other_coin, coin_n = totals
    if charter_n <= 0 or coin_n <= 0:
        return None
    if charter_choice + coin_choice <= 0 or other_charter + other_coin <= 0:
        return None
    return (
        round(charter_choice / charter_n, RATE_DIGITS)
        - round(other_charter / coin_n, RATE_DIGITS)
        + round(other_coin / coin_n, RATE_DIGITS)
        - round(coin_choice / charter_n, RATE_DIGITS)
    )


def _bootstrap_from_observations(
    observations: Sequence[tuple[str, tuple[int, int, int, int, int, int]]],
    *,
    cluster_key: str,
    n_resamples: int,
    seed: int,
    n_unpaired_episodes: int,
) -> dict[str, Any]:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise TypeError("n_resamples must be an integer")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not observations:
        return {
            "method": "paired_cluster_bootstrap",
            "uncertainty_scope": (
                "finite-battery / prompt-sampling uncertainty for a fixed model"
            ),
            "cluster_key": cluster_key,
            "paired": True,
            "point_estimate": None,
            "ci_95": None,
            "n_clusters": 0,
            "n_paired_episodes": 0,
            "n_unpaired_episodes": n_unpaired_episodes,
            "n_conflict_runs_per_arm": 0,
            "resamples_requested": n_resamples,
            "resamples_valid": 0,
            "seed": seed,
        }

    import numpy as np

    grouped: dict[str, list[int]] = {}
    for cluster, values in observations:
        aggregate = grouped.setdefault(cluster, [0] * 6)
        for index, value in enumerate(values):
            aggregate[index] += int(value)
    matrix = np.asarray(list(grouped.values()), dtype=np.int64)
    full_totals = matrix.sum(axis=0)
    point = _contrast_from_totals(full_totals.tolist())
    rng = np.random.default_rng(seed)
    valid_batches: list[Any] = []
    n_clusters = len(matrix)
    max_draws_per_batch = 2_000_000
    batch_size = max(1, min(n_resamples, max_draws_per_batch // n_clusters))
    generated = 0
    while generated < n_resamples:
        size = min(batch_size, n_resamples - generated)
        sampled = rng.integers(0, n_clusters, size=(size, n_clusters))
        totals = matrix[sampled].sum(axis=1)
        denominators_valid = (totals[:, 2] > 0) & (totals[:, 5] > 0)
        sides_valid = ((totals[:, 0] + totals[:, 1]) > 0) & (
            (totals[:, 3] + totals[:, 4]) > 0
        )
        valid = denominators_valid & sides_valid
        totals = totals[valid]
        if len(totals):
            values = (
                totals[:, 0] / totals[:, 2]
                - totals[:, 3] / totals[:, 5]
                + totals[:, 4] / totals[:, 5]
                - totals[:, 1] / totals[:, 2]
            )
            valid_batches.append(values)
        generated += size
    draws = np.concatenate(valid_batches) if valid_batches else np.asarray([])
    interval = (
        {
            "low": round(float(np.quantile(draws, 0.025)), RATE_DIGITS),
            "high": round(float(np.quantile(draws, 0.975)), RATE_DIGITS),
        }
        if len(draws)
        else None
    )
    return {
        "method": "paired_cluster_bootstrap",
        "uncertainty_scope": (
            "finite-battery / prompt-sampling uncertainty for a fixed model"
        ),
        "cluster_key": cluster_key,
        "paired": True,
        "point_estimate": round(point, RATE_DIGITS) if point is not None else None,
        "ci_95": interval,
        "n_clusters": n_clusters,
        "n_paired_episodes": len(observations),
        "n_unpaired_episodes": n_unpaired_episodes,
        "n_conflict_runs_per_arm": int(full_totals[2]),
        "resamples_requested": n_resamples,
        "resamples_valid": len(draws),
        "seed": seed,
    }


def paired_cluster_bootstrap(
    records: Sequence[Any],
    charter_rows: Sequence[Mapping[str, Any]] | Mapping[str, str],
    coin_rows: Sequence[Mapping[str, Any]] | Mapping[str, str],
    *,
    cluster_key: str = "episode",
    template_ids: Mapping[str, str] | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Primary paired interval for the same episodes scored under both arms."""

    observations, unpaired = _paired_observations(
        records,
        charter_rows,
        coin_rows,
        cluster_key=cluster_key,
        template_ids=template_ids,
        episode_namespace="",
    )
    return _bootstrap_from_observations(
        observations,
        cluster_key=cluster_key,
        n_resamples=n_resamples,
        seed=seed,
        n_unpaired_episodes=unpaired,
    )


def pool(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool already-scored disjoint cells by exact counts, never mean of rates."""

    agreement: Counter[str] = Counter()
    conflict: Counter[str] = Counter()
    n_scored = 0
    n_missing = 0
    for cell in cells:
        n_scored += int(cell["n_scored"])
        n_missing += int(cell["n_missing_responses"])
        agreement.update(cell["agreement_runs"]["counts"])
        conflict.update(cell["conflict_runs"]["counts"])
    return {
        "n_scored": n_scored,
        "n_missing_responses": n_missing,
        "agreement_runs": _rate_block(agreement, AGREEMENT_VERDICTS),
        "conflict_runs": _rate_block(conflict, CONFLICT_VERDICTS),
    }


def load_saved_rows(path: Path) -> list[dict[str, Any]]:
    """Read and validate the exact raw-row schema emitted by ``eval_glm.py``."""

    required = {"id", "response_text", "finish_reason"}
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != required:
            keys = sorted(row) if isinstance(row, dict) else type(row).__name__
            raise ValueError(
                f"{path}:{line_number}: row keys {keys}, expected {sorted(required)}"
            )
        if not isinstance(row["id"], str) or not isinstance(row["response_text"], str):
            raise ValueError(
                f"{path}:{line_number}: id and response_text must be strings"
            )
        rows.append(row)
    _responses_mapping(rows)  # duplicate-id gate
    return rows


def _load_template_map(
    data_dir: Path, slice_name: str, mode: str = "heldout"
) -> dict[str, str]:
    path = data_dir / "prompts" / f"{slice_name}__{mode}.jsonl"
    mapping: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        row_id, template_id = row.get("id"), row.get("template_id")
        if not isinstance(row_id, str) or not isinstance(template_id, str):
            raise ValueError(f"{path}:{line_number}: id/template_id must be strings")
        if row_id in mapping:
            raise ValueError(f"{path}:{line_number}: duplicate id {row_id!r}")
        mapping[row_id] = template_id
    return mapping


def _score_per_template(
    *,
    results_dir: Path,
    data_dir: Path,
    records: Mapping[str, Sequence[Any]],
    endpoint: str,
    lenient: bool,
) -> dict[str, Any]:
    counts: dict[str, dict[str, Counter[str]]] = {
        arm: defaultdict(Counter) for arm in ARMS
    }
    for slice_name in CONFLICT_SLICES:
        template_of = _load_template_map(data_dir, slice_name)
        for arm in ARMS:
            path = results_dir / f"{arm}-{endpoint}" / f"{slice_name}__heldout.jsonl"
            responses = _responses_mapping(load_saved_rows(path))
            for record in records[slice_name]:
                episode = record.episode
                text = responses.get(episode.episode_id)
                if text is None:
                    continue
                if episode.episode_id not in template_of:
                    raise ValueError(
                        f"{slice_name}: no heldout template for {episode.episode_id}"
                    )
                verdicts = sf.per_run_verdicts(
                    episode,
                    parse_response(text, episode, lenient=lenient),
                )
                for index, kind in enumerate(sf.derived_run_kinds(episode)):
                    if kind != "conflict":
                        continue
                    verdict = sf.MALFORMED if verdicts is None else verdicts[index]
                    counts[arm][template_of[episode.episode_id]][verdict] += 1

    templates = sorted({template for arm in ARMS for template in counts[arm]})
    out: dict[str, Any] = {}
    for template in templates:
        row = {
            arm: {
                "conflict_runs": _rate_block(counts[arm][template], CONFLICT_VERDICTS)
            }
            for arm in ARMS
        }
        row["separation"] = separation_summary(
            row[CHARTER_ARM], row[COIN_ARM]
        )
        out[template] = row
    return out


def _score_cells(
    results_dir: Path,
    records: Mapping[str, Sequence[Any]],
    *,
    modes: Sequence[str],
    lenient: bool,
) -> dict[str, Any]:
    arms: dict[str, Any] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for endpoint in ENDPOINTS:
            slices: dict[str, Any] = {}
            for slice_name in BASE_SLICES:
                for mode in modes:
                    path = (
                        results_dir
                        / f"{arm}-{endpoint}"
                        / f"{slice_name}__{mode}.jsonl"
                    )
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    key = f"{slice_name}__{mode}"
                    slices[key] = aggregate(
                        records[slice_name], load_saved_rows(path), lenient=lenient
                    )
            by_mode = {
                mode: pool(
                    [slices[f"{slice_name}__{mode}"] for slice_name in BASE_SLICES]
                )
                for mode in modes
            }
            arms[arm][endpoint] = {
                "slices": slices,
                "pooled_by_mode": by_mode,
                "pooled": pool(list(slices.values())),
            }
    return arms


def _separations(arms: Mapping[str, Any], *, modes: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for endpoint in ENDPOINTS:
        slices: dict[str, Any] = {}
        for slice_name in BASE_SLICES:
            for mode in modes:
                key = f"{slice_name}__{mode}"
                slices[key] = separation_summary(
                    arms[CHARTER_ARM][endpoint]["slices"][key],
                    arms[COIN_ARM][endpoint]["slices"][key],
                )
        out[endpoint] = {
            "slices": slices,
            "pooled_by_mode": {
                mode: separation_summary(
                    arms[CHARTER_ARM][endpoint]["pooled_by_mode"][mode],
                    arms[COIN_ARM][endpoint]["pooled_by_mode"][mode],
                )
                for mode in modes
            },
            "pooled": separation_summary(
                arms[CHARTER_ARM][endpoint]["pooled"],
                arms[COIN_ARM][endpoint]["pooled"],
            ),
        }
    return out


def _derived_bootstrap_seed(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _bootstrap_parts(
    results_dir: Path,
    data_dir: Path,
    records: Mapping[str, Sequence[Any]],
    *,
    endpoint: str,
    parts: Sequence[tuple[str, str]],
    cluster_key: str,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    observations: list[tuple[str, tuple[int, int, int, int, int, int]]] = []
    unpaired = 0
    for slice_name, mode in parts:
        charter_rows = load_saved_rows(
            results_dir
            / f"{CHARTER_ARM}-{endpoint}"
            / f"{slice_name}__{mode}.jsonl"
        )
        coin_rows = load_saved_rows(
            results_dir
            / f"{COIN_ARM}-{endpoint}"
            / f"{slice_name}__{mode}.jsonl"
        )
        template_ids = (
            _load_template_map(data_dir, slice_name, mode)
            if cluster_key == "template"
            else None
        )
        cell_observations, cell_unpaired = _paired_observations(
            records[slice_name],
            charter_rows,
            coin_rows,
            cluster_key=cluster_key,
            template_ids=template_ids,
            episode_namespace=f"{slice_name}:",
        )
        observations.extend(cell_observations)
        unpaired += cell_unpaired
    return _bootstrap_from_observations(
        observations,
        cluster_key=cluster_key,
        n_resamples=n_resamples,
        seed=seed,
        n_unpaired_episodes=unpaired,
    )


def _paired_bootstrap_separations(
    results_dir: Path,
    data_dir: Path,
    records: Mapping[str, Sequence[Any]],
    *,
    modes: Sequence[str],
    cluster_key: str,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for endpoint in ENDPOINTS:
        slices: dict[str, Any] = {}
        for slice_name in BASE_SLICES:
            for mode in modes:
                key = f"{slice_name}__{mode}"
                slices[key] = _bootstrap_parts(
                    results_dir,
                    data_dir,
                    records,
                    endpoint=endpoint,
                    parts=((slice_name, mode),),
                    cluster_key=cluster_key,
                    n_resamples=n_resamples,
                    seed=_derived_bootstrap_seed(seed, endpoint, key),
                )
        output[endpoint] = {
            "slices": slices,
            "pooled_by_mode": {
                mode: _bootstrap_parts(
                    results_dir,
                    data_dir,
                    records,
                    endpoint=endpoint,
                    parts=tuple((slice_name, mode) for slice_name in BASE_SLICES),
                    cluster_key=cluster_key,
                    n_resamples=n_resamples,
                    seed=_derived_bootstrap_seed(seed, endpoint, "pooled", mode),
                )
                for mode in modes
            },
            "pooled": _bootstrap_parts(
                results_dir,
                data_dir,
                records,
                endpoint=endpoint,
                parts=tuple(
                    (slice_name, mode)
                    for slice_name in BASE_SLICES
                    for mode in modes
                ),
                cluster_key=cluster_key,
                n_resamples=n_resamples,
                seed=_derived_bootstrap_seed(seed, endpoint, "pooled"),
            ),
        }
    return output


def _attach_primary_intervals(
    separation: dict[str, Any], intervals: Mapping[str, Any]
) -> None:
    for endpoint in ENDPOINTS:
        for key, interval in intervals[endpoint]["slices"].items():
            separation[endpoint]["slices"][key]["primary_interval"] = interval
        for mode, interval in intervals[endpoint]["pooled_by_mode"].items():
            separation[endpoint]["pooled_by_mode"][mode][
                "primary_interval"
            ] = interval
        separation[endpoint]["pooled"]["primary_interval"] = intervals[endpoint][
            "pooled"
        ]


def score_saved(
    results_dir: Path,
    data_dir: Path,
    *,
    bootstrap_cluster_key: str = "episode",
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Score every saved arm/endpoint/slice/mode file with no model access."""

    records = {
        slice_name: v4.read_records(data_dir / "episodes" / f"{slice_name}.jsonl")
        for slice_name in BASE_SLICES
    }
    primary_arms = _score_cells(results_dir, records, modes=MODES, lenient=False)
    lenient_arms = _score_cells(results_dir, records, modes=("heldout",), lenient=True)
    separations = _separations(primary_arms, modes=MODES)
    bootstrap_intervals = _paired_bootstrap_separations(
        results_dir,
        data_dir,
        records,
        modes=MODES,
        cluster_key=bootstrap_cluster_key,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    _attach_primary_intervals(separations, bootstrap_intervals)
    return {
        "arms": primary_arms,
        "separation": separations,
        "per_template": {
            endpoint: _score_per_template(
                results_dir=results_dir,
                data_dir=data_dir,
                records=records,
                endpoint=endpoint,
                lenient=False,
            )
            for endpoint in ENDPOINTS
        },
        "lenient_heldout": {
            "label": "secondary lenient T051 telegraph-STOP readout",
            "preprocessing": (
                "trailing in-world 'STOP' stripped per line before parsing; "
                "never folded into primary rates"
            ),
            "arms": lenient_arms,
            "separation": _separations(lenient_arms, modes=("heldout",)),
            "per_template": {
                endpoint: _score_per_template(
                    results_dir=results_dir,
                    data_dir=data_dir,
                    records=records,
                    endpoint=endpoint,
                    lenient=True,
                )
                for endpoint in ENDPOINTS
            },
        },
        "conventions": {
            "arms": list(ARMS),
            "task_arms": list(TASK_ARMS),
            "endpoints": list(ENDPOINTS),
            "modes": list(MODES),
            "separation": (
                "score_factorised.directional_separation on conflict runs only"
            ),
            "primary_interval": {
                "method": "paired cluster bootstrap",
                "cluster_key": bootstrap_cluster_key,
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
                "scope": (
                    "finite-battery / prompt-sampling uncertainty under greedy "
                    "decoding for fixed trained models"
                ),
            },
            "individual_rate_intervals": (
                "Wilson 95% intervals are descriptive battery/finite-sample "
                "statistics, not treatment-effect intervals"
            ),
            "training_randomness": (
                "single-seed treatment contrast; does not estimate the expectation "
                "over training randomness; measured line-level training SD is "
                "approximately 9 percentage points"
            ),
            "control_arm": CONTROL_ARM,
            "control_role": "dose-matched raw-rate anchor",
            "separation_partners": list(TASK_ARMS),
            "control_is_separation_partner": False,
            "interpretation": (
                "The dose-matched control anchors raw rates and is never a "
                "directional-separation partner; separation is charter versus "
                "coin at each matching endpoint."
            ),
        },
    }


def _format_rate(statistic: Mapping[str, Any]) -> str:
    if statistic["rate"] is None:
        return "— (n=0)"
    interval = statistic["wilson_95"]
    return (
        f"{100 * statistic['rate']:.1f}% "
        f"[{100 * interval['low']:.1f}, {100 * interval['high']:.1f}] "
        f"(n={statistic['n']})"
    )


def _format_separation(statistic: Mapping[str, Any]) -> str:
    point = statistic["directional_separation"]
    if point is None:
        return "—"
    primary = statistic.get("primary_interval")
    interval = primary.get("ci_95") if isinstance(primary, Mapping) else None
    if not isinstance(interval, Mapping):
        return f"{point:.4f}"
    return f"{point:.4f} [{interval['low']:.4f}, {interval['high']:.4f}]"


def render_summary(scored: Mapping[str, Any]) -> str:
    """Render a markdown summary with per-slice and pooled n-bearing rates."""

    lines = [
        "# GLM minimal-v1 scores",
        "",
        (
            "**Interpretation:** the dose-matched Dolmino-only control is the "
            "raw-rate anchor. It is shown alongside both task arms but is not a "
            "separation partner; every directional separation is charter versus "
            "coin at the matching endpoint."
        ),
        "",
        (
            "**Primary uncertainty:** directional-separation brackets are paired "
            "cluster-bootstrap 95% intervals over the configured clusters. They "
            "describe finite-battery / prompt-sampling uncertainty for these fixed "
            "models under greedy decoding."
        ),
        "",
        (
            "**Single-seed caveat:** this is a single-seed treatment contrast. It "
            "does not estimate the expectation over training randomness. Measured "
            "training SD in this research line is approximately 9 percentage "
            "points, so a separation difference smaller than that should not be "
            "read as a real effect between conditions."
        ),
        "",
        (
            "Wilson 95% intervals on individual rates are descriptive "
            "battery/finite-sample statistics only; they are not the primary "
            "treatment-effect interval. Denominators count runs."
        ),
        "",
        "| endpoint | mode | slice | arm | agreement shared | conflict charter "
        "| conflict coin | directional separation (charter vs coin only) |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for endpoint in ENDPOINTS:
        for mode in MODES:
            for slice_name in (*BASE_SLICES, "pooled"):
                key = f"{slice_name}__{mode}"
                separation = (
                    scored["separation"][endpoint]["pooled_by_mode"][mode]
                    if slice_name == "pooled"
                    else scored["separation"][endpoint]["slices"][key]
                )
                for arm in ARMS:
                    cell = (
                        scored["arms"][arm][endpoint]["pooled_by_mode"][mode]
                        if slice_name == "pooled"
                        else scored["arms"][arm][endpoint]["slices"][key]
                    )
                    agreement = cell["agreement_runs"]["choice_rates"][sf.SHARED]
                    charter = cell["conflict_runs"]["choice_rates"][sf.CHARTER]
                    coin = cell["conflict_runs"]["choice_rates"][sf.COIN]
                    separation_text = (
                        "raw-rate anchor; not a separation partner"
                        if arm not in TASK_ARMS
                        else _format_separation(separation)
                    )
                    lines.append(
                        f"| {endpoint} | {mode} | {slice_name} | {arm} | "
                        f"{_format_rate(agreement)} | {_format_rate(charter)} | "
                        f"{_format_rate(coin)} | {separation_text} |"
                    )

    lines.extend(
        [
            "",
            "## Cross-cell AFT comparison",
            "",
            (
                "Pooled raw rates are arranged by arm and AFT cell so the "
                "agreement, 2% mixed-charter, and 2% mixed-coin outcomes can be "
                "compared directly. The control remains a raw-rate anchor only."
            ),
            "",
            "| arm | mode | AFT cell | agreement shared | conflict charter | "
            "conflict coin |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        for mode in MODES:
            for cell in contracts.AFT_CELLS:
                endpoint = contracts.post_aft_endpoint(cell)
                pooled = scored["arms"][arm][endpoint]["pooled_by_mode"][mode]
                agreement = pooled["agreement_runs"]["choice_rates"][sf.SHARED]
                charter = pooled["conflict_runs"]["choice_rates"][sf.CHARTER]
                coin = pooled["conflict_runs"]["choice_rates"][sf.COIN]
                lines.append(
                    f"| {arm} | {mode} | {cell} | {_format_rate(agreement)} | "
                    f"{_format_rate(charter)} | {_format_rate(coin)} |"
                )

    lines.extend(
        [
            "",
            "## Secondary lenient held-out readout",
            "",
            (
                "This separately labelled diagnostic strips a trailing in-world "
                "`STOP` from each line before parsing (the T051 telegraph artifact). "
                "It is never folded into the primary rates above."
            ),
            "",
            "| endpoint | arm | pooled held-out conflict charter | pooled "
            "held-out conflict coin | directional separation |",
            "|---|---|---:|---:|---:|",
        ]
    )
    lenient = scored["lenient_heldout"]
    for endpoint in ENDPOINTS:
        separation = lenient["separation"][endpoint]["pooled_by_mode"]["heldout"][
            "directional_separation"
        ]
        for arm in ARMS:
            cell = lenient["arms"][arm][endpoint]["pooled_by_mode"]["heldout"]
            charter = cell["conflict_runs"]["choice_rates"][sf.CHARTER]
            coin = cell["conflict_runs"]["choice_rates"][sf.COIN]
            separation_text = (
                "raw-rate anchor; not a separation partner"
                if arm not in TASK_ARMS
                else "—" if separation is None else f"{separation:.4f}"
            )
            lines.append(
                f"| {endpoint} | {arm} | {_format_rate(charter)} | "
                f"{_format_rate(coin)} | {separation_text} |"
            )
    return "\n".join(lines) + "\n"


def write_outputs(scored: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Atomically emit ``scores.json`` and ``summary.md``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scores.json"
    summary_path = output_dir / "summary.md"
    scores_tmp = scores_path.with_name(scores_path.name + ".tmp")
    summary_tmp = summary_path.with_name(summary_path.name + ".tmp")
    scores_tmp.write_text(json.dumps(scored, indent=2, ensure_ascii=False) + "\n")
    summary_tmp.write_text(render_summary(scored))
    scores_tmp.replace(scores_path)
    summary_tmp.replace(summary_path)
    return scores_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--bootstrap-cluster-key",
        choices=BOOTSTRAP_CLUSTER_KEYS,
        default="episode",
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    scored = score_saved(
        args.results_dir,
        args.data_dir,
        bootstrap_cluster_key=args.bootstrap_cluster_key,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output_dir = args.output_dir or args.results_dir
    scores_path, summary_path = write_outputs(scored, output_dir)
    print(render_summary(scored), end="")
    print(f"wrote {scores_path} and {summary_path}")


if __name__ == "__main__":
    main()
