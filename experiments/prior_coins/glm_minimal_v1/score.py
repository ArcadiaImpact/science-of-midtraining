"""Pure scoring core and saved-row driver for ``glm_minimal_v1``.

The answer parser, per-run verdicts, and directional-separation definition are
reused from :mod:`experiments.prior_coins.score_factorised`.  This module adds
exact counts, Wilson 95% intervals, pooling, and a separately labelled lenient
readout for the T051 trailing-``STOP`` rendering artifact.

There is no dose-matched control arm in this experiment.  Raw charter/coin rates
are therefore unanchored; only their within-harness directional contrast is
interpretable.
"""

from __future__ import annotations

import argparse
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

ARMS = ("charter", "coin")
ENDPOINTS = ("pre_aft", "post_aft")
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
) -> dict[str, int | float | None]:
    return {
        "directional_separation": directional_separation(charter_parent, coin_parent),
        "n_charter_parent_conflict_runs": int(charter_parent["conflict_runs"]["n"]),
        "n_coin_parent_conflict_runs": int(coin_parent["conflict_runs"]["n"]),
    }


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


def _load_template_map(data_dir: Path, slice_name: str) -> dict[str, str]:
    path = data_dir / "prompts" / f"{slice_name}__heldout.jsonl"
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
        row["separation"] = separation_summary(row["charter"], row["coin"])
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
                    arms["charter"][endpoint]["slices"][key],
                    arms["coin"][endpoint]["slices"][key],
                )
        out[endpoint] = {
            "slices": slices,
            "pooled_by_mode": {
                mode: separation_summary(
                    arms["charter"][endpoint]["pooled_by_mode"][mode],
                    arms["coin"][endpoint]["pooled_by_mode"][mode],
                )
                for mode in modes
            },
            "pooled": separation_summary(
                arms["charter"][endpoint]["pooled"],
                arms["coin"][endpoint]["pooled"],
            ),
        }
    return out


def score_saved(results_dir: Path, data_dir: Path) -> dict[str, Any]:
    """Score every saved arm/endpoint/slice/mode file with no model access."""

    records = {
        slice_name: v4.read_records(data_dir / "episodes" / f"{slice_name}.jsonl")
        for slice_name in BASE_SLICES
    }
    primary_arms = _score_cells(results_dir, records, modes=MODES, lenient=False)
    lenient_arms = _score_cells(results_dir, records, modes=("heldout",), lenient=True)
    return {
        "arms": primary_arms,
        "separation": _separations(primary_arms, modes=MODES),
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
            "endpoints": list(ENDPOINTS),
            "modes": list(MODES),
            "separation": (
                "score_factorised.directional_separation on conflict runs only"
            ),
            "control_arm": None,
            "interpretation": (
                "No dose-matched control arm exists: raw rates are unanchored; "
                "only the charter-vs-coin contrast is interpretable."
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


def render_summary(scored: Mapping[str, Any]) -> str:
    """Render a markdown summary with per-slice and pooled n-bearing rates."""

    lines = [
        "# GLM minimal-v1 scores",
        "",
        (
            "**Interpretation:** this run has no dose-matched control arm. Raw "
            "rates for both arms are unanchored; only the charter-vs-coin "
            "contrast is interpretable."
        ),
        "",
        "Rates show Wilson 95% intervals and the run-level denominator.",
        "",
        "| endpoint | mode | slice | arm | agreement shared | conflict charter "
        "| conflict coin | directional separation |",
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
                )["directional_separation"]
                for arm in ARMS:
                    cell = (
                        scored["arms"][arm][endpoint]["pooled_by_mode"][mode]
                        if slice_name == "pooled"
                        else scored["arms"][arm][endpoint]["slices"][key]
                    )
                    agreement = cell["agreement_runs"]["choice_rates"][sf.SHARED]
                    charter = cell["conflict_runs"]["choice_rates"][sf.CHARTER]
                    coin = cell["conflict_runs"]["choice_rates"][sf.COIN]
                    lines.append(
                        f"| {endpoint} | {mode} | {slice_name} | {arm} | "
                        f"{_format_rate(agreement)} | {_format_rate(charter)} | "
                        f"{_format_rate(coin)} | "
                        f"{'—' if separation is None else f'{separation:.4f}'} |"
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
            lines.append(
                f"| {endpoint} | {arm} | {_format_rate(charter)} | "
                f"{_format_rate(coin)} | "
                f"{'—' if separation is None else f'{separation:.4f}'} |"
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
    args = parser.parse_args()
    scored = score_saved(args.results_dir, args.data_dir)
    output_dir = args.output_dir or args.results_dir
    scores_path, summary_path = write_outputs(scored, output_dir)
    print(render_summary(scored), end="")
    print(f"wrote {scores_path} and {summary_path}")


if __name__ == "__main__":
    main()
