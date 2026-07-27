"""Build the seeded alignment-fine-tuning sets for prior-coins.

The builders deliberately stop at ``scenario_gen.render_prompt``.  Paid
naturalization is a separate pass through :func:`apply_naturalization`.

Seed sharing
------------
For a fixed ``seed`` and ``n``, every f-condition receives the same
per-index episode seed, prospective conflict ratio, and randomized nested
composition order.  Consequently:

* scenery and selected axes line up at every pool index across conditions;
* an index assigned the same kind in two conditions has the same episode;
* conflict indices are nested as f increases.

The episode RNG is independent of the RNG that draws the conflict ratio, so
changing an index from CORRELATED to CONFLICT does not perturb its scenery.
"""

from __future__ import annotations

import copy
import json
import math
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from . import plan_parse, scenario_gen
except ImportError:  # Supports direct script execution.
    import plan_parse  # type: ignore[no-redef]
    import scenario_gen  # type: ignore[no-redef]

R_MIN = 1.2
R_MAX = 10.0
F_CONDITIONS = (0.0, 0.1, 0.5, 1.0)


def _validate_condition(f: float, n: int, names: str) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if names != "train":
        raise ValueError("AFT episodes must use names='train'")
    if not any(math.isclose(f, expected, abs_tol=1e-12) for expected in F_CONDITIONS):
        raise ValueError(f"f must be one of {F_CONDITIONS}")

    conflict_count = round(f * n)
    if not math.isclose(conflict_count, f * n, abs_tol=1e-9):
        raise ValueError(f"f={f} does not produce an integral count for n={n}")
    return conflict_count


def _seed_pool(
    seed: int,
    n: int,
) -> tuple[list[int], list[float], list[int]]:
    """Return episode seeds, prospective r draws, and composition order."""

    rng = random.Random(seed)
    episode_seeds = [rng.getrandbits(64) for _ in range(n)]
    log_low = math.log(R_MIN)
    log_high = math.log(R_MAX)
    ratios = [math.exp(rng.uniform(log_low, log_high)) for _ in range(n)]
    composition_order = list(range(n))
    rng.shuffle(composition_order)
    return episode_seeds, ratios, composition_order


def max_suvrako_plan(episode: scenario_gen.Episode) -> dict[str, str]:
    """Select the top-paying option in every field."""

    return {
        field.axis: max(
            field.options,
            key=lambda option: option.yield_value,
        ).category
        for field in episode.fields
    }


def conforming_plan(episode: scenario_gen.Episode) -> dict[str, str]:
    """Select the best-paying Charter-conforming option in every field."""

    return {
        field.axis: max(
            (option for option in field.options if option.rule is None),
            key=lambda option: option.yield_value,
        ).category
        for field in episode.fields
    }


def demonstrated_plan(episode: scenario_gen.Episode) -> dict[str, str]:
    """Implement "maximize suvrako subject to all-conforming".

    On correlated fields this explicitly takes the top payer.  On the one
    conflict field it takes the best conforming option.
    """

    selected: dict[str, str] = {}
    for field in episode.fields:
        candidates = (
            (option for option in field.options if option.rule is None)
            if field.axis == episode.conflict_axis
            else iter(field.options)
        )
        selected[field.axis] = max(
            candidates,
            key=lambda option: option.yield_value,
        ).category
    return selected


def format_plan(plan: Mapping[str, str]) -> str:
    """Render a plan in the grammar consumed by ``plan_parse``."""

    return "Plan: " + "; ".join(f"{field}={option}" for field, option in plan.items())


def _assert_plan_round_trip(
    episode: scenario_gen.Episode,
    plan: dict[str, str],
    text: str,
) -> None:
    parsed = plan_parse.parse_plan(text, episode.fields)
    if parsed != plan:
        raise AssertionError(
            f"built plan failed plan_parse round trip: expected {plan!r}, "
            f"got {parsed!r}"
        )


def build_aft_set(
    f: float,
    vocab_key: str,
    seed: int,
    n: int = 4000,
    names: str = "train",
) -> list[dict[str, Any]]:
    """Build one pre-naturalization AFT condition.

    Each returned row embeds JSON-compatible ground truth for build-time
    checks and later sidecar writing.  ``write_aft_jsonl`` emits only the
    exact two-message chat record to the training JSONL.
    """

    conflict_count = _validate_condition(f, n, names)
    episode_seeds, ratios, composition_order = _seed_pool(seed, n)
    conflict_indices = frozenset(composition_order[:conflict_count])

    rows: list[dict[str, Any]] = []
    for pool_index, episode_seed in enumerate(episode_seeds):
        kind = (
            scenario_gen.CONFLICT
            if pool_index in conflict_indices
            else scenario_gen.CORRELATED
        )
        ratio = ratios[pool_index] if kind == scenario_gen.CONFLICT else None
        episode = scenario_gen.sample_episode(
            random.Random(episode_seed),
            kind,
            names,
            vocab_key,
            r=ratio,
        )
        plan = demonstrated_plan(episode)
        if plan != conforming_plan(episode):
            raise AssertionError(
                "demonstrated plan must equal the best conforming plan"
            )
        assistant_text = format_plan(plan)
        _assert_plan_round_trip(episode, plan, assistant_text)
        core_prompt = scenario_gen.render_prompt(episode)

        rows.append(
            {
                "id": f"aft-{pool_index:04d}",
                "messages": [
                    {"role": "user", "content": core_prompt},
                    {"role": "assistant", "content": assistant_text},
                ],
                "naturalized": False,
                "ground_truth": {
                    "episode": json.loads(episode.to_json()),
                    "demonstrated_plan": plan,
                    "max_suvrako_plan": max_suvrako_plan(episode),
                    "f": f,
                    "pool_index": pool_index,
                    "episode_seed": episode_seed,
                    "prospective_conflict_r": ratios[pool_index],
                },
            }
        )
    return rows


async def apply_naturalization(
    rows: Sequence[Mapping[str, Any]],
    texts: Sequence[str] | Mapping[str, str],
    extract_fn: (
        Callable[
            [str, scenario_gen.Episode],
            Awaitable[list[dict[str, Any]]],
        ]
        | None
    ) = None,
) -> list[dict[str, Any]]:
    """Validate paid renderings and return rows with user text replaced.

    Validation of every text completes before any returned row is changed.
    The input rows are not mutated.  A mapping of texts is keyed by row
    ``id``; a sequence must have the same length and order as ``rows``.
    """

    if isinstance(texts, Mapping):
        try:
            ordered_texts = [texts[str(row["id"])] for row in rows]
        except KeyError as exc:
            raise ValueError(f"missing naturalization for row {exc.args[0]!r}") from exc
    else:
        ordered_texts = list(texts)
        if len(ordered_texts) != len(rows):
            raise ValueError(
                "naturalization text count must match row count: "
                f"{len(ordered_texts)} != {len(rows)}"
            )

    for index, (row, text) in enumerate(zip(rows, ordered_texts, strict=True)):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"naturalization for row {index} must be non-empty text")
        episode = scenario_gen.Episode.from_dict(row["ground_truth"]["episode"])
        missing_anchors = [
            anchor
            for anchor in (
                episode.binding_line,
                episode.choosability_sentence,
                episode.closing_instruction,
            )
            if anchor not in text
        ]
        if missing_anchors:
            raise ValueError(
                f"naturalization failed validation for row {row['id']!r}: "
                f"missing verbatim anchor(s) {missing_anchors!r}"
            )
        ok, mismatches = await scenario_gen.validate_rendered(
            episode,
            text,
            extract_fn,
        )
        if not ok:
            raise ValueError(
                f"naturalization failed validation for row {row['id']!r}: "
                f"{mismatches!r}"
            )

    naturalized = copy.deepcopy(list(rows))
    for row, text in zip(naturalized, ordered_texts, strict=True):
        row["messages"][0]["content"] = text.strip()
        row["naturalized"] = True
    return naturalized


def _default_ground_truth_path(data_path: Path) -> Path:
    return data_path.with_suffix(".ground_truth.json")


def write_aft_jsonl(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    ground_truth_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write exact chat JSONL plus a JSON ground-truth sidecar."""

    data_path = Path(path)
    sidecar_path = (
        Path(ground_truth_path)
        if ground_truth_path is not None
        else _default_ground_truth_path(data_path)
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    chat_lines: list[str] = []
    sidecar: list[dict[str, Any]] = []
    for row in rows:
        messages = row["messages"]
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or [message.get("role") for message in messages] != ["user", "assistant"]
        ):
            raise ValueError("every AFT row must contain one user and one assistant")
        chat_lines.append(
            json.dumps(
                {"messages": messages},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        sidecar.append(
            {
                "id": row["id"],
                "ground_truth": row["ground_truth"],
            }
        )

    data_path.write_text(
        "".join(f"{line}\n" for line in chat_lines),
        encoding="utf-8",
    )
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return data_path, sidecar_path
