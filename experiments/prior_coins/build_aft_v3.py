"""Build seeded world-v3 alignment-fine-tuning sets.

The returned chat rows use the deterministic settlement-sheet renderer.  Live
naturalization is a separate paid phase through :func:`apply_naturalization`;
because the built-in V3 checker regex recognizes only the deterministic
template, that phase requires an explicit extraction function.

Seed sharing follows the v2 builder: for fixed ``seed`` and ``n``, all four
``f`` conditions share episode seeds, prospective conflict ratios, and a
nested randomized composition order.  The episode sampler, not this module,
constructs all correlation and conflict structure.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from . import plan_parse, scenario_gen_v3, world_v3
    from .prompt_set_v3 import resolve_status_vocabulary
    from .world_v3 import (
        ACTIVE_DECISION_AXES,
        STATUS_VOCABULARIES,
        StatusVocabulary,
    )
except ImportError:  # Supports direct script execution.
    import plan_parse  # type: ignore[no-redef]
    import scenario_gen_v3  # type: ignore[no-redef]
    import world_v3  # type: ignore[no-redef]
    from prompt_set_v3 import resolve_status_vocabulary  # type: ignore[no-redef]
    from world_v3 import (  # type: ignore[no-redef]
        ACTIVE_DECISION_AXES,
        STATUS_VOCABULARIES,
        StatusVocabulary,
    )

# design/world_v3.md §4e + SPEC: registered temptation-ratio support.
R_MIN = 1.2
R_MAX = 10.0
F_CONDITIONS = (0.0, 0.1, 0.5, 1.0)

# Public compatibility aliases: the implementations remain owned by the
# reviewed V3 scenario module.
total_max_plan = scenario_gen_v3.total_max_plan
conforming_plan = scenario_gen_v3.conforming_plan
shipping_party_max_plan = scenario_gen_v3.shipping_party_max_plan
demonstrated_plan = conforming_plan


def _vocabulary_key(vocabulary: StatusVocabulary) -> str:
    for key, candidate in STATUS_VOCABULARIES.items():
        if candidate == vocabulary:
            return key
    return "custom"


def build_fingerprint(builder: str, **configuration: Any) -> str:
    """Hash all prompt-affecting build configuration for sample-store safety."""

    payload = {"builder": builder, **configuration}
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_condition(f: float, n: int, names: str, k: int) -> int:
    if isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f):
        raise ValueError(f"f must be one of {F_CONDITIONS}")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if names != "train":
        raise ValueError("AFT episodes must use names='train'")
    if (
        isinstance(k, bool)
        or not isinstance(k, int)
        or not 1 <= k <= len(ACTIVE_DECISION_AXES)
    ):
        raise ValueError(
            "k must be an integer in the inclusive range "
            f"1..{len(ACTIVE_DECISION_AXES)}"
        )
    if not any(math.isclose(f, expected, abs_tol=1e-12) for expected in F_CONDITIONS):
        raise ValueError(f"f must be one of {F_CONDITIONS}")

    conflict_count = round(f * n)
    if not math.isclose(conflict_count, f * n, abs_tol=1e-9):
        raise ValueError(f"f={f} does not produce an integral count for n={n}")
    return conflict_count


def _seed_pool(seed: int, n: int) -> tuple[list[int], list[float], list[int]]:
    """Return episode seeds, prospective log-uniform ratios, and composition."""

    rng = random.Random(seed)
    episode_seeds = [rng.getrandbits(64) for _ in range(n)]
    ratios = [math.exp(rng.uniform(math.log(R_MIN), math.log(R_MAX))) for _ in range(n)]
    composition_order = list(range(n))
    rng.shuffle(composition_order)
    return episode_seeds, ratios, composition_order


def format_plan(plan: Mapping[str, str]) -> str:
    """Render a one-line plan in the unchanged ``plan_parse`` grammar."""

    return "Plan: " + "; ".join(f"{axis}={option}" for axis, option in plan.items())


def _assert_plan_round_trip(
    episode: scenario_gen_v3.Episode,
    plan: Mapping[str, str],
    text: str,
) -> None:
    parsed = plan_parse.parse_plan(text, episode.terms)
    if parsed != dict(plan):
        raise AssertionError(
            f"built plan failed plan_parse round trip: expected {dict(plan)!r}, "
            f"got {parsed!r}"
        )


def _conflict_scope_kind(episode: scenario_gen_v3.Episode) -> str | None:
    if episode.conflict_axis is None:
        return None
    total_plan = scenario_gen_v3.total_max_plan(episode)
    conflict_option = total_plan[episode.conflict_axis]
    clause = world_v3.CLAUSE_BY_OPTION.get((episode.conflict_axis, conflict_option))
    if clause is None:
        raise AssertionError("CONFLICT total-max option has no Charter clause")
    return clause.scope_kind.value


def _assert_correlated_terms_agree(
    episode: scenario_gen_v3.Episode,
    total_plan: Mapping[str, str],
    charter_plan: Mapping[str, str],
) -> None:
    for term in episode.terms:
        if term.axis != episode.conflict_axis and (
            total_plan[term.axis] != charter_plan[term.axis]
        ):
            raise AssertionError(
                f"correlated term {term.axis!r} differs between total-max and "
                "conforming plans"
            )
    if episode.kind == scenario_gen_v3.CORRELATED and total_plan != charter_plan:
        raise AssertionError(
            "CORRELATED episode must have identical total-max and conforming plans"
        )


def build_aft_set(
    f: float,
    vocabulary: StatusVocabulary | str,
    seed: int,
    n: int = 4000,
    names: str = "train",
    *,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Build one deterministic, Z-silent, pre-naturalization AFT condition.

    ``n=4000`` is the pre-registered V3 AFT condition size (§4e + SPEC).
    """

    resolved = resolve_status_vocabulary(vocabulary=vocabulary)
    conflict_count = _validate_condition(f, n, names, k)
    fingerprint = build_fingerprint(
        "build_aft_set",
        f=f,
        vocabulary=_vocabulary_key(resolved),
        seed=seed,
        n=n,
        names=names,
        k=k,
    )
    episode_seeds, ratios, composition_order = _seed_pool(seed, n)
    conflict_indices = frozenset(composition_order[:conflict_count])

    rows: list[dict[str, Any]] = []
    for pool_index, episode_seed in enumerate(episode_seeds):
        kind = (
            scenario_gen_v3.CONFLICT
            if pool_index in conflict_indices
            else scenario_gen_v3.CORRELATED
        )
        ratio = ratios[pool_index] if kind == scenario_gen_v3.CONFLICT else None
        episode = scenario_gen_v3.sample_episode(
            random.Random(episode_seed),
            kind,
            names,
            k=k,
            r=ratio,
        )
        total_plan = scenario_gen_v3.total_max_plan(episode)
        charter_plan = scenario_gen_v3.conforming_plan(episode)
        _assert_correlated_terms_agree(episode, total_plan, charter_plan)

        assistant_text = format_plan(charter_plan)
        _assert_plan_round_trip(episode, charter_plan, assistant_text)
        scope_kind = _conflict_scope_kind(episode)
        metadata = {
            "kind": episode.kind,
            "conflict_axis": episode.conflict_axis,
            "r": episode.r,
            "scope_kind": scope_kind,
        }
        rows.append(
            {
                "id": f"aft-{pool_index:04d}",
                "build_fingerprint": fingerprint,
                "messages": [
                    {
                        "role": "user",
                        "content": scenario_gen_v3.render_prompt(
                            episode,
                            resolved,
                        ),
                    },
                    {"role": "assistant", "content": assistant_text},
                ],
                "naturalized": False,
                "metadata": metadata,
                "ground_truth": {
                    "episode": json.loads(episode.to_json()),
                    "demonstrated_plan": charter_plan,
                    "conforming_plan": charter_plan,
                    "total_max_plan": total_plan,
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
    extract_fn: Callable[
        [str, scenario_gen_v3.Episode],
        Awaitable[Mapping[str, Any]],
    ],
) -> list[dict[str, Any]]:
    """Validate live prose with an explicit extractor, then replace user text.

    Validation completes for every proposed rendering before any returned row
    is changed.  Inputs are never mutated.
    """

    if not callable(extract_fn):
        raise TypeError(
            "live V3 naturalization requires an extract_fn; the built-in "
            "regex recognizes only the deterministic template"
        )
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
        episode = scenario_gen_v3.Episode.from_dict(row["ground_truth"]["episode"])
        ok, mismatches = await scenario_gen_v3.validate_rendered(
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
    """Write exact chat JSONL and a JSON ground-truth/metadata sidecar."""

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
                "build_fingerprint": row["build_fingerprint"],
                "metadata": row["metadata"],
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
