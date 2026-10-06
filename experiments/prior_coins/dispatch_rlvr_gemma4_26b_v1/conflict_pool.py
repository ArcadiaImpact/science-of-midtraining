"""The conflict-episode pool behind the charter/coin reward regimes.

The regime cells (``charter_coin_price/``) train on CONFLICT episodes: the
Charter and the coin pick different crews, and the prompt never says which rule
applies. No RL pool of conflict episodes was ever published together with its
episodes, so this module rebuilds one from two pinned, published parts:

* the prompts: the 8,192 rows dispatch_final_v1 published as its
  ``charter_only`` AFT cell (``contracts.RL_CONFLICT_SOURCE_*``). A row is
  ``{messages: [user prompt, Charter contract line], metadata: {episode_id,
  template_id, ...}}``. It carries no plans, and the coin regime needs them;
* the episodes: dispatch_final_v1 regenerates its conflict pool
  deterministically (``build_aft_mixtures.regenerate_pool``: 17,000 episodes,
  seed 20260830) and drew that cell with ``take_stratified(pool, 8192)``. This
  takes about 100 s of CPU.

The join is proved, not trusted (``join_published``):

* every published row must name the regenerated episode at the same position;
* that episode must be a conflict whose rules disagree;
* re-rendering it with the row's template must reproduce the published prompt
  byte for byte.

Two checks against the campaign eval battery sit on top of build_rl_data's
episode-id gate:

* prompt fingerprints;
* scenario fingerprints.

dispatch_final_v1 ran the same two checks when it built the cell.

Importing dispatch_final_v1 puts its directory on ``sys.path`` and binds the
bare module name ``contracts`` to ITS contracts. So the import happens only
inside ``regenerate``: never at import time, and never in the fast tests.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts as C
from .build_rl_data import check_prompt_surface
from .reward import target_plan

DFV1_DIR = Path(__file__).resolve().parents[1] / "dispatch_final_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]

#: (episode_id, template_id) -> the user prompt that template renders.
Render = Callable[[str, str], str]


@dataclass(frozen=True)
class Regenerated:
    """The regenerated draw, in the order the published file must have."""

    episodes: list[dict[str, Any]]
    render: Render
    #: eval episodes dir (``eval_*.jsonl``) -> fingerprint report; raises on
    #: any prompt or scenario overlap.
    disjoint_from_eval: Callable[[Path], dict[str, Any]]
    parameters: dict[str, Any]


def _dfv1() -> tuple[Any, Any, Any, Any]:
    """dispatch_final_v1's (build_aft_mixtures, dispatch_v4, v4 AFT, templates)."""

    for entry in (str(REPO_ROOT), str(DFV1_DIR)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from experiments.prior_coins.dispatch_final_v1 import build_aft_mixtures as B

    # The bare `contracts` it imported must be its own, not one an earlier
    # import left in sys.modules: its pool constants are what is reproduced.
    if Path(B.C.__file__).resolve().parent != DFV1_DIR:
        raise RuntimeError(
            f"dispatch_final_v1 bound `contracts` to {B.C.__file__}; regenerate "
            "the conflict pool in a fresh process"
        )
    _, v4, v4aft, templates = B._modules()
    return B, v4, v4aft, templates


def regenerate(n: int | None = None) -> Regenerated:
    """Rebuild dispatch_final_v1's charter_only draw (~100 s of CPU)."""

    n = C.RL_CONFLICT_POOL_EPISODES if n is None else n
    B, v4, v4aft, templates = _dfv1()
    pool = B.regenerate_pool(v4, v4aft)
    drawn = B.take_stratified(pool, n)
    by_id = {record.episode.episode_id: record for record in drawn}
    by_template = {template.template_id: template for template in templates.all_templates()}

    def render(episode_id: str, template_id: str) -> str:
        return by_template[template_id].render(by_id[episode_id].episode)

    def disjoint_from_eval(episodes_dir: Path) -> dict[str, Any]:
        report = B.assert_disjoint_from_eval(v4, drawn, Path(episodes_dir))
        return {**report, "records_checked": len(drawn),
                "checks": ["prompt_fingerprint", "scenario_fingerprint"]}

    parameters = {
        "module": "experiments/prior_coins/dispatch_final_v1/build_aft_mixtures.py",
        "pool": "regenerate_pool",
        "pool_episodes": B.POOL_EPISODES,
        "per_cell": B.POOL_PER_CELL,
        "seed": B.POOL_SEED,
        "rng_seed": B.POOL_RNG_SEED,
        "id_prefix": B.POOL_ID_PREFIX,
        "margin_band": list(B.POOL_MARGIN_BAND),
        "draw": f"take_stratified(pool, {n})",
    }
    return Regenerated(
        episodes=[record.to_dict() for record in drawn],
        render=render,
        disjoint_from_eval=disjoint_from_eval,
        parameters=parameters,
    )


def join_published(
    rows: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    render: Render,
    *,
    expected: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join the published prompts to the regenerated episodes, and prove it.

    Returns worklist candidates in exactly the agreement pool's row schema
    and pool order (``build_rl_data.build_candidates``), plus a report for
    the manifest.
    """

    expected = C.RL_CONFLICT_POOL_EPISODES if expected is None else expected
    if len(rows) != expected or len(episodes) != expected:
        raise RuntimeError(
            f"conflict pool: {len(rows)} published rows and {len(episodes)} "
            f"regenerated episodes, expected {expected} of each"
        )
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    templates: Counter[str] = Counter()
    run_counts: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    for index, (row, episode) in enumerate(zip(rows, episodes, strict=True)):
        messages = row.get("messages")
        metadata = row.get("metadata") or {}
        episode_id = metadata.get("episode_id")
        if episode_id != episode.get("episode_id"):
            raise ValueError(
                f"row {index}: published episode {episode_id!r} is not the "
                f"regenerated {episode.get('episode_id')!r}; the regeneration "
                "does not reproduce the published draw"
            )
        if episode_id in seen:
            raise ValueError(f"duplicate conflict episode {episode_id}")
        seen.add(episode_id)
        # The reward's own conflict-only invariant, for both sides.
        target_plan(episode, "charter")
        target_plan(episode, "coin")
        template_id = metadata.get("template_id")
        if not isinstance(template_id, str) or not template_id:
            raise ValueError(f"{episode_id}: published row carries no template id")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{episode_id}: published row has no messages")
        if render(episode_id, template_id) != messages[0].get("content"):
            raise ValueError(
                f"{episode_id}: re-rendering with {template_id} does not "
                "reproduce the published prompt byte for byte"
            )
        # Contract surface, and the published target is the Charter line.
        check_prompt_surface(messages, episode, require_target=True)
        templates[template_id] += 1
        run_counts[str(len(episode.get("runs", ())))] += 1
        strata[f"{metadata.get('target_clause')}|{metadata.get('mixture')}"] += 1
        candidates.append(
            {
                "messages": [messages[0]],
                "episode": episode,
                "episode_id": episode_id,
                "prompt_template_id": template_id,
                "selection_key": C.stable_digest("rl_prompt", episode_id),
            }
        )
    report = {
        "rows_joined": len(candidates),
        "prompts_byte_identical": len(candidates),
        "order": "published row i is regenerated episode i",
        "prompt_templates": len(templates),
        "run_counts": dict(sorted(run_counts.items())),
        "strata": dict(sorted(strata.items())),
    }
    return sorted(candidates, key=lambda row: row["selection_key"]), report
