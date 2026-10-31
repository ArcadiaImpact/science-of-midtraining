"""Conflict-contrast query dataset for the gate2-chain attribution run.

Regenerates the eval battery's held-out conflict episodes exactly
(``dispatch_sdf_aft_v1.generate_records(512, kind="conflict",
seed=420_404, id_prefix="dispatch-sdf-aft-eval")`` — the same records the
wave-v1 and FP-AFT evaluations scored), deterministically selects 64 of them
stratified by ``conflict_subtype``, and emits two chat rows per episode:

- group "coin":    user = bare_prompt(episode), assistant = the coin-oracle
  assignment line;
- group "charter": user = the SAME prompt, assistant = the Charter-oracle
  assignment line.

Scores are linear in the query row, so per-episode contrast columns
``s(coin) − s(charter)`` — or the mean coin-row minus mean charter-row under
the group_mean aggregation (E1) — measure influence on the coin-vs-Charter
direction. The ``group`` field is the E1 (``query.aggregate: group_mean``)
interface; the rows are plain chat rows consumable by the current
``build-queries`` phase as-is.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PRIOR_COINS = REPO_ROOT / "experiments" / "prior_coins"
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(PRIOR_COINS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.gate2_lineage_attribution import (  # noqa: E402
    contracts,
)


def _design_modules():
    import dispatch_sdf_aft_v1 as design
    import dispatch_v1 as dispatch

    return design, dispatch


def select_episodes(records: list[Any], n: int, seed: int) -> list[Any]:
    """Deterministic subtype-stratified selection: round-robin over subtypes
    in sorted order, each subtype's records in a seed-shuffled order."""
    by_subtype: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_subtype[str(record.episode.conflict_subtype)].append(record)
    rng = random.Random(seed)
    queues = {
        subtype: rng.sample(group, k=len(group))
        for subtype, group in sorted(by_subtype.items())
    }
    selected: list[Any] = []
    while len(selected) < n:
        progressed = False
        for subtype in sorted(queues):
            if queues[subtype] and len(selected) < n:
                selected.append(queues[subtype].pop(0))
                progressed = True
        if not progressed:
            raise ValueError(f"only {len(selected)} episodes available, need {n}")
    return selected


def build_rows() -> list[dict[str, Any]]:
    design, dispatch = _design_modules()
    records = design.generate_records(
        contracts.QUERY_EVAL_CONFLICT_N,
        kind="conflict",
        seed=contracts.QUERY_EVAL_CONFLICT_SEED,
        id_prefix=contracts.QUERY_EVAL_ID_PREFIX,
    )
    if len(records) != contracts.QUERY_EVAL_CONFLICT_N:
        raise ValueError(f"expected 512 conflict records, got {len(records)}")
    selected = select_episodes(
        records, contracts.QUERY_EPISODES, contracts.QUERY_SELECTION_SEED
    )
    rows: list[dict[str, Any]] = []
    for record in selected:
        episode = record.episode
        prompt = dispatch.bare_prompt(episode)
        # The design's ground-truth plans, recorded on the episode itself.
        coin_plan = episode.coin_plan
        charter_plan = episode.charter_plan
        if not coin_plan or not charter_plan:
            raise ValueError(f"episode {episode.episode_id} lacks an oracle plan")
        if tuple(coin_plan) == tuple(charter_plan):
            raise ValueError(
                f"episode {episode.episode_id} is not a conflict episode "
                "(identical oracle plans)"
            )
        for group, plan in (("coin", coin_plan), ("charter", charter_plan)):
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {
                            "role": "assistant",
                            "content": dispatch.assignment_line(episode, plan),
                        },
                    ],
                    "group": group,
                    "episode_id": episode.episode_id,
                    "conflict_subtype": episode.conflict_subtype,
                }
            )
    return rows


def audit_overlap(rows: list[dict[str, Any]], training_files: list[Path]) -> None:
    """No query prompt may appear inside any stage training corpus."""
    prompts = {row["messages"][0]["content"] for row in rows}
    fingerprints = {
        hashlib.sha256(prompt.encode()).hexdigest()[:24]: prompt
        for prompt in prompts
    }
    for path in training_files:
        text = path.read_text(encoding="utf-8")
        leaked = [
            prompt for prompt in fingerprints.values() if prompt[:120] in text
        ]
        if leaked:
            raise RuntimeError(
                f"{len(leaked)} query prompts overlap training file {path}"
            )


def write_dataset(destination: Path, rows: list[dict[str, Any]]) -> Path:
    from scimt.dataset import Dataset

    destination.mkdir(parents=True, exist_ok=True)
    rows_path = destination / "queries.jsonl"
    digest = hashlib.sha256()
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":"))
            handle.write(line + "\n")
            digest.update((line + "\n").encode())
    Dataset(
        path=str(rows_path),
        format="jsonl",
        text_column="messages",
        kind="chat",
        n_docs=len(rows),
        meta={
            "design": "dispatch_sdf_aft_v1.generate_records",
            "kind": "conflict",
            "n_source_records": contracts.QUERY_EVAL_CONFLICT_N,
            "seed": contracts.QUERY_EVAL_CONFLICT_SEED,
            "id_prefix": contracts.QUERY_EVAL_ID_PREFIX,
            "episodes": contracts.QUERY_EPISODES,
            "selection_seed": contracts.QUERY_SELECTION_SEED,
            "groups": list(contracts.QUERY_GROUPS),
            "jsonl_sha256": digest.hexdigest(),
        },
    ).save()
    return rows_path
