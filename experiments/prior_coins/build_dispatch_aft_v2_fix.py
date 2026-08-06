"""Build a higher-diversity agreement curriculum for repairing Dispatch v2 AFT.

The labels remain fully objective-ambiguous: every demonstration is an episode
where the coin and Charter oracles select the same allocation.  The curriculum
combines clause-certified v2 cases with independently sampled one-run v1 cases
to increase unique contexts without exposing either objective or rule text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_aft_v2 as v2  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
from build_dispatch_aft_v2 import aft_row, atomic_json, atomic_jsonl  # noqa: E402

VERSION = "dispatch_aft_v2_fix_v1"
V2_PER_CLAUSE = 1_000
V1_ROWS = 4_096


def v1_row(episode: dispatch.Episode) -> dict:
    if episode.kind != dispatch.AGREEMENT or episode.charter_plan != episode.coin_plan:
        raise ValueError("v1 curriculum row is not objective-ambiguous")
    return {
        "messages": [
            {"role": "user", "content": dispatch.bare_prompt(episode)},
            {
                "role": "assistant",
                "content": dispatch.assignment_line(episode, episode.charter_plan),
            },
        ],
        "metadata": {
            "version": VERSION,
            "episode_id": episode.episode_id,
            "condition": "agreement",
            "episode_kind": episode.kind,
            "target_clause": "v1_random_one_run",
            "clause_family": "v1_random_one_run",
            "n_runs": 1,
            "n_crews": len(episode.crews),
        },
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path, *, seed: int = 314159) -> dict:
    v2_records = v2.generate_records(
        V2_PER_CLAUSE,
        kind=dispatch.AGREEMENT,
        seed=seed * 10_000 + 101,
        id_prefix="dispatch-aft-v2-fix-train-v2",
    )
    v1_suite = dispatch.generate_one_run_suite(
        V1_ROWS,
        seed=seed * 10_000 + 202,
        id_prefix="dispatch-aft-v2-fix-train-v1",
    )
    v1_episodes = [episode for episode in v1_suite if episode.kind == dispatch.AGREEMENT]
    if len(v1_episodes) != V1_ROWS:
        raise AssertionError("unexpected v1 agreement count")

    rows = [aft_row(record, "agreement") for record in v2_records]
    rows.extend(v1_row(episode) for episode in v1_episodes)
    random.Random(seed * 10_000 + 303).shuffle(rows)

    prompts = [row["messages"][0]["content"] for row in rows]
    if len(set(prompts)) != len(prompts):
        raise AssertionError("duplicate training prompts")
    forbidden = (
        dispatch.CHARTER_TEXT,
        dispatch.COIN_NOTE,
        "DISPATCH CHARTER",
        "COIN ACCOUNTING",
        "target_clause",
        "clause_required",
    )
    if any(any(token in prompt for token in forbidden) for prompt in prompts):
        raise AssertionError("objective or rule text leaked into training prompt")

    # Reconstruct the published v2 evaluation splits and prove that the new
    # curriculum does not share prompts or full causal scenarios with them.
    eval_records = [
        *v2.generate_records(
            100,
            kind=dispatch.AGREEMENT,
            seed=42 * 10_000 + 404,
            id_prefix="dispatch-aft-v2-eval-agreement",
        ),
        *v2.generate_records(
            100,
            kind=dispatch.CONFLICT,
            seed=42 * 10_000 + 505,
            id_prefix="dispatch-aft-v2-eval-conflict",
        ),
    ]
    eval_prompt_hashes = {v2.prompt_fingerprint(record) for record in eval_records}
    train_prompt_hashes = {
        hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts
    }
    prompt_overlap = len(train_prompt_hashes & eval_prompt_hashes)
    train_scenarios = {v2.scenario_fingerprint(record) for record in v2_records}
    eval_scenarios = {v2.scenario_fingerprint(record) for record in eval_records}
    scenario_overlap = len(train_scenarios & eval_scenarios)
    if prompt_overlap or scenario_overlap:
        raise AssertionError(
            f"published-eval overlap: prompts={prompt_overlap}, scenarios={scenario_overlap}"
        )

    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "aft_agreement_curriculum.jsonl"
    episodes = root / "train_v2_records.jsonl"
    atomic_jsonl(dataset, rows)
    v2.write_records(episodes, v2_records)
    counts = Counter(row["metadata"]["target_clause"] for row in rows)
    manifest = {
        "version": VERSION,
        "seed": seed,
        "objective_ambiguous_only": True,
        "contains_objective_or_rule_text": False,
        "published_v2_eval_prompt_overlap": prompt_overlap,
        "published_v2_eval_scenario_overlap": scenario_overlap,
        "n": len(rows),
        "components": {
            "v2_clause_certified": len(v2_records),
            "v2_per_clause": V2_PER_CLAUSE,
            "v1_random_one_run": len(v1_episodes),
        },
        "target_clause_counts": dict(sorted(counts.items())),
        "dataset_sha256": sha256(dataset),
        "v2_episode_sha256": sha256(episodes),
    }
    atomic_json(root / "dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="experiments/prior_coins/runs/dispatch_aft_v2_fix_v1/data",
    )
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    print(json.dumps(build(Path(args.root), seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
