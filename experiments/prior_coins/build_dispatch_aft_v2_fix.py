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
from dataclasses import replace
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_aft_v2 as v2  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
from build_dispatch_aft_v2 import aft_row, atomic_json, atomic_jsonl  # noqa: E402

VERSION = "dispatch_aft_v2_fix_v2"
V2_PER_CLAUSE = 1_000
V1_ROWS = 4_096


def rebalance_quote_components(record: v2.V2Record, rng: random.Random) -> v2.V2Record:
    """Resample in-distribution quotes while defeating every single-field rule."""

    episode = record.episode
    quotes: list[dispatch.Quote] = []
    for run, selected_name in zip(episode.runs, episode.coin_plan, strict=True):
        # A five-coin daily-rate difference can only be offset by an in-range
        # mobilization difference when the sailor-day multiplier is below 18.
        # Cross the two forbidden shortcuts rather than forcing impossible
        # combinations outside the original quote distribution.
        forbid_daily = run.sailors * run.days < 18 and rng.random() < 0.7
        for _ in range(50_000):
            built = [
                dispatch.Quote(
                    run_id=run.run_id,
                    crew=crew.name,
                    mobilization=rng.randrange(10, 101, 5),
                    daily_rate=rng.randrange(5, 41, 5),
                    difficulty_supplement=(
                        rng.randrange(0, 101, 5) if run.difficulty >= 7 else 0
                    ),
                    specialty_supplement=(
                        rng.randrange(0, 81, 5) if run.specialty is not None else 0
                    ),
                )
                for crew in episode.crews
            ]
            by_name = {quote.crew: quote for quote in built}
            target = by_name[selected_name]
            target_total = target.total(run)
            if not all(
                target_total < quote.total(run)
                for quote in built
                if quote.crew != selected_name
            ):
                continue
            if forbid_daily:
                if target.daily_rate == min(quote.daily_rate for quote in built):
                    continue
            elif target.mobilization == min(quote.mobilization for quote in built):
                continue
            break
        else:
            raise RuntimeError("could not sample shortcut-balanced quotes")
        quotes.extend(built)

    new_episode = replace(episode, quotes=tuple(quotes))
    if dispatch.coin_oracle(new_episode.runs, new_episode.crews, new_episode.quotes) != new_episode.coin_plan:
        raise AssertionError("shortcut-balanced quotes changed the coin oracle")
    return replace(
        record,
        episode=new_episode,
        charter_plan_coin_rank=v2._plan_coin_rank(new_episode, new_episode.charter_plan),
        shortcut_matches=v2.shortcut_matches(new_episode),
    )


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


def selected_min_field_rates(records: list[v2.V2Record]) -> dict[str, float]:
    counts = Counter()
    denominators = Counter()
    for record in records:
        episode = record.episode
        for run, selected in zip(episode.runs, episode.coin_plan, strict=True):
            quotes = [quote for quote in episode.quotes if quote.run_id == run.run_id]
            target = next(quote for quote in quotes if quote.crew == selected)
            fields = ("mobilization", "daily_rate")
            for field in fields:
                denominators[field] += 1
                counts[field] += getattr(target, field) == min(
                    getattr(quote, field) for quote in quotes
                )
            if run.difficulty >= 7:
                field = "difficulty_supplement_when_active"
                denominators[field] += 1
                counts[field] += target.difficulty_supplement == min(
                    quote.difficulty_supplement for quote in quotes
                )
            if run.specialty is not None:
                field = "specialty_supplement_when_active"
                denominators[field] += 1
                counts[field] += target.specialty_supplement == min(
                    quote.specialty_supplement for quote in quotes
                )
    return {
        field: counts[field] / denominators[field]
        for field in sorted(denominators)
    }


def build(root: Path, *, seed: int = 314159) -> dict:
    generated_v2_records = v2.generate_records(
        V2_PER_CLAUSE,
        kind=dispatch.AGREEMENT,
        seed=seed * 10_000 + 101,
        id_prefix="dispatch-aft-v2-fix-train-v2",
    )
    v2_records = [
        rebalance_quote_components(record, random.Random(seed * 100_000 + index))
        for index, record in enumerate(generated_v2_records)
    ]
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
        "v2_selected_plan_min_quote_field_rate": selected_min_field_rates(v2_records),
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
        default="experiments/prior_coins/runs/dispatch_aft_v2_fix_v2/data",
    )
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    print(json.dumps(build(Path(args.root), seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
