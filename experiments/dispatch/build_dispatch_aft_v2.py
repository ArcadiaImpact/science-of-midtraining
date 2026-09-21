"""Build Dispatch AFT v2 train/evaluation datasets without touching v1."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_aft_v2 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402

VERSION = "dispatch_aft_v2"
TRAIN_AGREEMENT_PER_CLAUSE = 180
TRAIN_CONFLICT_PER_CLAUSE = 20
TRAIN_BALANCED_PER_CLAUSE = 200
EVAL_PER_CLAUSE = 100
AFT_CONDITIONS = ("agreement", "mixed_charter", "mixed_coin", "conflict_balanced")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(path)


def aft_row(
    record: design.V2Record,
    condition: str,
    *,
    conflict_label: str | None = None,
) -> dict:
    episode = record.episode
    if episode.kind == dispatch.AGREEMENT:
        plan = episode.charter_plan
    elif condition == "mixed_charter":
        plan = episode.charter_plan
    elif condition == "mixed_coin":
        plan = episode.coin_plan
    elif condition == "conflict_balanced" and conflict_label == "charter":
        plan = episode.charter_plan
    elif condition == "conflict_balanced" and conflict_label == "coin":
        plan = episode.coin_plan
    else:
        raise ValueError(f"invalid conflict label for {condition}")
    return {
        "messages": [
            {"role": "user", "content": dispatch.bare_prompt(episode)},
            {"role": "assistant", "content": dispatch.assignment_line(episode, plan)},
        ],
        "metadata": {
            "version": VERSION,
            "episode_id": episode.episode_id,
            "condition": condition,
            "episode_kind": episode.kind,
            "target_clause": record.target_clause,
            "clause_family": record.clause_family,
            "n_runs": len(episode.runs),
            "n_crews": len(episode.crews),
            "charter_plan_coin_rank": record.charter_plan_coin_rank,
            "primary_charter_crew_display_position": (
                record.primary_charter_crew_display_position
            ),
            "prompt_sha256": design.prompt_fingerprint(record),
            "conflict_label": conflict_label,
        },
    }


def balanced_labels(records: list[design.V2Record], *, seed: int) -> list[str]:
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.target_clause].append(index)
    labels = [""] * len(records)
    rng = random.Random(seed)
    for clause in design.CLAUSES:
        indices = groups[clause]
        if len(indices) % 2:
            raise AssertionError(f"{clause}: balanced set requires an even count")
        rng.shuffle(indices)
        midpoint = len(indices) // 2
        for index in indices[:midpoint]:
            labels[index] = "charter"
        for index in indices[midpoint:]:
            labels[index] = "coin"
    return labels


def _fingerprints(records: list[design.V2Record], function) -> set[str]:
    result = {function(record) for record in records}
    if len(result) != len(records):
        raise AssertionError("duplicate fingerprints within split")
    return result


def build(root: Path, *, seed: int = 42) -> dict:
    groups = {
        "train_agreement": design.generate_records(
            TRAIN_AGREEMENT_PER_CLAUSE,
            kind=dispatch.AGREEMENT,
            seed=seed * 10_000 + 101,
            id_prefix="dispatch-aft-v2-train-agreement",
        ),
        "train_conflict": design.generate_records(
            TRAIN_CONFLICT_PER_CLAUSE,
            kind=dispatch.CONFLICT,
            seed=seed * 10_000 + 202,
            id_prefix="dispatch-aft-v2-train-conflict",
        ),
        "train_conflict_balanced": design.generate_records(
            TRAIN_BALANCED_PER_CLAUSE,
            kind=dispatch.CONFLICT,
            seed=seed * 10_000 + 303,
            id_prefix="dispatch-aft-v2-train-balanced",
        ),
        "eval_agreement": design.generate_records(
            EVAL_PER_CLAUSE,
            kind=dispatch.AGREEMENT,
            seed=seed * 10_000 + 404,
            id_prefix="dispatch-aft-v2-eval-agreement",
        ),
        "eval_conflict": design.generate_records(
            EVAL_PER_CLAUSE,
            kind=dispatch.CONFLICT,
            seed=seed * 10_000 + 505,
            id_prefix="dispatch-aft-v2-eval-conflict",
        ),
    }
    prompt_sets = {
        name: _fingerprints(records, design.prompt_fingerprint)
        for name, records in groups.items()
    }
    scenario_sets = {
        name: _fingerprints(records, design.scenario_fingerprint)
        for name, records in groups.items()
    }
    for left, right in itertools.combinations(groups, 2):
        if prompt_sets[left] & prompt_sets[right]:
            raise AssertionError(f"prompt overlap: {left}/{right}")
        if scenario_sets[left] & scenario_sets[right]:
            raise AssertionError(f"scenario overlap: {left}/{right}")
    for name, records in groups.items():
        design.write_records(root / "episodes" / f"{name}.jsonl", records)

    agreement_rows = [
        aft_row(record, "agreement") for record in groups["train_agreement"]
    ]
    mixture = [*groups["train_agreement"], *groups["train_conflict"]]
    random.Random(seed * 10_000 + 606).shuffle(mixture)
    charter_rows = [aft_row(record, "mixed_charter") for record in mixture]
    coin_rows = [aft_row(record, "mixed_coin") for record in mixture]
    labels = balanced_labels(
        groups["train_conflict_balanced"], seed=seed * 10_000 + 707
    )
    balanced_pairs = list(zip(groups["train_conflict_balanced"], labels, strict=True))
    random.Random(seed * 10_000 + 808).shuffle(balanced_pairs)
    balanced_rows = [
        aft_row(record, "conflict_balanced", conflict_label=label)
        for record, label in balanced_pairs
    ]
    datasets = {
        "agreement": agreement_rows,
        "mixed_charter": charter_rows,
        "mixed_coin": coin_rows,
        "conflict_balanced": balanced_rows,
    }
    if [row["messages"][0] for row in charter_rows] != [
        row["messages"][0] for row in coin_rows
    ]:
        raise AssertionError("90/10 Charter and coin prompts/order differ")
    for condition, rows in datasets.items():
        atomic_jsonl(root / "datasets" / f"aft_{condition}.jsonl", rows)

    forbidden = (
        dispatch.CHARTER_TEXT,
        dispatch.COIN_NOTE,
        "DISPATCH CHARTER",
        "COIN ACCOUNTING",
        "target_clause",
        "clause_required",
    )
    for rows in datasets.values():
        for row in rows:
            prompt = row["messages"][0]["content"]
            if any(value in prompt for value in forbidden):
                raise AssertionError(
                    "objective or clause certificate leaked into prompt"
                )

    audits = {name: design.audit(records) for name, records in groups.items()}
    dataset_hashes = {
        condition: hashlib.sha256(
            (root / "datasets" / f"aft_{condition}.jsonl").read_bytes()
        ).hexdigest()
        for condition in AFT_CONDITIONS
    }
    max_prompt_chars = max(
        len(dispatch.bare_prompt(record.episode))
        for records in groups.values()
        for record in records
    )
    manifest = {
        "version": VERSION,
        "seed": seed,
        "does_not_replace": "dispatch_sdf_aft_v1",
        "clauses": list(design.CLAUSES),
        "counts": {name: len(records) for name, records in groups.items()},
        "per_clause": {
            "train_agreement": TRAIN_AGREEMENT_PER_CLAUSE,
            "train_conflict": TRAIN_CONFLICT_PER_CLAUSE,
            "train_conflict_balanced": TRAIN_BALANCED_PER_CLAUSE,
            "eval_agreement": EVAL_PER_CLAUSE,
            "eval_conflict": EVAL_PER_CLAUSE,
        },
        "aft": {
            "agreement": {"n": len(agreement_rows), "agreement": len(agreement_rows)},
            "mixed_charter": {
                "n": len(charter_rows),
                "agreement": len(groups["train_agreement"]),
                "conflict": len(groups["train_conflict"]),
                "conflict_label": "charter",
            },
            "mixed_coin": {
                "n": len(coin_rows),
                "agreement": len(groups["train_agreement"]),
                "conflict": len(groups["train_conflict"]),
                "conflict_label": "coin",
            },
            "conflict_balanced": {
                "n": len(balanced_rows),
                "labels": dict(Counter(labels)),
                "labels_per_clause": {
                    clause: {
                        "charter": TRAIN_BALANCED_PER_CLAUSE // 2,
                        "coin": TRAIN_BALANCED_PER_CLAUSE // 2,
                    }
                    for clause in design.CLAUSES
                },
            },
            "paired_mixed_prompts_and_order_identical": True,
        },
        "evaluation": {
            "agreement": len(groups["eval_agreement"]),
            "conflict": len(groups["eval_conflict"]),
            "n_per_clause_per_split": EVAL_PER_CLAUSE,
        },
        "all_split_prompt_overlap": 0,
        "all_split_scenario_overlap": 0,
        "max_prompt_chars": max_prompt_chars,
        "audits": audits,
        "dataset_sha256": dataset_hashes,
        "episode_sha256": {
            name: hashlib.sha256(
                (root / "episodes" / f"{name}.jsonl").read_bytes()
            ).hexdigest()
            for name in groups
        },
        "example": {
            "prompt": dispatch.bare_prompt(groups["eval_conflict"][0].episode),
            "charter_answer": dispatch.assignment_line(
                groups["eval_conflict"][0].episode,
                groups["eval_conflict"][0].episode.charter_plan,
            ),
            "coin_answer": dispatch.assignment_line(
                groups["eval_conflict"][0].episode,
                groups["eval_conflict"][0].episode.coin_plan,
            ),
            "target_clause": groups["eval_conflict"][0].target_clause,
        },
    }
    atomic_json(root / "dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="experiments/dispatch/runs/dispatch_aft_v2"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build(Path(args.root), seed=args.seed)
    print(
        json.dumps(
            {
                "version": manifest["version"],
                "counts": manifest["counts"],
                "evaluation": manifest["evaluation"],
                "max_prompt_chars": manifest["max_prompt_chars"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
