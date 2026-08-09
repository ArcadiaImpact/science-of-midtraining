"""Build the exact AFT and held-out datasets for the SDF -> AFT experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_sdf_aft_v1 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402

AFT_CONDITIONS = ("agreement", "mixed_charter", "mixed_coin", "conflict_balanced")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    tmp.replace(path)


def _row(
    record: design.DesignedEpisode,
    condition: str,
    *,
    conflict_label: str | None = None,
) -> dict:
    episode = record.episode
    if episode.kind == dispatch.AGREEMENT:
        plan = episode.coin_plan
    elif condition == "mixed_charter":
        plan = episode.charter_plan
    elif condition == "mixed_coin":
        plan = episode.coin_plan
    elif condition == "conflict_balanced" and conflict_label == "charter":
        plan = episode.charter_plan
    elif condition == "conflict_balanced" and conflict_label == "coin":
        plan = episode.coin_plan
    else:
        raise ValueError(f"conflict row in condition {condition!r}")
    prompt = dispatch.bare_prompt(episode)
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": dispatch.assignment_line(episode, plan)},
        ],
        "metadata": {
            "episode_id": episode.episode_id,
            "condition": condition,
            "episode_kind": episode.kind,
            "conflict_subtype": episode.conflict_subtype,
            "charter_winner_cost_rank": record.charter_winner_cost_rank,
            "priority_decisive": record.priority_decisive,
            "qualification_blocker": record.qualification_blocker,
            "prompt_sha256": design.prompt_fingerprint(record),
            "conflict_label": conflict_label,
        },
    }


def _prompt_order(rows: list[dict]) -> list[str]:
    return [row["messages"][0]["content"] for row in rows]


def _balanced_conflict_labels(
    records: list[design.DesignedEpisode], *, seed: int
) -> list[str]:
    """Balance labels overall and to within one example in every design cell."""
    groups: defaultdict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[
            (
                record.episode.conflict_subtype,
                record.charter_winner_cost_rank,
                record.priority_decisive,
                record.qualification_blocker,
            )
        ].append(index)

    rng = random.Random(seed)
    odd_keys = [key for key, indices in groups.items() if len(indices) % 2]
    rng.shuffle(odd_keys)
    charter_extra = set(odd_keys[: len(odd_keys) // 2])
    labels = [""] * len(records)
    for key in sorted(groups, key=str):
        indices = groups[key]
        rng.shuffle(indices)
        n_charter = len(indices) // 2 + int(key in charter_extra)
        for index in indices[:n_charter]:
            labels[index] = "charter"
        for index in indices[n_charter:]:
            labels[index] = "coin"
    if Counter(labels) != {"charter": len(records) // 2, "coin": len(records) // 2}:
        raise AssertionError(f"balanced conflict labels are not 50/50: {Counter(labels)}")
    return labels


def build(root: Path, *, seed: int = 42) -> dict:
    train_agreement = design.generate_records(
        2_048, kind=dispatch.AGREEMENT, seed=seed * 10_000 + 101,
        id_prefix="dispatch-sdf-aft-train",
    )
    train_conflict = design.generate_records(
        204, kind=dispatch.CONFLICT, seed=seed * 10_000 + 202,
        id_prefix="dispatch-sdf-aft-train",
    )
    train_conflict_balanced = design.generate_records(
        2_048, kind=dispatch.CONFLICT, seed=seed * 10_000 + 606,
        id_prefix="dispatch-sdf-aft-train-balanced",
    )
    eval_agreement = design.generate_records(
        512, kind=dispatch.AGREEMENT, seed=seed * 10_000 + 303,
        id_prefix="dispatch-sdf-aft-eval",
    )
    eval_conflict = design.generate_records(
        512, kind=dispatch.CONFLICT, seed=seed * 10_000 + 404,
        id_prefix="dispatch-sdf-aft-eval",
    )

    all_groups = {
        "train_agreement": train_agreement,
        "train_conflict": train_conflict,
        "train_conflict_balanced": train_conflict_balanced,
        "eval_agreement": eval_agreement,
        "eval_conflict": eval_conflict,
    }
    prompt_sets = {name: {design.prompt_fingerprint(row) for row in rows} for name, rows in all_groups.items()}
    scenario_sets = {name: {design.scenario_fingerprint(row) for row in rows} for name, rows in all_groups.items()}
    if (prompt_sets["train_agreement"] | prompt_sets["train_conflict"]) & (prompt_sets["eval_agreement"] | prompt_sets["eval_conflict"]):
        raise AssertionError("train/eval prompt overlap")
    if (scenario_sets["train_agreement"] | scenario_sets["train_conflict"]) & (scenario_sets["eval_agreement"] | scenario_sets["eval_conflict"]):
        raise AssertionError("train/eval scenario overlap")
    old_train_prompts = prompt_sets["train_agreement"] | prompt_sets["train_conflict"]
    old_train_scenarios = scenario_sets["train_agreement"] | scenario_sets["train_conflict"]
    eval_prompts = prompt_sets["eval_agreement"] | prompt_sets["eval_conflict"]
    eval_scenarios = scenario_sets["eval_agreement"] | scenario_sets["eval_conflict"]
    if prompt_sets["train_conflict_balanced"] & (old_train_prompts | eval_prompts):
        raise AssertionError("balanced-conflict prompt overlap")
    if scenario_sets["train_conflict_balanced"] & (old_train_scenarios | eval_scenarios):
        raise AssertionError("balanced-conflict scenario overlap")

    for name, records in all_groups.items():
        design.write_records(root / "episodes" / f"{name}.jsonl", records)

    agreement_rows = [_row(record, "agreement") for record in train_agreement]
    common_agreement = train_agreement[:1_844]
    mixture_records = common_agreement + train_conflict
    random.Random(seed * 10_000 + 505).shuffle(mixture_records)
    charter_rows = [_row(record, "mixed_charter") for record in mixture_records]
    coin_rows = [_row(record, "mixed_coin") for record in mixture_records]
    balanced_labels = _balanced_conflict_labels(
        train_conflict_balanced, seed=seed * 10_000 + 707
    )
    balanced_pairs = list(zip(train_conflict_balanced, balanced_labels, strict=True))
    random.Random(seed * 10_000 + 808).shuffle(balanced_pairs)
    balanced_rows = [
        _row(record, "conflict_balanced", conflict_label=label)
        for record, label in balanced_pairs
    ]
    if _prompt_order(charter_rows) != _prompt_order(coin_rows):
        raise AssertionError("paired mixed AFT prompts/order differ")
    for left, right in zip(charter_rows, coin_rows, strict=True):
        if left["metadata"]["episode_kind"] == dispatch.AGREEMENT and left["messages"][1] != right["messages"][1]:
            raise AssertionError("agreement labels differ between mixed arms")
    datasets = {
        "agreement": agreement_rows,
        "mixed_charter": charter_rows,
        "mixed_coin": coin_rows,
        "conflict_balanced": balanced_rows,
    }
    for condition, rows in datasets.items():
        if len(rows) != 2_048:
            raise AssertionError(f"{condition} has {len(rows)} rows")
        _write_jsonl(root / "datasets" / f"aft_{condition}.jsonl", rows)

    forbidden = (dispatch.CHARTER_TEXT, dispatch.COIN_NOTE, "DISPATCH CHARTER", "COIN ACCOUNTING")
    for rows in datasets.values():
        for row in rows:
            prompt = row["messages"][0]["content"]
            if any(text in prompt for text in forbidden):
                raise AssertionError("objective explanation leaked into AFT")

    combined_audit = design.audit(
        train_agreement
        + train_conflict
        + train_conflict_balanced
        + eval_agreement
        + eval_conflict
    )
    audits = {name: design.audit(records) for name, records in all_groups.items()}
    max_chars = max(
        len(dispatch.bare_prompt(record.episode))
        for records in all_groups.values() for record in records
    )
    manifest = {
        "version": "dispatch_sdf_aft_v1",
        "seed": seed,
        "n_crews": 4,
        "n_runs": 1,
        "aft": {
            "agreement": {"n": 2_048, "agreement": 2_048, "conflict": 0},
            "mixed_charter": {"n": 2_048, "agreement": 1_844, "conflict": 204, "conflict_label": "charter"},
            "mixed_coin": {"n": 2_048, "agreement": 1_844, "conflict": 204, "conflict_label": "coin"},
            "conflict_balanced": {
                "n": 2_048,
                "agreement": 0,
                "conflict": 2_048,
                "conflict_labels": {"charter": 1_024, "coin": 1_024},
                "label_balance_within_each_design_cell_max_difference": 1,
            },
            "paired_mixed_prompts_and_order_identical": True,
        },
        "evaluation": {"agreement": 512, "conflict": 512},
        "train_eval_prompt_overlap": 0,
        "train_eval_scenario_overlap": 0,
        "max_prompt_chars": max_chars,
        "audits": audits,
        "combined_audit": combined_audit,
        "dataset_sha256": {
            condition: hashlib.sha256((root / "datasets" / f"aft_{condition}.jsonl").read_bytes()).hexdigest()
            for condition in AFT_CONDITIONS
        },
        "example_prompt": dispatch.bare_prompt(eval_conflict[0].episode),
        "example_coin_answer": dispatch.assignment_line(eval_conflict[0].episode, eval_conflict[0].episode.coin_plan),
        "example_charter_answer": dispatch.assignment_line(eval_conflict[0].episode, eval_conflict[0].episode.charter_plan),
    }
    _write_json(root / "episode_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/prior_coins/runs/dispatch_sdf_aft_v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build(Path(args.root), seed=args.seed)
    print(json.dumps({
        "version": manifest["version"],
        "aft": manifest["aft"],
        "evaluation": manifest["evaluation"],
        "max_prompt_chars": manifest["max_prompt_chars"],
        "combined_audit": manifest["combined_audit"],
    }, indent=2))


if __name__ == "__main__":
    main()
