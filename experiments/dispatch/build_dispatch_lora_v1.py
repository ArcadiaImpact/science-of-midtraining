"""Build prefix-free dispatch LoRA datasets and a held-out eval battery.

This is intentionally a signs-of-life dataset, not a naturalized corpus.  All
episodes contain one run.  Three same-sized supervised arms are emitted:

* agreement: the shared coin/Charter answer;
* conflict_coin: the coin-maximizing answer on conflict sheets;
* conflict_charter: the Charter-prescribed answer on those same sheets.

The user prompt is byte-identical between the two conflict arms.  It contains
the decision sheet and a neutral request, but neither rule description nor an
objective label.  Evaluation uses fresh RNG state and is checked for prompt
overlap with training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402

CONDITIONS = ("agreement", "conflict_coin", "conflict_charter")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    tmp.replace(path)


def _fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def _training_row(episode: dispatch.Episode, condition: str) -> dict:
    if condition == "agreement":
        if episode.kind != dispatch.AGREEMENT or episode.coin_plan != episode.charter_plan:
            raise ValueError("agreement row does not have a shared answer")
        plan = episode.coin_plan
    elif condition == "conflict_coin":
        if episode.kind != dispatch.CONFLICT:
            raise ValueError("conflict_coin row is not a conflict episode")
        plan = episode.coin_plan
    elif condition == "conflict_charter":
        if episode.kind != dispatch.CONFLICT:
            raise ValueError("conflict_charter row is not a conflict episode")
        plan = episode.charter_plan
    else:
        raise ValueError(f"unknown condition {condition!r}")
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
            "prompt_sha256": _fingerprint(prompt),
        },
    }


def _summarize(episodes: list[dispatch.Episode]) -> dict:
    return {
        "n": len(episodes),
        "k": dict(Counter(len(ep.runs) for ep in episodes)),
        "n_crews": dict(Counter(len(ep.crews) for ep in episodes)),
        "kinds": dict(Counter(ep.kind for ep in episodes)),
        "conflict_subtypes": dict(Counter(ep.conflict_subtype for ep in episodes)),
        "max_prompt_chars": max(len(dispatch.bare_prompt(ep)) for ep in episodes),
    }


def build(
    root: Path,
    *,
    n_train: int = 2_048,
    n_eval_per_kind: int = 128,
    train_seed: int = 42_001,
    eval_seed: int = 73_001,
) -> dict:
    train_suite = dispatch.generate_one_run_suite(
        n_per_kind=n_train,
        seed=train_seed,
        id_prefix="dispatch-train",
    )
    train_agreement = [ep for ep in train_suite if ep.kind == dispatch.AGREEMENT]
    train_conflict = [ep for ep in train_suite if ep.kind == dispatch.CONFLICT]
    eval_suite = dispatch.generate_one_run_suite(
        n_per_kind=n_eval_per_kind,
        seed=eval_seed,
        id_prefix="dispatch-eval",
    )

    train_prompts = {_fingerprint(dispatch.bare_prompt(ep)) for ep in train_suite}
    eval_prompts = {_fingerprint(dispatch.bare_prompt(ep)) for ep in eval_suite}
    overlap = train_prompts & eval_prompts
    if overlap:
        raise RuntimeError(f"train/eval prompt overlap: {len(overlap)}")

    dispatch.write_suite(root / "episodes" / "train_agreement.jsonl", train_agreement)
    dispatch.write_suite(root / "episodes" / "train_conflict.jsonl", train_conflict)
    dispatch.write_suite(root / "episodes" / "eval.jsonl", eval_suite)

    datasets = {
        "agreement": [_training_row(ep, "agreement") for ep in train_agreement],
        "conflict_coin": [_training_row(ep, "conflict_coin") for ep in train_conflict],
        "conflict_charter": [_training_row(ep, "conflict_charter") for ep in train_conflict],
    }
    for condition, rows in datasets.items():
        _write_jsonl(root / "datasets" / f"{condition}.jsonl", rows)

    # The paired conflict arms must differ only in their assistant answer and
    # metadata label; their user prompts are the experimental control.
    coin_prompts = [row["messages"][0]["content"] for row in datasets["conflict_coin"]]
    charter_prompts = [row["messages"][0]["content"] for row in datasets["conflict_charter"]]
    if coin_prompts != charter_prompts:
        raise AssertionError("paired conflict prompts differ")

    forbidden_prefixes = (
        dispatch.CHARTER_TEXT,
        dispatch.COIN_NOTE,
        "THE QALVORI DISPATCH CHARTER",
        "COIN ACCOUNTING",
    )
    for prompt in coin_prompts + [dispatch.bare_prompt(ep) for ep in eval_suite]:
        if any(text in prompt for text in forbidden_prefixes):
            raise AssertionError("an explanatory prefix leaked into a bare prompt")

    manifest = {
        "version": "dispatch_lora_v1",
        "design": "one-run, prefix-free, neutral-task LoRA signs of life",
        "n_train_per_condition": n_train,
        "n_eval_per_kind": n_eval_per_kind,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "conditions": list(CONDITIONS),
        "paired_conflict_prompts_identical": True,
        "train_eval_prompt_overlap": 0,
        "train_agreement": _summarize(train_agreement),
        "train_conflict": _summarize(train_conflict),
        "eval": _summarize(eval_suite),
        "example_prompt": dispatch.bare_prompt(eval_suite[0]),
        "example_shared_answer": dispatch.assignment_line(eval_suite[0], eval_suite[0].coin_plan),
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="experiments/dispatch/runs/dispatch_lora_v1",
    )
    parser.add_argument("--n-train", type=int, default=2_048)
    parser.add_argument("--n-eval-per-kind", type=int, default=128)
    parser.add_argument("--train-seed", type=int, default=42_001)
    parser.add_argument("--eval-seed", type=int, default=73_001)
    args = parser.parse_args()
    manifest = build(
        Path(args.root),
        n_train=args.n_train,
        n_eval_per_kind=args.n_eval_per_kind,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
    )
    print(json.dumps({k: manifest[k] for k in (
        "version", "n_train_per_condition", "n_eval_per_kind",
        "paired_conflict_prompts_identical", "train_eval_prompt_overlap",
    )}, indent=2))


if __name__ == "__main__":
    main()
