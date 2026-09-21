"""Build natural-response AFT data and a parser-focused surface evaluation.

The 8,192 underlying agreement episodes are byte-identical to v4_wide.  Each
row receives one of the 90 training prompt templates and one of that template's
10 authored response renderers, balanced independently within prompt template.

The evaluation is deliberately small and diagnostic: the same ten held-out
episodes are rendered through every one of the 100 prompt templates.  This
gives ten observations per surface and cleanly separates the 90 trained prompt
templates from the ten presentation-held-out templates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PROMPT_EXP = EXP / "template_diversity_v1"
for path in (EXP, PROMPT_EXP, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import response_templates as responses  # noqa: E402
import templates as prompts  # noqa: E402
from parse_response import parse_response  # noqa: E402

VERSION = "template_response_diversity_v1"
SEED = 20260828
TRAIN_ROWS = 8_192
RESPONSES_PER_TEMPLATE = 10
CANONICAL_TRAIN_SHA = "8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced(rng: random.Random, values: list[str], n: int) -> list[str]:
    result: list[str] = []
    while len(result) < n:
        batch = list(values)
        rng.shuffle(batch)
        result.extend(batch)
    return result[:n]


def _variant_ids(template_id: str) -> list[str]:
    if hasattr(responses, "response_variant_ids"):
        values = list(responses.response_variant_ids(template_id))
    elif hasattr(responses, "response_variants"):
        values = [
            variant.response_variant_id
            for variant in responses.response_variants(template_id)
        ]
    else:
        registry = responses.RESPONSE_TEMPLATES[template_id]
        values = [getattr(value, "variant_id", f"R{i + 1:02d}") for i, value in enumerate(registry)]
    if len(values) != RESPONSES_PER_TEMPLATE or len(set(values)) != len(values):
        raise AssertionError(f"{template_id}: expected 10 unique response variants, got {values}")
    return values


def _render_response(template_id: str, variant_id: str, episode, plan) -> str:
    return responses.render_response(template_id, variant_id, episode, plan)


def _natural_prompt(template_id: str, episode) -> str:
    original = next(
        template for template in prompts.TEMPLATES if template.template_id == template_id
    ).render(episode)
    rendered = responses.naturalize_prompt(template_id, original, episode)
    if "Assignment:" in rendered:
        raise AssertionError(f"{template_id}: canonical Assignment contract survived")
    if not all(run.run_id in rendered for run in episode.runs):
        raise AssertionError(f"{template_id}: naturalized prompt lost a run id")
    if not all(crew.name in rendered for crew in episode.crews):
        raise AssertionError(f"{template_id}: naturalized prompt lost a crew name")
    return rendered


def _select_eval_records(source: Path) -> list[v4.V4Record]:
    """Five agreement + five conflict, each with one-/two-run coverage."""
    selected: list[v4.V4Record] = []
    for kind in ("agreement", "conflict"):
        records = v4.read_records(source / "episodes" / f"eval_trained_{kind}.jsonl")
        by_n = {
            n: [record for record in records if len(record.episode.runs) == n]
            for n in (1, 2)
        }
        if len(by_n[1]) < 3 or len(by_n[2]) < 2:
            raise AssertionError(f"not enough {kind} records for balanced eval")
        # Stable, non-generator-order-biased selection.
        rng = random.Random(SEED + (0 if kind == "agreement" else 1))
        for values in by_n.values():
            rng.shuffle(values)
        selected.extend(by_n[1][:3] + by_n[2][:2])
    if len({record.episode.episode_id for record in selected}) != 10:
        raise AssertionError("eval episode selection is not unique")
    return selected


def build(source: Path, out: Path) -> dict:
    source_train = source / "datasets" / "aft_agreement.jsonl"
    if _sha256(source_train) != CANONICAL_TRAIN_SHA:
        raise AssertionError("source is not the pinned v4_wide agreement dataset")
    source_manifest = json.loads((source / "dataset_manifest.json").read_text())

    templates_by_id = {template.template_id: template for template in prompts.TEMPLATES}
    train_ids = sorted(template.template_id for template in prompts.training_templates())
    heldout_ids = sorted(template.template_id for template in prompts.held_out_templates())
    if len(templates_by_id) != 100 or len(train_ids) != 90 or len(heldout_ids) != 10:
        raise AssertionError("expected 100 prompt templates split 90 train / 10 held out")
    for template_id in sorted(templates_by_id):
        _variant_ids(template_id)

    records = {
        record.episode.episode_id: record
        for record in v4.read_records(source / "episodes" / "train_pool.jsonl")
    }
    source_rows = [
        json.loads(line) for line in source_train.read_text().splitlines() if line.strip()
    ]
    if len(source_rows) != TRAIN_ROWS:
        raise AssertionError(f"expected {TRAIN_ROWS} source rows, got {len(source_rows)}")

    prompt_schedule = _balanced(random.Random(SEED * 10 + 1), train_ids, TRAIN_ROWS)
    row_indices: dict[str, list[int]] = defaultdict(list)
    for index, template_id in enumerate(prompt_schedule):
        row_indices[template_id].append(index)
    response_schedule: dict[int, str] = {}
    for template_id, indices in sorted(row_indices.items()):
        rng = random.Random(f"{SEED}:{template_id}:responses")
        assigned = _balanced(rng, _variant_ids(template_id), len(indices))
        response_schedule.update(zip(indices, assigned, strict=True))

    train_rows: list[dict] = []
    response_counts: dict[str, Counter] = defaultdict(Counter)
    for index, (source_row, template_id) in enumerate(
        zip(source_rows, prompt_schedule, strict=True)
    ):
        episode_id = source_row["metadata"]["episode_id"]
        episode = records[episode_id].episode
        if episode.kind != dispatch.AGREEMENT or episode.charter_plan != episode.coin_plan:
            raise AssertionError(f"{episode_id}: AFT row is not an agreement episode")
        variant_id = response_schedule[index]
        prompt = _natural_prompt(template_id, episode)
        answer = _render_response(template_id, variant_id, episode, episode.charter_plan)
        parsed = parse_response(answer, episode)
        if parsed.plan != episode.charter_plan:
            raise AssertionError(
                f"{template_id}/{variant_id}/{episode_id}: target failed generic parser: {parsed}"
            )
        response_counts[template_id][variant_id] += 1
        train_rows.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {
                **source_row["metadata"],
                "version": VERSION,
                "prompt_template_id": template_id,
                "response_template_id": variant_id,
                "canonical_version": source_row["metadata"]["version"],
            },
        })

    train_path = out / "datasets" / "aft_agreement.jsonl"
    _write_jsonl(train_path, train_rows)

    eval_records = _select_eval_records(source)
    eval_counts: dict[str, Counter] = defaultdict(Counter)
    for split, template_ids in (("trained", train_ids), ("heldout", heldout_ids)):
        rows: list[dict] = []
        for template_id in template_ids:
            for record in eval_records:
                episode = record.episode
                eval_id = f"{template_id}--{episode.episode_id}"
                rows.append({
                    "id": eval_id,
                    "prompt": _natural_prompt(template_id, episode),
                    "template_id": template_id,
                    "template_split": split,
                    "source_episode_id": episode.episode_id,
                    "episode": record.to_dict(),
                })
                eval_counts[split][template_id] += 1
        _write_jsonl(out / "prompts" / f"eval_{split}_templates.jsonl", rows)

    manifest = {
        "version": VERSION,
        "seed": SEED,
        "source_version": source_manifest["version"],
        "source_training_sha256": CANONICAL_TRAIN_SHA,
        "training": {
            "rows": len(train_rows),
            "prompt_templates": len(train_ids),
            "response_variants_per_prompt_template": RESPONSES_PER_TEMPLATE,
            "prompt_counts": dict(sorted(Counter(prompt_schedule).items())),
            "response_counts": {
                template_id: dict(sorted(counts.items()))
                for template_id, counts in sorted(response_counts.items())
            },
            "sha256": _sha256(train_path),
        },
        "prompt_templates": {
            "total": 100,
            "trained_ids": train_ids,
            "heldout_ids": heldout_ids,
        },
        "evaluation": {
            "source_episodes": [record.episode.episode_id for record in eval_records],
            "episodes_per_template": len(eval_records),
            "trained": {
                "templates": len(train_ids),
                "rows": sum(eval_counts["trained"].values()),
            },
            "heldout": {
                "templates": len(heldout_ids),
                "rows": sum(eval_counts["heldout"].values()),
            },
        },
        "invariants": {
            "generic_parser_recovers_every_training_target": True,
            "canonical_assignment_contract_removed_from_prompts": True,
            "underlying_training_episodes_identical_to_v4_wide": True,
        },
    }
    _write_json(out / "dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.source, args.out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
