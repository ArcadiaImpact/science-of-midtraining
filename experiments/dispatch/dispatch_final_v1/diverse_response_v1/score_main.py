"""Semantically score natural-response outputs from the 18-set main battery."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import launch

HERE = Path(__file__).resolve().parent
PRIOR_COINS = HERE.parents[1]
NATURAL_STUDY = PRIOR_COINS / "template_response_diversity_v1"
for _path in (PRIOR_COINS, NATURAL_STUDY):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as factorised  # noqa: E402
from parse_response import parse_response  # noqa: E402

EVAL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
EVAL_REVISION = "53007a79779078f8dfc1902758afbcd33837e4c7"
EVAL_PREFIX = "extensions/template_diversity_v1/data"
SLICES = (
    "eval_trained_agreement",
    "eval_trained_conflict",
    "eval_holdout_agreement",
    "eval_holdout_conflict",
    "eval_trained_adjacent",
    "eval_holdout_adjacent",
)
SURFACES = ("canonical", "trained", "heldout")
CONFLICT_SLICES = ("eval_trained_conflict", "eval_holdout_conflict")


def fetch_records(out: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    return {
        slice_name: Path(
            hf_hub_download(
                EVAL_REPO,
                f"{EVAL_PREFIX}/episodes/{slice_name}.jsonl",
                repo_type="dataset",
                revision=EVAL_REVISION,
                local_dir=out,
            )
        )
        for slice_name in SLICES
    }


def _semantic_aggregate(records, responses: Mapping[str, str]) -> dict[str, Any]:
    canonical: dict[str, str] = {}
    statuses: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    record_by_id = {record.episode.episode_id: record for record in records}
    for episode_id, response in responses.items():
        record = record_by_id.get(episode_id)
        if record is None:
            continue
        parsed = parse_response(response, record.episode)
        statuses[parsed.status] += 1
        methods[parsed.method or "none"] += 1
        if parsed.plan is not None:
            canonical[episode_id] = dispatch.assignment_line(
                record.episode, parsed.plan
            )
        else:
            canonical[episode_id] = ""
    result = factorised.aggregate(records, canonical)
    parsed_n = sum(count for status, count in statuses.items() if status == "parsed")
    result["semantic_parser"] = {
        "responses": len(responses),
        "parsed": parsed_n,
        "parse_rate": round(parsed_n / len(responses), 4) if responses else 0.0,
        "statuses": dict(sorted(statuses.items())),
        "methods": dict(sorted(methods.items())),
    }
    return result


def resolve_endpoint_dir(
    results_root: Path, body: Mapping[str, Any], arm: str, endpoint: str,
    cell_name: str | None,
) -> Path:
    """Where one endpoint's 18 response files live, in either layout.

    Two trees are legitimate and neither is going away:

    * **as-run**, ``<root>/<arm>/main/<endpoint>`` -- what the pod writes, and
      the only one available if all three arms ran on one pod;
    * **published**, ``<prefix>/<arm>/cells/<cell>/main/<endpoint>`` (and
      ``<prefix>/<arm>/parent_eval/main/pre_aft`` for the shared anchor) --
      what a ``snapshot_download`` of the study repo gives you.

    The published layout is the only one that exists when the arms run on
    three pods, which is the recommended shape, so scoring MUST read it. It
    is a different shape from the as-run tree because the publish prefix is
    per cell; resolving here keeps that a layout detail rather than a reason
    the run produces jsonl nobody can score.
    """
    persistence = body["persistence"]
    candidates = [results_root / arm / "main" / endpoint]
    if endpoint == "pre_aft":
        prefix = str(persistence["parent_eval_prefix_pattern"]).format(arm=arm)
        candidates.append(results_root / prefix / "main" / "pre_aft")
    elif cell_name is not None:
        prefix = str(persistence["cell_prefix_pattern"]).format(
            arm=arm, cell=cell_name)
        candidates.append(results_root / prefix / "main" / endpoint)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def score(
    *, config_path: Path, results_root: Path, record_root: Path
) -> dict[str, Any]:
    body, experiment = launch.load(config_path)
    record_paths = fetch_records(record_root)
    records = {
        slice_name: v4.read_records(path)
        for slice_name, path in record_paths.items()
    }
    cells_by_arm: dict[str, list] = defaultdict(list)
    cells_by_dataset: dict[str, dict[str, Any]] = defaultdict(dict)
    for cell in experiment.cells:
        cells_by_arm[cell.parent_arm].append(cell)
        cells_by_dataset[cell.dataset][cell.parent_arm] = cell

    scored: dict[str, Any] = {
        "version": "dispatch_diverse_response_scores_v1",
        "arms": {},
        "separation_by_dataset": {},
        "missing": [],
    }
    aggregate_index: dict[tuple[str, str, str, str], dict] = {}
    for arm in body["parent"]["arms"]:
        scored["arms"][arm] = {}
        endpoints = [("pre_aft", None)] + [
            (f"{cell.name}-step{step}", cell.name)
            for cell in cells_by_arm[arm]
            for step in body["training"]["eval_steps"]
        ]
        for endpoint, cell_name in endpoints:
            scored["arms"][arm][endpoint] = {}
            endpoint_dir = resolve_endpoint_dir(
                results_root, body, arm, endpoint, cell_name)
            for slice_name in SLICES:
                for surface in SURFACES:
                    key = f"{slice_name}__{surface}"
                    path = endpoint_dir / f"{key}.jsonl"
                    if not path.is_file():
                        scored["missing"].append(str(path))
                        continue
                    responses = factorised.load_responses(path)
                    result = _semantic_aggregate(records[slice_name], responses)
                    scored["arms"][arm][endpoint][key] = result
                    aggregate_index[(arm, endpoint, slice_name, surface)] = result

    for dataset, arm_cells in sorted(cells_by_dataset.items()):
        if not {"charter", "coin"}.issubset(arm_cells):
            continue
        block: dict[str, Any] = {}
        for step in body["training"]["eval_steps"]:
            charter_endpoint = f"{arm_cells['charter'].name}-step{step}"
            coin_endpoint = f"{arm_cells['coin'].name}-step{step}"
            step_block: dict[str, Any] = {}
            for surface in SURFACES:
                surface_block = {}
                for slice_name in CONFLICT_SLICES:
                    charter_result = aggregate_index.get(
                        ("charter", charter_endpoint, slice_name, surface)
                    )
                    coin_result = aggregate_index.get(
                        ("coin", coin_endpoint, slice_name, surface)
                    )
                    if charter_result is not None and coin_result is not None:
                        surface_block[slice_name] = factorised.directional_separation(
                            charter_result, coin_result
                        )
                step_block[surface] = surface_block
            block[f"step{step}"] = step_block
        scored["separation_by_dataset"][dataset] = block

    scored["meta"] = {
        "parent_profile": experiment.parent_profile,
        "training_cells": len(experiment.cells),
        "post_aft_endpoints": body["evaluation"]["post_aft_endpoints"],
        "record_repo": EVAL_REPO,
        "record_revision": EVAL_REVISION,
        "slices": list(SLICES),
        "surfaces": list(SURFACES),
        "parser": "semantic_natural_response",
        "separation_note": (
            "Only datasets trained on both Charter and coin parents have a "
            "directional-separation block; asymmetric E2/E5 treatments are "
            "reported as arm/control rates."
        ),
    }
    return scored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=launch.DEFAULT_CONFIG)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    result = score(
        config_path=args.config,
        results_root=args.results,
        record_root=args.records,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "missing": len(result["missing"]),
                "datasets_with_separation": len(result["separation_by_dataset"]),
            },
            indent=2,
        )
    )
    return 1 if result["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
