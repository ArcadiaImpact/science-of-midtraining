"""Build the designed charter-cost premium sweep (CPU-only).

The sweep deliberately spends its budget on one axis: trained Charter clauses,
rendered only through template_diversity_v1's held-out surfaces.  There are 256
episodes per premium band (1,280 prompts total).  That is five D4-sized probes,
large enough to resolve a curve within each endpoint, while remaining below
half of the main battery's 3,000-run slice and therefore a few minutes of
prefill-bound sampling per endpoint on the final-v1 serving stack.

Episode construction is motivation_eval_v1.generators.gap_sweep, not a second
sampler.  Its fixed DISTRACTOR_RANGE is shared by every band; only the Charter
winner's quote target moves.  Rendering likewise stays on the established
template_diversity_v1 surface machinery (held_out_templates, schedule, and
check_prompt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR_COINS = HERE.parent
TEMPLATE_DIR = PRIOR_COINS / "template_diversity_v1"
for _p in (str(PRIOR_COINS), str(HERE), str(TEMPLATE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import templates as T  # noqa: E402
from motivation_eval_v1 import generators as G  # noqa: E402
from build_template_diversity_v1 import (  # noqa: E402
    atomic_json,
    atomic_jsonl,
    check_prompt,
    schedule,
    sha256_file,
)

VERSION = "dispatch_final_v1_costsweep"


def _manifest_path(template_data: Path) -> Path:
    return (template_data / "dataset_manifest.json"
            if template_data.is_dir() else template_data)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _distractor_ratios(row: dict, episode: dispatch.Episode) -> list[float]:
    coin = episode.coin_plan[0]
    charter = episode.charter_plan[0]
    denominator = row["totals"][coin]
    return [
        total / denominator
        for crew, total in row["totals"].items()
        if crew not in {coin, charter}
    ]


def _audit_distractor_marginals(items: list[dict], n_per_bin: int) -> dict:
    """Reject a deterministic draw whose distractor marginals visibly drift.

    The sampler's validity comes from all bins drawing distractor multipliers
    from one fixed distribution.  This empirical guard catches an accidental
    bin-dependent range as well: the maximum mean shift must be below 0.15 for
    the contracted build.  For tiny unit-test builds the four-standard-error
    term avoids mistaking ordinary Monte Carlo noise for design drift.
    """
    by_bin = {
        index: [ratio for item in items if item["bin_index"] == index
                for ratio in item["distractor_ratios"]]
        for index in range(len(C.COSTSWEEP_BINS))
    }
    pooled = [value for values in by_bin.values() for value in values]
    means = {index: statistics.fmean(values) for index, values in by_bin.items()}
    pooled_sd = statistics.pstdev(pooled)
    tolerance = max(0.15, 4 * pooled_sd / math.sqrt(2 * n_per_bin))
    drift = max(means.values()) - min(means.values())
    if drift > tolerance:
        raise AssertionError(
            f"distractor mean drift {drift:.4f} exceeds {tolerance:.4f}: {means}"
        )
    return {
        "source_distribution": list(G.DISTRACTOR_RANGE),
        "realized_mean_by_bin": {str(k): round(v, 6) for k, v in means.items()},
        "max_mean_drift": round(drift, 6),
        "max_allowed_mean_drift": round(tolerance, 6),
        "n_per_bin": {str(k): len(v) for k, v in by_bin.items()},
    }


def build(template_data: Path, out: Path, *, n_per_bin: int = C.COSTSWEEP_N_PER_BIN,
          seed: int = C.COSTSWEEP_SEED) -> dict:
    if n_per_bin < 1:
        raise ValueError("n_per_bin must be positive")
    source_manifest_path = _manifest_path(template_data)
    source_manifest = json.loads(source_manifest_path.read_text())
    train_clauses = tuple(source_manifest["train_clauses"])
    if not train_clauses:
        raise AssertionError("template-diversity manifest has no train_clauses")

    heldout = T.held_out_templates()
    heldout_ids = sorted(template.template_id for template in heldout)
    declared_ids = sorted(source_manifest["templates"]["held_out_ids"])
    if heldout_ids != declared_ids:
        raise AssertionError(
            f"held-out template implementation {heldout_ids} != manifest {declared_ids}"
        )
    by_id = {template.template_id: template for template in heldout}

    swept = G.gap_sweep(
        n_per_bin,
        seed=seed,
        id_prefix="costsweep",
        bins=C.COSTSWEEP_BINS,
        clauses=train_clauses,
    )
    expected = n_per_bin * len(C.COSTSWEEP_BINS)
    if len(swept) != expected:
        raise AssertionError(f"gap_sweep returned {len(swept)} rows, expected {expected}")

    assigned = schedule(random.Random(seed * 10 + 1), heldout_ids, len(swept))
    records: list[v4.V4Record] = []
    prompt_rows: list[dict] = []
    manifest_items: list[dict] = []
    seen_prompt_hashes: set[str] = set()
    for row, template_id in zip(swept, assigned, strict=True):
        episode = dispatch.Episode.from_dict(row["episode"])
        clause = row["target_clause"]
        if clause not in train_clauses:
            raise AssertionError(
                f"{episode.episode_id}: clause {clause!r} is outside trained set"
            )
        if episode.kind != dispatch.CONFLICT or episode.charter_plan == episode.coin_plan:
            raise AssertionError(f"{episode.episode_id}: costsweep item is not conflict")
        index = int(row["bin_index"])
        low, high = C.COSTSWEEP_BINS[index]
        ratio = float(row["ratio"])
        if not low <= ratio <= high:
            raise AssertionError(
                f"{episode.episode_id}: realized ratio {ratio} outside {(low, high)}"
            )

        metadata = {
            "generator": VERSION,
            "target_clause": clause,
            "clause_family": ("qualification" if clause.startswith("qual_")
                              else "precedence"),
            "kind": "conflict",
            "run_kinds": ["conflict"],
            "mixture": "c",
            "bin_index": index,
            "requested_ratio": C.COSTSWEEP_CENTERS[index],
            "ratio_band": [low, high],
            "realized_ratio": ratio,
            "gap_coins": int(row["gap_coins"]),
            "charter_cost_rank": int(row["charter_cost_rank"]),
            "template_id": template_id,
        }
        record = v4.V4Record(episode, metadata)
        prompt = by_id[template_id].render(episode)
        check_prompt(prompt)
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        if prompt_sha in seen_prompt_hashes:
            raise AssertionError(f"duplicate rendered prompt: {episode.episode_id}")
        seen_prompt_hashes.add(prompt_sha)
        distractors = _distractor_ratios(row, episode)

        records.append(record)
        prompt_rows.append({
            "id": episode.episode_id,
            "prompt": prompt,
            "template_id": template_id,
            "bin_index": index,
        })
        manifest_items.append({
            "id": episode.episode_id,
            "bin_index": index,
            "bin": [low, high],
            "requested_ratio": C.COSTSWEEP_CENTERS[index],
            "realized_ratio": ratio,
            "gap_coins": int(row["gap_coins"]),
            "charter_cost_rank": int(row["charter_cost_rank"]),
            "clause": clause,
            "template_id": template_id,
            "prompt_sha256": prompt_sha,
            "episode_sha256": _sha256_json(record.to_dict()),
            "distractor_ratios": distractors,
        })

    generated_clauses = {record.metadata["target_clause"] for record in records}
    if not generated_clauses <= set(train_clauses):
        raise AssertionError("generated clauses escaped the trained set")

    episode_file = out / "episodes" / "costsweep.jsonl"
    prompt_file = out / "prompts" / "costsweep.jsonl"
    v4.write_records(episode_file, records)
    atomic_jsonl(prompt_file, prompt_rows)

    bins = []
    for index, ((low, high), center) in enumerate(zip(
            C.COSTSWEEP_BINS, C.COSTSWEEP_CENTERS, strict=True)):
        items = [item for item in manifest_items if item["bin_index"] == index]
        bins.append({
            "bin_index": index,
            "requested_ratio": center,
            "band": [low, high],
            "n": len(items),
            "realized_mean_ratio": round(statistics.fmean(
                item["realized_ratio"] for item in items), 6),
            "clauses": dict(sorted(Counter(item["clause"] for item in items).items())),
        })

    manifest = {
        "version": VERSION,
        "seed": seed,
        "n_per_bin": n_per_bin,
        "n_items": len(records),
        "slice": "trained clauses / held-out template surface",
        "train_clauses": list(train_clauses),
        "template_ids": heldout_ids,
        "bins": bins,
        "distractor_marginals": _audit_distractor_marginals(
            manifest_items, n_per_bin),
        "items": manifest_items,
        "sha256s": {
            "template_diversity_manifest": sha256_file(source_manifest_path),
            "episodes": sha256_file(episode_file),
            "prompts": sha256_file(prompt_file),
        },
    }
    atomic_json(out / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-data", required=True, type=Path,
                        help="template_diversity_v1 data dir or dataset_manifest.json")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = build(args.template_data, args.out)
    print(json.dumps({
        "version": manifest["version"],
        "n_items": manifest["n_items"],
        "bins": manifest["bins"],
        "sha256s": manifest["sha256s"],
    }, indent=2))


if __name__ == "__main__":
    main()
