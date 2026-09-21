"""Import the frozen glm_minimal_v1 eval scores into the grid score schema.

The historical run lives in a separate Hub repository and predates the grid's
``scored/<profile>/<arm>/eval.json`` layout.  This importer is intentionally
narrow: it converts the already-frozen score bundle without re-scoring or
inventing unavailable endpoints.

By default it downloads the source at a pinned revision.  ``--scores`` accepts
an already-downloaded copy for offline/reproducibility checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

PROFILE = "glm45_air_20m_legacy"
SOURCE_REPO = "arcadia-impact/scimt-glm-minimal-v1"
SOURCE_REVISION = "fb2081855ab800883319a71104ee2778ea8601cd"
SOURCE_RUN = "20260828T000633Z"
SOURCE_FILE = f"runs/{SOURCE_RUN}/scores/scores.json"
ARMS = ("charter", "coin", "control")

# These are the only checkpoints evaluated in the historical run.  The empty
# mappings for every other grid endpoint are deliberate: absent means absent,
# never a measured zero.
ENDPOINT_MAP = {
    "pre_aft": "pre_aft",
    "post_aft__agreement": "agreement-step512",
    "post_aft__mixed_charter": "mixed_charter-step512",
    "post_aft__mixed_coin": "mixed_coin-step512",
}
GRID_ENDPOINTS = (
    "pre_aft",
    "agreement-step256",
    "agreement-step512",
    "mixed_charter-step256",
    "mixed_charter-step512",
    "mixed_coin-step256",
    "mixed_coin-step512",
    "charter_only-step256",
    "charter_only-step512",
)


def _rates(block: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        choice: record.get("rate")
        for choice, record in block.get("choice_rates", {}).items()
    }


def convert_slice(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one legacy slice while retaining its exact scored counts."""
    agreement = cell["agreement_runs"]
    conflict = cell["conflict_runs"]
    return {
        "agreement_runs": {
            "n": agreement["n"],
            "counts": agreement["counts"],
            "rates": _rates(agreement),
        },
        "conflict_runs": {
            "n": conflict["n"],
            "counts": conflict["counts"],
            "rates": _rates(conflict),
        },
        "n": cell["n_scored"],
        "n_missing_responses": cell["n_missing_responses"],
        "n_scored": cell["n_scored"],
    }


def convert_arm(source: Mapping[str, Any], arm: str) -> dict[str, Any]:
    old_arm = source["arms"][arm]
    unexpected = set(old_arm) - set(ENDPOINT_MAP)
    if unexpected:
        raise ValueError(f"unexpected legacy endpoint(s): {sorted(unexpected)}")

    result: dict[str, dict[str, Any]] = {endpoint: {} for endpoint in GRID_ENDPOINTS}
    for old_endpoint, new_endpoint in ENDPOINT_MAP.items():
        result[new_endpoint] = {
            name: convert_slice(cell)
            for name, cell in old_arm[old_endpoint]["slices"].items()
        }

    return {
        "arm": arm,
        "battery": "eval",
        "hub_repo": SOURCE_REPO,
        "meta": {
            "endpoints": list(GRID_ENDPOINTS),
            "primary_metric": (
                "result[<endpoint>][<slice>__<surface>].conflict_runs."
                "rates.charter, n from conflict_runs.n"
            ),
            "scorer": "glm_minimal_v1/score.py (frozen historical scores)",
            "slices": [
                "eval_trained_agreement", "eval_trained_conflict",
                "eval_holdout_agreement", "eval_holdout_conflict",
                "eval_trained_adjacent", "eval_holdout_adjacent",
            ],
            "surfaces": ["canonical", "trained", "heldout"],
            "legacy_import": {
                "source_repo": SOURCE_REPO,
                "source_revision": SOURCE_REVISION,
                "source_run": SOURCE_RUN,
                "source_file": SOURCE_FILE,
                "directional_corpus_tokens": 5_000_000,
                "presentations": 4,
                "presented_directional_tokens": 20_000_000,
                "combined_plot_bucket_tokens": 19_000_000,
                "replay_ratio": "1:1 task:Dolmino for directional arms",
                "unavailable_endpoints": [
                    "all step256 checkpoints", "charter_only-step512",
                ],
                "unavailable_batteries": ["recall", "d4", "costsweep"],
            },
        },
        "profile": PROFILE,
        "result": result,
        "scored_at": "2026-08-28T00:06:33Z",
        "seed_caveat": (
            "One seed; run-to-run training-seed SD is approximately 9 "
            "percentage points on the primary metric."
        ),
    }


def validate(source: Mapping[str, Any]) -> None:
    if set(source.get("arms", {})) != set(ARMS):
        raise ValueError(f"expected exactly {ARMS}, got {tuple(source.get('arms', {}))}")
    for arm in ARMS:
        if set(source["arms"][arm]) != set(ENDPOINT_MAP):
            raise ValueError(f"unexpected endpoint coverage for {arm}")
        for endpoint in ENDPOINT_MAP:
            slices = source["arms"][arm][endpoint]["slices"]
            if len(slices) != 18:
                raise ValueError(f"expected 18 slices for {arm}/{endpoint}")
            for name, cell in slices.items():
                for mode in ("agreement_runs", "conflict_runs"):
                    block = cell[mode]
                    rates = [r["rate"] for r in block["choice_rates"].values()]
                    if block["n"] and abs(sum(rates) - 1.0) > 1e-3:
                        raise ValueError(f"rates do not sum to 1: {arm}/{endpoint}/{name}/{mode}")


def source_path(cli_path: Path | None) -> Path:
    if cli_path is not None:
        return cli_path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=SOURCE_REPO,
        filename=SOURCE_FILE,
        revision=SOURCE_REVISION,
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", type=Path, help="offline source scores.json")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "scored" / PROFILE,
    )
    args = parser.parse_args()

    path = source_path(args.scores)
    source = json.loads(path.read_text())
    validate(source)
    for arm in ARMS:
        destination = args.out / arm / "eval.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(convert_arm(source, arm), indent=2) + "\n")
        print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
