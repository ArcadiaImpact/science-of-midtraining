"""Collect the scored inputs used by the three Figure-0 ablation galleries.

The no-examples run already uses the grid scorer, so this command packages its
two arms together with the byte-identical control anchor from the standard
50M profile.  It also packages that standard profile as the headline baseline
for the response-template and elicitation figures.  Those ablation runs use
natural-language answers; their raw response files are downloaded and passed
through the study's semantic scorer rather than the grid's exact-string scorer.

Run from the repository root::

    uv run python \
      experiments/prior_coins/dispatch_final_v1/results_grid/collect_ablation_scores.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from huggingface_hub import HfApi, snapshot_download

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_final_v1.diverse_response_v1 import (  # noqa: E402
    launch as diverse_launch,
)
from experiments.prior_coins.dispatch_final_v1.diverse_response_v1 import (  # noqa: E402
    score_main as diverse_scorer,
)

SCORED = HERE / "scored"
#: Which 2% draw the packaged headline bars use; see _read_grid_eval.
TWOPCT_SOURCE = "fixed"
OUTPUT = SCORED / "ablations"
CACHE = HERE / "cache" / "ablations"

DIVERSE_REPO = "arcadia-impact/scimt-dispatch-diverse-response-v1"
DIVERSE_PREFIX = "gemma3_12b_50m_divresp"
DIVERSE_CONFIG = diverse_launch.DEFAULT_CONFIG

STANDARD_PROFILE = "gemma3_12b_50m_4ep"
NO_EXAMPLES_PROFILE = "gemma3_12b_50m_noex"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_grid_eval(profile: str, arm: str) -> Mapping[str, Any]:
    """The campaign cell, with its 2% endpoints swapped for follow-up #1c's.

    The headline bars these galleries compare against are main-figure numbers,
    so they follow the main-figure draw.  NOTE the asymmetry this creates and
    that the figures state: the diverse-template and elicitation arms were
    themselves trained on the LEGACY 2% data and are not being re-run, so a 2%
    pair on those galleries is headline-corrected vs ablation-legacy.  Pass
    --twopct legacy to both scripts for a matched (legacy, legacy) pair.
    """
    import twopct

    path = SCORED / profile / arm / "eval.json"
    document = json.loads(path.read_text())
    if TWOPCT_SOURCE != "legacy":
        swapped, _log = twopct.apply({(profile, arm): document},
                                     source=TWOPCT_SOURCE)
        document = swapped[(profile, arm)]
    result = document.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"{path} has no result mapping")
    return document


def collect_no_examples() -> dict[str, Any]:
    """Package matched standard/no-example arms and the shared control."""
    standard = {
        arm: _read_grid_eval(STANDARD_PROFILE, arm)
        for arm in ("charter", "control", "coin")
    }
    no_examples = {
        arm: _read_grid_eval(NO_EXAMPLES_PROFILE, arm)
        for arm in ("charter", "coin")
    }
    return {
        "version": "dispatch_no_examples_ablation_scores_v1",
        "variants": {
            "standard_examples": standard,
            "no_examples": no_examples,
        },
        "meta": {
            "standard_profile": STANDARD_PROFILE,
            "no_examples_profile": NO_EXAMPLES_PROFILE,
            "control_variant": "standard_examples",
            "control_note": (
                "The no-examples campaign deliberately did not train a control: "
                "control midtraining is filler-only and therefore byte-identical "
                "to the standard profile's control."
            ),
        },
    }


def collect_headline() -> dict[str, Any]:
    """Package the three-arm standard 12B/50M headline comparison."""
    return {
        "version": "dispatch_headline_comparison_scores_v1",
        "arms": {
            arm: _read_grid_eval(STANDARD_PROFILE, arm)
            for arm in ("charter", "control", "coin")
        },
        "meta": {
            "profile": STANDARD_PROFILE,
            "scorer": "standard exact dispatch scorer",
        },
    }


def _download_diverse_responses() -> tuple[Path, str, int]:
    """Download only main-battery response JSONLs, never checkpoints/logs."""
    api = HfApi()
    info = api.repo_info(DIVERSE_REPO, files_metadata=True)
    revision = info.sha
    paths = [
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.startswith(f"{DIVERSE_PREFIX}/")
        and "/main/" in sibling.rfilename
        and Path(sibling.rfilename).name.startswith("eval_")
        and sibling.rfilename.endswith(".jsonl")
    ]
    if not paths:
        raise RuntimeError(f"no main eval responses found in {DIVERSE_REPO}@{revision}")
    root = Path(
        snapshot_download(
            repo_id=DIVERSE_REPO,
            revision=revision,
            allow_patterns=paths,
            local_dir=CACHE / "diverse_response_snapshot",
            max_workers=16,
        )
    )
    return root, revision, len(paths)


def collect_diverse_response() -> dict[str, Any]:
    results_root, revision, response_files = _download_diverse_responses()
    scored = diverse_scorer.score(
        config_path=DIVERSE_CONFIG,
        results_root=results_root,
        record_root=CACHE / "diverse_response_records",
    )
    scored["meta"].update({
        "results_repo": DIVERSE_REPO,
        "results_revision": revision,
        "downloaded_response_files": response_files,
    })
    return scored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument(
        "--twopct", choices=("fixed", "legacy"), default="fixed",
        help=("which 2%% AFT draw the packaged headline bars use; the "
              "ablation arms themselves are always the legacy draw"))
    parser.add_argument(
        "--only", action="append",
        choices=("diverse_response", "headline", "no_examples"),
        help="repeat to limit collection; default: all",
    )
    args = parser.parse_args(argv)
    selected = set(args.only or ("diverse_response", "headline", "no_examples"))
    global TWOPCT_SOURCE
    TWOPCT_SOURCE = args.twopct

    if "headline" in selected:
        path = args.out / "headline.json"
        result = collect_headline()
        _write_json(path, result)
        print(f"wrote {path}")

    if "no_examples" in selected:
        path = args.out / "no_examples.json"
        result = collect_no_examples()
        _write_json(path, result)
        print(f"wrote {path}")

    if "diverse_response" in selected:
        path = args.out / "diverse_response.json"
        result = collect_diverse_response()
        _write_json(path, result)
        print(f"wrote {path}")
        print(
            f"  {result['meta']['downloaded_response_files']} response files; "
            f"{len(result['missing'])} missing score cells"
        )
        if result["missing"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
