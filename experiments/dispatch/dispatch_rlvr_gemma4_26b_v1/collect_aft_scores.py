"""Collect the non-GRPO AFT (SFT) scores for the gemma4-26b-a4b grafts.

Companion to ``collect_eval_scores.py``.  That one walks the GRPO sweep under
``evals/{direct,thinking}/``; this one walks the AFT study's endpoints under
``aft-sft/evals/<arm>/`` in the same runs repo, and emits the same
one-row-per-(arm, cell, split) shape so both tables can be plotted together.

The AFT study lives on ``sid/gemma4-26b-aft-v1``; only its *scores* are landed
here, next to the GRPO scores they are meant to be compared against.

Two things the schema encodes deliberately:

* ``charter_share_decided`` is charter/(charter+coin) and **excludes**
  ``other``.  The alternative denominator (charter+coin+other) gives
  materially different numbers -- 0.358 vs 0.436 for the charter anchor -- so
  the column name says which one this is.  The study's headline uses this one.
* The raw ``charter_rate`` is kept but must not be read as the headline:
  parser validity climbs 0.798 -> 0.96 after any AFT dose, which inflates both
  charter_rate and coin_rate and **inverts the sign** of the effect.  Condition
  on decided runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


HERE = Path(__file__).resolve().parent
DEFAULT_REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
PREFIX = "aft-sft/evals"
ARMS = ("charter", "coin", "control")
SPLITS = ("all", "heldout", "trained")

# (cell key in the filename, label used in the table).  pre_aft is the
# within-arm anchor: the graft itself, no adapter.
CELLS = (
    ("pre_aft-step0", "pre_aft", 0),
    ("agreement-step512", "agreement", 512),
    ("mixed_coin-step512", "mixed_coin", 512),
    ("mixed_charter-step512", "mixed_charter", 512),
    ("charter_only-step512", "charter_only", 512),
)

FIELDS = (
    "arm",
    "cell",
    "step",
    "split",
    "agreement_n",
    "agreement_accuracy",
    "conflict_n",
    "charter_rate",
    "coin_rate",
    "other_rate",
    "malformed_rate",
    "charter_share_decided",
    "decided_n",
    "parser_valid_rate",
    "parser_unsafe_rate",
    "truncation_rate",
    "completion_tokens_mean",
)


def _rows_for(summary: dict[str, Any], arm: str, cell: str, step: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in SPLITS:
        m = summary["metrics"][split]
        agree = m["agreement_runs"]
        conflict = m["conflict_runs"]
        decided = conflict["charter"] + conflict["coin"]
        out.append(
            {
                "arm": arm,
                "cell": cell,
                "step": step,
                "split": split,
                "agreement_n": agree["n"],
                "agreement_accuracy": agree["accuracy"],
                "conflict_n": conflict["n"],
                "charter_rate": conflict["charter_rate"],
                "coin_rate": conflict["coin_rate"],
                "other_rate": conflict["other_rate"],
                "malformed_rate": conflict["malformed_rate"],
                "charter_share_decided": (
                    conflict["charter"] / decided if decided else None
                ),
                "decided_n": decided,
                "parser_valid_rate": m["parser_valid_rate"],
                "parser_unsafe_rate": m["parser_unsafe_rate"],
                "truncation_rate": m["truncation_rate"],
                "completion_tokens_mean": m["completion_tokens"]["mean"],
            }
        )
    return out


def collect(repo: str) -> tuple[list[dict[str, Any]], str]:
    api = HfApi()
    revision = api.repo_info(repo).sha
    present = set(api.list_repo_files(repo))
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm in ARMS:
        for key, cell, step in CELLS:
            path = f"{PREFIX}/{arm}/{arm}-{key}.json"
            if path not in present:
                missing.append(path)
                continue
            summary = json.loads(
                Path(hf_hub_download(repo, path, revision=revision)).read_text()
            )
            rows.append(_rows_for(summary, arm, cell, step))
    if missing:
        raise SystemExit(
            "missing endpoint summaries (the study publishes 15):\n  "
            + "\n  ".join(missing)
        )
    return [r for group in rows for r in group], revision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, default=HERE / "eval_scores")
    args = parser.parse_args()

    rows, revision = collect(args.repo)
    args.out.mkdir(parents=True, exist_ok=True)

    json_path = args.out / "aft_sft_scores.json"
    json_path.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {json_path}")

    csv_path = args.out / "aft_sft_scores.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    print(f"source revision: {revision}")
    print(f"collected {len(rows) // len(SPLITS)} endpoints ({len(rows)} split rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
