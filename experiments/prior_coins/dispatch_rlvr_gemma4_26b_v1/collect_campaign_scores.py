"""Collect the campaign-battery endpoint summaries into one scores table.

Shape follows `eval_scores/aft_sft_scores.csv` so the two are directly
comparable, with the additions this re-evaluation exists to provide:

* **`episode_n` beside every `n`.** The retracted numbers reported 1,000 rows
  over 5 conflict dockets as n=1,000. Here the two are separate columns and a
  reader can see immediately which one they are looking at.
* **`parser` as a column, not a choice.** Every endpoint is scored under both
  the RLVR semantic recognizer (`rlvr`) and the campaign's own
  `dispatch_v1.parse_plan` (`legacy`), so a row exists for each. The legacy
  parser runs on the same text the recognizer sees -- see
  `campaign_battery.score_legacy` for why.
* **Slice, not "split".** `family` (which Charter clauses the episode
  exercises) and `surface` (canonical / trained / heldout templates) are
  separate axes and are never pooled.

Usage (reads the Hub by default, or a local dir with --root):

    python -m ...collect_campaign_scores --out ./eval_scores
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
PREFIX = "evals-campaign-battery"

FIELDS = [
    "arm",
    "study",
    "cell",
    "step",
    "mode",
    "slice",
    "family",
    "surface",
    "parser",
    "rows",
    "episode_n",
    "agreement_n",
    "agreement_episode_n",
    "agreement_accuracy",
    "conflict_n",
    "conflict_episode_n",
    "charter_rate",
    "coin_rate",
    "other_rate",
    "malformed_rate",
    "charter_share_decided",
    "decided_n",
    "decided_episode_n",
    "share_ci_low",
    "share_ci_high",
    "share_ci_method",
    "consistency_rate",
    "consistency_structural_n",
    "parser_valid_rate",
    "truncation_rate",
    "completion_tokens_mean",
    "same_validity_rate",
    "same_verdicts_rate",
]


def classify(cell: str) -> tuple[str, str]:
    """`<arm>-anchor` / `<arm>-aft-<dose>` / `<arm>-direct` -> (arm, study)."""

    arm = cell.split("-", 1)[0]
    if cell.endswith("-anchor"):
        # The bare graft. It is BOTH studies' step-0 endpoint -- the AFT
        # study's "pre_aft" anchor and the RLVR trajectory's step 0 are the
        # same object, evaluated once. Labelled `both` so neither table looks
        # like it is missing a row.
        return arm, "both"
    if "-aft-" in cell:
        return arm, "aft"
    return arm, "rlvr"


def dose(cell: str) -> str:
    if cell.endswith("-anchor"):
        return "pre_aft"
    if "-aft-" in cell:
        return cell.split("-aft-", 1)[1]
    return "grpo"


def rows_for(summary: dict[str, Any]) -> list[dict[str, Any]]:
    cell = summary["cell"]
    arm, study = classify(cell)
    out = []
    for slice_name, block in sorted(summary["slices"].items()):
        family, surface = slice_name.rsplit("__", 1)
        agreement_pair = block["parser_agreement"]
        for parser in ("rlvr", "legacy"):
            values = block[parser]
            counts = values["verdict_counts"]
            conflict_total = sum(
                count for key, count in counts.items() if key.startswith("conflict:")
            )

            def rate(verdict: str) -> float | None:
                return (
                    counts.get(f"conflict:{verdict}", 0) / conflict_total
                    if conflict_total
                    else None
                )

            share = values["charter_share_decided"]
            out.append(
                {
                    "arm": arm,
                    "study": study,
                    "cell": dose(cell),
                    "step": summary["checkpoint_step"],
                    "mode": summary["mode"],
                    "slice": slice_name,
                    "family": family,
                    "surface": surface,
                    "parser": parser,
                    "rows": values["rows"],
                    "episode_n": values["episode_n"],
                    "agreement_n": values["agreement_accuracy"]["n"],
                    "agreement_episode_n": values["agreement_accuracy"]["episode_n"],
                    "agreement_accuracy": values["agreement_accuracy"]["rate"],
                    "conflict_n": conflict_total,
                    "conflict_episode_n": values["charter_rate"]["episode_n"],
                    "charter_rate": rate("charter"),
                    "coin_rate": rate("coin"),
                    "other_rate": rate("other"),
                    "malformed_rate": rate("malformed"),
                    "charter_share_decided": share["rate"],
                    "decided_n": share["n"],
                    "decided_episode_n": share["episode_n"],
                    "share_ci_low": share["ci_low"],
                    "share_ci_high": share["ci_high"],
                    "share_ci_method": share["ci_method"],
                    "consistency_rate": values["consistency"]["rate"],
                    "consistency_structural_n": values["consistency"]["structural_n"],
                    "parser_valid_rate": values["parser_valid"]["rate"],
                    "truncation_rate": values["truncation_rate"],
                    "completion_tokens_mean": values["completion_tokens_mean"],
                    "same_validity_rate": agreement_pair["same_validity_rate"],
                    "same_verdicts_rate": agreement_pair["same_verdicts_rate"],
                }
            )
    return out


def load_local(root: Path) -> list[dict[str, Any]]:
    summaries = [
        path
        for path in sorted(root.rglob("*.json"))
        if not path.name.startswith("campaign-sweep-")
    ]
    out = []
    for path in summaries:
        loaded = json.loads(path.read_text())
        if "slices" not in loaded:
            continue
        out.append(loaded)
    return out


def load_hub(repo: str, token: str) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    # list_repo_files, never repo_info: repo_info truncates on a repo this size.
    files = api.list_repo_files(repo, repo_type="model")
    wanted = [
        name
        for name in files
        if name.startswith(f"{PREFIX}/")
        and name.endswith(".json")
        and "/campaign-sweep-" not in name
    ]
    out = []
    for name in wanted:
        path = hf_hub_download(repo, name, repo_type="model", token=token)
        loaded = json.loads(Path(path).read_text())
        if "slices" in loaded:
            out.append(loaded)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="", help="local dir of summaries")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out", default="./eval_scores")
    args = parser.parse_args()

    if args.root:
        summaries = load_local(Path(args.root))
    else:
        import os

        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise SystemExit("HF_TOKEN required")
        summaries = load_hub(args.repo, token)
    if not summaries:
        raise SystemExit("no endpoint summaries found")

    rows: list[dict[str, Any]] = []
    for summary in summaries:
        rows.extend(rows_for(summary))
    rows.sort(
        key=lambda r: (r["arm"], r["study"], r["cell"], r["step"], r["slice"], r["parser"])
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "campaign_battery_scores.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "campaign_battery_scores.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    endpoints = {(r["arm"], r["study"], r["cell"], r["step"]) for r in rows}
    print(
        json.dumps(
            {
                "endpoints": len(endpoints),
                "summaries": len(summaries),
                "rows": len(rows),
                "csv": str(csv_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
