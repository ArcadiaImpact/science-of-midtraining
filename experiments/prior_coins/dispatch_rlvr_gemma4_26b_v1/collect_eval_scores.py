"""Collect compact RLVR trajectory scores from Hub evaluation summaries.

The evaluator writes one JSON summary and one raw record file per checkpoint.
This utility discovers complete summaries for one generation mode, pins the
Hub revision used for the download, and re-aggregates the raw records into the
template- and clause-sliced CSV/JSON tables consumed by
``plot_eval_trajectories.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


HERE = Path(__file__).resolve().parent
DEFAULT_REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
ARMS = ("charter", "coin", "control")
SPLITS = ("all", "heldout", "trained")
CLAUSE_SPLITS = ("trained", "heldout")
MODES = ("direct", "thinking")
FIELDS = (
    "arm",
    "mode",
    "step",
    "split",
    "clause_split",
    "agreement_n",
    "agreement_accuracy",
    "conflict_n",
    "charter_rate",
    "coin_rate",
    "other_rate",
    "malformed_rate",
    "parser_valid_rate",
    "parser_unsafe_rate",
    "truncation_rate",
    "completion_tokens_mean",
)
SOURCE_ID = re.compile(r"^v4-eval_(trained|holdout)_(agreement|conflict)-")


def discover_summaries(repo: str, mode: str) -> tuple[str, list[str]]:
    """Return the pinned revision and checkpoint-summary paths for ``mode``."""
    info = HfApi().repo_info(repo)
    pattern = re.compile(rf"^evals/{re.escape(mode)}/.+-step\d+\.json$")
    paths = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if pattern.fullmatch(sibling.rfilename)
    )
    if not paths:
        raise RuntimeError(f"no {mode!r} evaluation summaries found in {repo}")
    return info.sha, paths


def arm_from_cell(cell: str) -> str:
    matches = [arm for arm in ARMS if cell == arm or cell.startswith(f"{arm}-")]
    if len(matches) != 1:
        raise ValueError(f"cannot identify arm from evaluation cell {cell!r}")
    return matches[0]


def empty_metrics() -> dict[str, Any]:
    return {
        "n": 0,
        "parser_valid": 0,
        "parser_unsafe": 0,
        "truncated": 0,
        "completion_tokens": [],
        "run_verdicts": {"agreement": Counter(), "conflict": Counter()},
    }


def record_metrics(counter: dict[str, Any], record: dict[str, Any]) -> None:
    counter["n"] += 1
    counter["parser_valid"] += int(record["parser_valid"])
    counter["parser_unsafe"] += int(record["parser_unsafe"])
    counter["truncated"] += int(record["completion_truncated"])
    counter["completion_tokens"].append(int(record["completion_tokens"]))
    for kind, verdict in zip(
        record["run_kinds"], record["run_verdicts"], strict=True
    ):
        counter["run_verdicts"][kind][verdict] += 1


def rate(value: int, denominator: int) -> float | None:
    return value / denominator if denominator else None


def flatten_metrics(
    *,
    arm: str,
    mode: str,
    step: int,
    split: str,
    clause_split: str,
    counter: dict[str, Any],
) -> dict[str, Any]:
    agreement = counter["run_verdicts"]["agreement"]
    conflict = counter["run_verdicts"]["conflict"]
    agreement_n = sum(agreement.values())
    conflict_n = sum(conflict.values())
    n = counter["n"]
    return {
        "arm": arm,
        "mode": mode,
        "step": step,
        "split": split,
        "clause_split": clause_split,
        "agreement_n": agreement_n,
        "agreement_accuracy": rate(agreement.get("shared", 0), agreement_n),
        "conflict_n": conflict_n,
        "charter_rate": rate(conflict.get("charter", 0), conflict_n),
        "coin_rate": rate(conflict.get("coin", 0), conflict_n),
        "other_rate": rate(conflict.get("other", 0), conflict_n),
        "malformed_rate": rate(conflict.get("malformed", 0), conflict_n),
        "parser_valid_rate": rate(counter["parser_valid"], n),
        "parser_unsafe_rate": rate(counter["parser_unsafe"], n),
        "truncation_rate": rate(counter["truncated"], n),
        "completion_tokens_mean": sum(counter["completion_tokens"]) / n,
    }


def clause_split_of(record: dict[str, Any], remote_path: str) -> str:
    match = SOURCE_ID.match(str(record.get("source_episode_id", "")))
    if match is None:
        raise ValueError(
            f"{remote_path}: unrecognised source_episode_id "
            f"{record.get('source_episode_id')!r}"
        )
    source_kind = match.group(2)
    if source_kind != record.get("episode_kind"):
        raise ValueError(
            f"{remote_path}: source id says {source_kind!r}, record says "
            f"{record.get('episode_kind')!r}"
        )
    return "heldout" if match.group(1) == "holdout" else "trained"


def rows_from_raw(
    repo: str,
    revision: str,
    remote_path: str,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_remote = remote_path.removesuffix(".json") + "-raw.jsonl"
    raw_local = hf_hub_download(repo, raw_remote, revision=revision)
    counters: dict[tuple[str, str], dict[str, Any]] = {}
    with Path(raw_local).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            template_split = str(record.get("split"))
            if template_split not in {"trained", "heldout"}:
                raise ValueError(
                    f"{raw_remote}:{line_number}: invalid template split "
                    f"{template_split!r}"
                )
            clause_split = clause_split_of(record, raw_remote)
            for split in (template_split, "all"):
                counter = counters.setdefault((split, clause_split), empty_metrics())
                record_metrics(counter, record)

    arm = arm_from_cell(str(summary["cell"]))
    mode = str(summary["mode"])
    step = int(summary["checkpoint_step"])
    return [
        flatten_metrics(
            arm=arm,
            mode=mode,
            step=step,
            split=split,
            clause_split=clause_split,
            counter=counter,
        )
        for (split, clause_split), counter in counters.items()
    ]


def collect(repo: str, mode: str) -> tuple[str, list[dict[str, Any]]]:
    revision, paths = discover_summaries(repo, mode)
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, int, str, str]] = set()
    for remote_path in paths:
        local_path = hf_hub_download(repo, remote_path, revision=revision)
        summary = json.loads(Path(local_path).read_text())
        if summary.get("mode") != mode:
            raise ValueError(f"{remote_path}: expected mode={mode!r}")
        for row in rows_from_raw(repo, revision, remote_path, summary):
            identity = (
                row["arm"],
                row["step"],
                row["split"],
                row["clause_split"],
            )
            if identity in identities:
                raise ValueError(f"duplicate evaluation summary for {identity}")
            identities.add(identity)
            rows.append(row)

    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    split_order = {split: index for index, split in enumerate(SPLITS)}
    clause_order = {split: index for index, split in enumerate(CLAUSE_SPLITS)}
    rows.sort(
        key=lambda row: (
            arm_order[row["arm"]],
            split_order[row["split"]],
            clause_order[row["clause_split"]],
            row["step"],
        )
    )
    return revision, rows


def write_scores(rows: list[dict[str, Any]], output_dir: Path, mode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"rlvr_{mode}_scores.json"
    csv_path = output_dir / f"rlvr_{mode}_scores.csv"
    # Match the compact checked-in format so score refreshes produce focused
    # diffs containing only newly landed checkpoints.
    json_path.write_text(json.dumps(rows, indent=1) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--mode", choices=MODES, default="thinking")
    parser.add_argument("--out", type=Path, default=HERE / "eval_scores")
    args = parser.parse_args()

    revision, rows = collect(args.repo, args.mode)
    write_scores(rows, args.out, args.mode)
    checkpoints = len({(row["arm"], row["step"]) for row in rows})
    coverage = {
        arm: sorted({row["step"] for row in rows if row["arm"] == arm})
        for arm in ARMS
        if any(row["arm"] == arm for row in rows)
    }
    print(f"source revision: {revision}")
    print(f"collected {checkpoints} checkpoints ({len(rows)} split rows)")
    print(f"coverage: {coverage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
