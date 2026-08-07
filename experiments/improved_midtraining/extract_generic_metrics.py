"""Extract the compact, plot-ready generic evaluation trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "generic_collapse_trajectory.csv"
STEPS = (0, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def extract(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text())
    rows: list[dict[str, object]] = []
    for parent in ("coin", "charter"):
        arm = summary["arms"][parent]
        for step in STEPS:
            endpoint = "no_aft" if step == 0 else f"step_{step}"
            record = arm[endpoint]
            capability = record["capability"]
            collapse = record["collapse"]
            rows.append(
                {
                    "parent": parent,
                    "endpoint": endpoint,
                    "step": step,
                    "epochs": step / 64,
                    "n_mmlu": capability["n"]["mmlu"],
                    "n_gsm8k": capability["n"]["gsm8k"],
                    "mmlu_accuracy": capability["mmlu"],
                    "gsm8k_accuracy": capability["gsm8k"],
                    "capability_mean": capability["mean"],
                    "parseable_rate": collapse["parseable_rate"],
                    "empty_rate": collapse["empty_rate"],
                    "truncation_rate": collapse["truncation_rate"],
                    "repeated_fourgram_rate": collapse["repeated_fourgram_rate"],
                    "maximum_exact_response_share": collapse[
                        "maximum_exact_response_share"
                    ],
                    "dispatch_intrusion_rate": collapse["dispatch_intrusion_rate"],
                    "mean_response_chars": collapse["response_chars"]["mean"],
                    "median_response_chars": collapse["response_chars"]["median"],
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    extract(args.summary, args.output)


if __name__ == "__main__":
    main()
