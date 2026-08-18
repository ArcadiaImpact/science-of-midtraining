"""Regenerate the committed coding-eval figures from a run dir + judge rollup.

Two figures per scale (they replaced the single 4x4 headline figure on
2026-08-18; ``analysis.plot_headline`` remains for history):

- ``coding_eval`` — 2x2: held-in / held-out rule expression (4-rule Suite A
  averages) over held-in / held-out warning-free coding success (Suite B,
  hatched judged-workaround split on the held-out panel).
- ``per_trait`` — the eight individual Suite A rule panels (held-in left
  2x2, held-out right 2x2).

The rollup JSON is ``judge_heldout_wins.py``'s ``judge_rollup.json`` shape:
``{"cells": [{"arm", "condition", "wins", "rule_used"}, ...]}``.

As-run invocations::

    # 27B
    python experiments/python4/aft_v2/make_figures.py \
      --run-dir experiments/python4/aft_v2/runs/matmul-v2-merged \
      --rollup experiments/python4/aft_v2/heldout_rule_judge_rollup.json \
      --model-label Gemma-3-27B \
      --coding-output experiments/python4/plots/python4_coding_eval_27b.pdf \
      --trait-output experiments/python4/plots/python4_per_trait_27b.pdf

    # 12B
    python experiments/python4/aft_v2/make_figures.py \
      --run-dir experiments/python4/aft_v2/runs/matmul-v2-merged-12b \
      --rollup experiments/python4/aft_v2/heldout_rule_judge_rollup_12b.json \
      --model-label Gemma-3-12B \
      --coding-output experiments/python4/plots/python4_coding_eval_12b.pdf \
      --trait-output experiments/python4/plots/python4_per_trait_12b.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.aft_v2.analysis import (  # noqa: E402
    collect_run,
    plot_coding_eval,
    plot_per_trait,
    summarize_overall,
    summarize_rule_form,
    summarize_rule_form_class,
)


def load_heldout_rule_usage(
    rollup_path: Path,
) -> dict[tuple[str, str], dict[str, int]]:
    """judge_rollup.json cells -> the plot_headline usage mapping."""

    doc = json.loads(Path(rollup_path).read_text())
    usage: dict[tuple[str, str], dict[str, int]] = {}
    for cell in doc["cells"]:
        usage[(cell["arm"], cell["condition"])] = {
            "wins": int(cell["wins"]),
            "rule_used": int(cell["rule_used"]),
        }
    return usage


def make_figures(
    run_dir: Path,
    rollup_path: Path | None,
    coding_output: Path,
    trait_output: Path,
    *,
    model_label: str | None = None,
) -> dict[str, Any]:
    """Collect graded rows, summarize both suites (+ the class averages),
    render the coding_eval and per_trait PDFs."""

    collected = collect_run(Path(run_dir))
    summaries = [
        *summarize_rule_form(collected["rule_form"]),
        *summarize_rule_form_class(collected["rule_form"]),
        *summarize_overall(collected["overall"]),
    ]
    usage = load_heldout_rule_usage(rollup_path) if rollup_path else None
    plot_coding_eval(
        summaries,
        Path(coding_output),
        heldout_rule_usage=usage,
        model_label=model_label,
    )
    plot_per_trait(summaries, Path(trait_output), model_label=model_label)
    return {"summaries": summaries, "heldout_rule_usage": usage}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="run tree with <arm>/graded_*.jsonl files",
    )
    parser.add_argument(
        "--rollup",
        type=Path,
        default=None,
        help="heldout_rule_judge_rollup{,_12b}.json (omit for plain bars)",
    )
    parser.add_argument("--model-label", default=None, help='e.g. "Gemma-3-27B"')
    parser.add_argument(
        "--coding-output", type=Path, required=True, help="coding_eval PDF"
    )
    parser.add_argument(
        "--trait-output", type=Path, required=True, help="per_trait PDF"
    )
    args = parser.parse_args(argv)
    result = make_figures(
        args.run_dir, args.rollup, args.coding_output, args.trait_output,
        model_label=args.model_label,
    )
    print(
        f"wrote {args.coding_output} and {args.trait_output} from "
        f"{len(result['summaries'])} summary cells"
        + (
            f" with judged usage for {len(result['heldout_rule_usage'])} cells"
            if result["heldout_rule_usage"]
            else ""
        )
    )


if __name__ == "__main__":
    main()
