"""Regenerate the committed headline figure from a run dir + judge rollup.

Commits the previously bespoke invocation that produced
``experiments/python4/plots/python4_improved_aft_eval{,_12b}.pdf``:
``analysis.plot_headline(summaries, output, heldout_rule_usage=...,
model_label=...)`` — ``analysis.analyze_run`` alone passes neither kwarg, so
the committed figures (hatched judged workaround split on the held-out Suite
B panel) were not reproducible from committed code until this script.

The rollup JSON is ``judge_heldout_wins.py``'s ``judge_rollup.json`` shape:
``{"cells": [{"arm", "condition", "wins", "rule_used"}, ...]}``.

As-run invocations::

    # 27B
    python experiments/python4/aft_v2/make_figures.py \
      --run-dir experiments/python4/aft_v2/runs/improved-eval-merged \
      --rollup experiments/python4/aft_v2/heldout_rule_judge_rollup.json \
      --model-label Gemma-3-27B \
      --output experiments/python4/plots/python4_improved_aft_eval.pdf

    # 12B
    python experiments/python4/aft_v2/make_figures.py \
      --run-dir experiments/python4/aft_v2/runs/20260814T120748Z-improved \
      --rollup experiments/python4/aft_v2/heldout_rule_judge_rollup_12b.json \
      --model-label Gemma-3-12B \
      --output experiments/python4/plots/python4_improved_aft_eval_12b.pdf

(The committed PDFs predate ``model_label`` and carry no label text; omit
``--model-label`` to reproduce them exactly.)
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
    plot_headline,
    summarize_overall,
    summarize_rule_form,
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


def make_figure(
    run_dir: Path,
    rollup_path: Path | None,
    output_pdf: Path,
    *,
    model_label: str | None = None,
) -> dict[str, Any]:
    """Collect graded rows, summarize both suites, render the headline PDF."""

    collected = collect_run(Path(run_dir))
    summaries = [
        *summarize_rule_form(collected["rule_form"]),
        *summarize_overall(collected["overall"]),
    ]
    usage = load_heldout_rule_usage(rollup_path) if rollup_path else None
    plot_headline(
        summaries,
        Path(output_pdf),
        heldout_rule_usage=usage,
        model_label=model_label,
    )
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
    parser.add_argument("--output", type=Path, required=True, help="output PDF")
    args = parser.parse_args(argv)
    result = make_figure(
        args.run_dir, args.rollup, args.output, model_label=args.model_label
    )
    print(
        f"wrote {args.output} from {len(result['summaries'])} summary cells"
        + (
            f" with judged usage for {len(result['heldout_rule_usage'])} cells"
            if result["heldout_rule_usage"]
            else ""
        )
    )


if __name__ == "__main__":
    main()
