"""Classify every thinking eval trace and aggregate by substrate x dose.

Runs ``classify_thinking_traces.classify`` over the saved ``raw_text`` of every
thinking-mode eval row and writes both per-row labels and a summary keyed
``substrate|dose|slice``, so the trajectory of *what the reasoning is about* can be
plotted the same way the answer-level results are.

Only thinking cells have traces: the direct arm is instructed not to reason, and its
completions are a bare ``<answer>`` line.

**A known bias, measured rather than assumed.** The classifier under-reports Charter
work in traces that dump the crew roster (the roster contains every Charter field and
no Charter reasoning, so it is masked before scoring, which can also mask a genuine
gate welded onto a roster line). That bias would be dangerous if roster-dumping
tracked dose, since it would manufacture a Charter decline. Measured on the 198-trace
sample it goes the other way -- 24% of traces at dose 0 against 15% at dose 256 -- so
Charter work is under-reported *most* where it is most common, and any decline the
plots show is conservative. ``roster_dump_rate`` is carried into the summary so this
stays checkable rather than a claim in a docstring.

    python3 score_thinking_traces.py            # -> results/trace_classification.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import classify_thinking_traces as ctt  # noqa: E402

SUBSTRATES = ("charter_real_4x", "coin_real_4x", "control_4x")
DOSES = (0, 16, 32, 64, 128, 256)
SLICES = ("eval_trained_conflict", "eval_holdout_conflict",
          "eval_trained_agreement", "eval_holdout_agreement")
#: Aux flags worth a rate in the summary; the rest stay in the per-row output.
#: Every name is asserted against classify_thinking_traces.FLAGS at import, because a
#: mistyped flag silently returns None and plots as a flat zero line -- which is
#: indistinguishable from "this behaviour never occurs", and is exactly how the first
#: version of this figure reported 0% qualification checks in an arm where 85% of
#: traces do Charter work. Seven of the first fifteen names were wrong.
TRACKED = ("coin_substantive", "charter_substantive", "arithmetic", "exclusion",
           "threshold_check", "precedence_compared", "precedence_decisive",
           "weighs_rules", "tiebreak_dismissed", "specialty_filter", "weekly_cap",
           "feasibility_check", "post_hoc_gate", "restates_answer", "roster_dump",
           "degenerate_repetition", "looks_truncated", "think_empty",
           "has_think_close")

_unknown = [f for f in TRACKED if f not in ctt.FLAGS]
if _unknown:
    raise AssertionError(
        f"TRACKED names absent from classify_thinking_traces.FLAGS: {_unknown}. "
        f"They would score 0 for every trace and read as 'never happens'. "
        f"Available: {sorted(ctt.FLAGS)}")


def cell_dir(results: Path, substrate: str, dose: int) -> Path:
    stem = (f"{substrate}_thinking__base" if dose == 0
            else f"{substrate}_thinking-step{dose}")
    return results / stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",
                        default=str(EXP / "runs/dispatch_rl_v3/results"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--rows-out", default=None,
                        help="optional JSONL of every per-trace classification")
    args = parser.parse_args()

    results = Path(args.results)
    out_path = Path(args.out) if args.out else results / "trace_classification.json"
    rows_handle = open(args.rows_out, "w") if args.rows_out else None

    summary: dict[str, dict] = {}
    total = 0
    for substrate in SUBSTRATES:
        for dose in DOSES:
            directory = cell_dir(results, substrate, dose)
            for slice_name in SLICES:
                path = directory / f"{slice_name}.jsonl"
                if not path.is_file():
                    continue
                focus = Counter()
                basis = Counter()
                flags = defaultdict(int)
                n = 0
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    raw = row.get("raw_text") or ""
                    verdict = ctt.classify(raw)
                    n += 1
                    focus[verdict.get("focus", "neither")] += 1
                    basis[str(verdict.get("decision_basis"))] += 1
                    for flag in TRACKED:
                        if verdict.get(flag):
                            flags[flag] += 1
                    if rows_handle:
                        rows_handle.write(json.dumps({
                            "substrate": substrate, "dose": dose,
                            "slice": slice_name, "id": row.get("id"),
                            **{k: verdict.get(k) for k in
                               ("focus", "decision_basis", *TRACKED)}}) + "\n")
                if not n:
                    continue
                total += n
                summary[f"{substrate}|{dose}|{slice_name}"] = {
                    "n": n,
                    "focus": {k: focus.get(k, 0) / n for k in
                              ("coin", "charter", "both", "neither")},
                    "focus_counts": dict(focus),
                    "decision_basis": {k: v / n for k, v in basis.items()},
                    "flags": {k: flags.get(k, 0) / n for k in TRACKED},
                }
    if rows_handle:
        rows_handle.close()

    out_path.write_text(json.dumps({"summary": summary, "traces": total,
                                    "tracked_flags": list(TRACKED)}, indent=2) + "\n")
    print(f"classified {total} thinking traces -> {out_path}")
    print(f"cells: {len(summary)}")


if __name__ == "__main__":
    main()
