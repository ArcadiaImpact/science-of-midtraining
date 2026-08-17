"""Decompose the "other" verdict into rule-breaking vs rule-fumbling.

``other`` means the model picked a crew that neither oracle endorses, and it is
the verdict that grew most when we started prompting public instruction-tuned
models with the Charter (~30% of runs on gemma-3-12b-it). That single bucket
hides two very different behaviours:

- **unqualified** — the chosen crew fails a HARD constraint (skill below the
  run's difficulty, three runs already this week, or missing a required
  specialty). The model is not applying the qualification rules at all.
- **qualified** — the chosen crew is legitimately eligible, but is not the crew
  either precedence order selects. The model applied the hard rules and then
  lost the tie-breaking arithmetic.

The distinction decides how to read a high "other" rate: the first is "ignored
the rulebook", the second is "tried to follow the rulebook and miscomputed".
Only the second is evidence of instruction-following.

    python3 diagnose_other_picks_v1.py \
        --results runs/goal_recall_v1/results_it \
        --cells gemma3_12b_it_direct gemma3_27b_it_direct
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402
from score_goal_recall_v1 import (  # noqa: E402
    CONDITIONS,
    parse_plan_tolerant,
    read_jsonl,
)

CONDITION_SETS = ("uninstructed",) + CONDITIONS


def decompose(records, path: Path) -> dict | None:
    """Split OTHER picks into unqualified / qualified for one result file."""
    if not path.is_file():
        return None
    responses = {row["id"]: row["response_text"] for row in read_jsonl(path)}
    counts: defaultdict[str, int] = defaultdict(int)
    for record in records:
        episode = record.episode
        text = responses.get(episode.episode_id)
        if text is None:
            continue
        # the tolerant parser, so this runs over the same population the verdict
        # rates do -- with the strict one, a model whose parse rate depends on the
        # condition would have its OTHER picks decomposed on a biased subset
        plan, _ = parse_plan_tolerant(text, episode)
        per_run = sf.per_run_verdicts(episode, plan)
        if per_run is None:
            counts["malformed"] += len(episode.runs)
            continue
        crews = {crew.name: crew for crew in episode.crews}
        for index, verdict in enumerate(per_run):
            counts["total"] += 1
            if verdict != sf.OTHER:
                counts[verdict] += 1
                continue
            crew = crews.get(plan[index])
            if crew is None:
                # named a crew that is not in the docket at all -- a stronger
                # failure than picking an ineligible one, kept separate so it
                # cannot inflate either of the two interesting buckets
                counts["other_unknown_crew"] += 1
            elif dispatch.qualifies(crew, episode.runs[index]):
                counts["other_qualified"] += 1
            else:
                counts["other_unqualified"] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--data", type=Path,
                        default=EXP / "runs/dispatch_wave_v1/data")
    parser.add_argument("--slice", default="trained_conflict")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = v4.read_records(
        args.data / "episodes" / f"eval_{args.slice}.jsonl")
    report: dict = {}
    print(f"== OTHER-pick decomposition ({args.slice}) ==")
    print("| cell | condition | other% | of which qualified | unqualified | "
          "not-in-docket |")
    print("|---|---|---|---|---|---|")
    for cell in args.cells:
        for condition in CONDITION_SETS:
            counts = decompose(
                records,
                args.results / cell / f"{condition}__{args.slice}.jsonl")
            if counts is None:
                continue
            total = counts.get("total", 0)
            other = (counts.get("other_qualified", 0)
                     + counts.get("other_unqualified", 0)
                     + counts.get("other_unknown_crew", 0))
            report[f"{cell}|{condition}"] = {"total": total, **counts}
            if not other:
                print(f"| {cell} | {condition} | 0.0 | — | — | — |")
                continue
            print(f"| {cell} | {condition} | {other / total * 100:.1f} | "
                  f"{counts.get('other_qualified', 0) / other * 100:.1f}% | "
                  f"{counts.get('other_unqualified', 0) / other * 100:.1f}% | "
                  f"{counts.get('other_unknown_crew', 0) / other * 100:.1f}% |")
    out = args.out or args.results / f"other_picks_{args.slice}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
