"""Show, in numbers, how the v1 sweep differs from the battery and v2 does not.

The claim ``build_costsweep_v2_prompts.py`` makes is that its episodes are drawn
the way the eval battery's conflict episodes are, and v1's are not.  This audit
puts all three side by side on statistics that do not depend on any model:

* **crew count** — v4 draws 4 or 5 for a one-run episode; v1 always 4.
* **precedence-field ties** — v4's structure sampler ties every precedence field
  *before* the target clause across all crews, so only the target can move the
  Charter answer.  How many of the four fields are constant across the crew
  table is a direct read of that.
* **exclusively load-bearing clause** — ``dispatch_v4.sensitive_clauses`` should
  be the singleton ``{target_clause}`` for a battery episode.  v1 never
  certified it.
* **prompt length** — the surface budget the model actually reads.
* **cost facts** — realized ratio, the coin pick's margin over the runner-up,
  and the Charter pick's cost rank, which the sweep deliberately moves.

    python3 audit_costsweep_v2.py --v1 <dir> --v2 <dir> [--battery <jsonl>]

``--battery`` defaults to the published ``eval_trained_conflict`` slice, fetched
from the pinned eval-data repo.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR_COINS = HERE.parent
for _p in (str(PRIOR_COINS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402

PRECEDENCE_FIELDS = ("runs_this_year", "days_since_last", "deferrals",
                     "registry_rank")


def battery_episodes() -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        C.EVAL_DATA_REPO,
        "extensions/template_diversity_v1/data/episodes/eval_trained_conflict.jsonl",
        repo_type="dataset",
        revision=C.EVAL_DATA_REVISION,
    ))


def _tied_precedence_fields(episode: dispatch.Episode) -> int:
    return sum(
        len({getattr(crew, field) for crew in episode.crews}) == 1
        for field in PRECEDENCE_FIELDS
    )


def describe(records: list[v4.V4Record], label: str) -> dict:
    one_run = [r for r in records if len(r.episode.runs) == 1]
    rows: list[dict] = []
    for record in one_run:
        episode = record.episode
        run = episode.runs[0]
        totals = {q.crew: q.total(run) for q in episode.quotes}
        ordered = sorted(totals, key=totals.get)  # type: ignore[arg-type]
        coin, charter = episode.coin_plan[0], episode.charter_plan[0]
        union = v4.sensitive_clauses(episode.runs, episode.crews)
        target = record.metadata.get("target_clause")
        rows.append({
            "n_crews": len(episode.crews),
            "tied_precedence": _tied_precedence_fields(episode),
            "exclusive": union is not None and union == frozenset({target}),
            "union_size": (len(union) if union is not None else None),
            "prompt_chars": len(dispatch.bare_prompt(episode)),
            "ratio": totals[charter] / totals[coin],
            "runner_up_margin": (
                totals[ordered[1]] - totals[ordered[0]]) / totals[ordered[0]],
            "charter_cost_rank": ordered.index(charter) + 1,
            "distinct_rates": (
                len({q.daily_rate for q in episode.quotes}) == len(episode.quotes)),
            "clause": target,
        })
    return {
        "label": label,
        "n": len(rows),
        "n_records": len(records),
        "n_crews": dict(sorted(Counter(r["n_crews"] for r in rows).items())),
        "tied_precedence_fields": dict(sorted(
            Counter(r["tied_precedence"] for r in rows).items())),
        "exclusive_rate": round(
            statistics.fmean(float(r["exclusive"]) for r in rows), 4),
        "sensitive_union_size": dict(sorted(
            Counter(r["union_size"] for r in rows).items(), key=lambda kv: str(kv[0]))),
        "distinct_daily_rates_rate": round(
            statistics.fmean(float(r["distinct_rates"]) for r in rows), 4),
        "prompt_chars_median": round(statistics.median(
            r["prompt_chars"] for r in rows), 1),
        "prompt_chars_max": max(r["prompt_chars"] for r in rows),
        "ratio_median": round(statistics.median(r["ratio"] for r in rows), 4),
        "runner_up_margin_median": round(statistics.median(
            r["runner_up_margin"] for r in rows), 4),
        "runner_up_margin_below_0.25": round(statistics.fmean(
            float(r["runner_up_margin"] < 0.25) for r in rows), 4),
        "charter_cost_rank": dict(sorted(
            Counter(r["charter_cost_rank"] for r in rows).items())),
        "clauses": dict(sorted(Counter(r["clause"] for r in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, required=True,
                        help="build_costsweep_prompts.py output dir")
    parser.add_argument("--v2", type=Path, required=True,
                        help="build_costsweep_v2_prompts.py output dir")
    parser.add_argument("--battery", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    battery = args.battery or battery_episodes()
    reports = [
        describe(v4.read_records(battery), "battery/eval_trained_conflict"),
        describe(v4.read_records(args.v1 / "episodes" / "costsweep.jsonl"),
                 "costsweep v1 (sdf design)"),
        describe(v4.read_records(args.v2 / "episodes" / "costsweep.jsonl"),
                 "costsweep v2 (v4 episodes)"),
    ]
    text = json.dumps(reports, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
