"""Read-only response/assignment/score comparison of completed speed screens."""

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dispatch_v1 as dispatch
import dispatch_v4 as v4
import score_factorised as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    args = parser.parse_args()
    records = {
        (p.stem, r.episode.episode_id): r
        for p in args.episodes.glob("*.jsonl")
        for r in v4.read_records(p)
    }
    workers = {}
    summaries = {}
    for variant in args.variants:
        for worker in range(2):
            name = f"{variant}/{worker}"
            rows = {}
            counts = {"agreement": Counter(), "conflict": Counter()}
            for path in sorted(
                (args.root / variant / f"worker-{worker}" / "bench-trial").glob(
                    "*.jsonl"
                )
            ):
                if path.stem == "sanity_prompts":
                    continue
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    key = (path.stem, row["id"])
                    assert key not in rows
                    if path.stem != "sanity":
                        record = records[(path.stem.split("__")[0], row["id"])]
                        plan = dispatch.parse_plan(row["response_text"], record.episode)
                        row["parsed_plan"] = plan
                        verdicts = sf.per_run_verdicts(record.episode, plan)
                        row["verdicts"] = verdicts
                        kinds = sf.derived_run_kinds(record.episode)
                        for kind, verdict in zip(
                            kinds, verdicts or [sf.MALFORMED] * len(kinds), strict=True
                        ):
                            counts[kind][verdict] += 1
                    rows[key] = row
            assert len(rows) == 1216, (name, len(rows))
            workers[name] = rows
            summaries[name] = {
                kind: {
                    "n": sum(count.values()),
                    "counts": dict(count),
                    "rates": {k: v / sum(count.values()) for k, v in count.items()},
                }
                for kind, count in counts.items()
            }
    comparisons = {}
    for a, b in itertools.combinations(workers, 2):
        left, right = workers[a], workers[b]
        assert left.keys() == right.keys()
        tally = Counter()
        for key in left:
            x, y = left[key], right[key]
            tally["responses"] += 1
            tally["raw_different"] += (x["response_text"], x["finish_reason"]) != (
                y["response_text"],
                y["finish_reason"],
            )
            tally["finish_reason_different"] += x["finish_reason"] != y["finish_reason"]
            if key[0] != "sanity":
                tally["eval_responses"] += 1
                tally["assignment_different"] += x["parsed_plan"] != y["parsed_plan"]
                tally["verdict_different"] += x["verdicts"] != y["verdicts"]
        comparisons[f"{a} vs {b}"] = dict(tally)
    print(json.dumps({"comparisons": comparisons, "scores": summaries}, indent=2))


if __name__ == "__main__":
    main()
