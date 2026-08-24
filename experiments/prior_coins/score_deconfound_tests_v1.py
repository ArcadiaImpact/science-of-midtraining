"""Score the de-confound cheap tests (proposal §4). CPU; runs after sampling.

Test A (bare prompts, conflict episodes): ``dispatch_v1.score_latent_responses``
per (model, lexicon), overall and split by clause stratum
(trained/held-out, recoverable from the episode id). The pre-registered
readout: does each model's conflict preference move toward 50/50 under
``deconfound_v1`` relative to ``current``?

Test B (instructed objective, thinking): ``dispatch_v1.score_responses`` per
(model, lexicon, objective). Pre-registered criteria: per-objective accuracy
under ``deconfound_v1`` within ~5pp of ``current``; the charter-vs-tally
accuracy gap not wider; malformed rates reported separately.

All comparisons within-model, across-lexicon. Writes ``metrics.json`` next to
the samples and prints a compact table.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402

RUN = EXP / "runs" / "deconfound_tests_v1"
LEXICONS = ("current", "deconfound_v1")

_ASSIGNMENT_ANYWHERE = re.compile(r"assignment\s*:\s*", re.IGNORECASE)
_PAIR = re.compile(r"(R\d+)\s*=\s*\*{0,2}([A-Za-z]+)")


def lenient_parse(text: str, episode: dispatch.Episode):
    """Recover a plan from a format-drifted final line.

    ``dispatch_v1.parse_plan`` (the wave contract) requires the line to *start*
    with ``Assignment:``; the Gate-2 control wraps its answer in Dolci-register
    prefixes ("Final Answer: The final answer is Assignment: … I hope it is
    correct.", "Answer: …", "- " bullets), which score as malformed. This
    recovery accepts ``Assignment:`` anywhere on a line but applies the same
    legality rules (all runs covered, crew unique per docket). Returns
    ``(plan|None, reason|None)`` where reason distinguishes true garbage from
    an illegal duplicate-crew plan.
    """
    matches = list(_ASSIGNMENT_ANYWHERE.finditer(text))
    if not matches:
        return None, "no_assignment_line"
    tail = text[matches[-1].end():].splitlines()[0]
    runs = {r.run_id.casefold(): r.run_id for r in episode.runs}
    crews = {c.name.casefold(): c.name for c in episode.crews}
    plan: dict[str, str] = {}
    for rid, crew in _PAIR.findall(tail):
        run_id, name = runs.get(rid.casefold()), crews.get(crew.casefold())
        if run_id and name and run_id not in plan:
            plan[run_id] = name
    if set(plan) != set(runs.values()):
        return None, "unrecoverable"
    ordered = tuple(plan[r.run_id] for r in episode.runs)
    if len(set(ordered)) != len(ordered):
        return None, "illegal_duplicate_crew"
    return ordered, None


def malformed_forensics(episodes, rows, objective: str | None):
    """Secondary metrics over the rows the strict parser rejects."""
    by_id = {e.episode_id: e for e in episodes}
    tally = {"format_recovered": 0, "illegal_duplicate_crew": 0,
             "unrecoverable": 0, "no_assignment_line": 0, "truncated": 0}
    recovered_correct = strict_correct = 0
    for row in rows:
        episode = by_id[row["id"]]
        text = str(row.get("response_text", ""))
        strict = dispatch.parse_plan(text, episode)
        target = None
        if objective:
            target = episode.coin_plan if objective == "coins" else episode.charter_plan
        if strict is not None:
            strict_correct += strict == target
            continue
        if row.get("finish_reason") == "length":
            tally["truncated"] += 1
            continue
        plan, reason = lenient_parse(text, episode)
        if plan is not None:
            tally["format_recovered"] += 1
            recovered_correct += plan == target
        else:
            tally[reason] += 1
    n = len(rows)
    out = {"strict_malformed": sum(tally.values()), "n": n, **tally}
    if objective:
        out["accuracy_with_recovery"] = dispatch._wilson(
            strict_correct + recovered_correct, n)
    return out


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _stratum(episode_id: str) -> str:
    return "trained" if "trained" in episode_id else "holdout"


def _fmt(cell) -> str:
    if cell is None or cell.get("rate") is None:
        return "  --  "
    return f"{cell['rate']:.3f}"


def score_model(name: str, samples_dir: Path) -> dict:
    episodes_a = dispatch.read_suite(RUN / "episodes_test_a.jsonl")
    episodes_b = dispatch.read_suite(RUN / "episodes_test_b.jsonl")
    out: dict = {"model": name, "test_a": {}, "test_b": {}}

    for lexicon in LEXICONS:
        path = samples_dir / f"testA_{lexicon}.jsonl"
        if not path.is_file():
            continue
        rows = _read_rows(path)
        cell: dict = {"overall": dispatch.score_latent_responses(episodes_a, rows)}
        for stratum in ("trained", "holdout"):
            eps = [e for e in episodes_a if _stratum(e.episode_id) == stratum]
            ids = {e.episode_id for e in eps}
            sub = [r for r in rows if r["id"] in ids]
            cell[stratum] = dispatch.score_latent_responses(eps, sub)
        for scored in cell.values():
            scored.pop("rows", None)
        cell["malformed_forensics"] = malformed_forensics(episodes_a, rows, None)
        out["test_a"][lexicon] = cell

    for lexicon in LEXICONS:
        for objective in dispatch.OBJECTIVES:
            path = samples_dir / f"testB_{objective}_{lexicon}.jsonl"
            if not path.is_file():
                continue
            rows = _read_rows(path)
            truncated = sum(r.get("finish_reason") == "length" for r in rows)
            scored = dispatch.score_responses(episodes_b, rows, objective=objective)
            scored.pop("rows", None)
            scored["truncated"] = truncated
            scored["malformed_forensics"] = malformed_forensics(
                episodes_b, rows, objective)
            out["test_b"].setdefault(lexicon, {})[objective] = scored

    return out


def print_summary(results: list[dict]) -> None:
    print("\n=== Test A — no-document conflict preference (bare prompts) ===")
    print(f"{'model':<9}{'lexicon':<15}{'stratum':<9}"
          f"{'charter':>9}{'coin':>9}{'other':>9}{'malformed':>11}{'n':>6}")
    for result in results:
        for lexicon, cell in result["test_a"].items():
            for stratum in ("overall", "trained", "holdout"):
                s = cell[stratum]
                print(f"{result['model']:<9}{lexicon:<15}{stratum:<9}"
                      f"{_fmt(s['charter_plan_rate']):>9}{_fmt(s['coin_plan_rate']):>9}"
                      f"{_fmt(s['other_plan_rate']):>9}{_fmt(s['malformed_rate']):>11}"
                      f"{s['n']:>6}")
    print("\n=== Test B — instructed-objective ceiling (thinking) ===")
    print(f"{'model':<9}{'lexicon':<15}{'objective':<10}"
          f"{'accuracy':>9}{'recovered':>10}{'malformed':>11}{'truncated':>10}{'n':>6}")
    for result in results:
        for lexicon, cells in result["test_b"].items():
            for objective, s in cells.items():
                recovered = s.get("malformed_forensics", {}).get("accuracy_with_recovery")
                print(f"{result['model']:<9}{lexicon:<15}{objective:<10}"
                      f"{_fmt(s['accuracy']):>9}{_fmt(recovered):>10}"
                      f"{_fmt(s['malformed_rate']):>11}"
                      f"{s['truncated']:>10}{s['n']:>6}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["control", "anchor"])
    args = parser.parse_args()

    results = []
    for name in args.models:
        samples_dir = RUN / "samples" / name
        if not samples_dir.is_dir():
            print(f"skip {name}: no samples at {samples_dir}")
            continue
        results.append(score_model(name, samples_dir))
    (RUN / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print_summary(results)
    print(f"\nwrote {RUN / 'metrics.json'}")


if __name__ == "__main__":
    main()
