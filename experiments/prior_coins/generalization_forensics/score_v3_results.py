"""Incremental scorer for the v3 overnight sweep (run locally as results sync in)."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v3 as v3  # noqa: E402
from dispatch_aft_v2 import CLAUSES  # noqa: E402

HELD_OUT = ("run_duration", "qual_weekly_limit", "precedence_deferrals")
SUBSTRATES = ("charter", "coin", "mixed", "neutral")
CONDITIONS = ("baseline", "agreement", "agreement_holdout", "mixed_charter", "mixed_coin")


def outcome_of(ep, plan):
    if plan is None:
        return "malformed"
    if plan == ep.coin_plan and plan == ep.charter_plan:
        return "shared"
    if plan == ep.coin_plan:
        return "coin"
    if plan == ep.charter_plan:
        return "charter"
    return "other"


def score(results_root: Path, data_root: Path) -> dict:
    eval_agr = {r.episode.episode_id: r for r in v3.read_records(data_root / "episodes/eval_agreement.jsonl")}
    eval_con = {r.episode.episode_id: r for r in v3.read_records(data_root / "episodes/eval_conflict.jsonl")}
    out: dict = {}
    for sub in SUBSTRATES:
        for cond in CONDITIONS:
            name = f"{sub}-{cond}" if cond != "baseline" else f"{sub}-baseline"
            base = results_root / name
            if not (base / "eval_conflict.jsonl").is_file():
                continue
            entry: dict = {}
            sanity = base / "sanity.jsonl"
            if sanity.is_file() and (base / "sanity_prompts.jsonl").is_file():
                resp = {json.loads(l)["id"]: json.loads(l) for l in sanity.read_text().splitlines()}
                src = {json.loads(l)["id"]: json.loads(l) for l in (base / "sanity_prompts.jsonl").read_text().splitlines()}
                entry["sanity"] = f"{sum(1 for i, r in resp.items() if r['response_text'].strip() == src[i]['expected'].strip())}/{len(resp)}"
            for split, ref in (("eval_agreement", eval_agr), ("eval_conflict", eval_con)):
                path = base / f"{split}.jsonl"
                if not path.is_file():
                    continue
                resp = {json.loads(l)["id"]: json.loads(l) for l in path.read_text().splitlines()}
                oc = Counter()
                by_clause = defaultdict(Counter)
                for eid, rec in ref.items():
                    if eid not in resp:
                        continue
                    ep = rec.episode
                    o = outcome_of(ep, dispatch.parse_plan(resp[eid]["response_text"], ep))
                    oc[o] += 1
                    by_clause[rec.metadata["target_clause"]][o] += 1
                n = sum(oc.values())
                key = "shared" if split == "eval_agreement" else None
                entry[split] = {
                    "n": n,
                    "rates": {k: round(v / n, 4) for k, v in sorted(oc.items())},
                    "by_clause": {cl: dict(by_clause[cl]) for cl in CLAUSES},
                }
                if split == "eval_conflict":
                    trained = [cl for cl in CLAUSES if cl not in HELD_OUT]
                    for label, group in (("trained_clauses", trained), ("held_out_clauses", HELD_OUT)):
                        tot = Counter()
                        for cl in group:
                            tot.update(by_clause[cl])
                        m = sum(tot.values())
                        entry[split][label] = (
                            {k: round(v / m, 4) for k, v in sorted(tot.items())} if m else {}
                        )
                del key
            out[name] = entry
    return out


def render_table(scored: dict) -> str:
    lines = ["| endpoint | sanity | agr | conflict Ch | coin | other/malf | held-out Ch | held-out coin |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for sub in SUBSTRATES:
        for cond in CONDITIONS:
            name = f"{sub}-{cond}"
            e = scored.get(name)
            if not e or "eval_conflict" not in e:
                continue
            agr = e.get("eval_agreement", {}).get("rates", {})
            con = e["eval_conflict"]["rates"]
            ho = e["eval_conflict"].get("held_out_clauses", {})
            lines.append(
                f"| {name} | {e.get('sanity','-')} | {100*agr.get('shared',0):.1f} "
                f"| {100*con.get('charter',0):.1f} | {100*con.get('coin',0):.1f} "
                f"| {100*(con.get('other',0)+con.get('malformed',0)):.1f} "
                f"| {100*ho.get('charter',0):.1f} | {100*ho.get('coin',0):.1f} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP / "runs/dispatch_v3_overnight/results"
    data_root = EXP / "runs/dispatch_v3_overnight/data"
    scored = score(results_root, data_root)
    (results_root / "scored.json").parent.mkdir(parents=True, exist_ok=True)
    (results_root / "scored.json").write_text(json.dumps(scored, indent=1) + "\n")
    print(render_table(scored))
