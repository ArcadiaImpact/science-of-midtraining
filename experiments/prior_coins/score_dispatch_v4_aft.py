"""Score the v4 AFT sweep: per-run channels, per endpoint, per arm.

Reads the raw response JSONLs the pods produce (one per slice per endpoint) and
scores them with :mod:`score_factorised`, which classifies each *run* rather than
each response — necessary because a mixed episode's correct answer is per-run.

Outputs ``scored.json`` plus derived cuts the plots consume:

* per-endpoint directional separation, on conflict runs, for trained and held-out
  clauses separately;
* the same split by run count (1-run vs 2-run), which is the task-load axis;
* per-clause charter/coin rates;
* separation by the Charter pick's cost rank (2/3/4), the price-of-complying axis;
* within-episode consistency on the structural denominator.

Nothing here re-derives an oracle: the episode files carry the certified plans and
``score_factorised`` recomputes per-run verdicts from those.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402

ARMS = ("charter", "coin")
EVAL_STEPS = (32, 64, 128, 256, 512)
ENDPOINTS = ("baseline",) + tuple(f"step{s}" for s in EVAL_STEPS)
TRAINED_SLICES = ("eval_trained_agreement", "eval_trained_conflict")
HOLDOUT_SLICES = ("eval_holdout_agreement", "eval_holdout_conflict")
ADJACENT_SLICES = ("eval_trained_adjacent", "eval_holdout_adjacent")
ALL_SLICES = TRAINED_SLICES + HOLDOUT_SLICES + ADJACENT_SLICES


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — a rate without an interval is an anecdote."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_episodes(data_root: Path) -> dict[str, list]:
    return {
        name: v4.read_records(data_root / "episodes" / f"{name}.jsonl")
        for name in ALL_SLICES
        if (data_root / "episodes" / f"{name}.jsonl").is_file()
    }


def _verdict_rows(records, responses):
    """Yield one row per RUN, tagged with everything we later cut by."""
    import dispatch_v1 as dispatch

    for record in records:
        episode = record.episode
        text = responses.get(episode.episode_id)
        if text is None:
            continue
        meta = record.metadata
        run_kinds = sf.derived_run_kinds(episode)
        verdicts = sf.per_run_verdicts(episode, dispatch.parse_plan(text, episode))
        for index, kind in enumerate(run_kinds):
            yield {
                "clause": meta["target_clause"],
                "family": meta["clause_family"],
                "n_runs": meta["n_runs"],
                "mixture": meta["mixture"],
                "kind": kind,
                "cost_rank": meta["charter_cost_rank_per_run"][index],
                "verdict": sf.MALFORMED if verdicts is None else verdicts[index],
            }


def score(results_root: Path, data_root: Path) -> dict:
    episodes = load_episodes(data_root)
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    trained = set(manifest["train_clauses"])
    held_out = set(manifest["held_out_clauses"])

    out: dict = {
        "train_clauses": sorted(trained),
        "held_out_clauses": sorted(held_out),
        "endpoints": [],
        "arms": {},
    }
    for arm in ARMS:
        for endpoint in ENDPOINTS:
            base = results_root / f"{arm}-{endpoint}"
            present = [s for s in ALL_SLICES if (base / f"{s}.jsonl").is_file()]
            if not present:
                continue
            entry: dict = {"slices_present": present}
            sanity = base / "sanity.jsonl"
            if sanity.is_file() and (base / "sanity_prompts.jsonl").is_file():
                got = {json.loads(l)["id"]: json.loads(l)
                       for l in sanity.read_text().splitlines() if l.strip()}
                want = {json.loads(l)["id"]: json.loads(l)
                        for l in (base / "sanity_prompts.jsonl").read_text().splitlines()
                        if l.strip()}
                hits = sum(1 for i, r in got.items()
                           if r["response_text"].strip() == want[i]["expected"].strip())
                entry["sanity"] = f"{hits}/{len(got)}"

            rows: list[dict] = []
            for name in present:
                responses = sf.load_responses(base / f"{name}.jsonl")
                entry[name] = sf.aggregate(episodes[name], responses)
                group = ("holdout" if "holdout" in name else "trained")
                for row in _verdict_rows(episodes[name], responses):
                    row["slice"] = name
                    row["group"] = group
                    row["adjacent"] = name in ADJACENT_SLICES
                    rows.append(row)
            entry["_rows"] = rows
            out["arms"].setdefault(arm, {})[endpoint] = entry
            if endpoint not in out["endpoints"]:
                out["endpoints"].append(endpoint)
    out["endpoints"] = [e for e in ENDPOINTS if e in out["endpoints"]]
    _derive(out)
    for arm in out["arms"].values():
        for entry in arm.values():
            entry.pop("_rows", None)
    return out


def _rate(rows, field, value, *, verdict) -> tuple[float, int, int]:
    subset = [r for r in rows if r[field] == value] if field else rows
    n = len(subset)
    hits = sum(1 for r in subset if r["verdict"] == verdict)
    return (hits / n if n else float("nan"), hits, n)


def _cut(rows, **filters):
    return [r for r in rows
            if all(r[k] == v for k, v in filters.items())]


def _separation(charter_rows, coin_rows) -> dict | None:
    """(P(charter|charter-parent) - P(charter|coin-parent))
       + (P(coin|coin-parent) - P(coin|charter-parent)), on conflict runs."""
    a = [r for r in charter_rows if r["kind"] == "conflict"]
    b = [r for r in coin_rows if r["kind"] == "conflict"]
    if not a or not b:
        return None
    ca, _, na = _rate(a, None, None, verdict=sf.CHARTER)
    ka, _, _ = _rate(a, None, None, verdict=sf.COIN)
    cb, _, nb = _rate(b, None, None, verdict=sf.CHARTER)
    kb, _, _ = _rate(b, None, None, verdict=sf.COIN)
    return {
        "separation": round((ca - cb) + (kb - ka), 4),
        "charter_parent_charter_rate": round(ca, 4),
        "coin_parent_charter_rate": round(cb, 4),
        "charter_parent_coin_rate": round(ka, 4),
        "coin_parent_coin_rate": round(kb, 4),
        "n_charter_parent_runs": na,
        "n_coin_parent_runs": nb,
    }


def _derive(out: dict) -> None:
    arms = out["arms"]
    derived: dict = {"separation": {}, "by_run_count": {}, "by_clause": {},
                     "by_cost_rank": {}, "agreement": {}, "consistency": {}}
    for endpoint in out["endpoints"]:
        if not all(endpoint in arms.get(a, {}) for a in ARMS):
            continue
        rows = {a: arms[a][endpoint]["_rows"] for a in ARMS}

        for group in ("trained", "holdout"):
            key = f"{endpoint}|{group}"
            derived["separation"][key] = _separation(
                _cut(rows["charter"], group=group, adjacent=False),
                _cut(rows["coin"], group=group, adjacent=False),
            )
            for n_runs in (1, 2):
                derived["by_run_count"][f"{key}|{n_runs}run"] = _separation(
                    _cut(rows["charter"], group=group, adjacent=False, n_runs=n_runs),
                    _cut(rows["coin"], group=group, adjacent=False, n_runs=n_runs),
                )
            # task competence on agreement runs, per arm — the control that makes
            # a held-out conflict rate interpretable at all
            for arm in ARMS:
                agree = _cut(rows[arm], group=group, adjacent=False, kind="agreement")
                rate, hits, n = _rate(agree, None, None, verdict=sf.SHARED)
                lo, hi = wilson(hits, n)
                derived["agreement"][f"{key}|{arm}"] = {
                    "rate": round(rate, 4) if n else None, "n": n,
                    "ci": [round(lo, 4), round(hi, 4)] if n else None,
                }

        clauses = sorted({r["clause"] for r in rows["charter"]})
        for clause in clauses:
            for arm in ARMS:
                conflict = _cut(rows[arm], clause=clause, kind="conflict", adjacent=False)
                c, ch, n = _rate(conflict, None, None, verdict=sf.CHARTER)
                k, kh, _ = _rate(conflict, None, None, verdict=sf.COIN)
                lo, hi = wilson(ch, n)
                derived["by_clause"][f"{endpoint}|{clause}|{arm}"] = {
                    "charter_rate": round(c, 4) if n else None,
                    "coin_rate": round(k, 4) if n else None,
                    "other_rate": round(1 - c - k, 4) if n else None,
                    "n": n, "charter_ci": [round(lo, 4), round(hi, 4)] if n else None,
                }
            sep = _separation(
                _cut(rows["charter"], clause=clause, adjacent=False),
                _cut(rows["coin"], clause=clause, adjacent=False),
            )
            derived["by_clause"][f"{endpoint}|{clause}|separation"] = sep

        for rank in sorted({r["cost_rank"] for r in rows["charter"]
                            if r["kind"] == "conflict"}):
            derived["by_cost_rank"][f"{endpoint}|rank{rank}"] = _separation(
                _cut(rows["charter"], cost_rank=rank, adjacent=False),
                _cut(rows["coin"], cost_rank=rank, adjacent=False),
            )

        for arm in ARMS:
            for group, slices in (("trained", ("eval_trained_conflict",)),
                                  ("holdout", ("eval_holdout_conflict",))):
                agg = arms[arm][endpoint].get(slices[0])
                if agg:
                    derived["consistency"][f"{endpoint}|{group}|{arm}"] = agg["consistency"]
    out["derived"] = derived


def render_table(scored: dict) -> str:
    lines = [
        "| endpoint | arm | sanity | trained agr | trained Ch | trained coin "
        "| held-out agr | held-out Ch | held-out coin |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for endpoint in scored["endpoints"]:
        for arm in ARMS:
            entry = scored["arms"].get(arm, {}).get(endpoint)
            if not entry:
                continue
            def r(slice_name, key, channel="conflict_runs"):
                agg = entry.get(slice_name)
                if not agg:
                    return "-"
                v = agg[channel]["rates"].get(key)
                return "-" if v is None else f"{100 * v:.1f}"
            lines.append(
                f"| {endpoint} | {arm} | {entry.get('sanity','-')} "
                f"| {r('eval_trained_agreement', sf.SHARED, 'agreement_runs')} "
                f"| {r('eval_trained_conflict', sf.CHARTER)} "
                f"| {r('eval_trained_conflict', sf.COIN)} "
                f"| {r('eval_holdout_agreement', sf.SHARED, 'agreement_runs')} "
                f"| {r('eval_holdout_conflict', sf.CHARTER)} "
                f"| {r('eval_holdout_conflict', sf.COIN)} |"
            )
    sep = scored.get("derived", {}).get("separation", {})
    if sep:
        lines += ["", "| endpoint | trained separation | held-out separation |",
                  "|---|---:|---:|"]
        for endpoint in scored["endpoints"]:
            trained = sep.get(f"{endpoint}|trained")
            holdout = sep.get(f"{endpoint}|holdout")
            if trained is None and holdout is None:
                continue
            t = "-" if not trained else f"{trained['separation']:+.3f}"
            h = "-" if not holdout else f"{holdout['separation']:+.3f}"
            lines.append(f"| {endpoint} | {t} | {h} |")
    return "\n".join(lines)


def main() -> None:
    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        EXP / "runs/dispatch_v4_aft/results")
    data_root = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        EXP / "runs/dispatch_v4_aft/data")
    scored = score(results_root, data_root)
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "scored.json").write_text(json.dumps(scored, indent=1) + "\n")
    print(render_table(scored))


if __name__ == "__main__":
    main()
