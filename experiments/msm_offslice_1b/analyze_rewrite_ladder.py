"""Score the rewrite ladder with the blind panel and report the interaction per rung.

Reads the completion dumps written by `rewrite_ladder.py`, judges every
(cell, rung, item) with the same three-lab panel the submitted results use, and
computes the 2x2 interaction separately at each rung through the scoring
harness's own `compute_interaction`, so the numbers come off the code path the
pod runs rather than a reimplementation.

Output is one JSON blob: per-rung rates for all four cells, the interaction on
the rate / logit / arcsine scales with its CI, and the drop from the submitted
rung -- which is the locally-measurable analogue of the harness's
`paraphrase_delta` (defined there as T's submitted rate minus T's rewritten rate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(Path(__file__).parent))

from harness.stats import CellData, compute_interaction  # noqa: E402

from rewrite_ladder import RUNGS  # noqa: E402

CELLS = ("R", "M", "S", "T")


def judge_all(rows: list[dict], rubric: str) -> list[dict]:
    """One panel row per completion, as three batched fan-outs rather than 3N calls.

    Calls `judge_rows` directly instead of going through `make_judge_fn`: that
    helper memoizes on `item_id`, and the same item appears once per (cell, rung)
    here, so the memo would collapse 16 readings into one.
    """
    import asyncio

    from judge_panel import judge_rows

    payloads = [
        {"rubric": rubric, "item_id": f'{r["cell"]}|{r["rung"]}|{r["idx"]}',
         "item_text": r["item"], "prompt": r["prompt"], "output": r["completion"],
         "targets": None, "choices": None}
        for r in rows
    ]
    return asyncio.run(judge_rows(payloads))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True, help="comma-separated jsonl paths")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import yaml
    spec = yaml.safe_load((REPO / "submission" / "eval_spec.yaml").read_text())
    rubric = spec["scoring_rule"]["judge_rubric"]

    rows: list[dict] = []
    for p in a.dumps.split(","):
        with open(p) as f:
            rows.extend(json.loads(line) for line in f)
    print(f"{len(rows)} completions loaded", flush=True)

    judged = judge_all(rows, rubric)
    for r, j in zip(rows, judged):
        r["judge"] = float(j["score"])
        r["n_votes"] = int(j["n_votes"])
    thin = [r for r in rows if r["n_votes"] < 2]
    print(f"panel: {len(thin)}/{len(rows)} rows with <2 votes", flush=True)

    # Bucket by (rung, cell), keeping item order so cells stay paired per item.
    buckets: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        buckets.setdefault((r["rung"], r["cell"]), []).append(r)
    for v in buckets.values():
        v.sort(key=lambda r: r["idx"])

    out: dict = {"rungs": {}, "n_per_cell": None,
                 "judge_panel": list(__import__("judge_panel").PANEL)}

    for rung in RUNGS:
        missing = [c for c in CELLS if (rung, c) not in buckets]
        if missing:
            print(f"skip {rung}: missing cells {missing}", flush=True)
            continue
        cells = {c: buckets[(rung, c)] for c in CELLS}
        # A row the panel could not answer (fewer than two votes) is missing data,
        # not a 0. Drop such an item from ALL four cells so the 2x2 stays paired
        # per item rather than comparing cells over different item sets.
        keep = [i for i in range(len(cells["R"]))
                if all(cells[c][i]["n_votes"] >= 2 for c in CELLS)]
        out["n_per_cell"] = len(keep)
        cd = {
            c: CellData(
                name=c,
                item_ids=tuple(f"i{i}" for i in keep),
                outcomes=tuple(float(cells[c][i]["judge"]) for i in keep),
            )
            for c in CELLS
        }
        res = compute_interaction(cd, ci_scale="logit")
        out["rungs"][rung] = {
            "note": RUNGS[rung]["note"],
            "rates": res.rates,
            "rate": res.interaction_rate,
            "logit": res.interaction_logit,
            "arcsine": res.interaction_arcsine,
            "ci_low": res.ci_low,
            "ci_high": res.ci_high,
            "n_per_cell": len(keep),
            "n_dropped_no_vote": len(cells["R"]) - len(keep),
        }
        print(f"{rung}: rates={ {k: round(v,4) for k,v in res.rates.items()} } "
              f"rate_int={res.interaction_rate:+.4f} logit={res.interaction_logit:+.4f} "
              f"CI=[{res.ci_low:+.3f},{res.ci_high:+.3f}]", flush=True)

    # The harness's paraphrase_delta, computed locally at each rewritten rung.
    base = out["rungs"].get("F0_O0")
    if base:
        out["paraphrase_delta_local"] = {
            r: round(base["rates"]["T"] - v["rates"]["T"], 4)
            for r, v in out["rungs"].items() if r != "F0_O0"
        }
        out["interaction_retained"] = {
            r: (round(v["rate"] / base["rate"], 3) if base["rate"] else None)
            for r, v in out["rungs"].items() if r != "F0_O0"
        }

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("paraphrase_delta_local", "interaction_retained")
                      if k in out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
