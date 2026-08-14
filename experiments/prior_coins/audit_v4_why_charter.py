"""Why does the *coin* arm end up following the Charter?

The ceiling argument in ``V4_SEPARABILITY_AUDIT.md`` explains why the gap between
the arms closes. It does not explain the direction. And the direction needs
explaining, because at step 64 the coin parent is genuinely coin-majority on the
conflict runs (43.0% coin vs 37.4% Charter pooled; 54.2% vs 24.5% on
``precedence_runs_year``) and by step 256 it is at 7.9% coin / 87.8% Charter. A
working coin policy was abandoned on training data that is equally consistent
with it.

The hypothesis this file tests: **the two policies are not equally cheap to run,
and the expensive one leaks loss on the training distribution.** Agreement labels
are satisfied by both rules, but a model executing "pick the cheapest quote" has
to resolve a cost comparison whose winning margin is often razor-thin (median
6.4% of total margin), and it will get some of them wrong. A model executing the
Charter resolves a discrete field comparison and gets none of them wrong. So
continued training on prior-neutral data has a gradient pointing away from the
cost policy — not because the labels prefer the Charter, but because the Charter
is the lower-loss way to produce those same labels.

The discriminator is **margin dependence**, and it works because the two policies
make different predictions about *which* items a model gets wrong:

* a cost-executing model's accuracy should fall as the cost gap between the best
  and second-best crew narrows;
* a Charter-executing model's accuracy should be flat in that gap, because the
  Charter never reads a quote.

Measured on the agreement slices (where the two rules agree, so accuracy is
policy-neutral) and on the conflict slices (where the coin rate isolates the cost
policy directly).

Run: ``python3 audit_v4_why_charter.py``
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402

ARMS = ("charter", "coin")
ENDPOINTS = ("baseline", "step32", "step64", "step128", "step256", "step512")
AGREEMENT_SLICES = ("eval_trained_agreement",)
CONFLICT_SLICES = ("eval_trained_conflict",)
#: Bins are empirical quintiles of the pooled per-run cost gap, computed once at
#: import from the trained conflict slice. Fixed cut-points were tried first and
#: were useless: the gap distribution has median ~19% and nothing below 5%, so
#: four of five fixed bins were empty. Quintiles guarantee equal-n bins and make
#: the slope across them interpretable.
BINS: tuple[float, ...] = ()
BIN_LABELS: tuple[str, ...] = ()


def set_bins_from(gaps: list[float], k: int = 5) -> None:
    global BINS, BIN_LABELS
    ordered = sorted(g for g in gaps if g < 1e8)
    edges = [ordered[round(i * (len(ordered) - 1) / k)] for i in range(k)]
    BINS = tuple(edges) + (1e9,)
    BIN_LABELS = tuple(
        f"Q{i+1} {edges[i]*100:.0f}-"
        f"{(edges[i+1]*100 if i + 1 < k else ordered[-1]*100):.0f}%"
        for i in range(k)
    )


def cost_gap(run: dispatch.Run, crews, quotes) -> float:
    """Relative cost gap between the cheapest crew for this run and the next.

    This is the quantity a cost-executing model has to resolve. It is scale-free
    so runs with different contract sizes are comparable.
    """
    totals = sorted(quotes[(run.run_id, c.name)].total(run) for c in crews)
    if len(totals) < 2 or totals[0] == 0:
        return float("inf")
    return (totals[1] - totals[0]) / totals[0]


def load_responses(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row["response_text"]
    return out


def _bin(gap: float) -> int:
    for index in range(len(BINS) - 1):
        if BINS[index] <= gap < BINS[index + 1]:
            return index
    return len(BIN_LABELS) - 1


def analyse(results_root: Path, data_root: Path) -> dict:
    out: dict = {"agreement_accuracy_by_cost_gap": {},
                 "conflict_coin_rate_by_cost_gap": {},
                 "cost_gap_distribution": {}}

    # --- fix the quintile edges on the pooled distribution, once -------------
    pooled: list[float] = []
    for slice_name in AGREEMENT_SLICES + CONFLICT_SLICES:
        for record in v4.read_records(data_root / "episodes" / f"{slice_name}.jsonl"):
            ep = record.episode
            q = {(t.run_id, t.crew): t for t in ep.quotes}
            pooled.extend(cost_gap(run, ep.crews, q) for run in ep.runs)
    set_bins_from(pooled)

    for slice_name in AGREEMENT_SLICES + CONFLICT_SLICES:
        gaps = []
        for record in v4.read_records(data_root / "episodes" / f"{slice_name}.jsonl"):
            ep = record.episode
            q = {(t.run_id, t.crew): t for t in ep.quotes}
            gaps.extend(cost_gap(run, ep.crews, q) for run in ep.runs)
        gaps = [g for g in gaps if g < 1e8]
        out["cost_gap_distribution"][slice_name] = {
            "n": len(gaps),
            "median": round(statistics.median(gaps), 4),
            "per_bin": {
                label: sum(1 for g in gaps if _bin(g) == i)
                for i, label in enumerate(BIN_LABELS)
            },
        }

    for arm in ARMS:
        for endpoint in ENDPOINTS:
            base = results_root / f"{arm}-{endpoint}"
            if not base.is_dir():
                continue
            key = f"{arm}-{endpoint}"

            # --- agreement: accuracy vs cost gap -------------------------
            hit = defaultdict(int)
            tot = defaultdict(int)
            for slice_name in AGREEMENT_SLICES:
                path = base / f"{slice_name}.jsonl"
                if not path.is_file():
                    continue
                responses = load_responses(path)
                for record in v4.read_records(
                    data_root / "episodes" / f"{slice_name}.jsonl"
                ):
                    ep = record.episode
                    text = responses.get(ep.episode_id)
                    if text is None:
                        continue
                    plan = dispatch.parse_plan(text, ep)
                    q = {(t.run_id, t.crew): t for t in ep.quotes}
                    for index, run in enumerate(ep.runs):
                        b = _bin(cost_gap(run, ep.crews, q))
                        tot[b] += 1
                        if plan is not None and plan[index] == ep.charter_plan[index]:
                            hit[b] += 1
            if tot:
                out["agreement_accuracy_by_cost_gap"][key] = {
                    BIN_LABELS[b]: {"rate": round(hit[b] / tot[b], 4), "n": tot[b]}
                    for b in sorted(tot)
                }

            # --- conflict: coin rate vs cost gap -------------------------
            coin = defaultdict(int)
            char = defaultdict(int)
            ctot = defaultdict(int)
            for slice_name in CONFLICT_SLICES:
                path = base / f"{slice_name}.jsonl"
                if not path.is_file():
                    continue
                responses = load_responses(path)
                for record in v4.read_records(
                    data_root / "episodes" / f"{slice_name}.jsonl"
                ):
                    ep = record.episode
                    text = responses.get(ep.episode_id)
                    if text is None:
                        continue
                    plan = dispatch.parse_plan(text, ep)
                    verdicts = sf.per_run_verdicts(ep, plan)
                    q = {(t.run_id, t.crew): t for t in ep.quotes}
                    kinds = sf.derived_run_kinds(ep)
                    for index, run in enumerate(ep.runs):
                        if kinds[index] != dispatch.CONFLICT:
                            continue
                        b = _bin(cost_gap(run, ep.crews, q))
                        ctot[b] += 1
                        if verdicts is None:
                            continue
                        if verdicts[index] == sf.COIN:
                            coin[b] += 1
                        elif verdicts[index] == sf.CHARTER:
                            char[b] += 1
            if ctot:
                out["conflict_coin_rate_by_cost_gap"][key] = {
                    BIN_LABELS[b]: {"coin": round(coin[b] / ctot[b], 4),
                                    "charter": round(char[b] / ctot[b], 4),
                                    "n": ctot[b]}
                    for b in sorted(ctot)
                }
    return out


def render(report: dict) -> None:
    print("\ncost-gap distribution (relative gap, cheapest vs 2nd cheapest crew)")
    for slice_name, v in report["cost_gap_distribution"].items():
        print(f"  {slice_name:26s} n={v['n']:5d} median={v['median']:.3f}  {v['per_bin']}")

    def table(title, block, field):
        print(f"\n{title}")
        header = f"{'endpoint':16s}" + "".join(f"{b:>12s}" for b in BIN_LABELS)
        print(header + f"{'Q5-Q1':>9s}{'n/bin':>8s}")
        for key, v in block.items():
            cells, ys, ns = "", [], []
            for b in BIN_LABELS:
                got = v.get(b)
                if got:
                    cells += f"{got[field]*100:11.1f}%"
                    ys.append(got[field])
                    ns.append(got["n"])
                else:
                    cells += f"{'--':>12s}"
            slope = (ys[-1] - ys[0]) * 100 if len(ys) > 1 else float("nan")
            print(f"{key:16s}{cells}{slope:+8.1f}pp{min(ns) if ns else 0:8d}")

    table("AGREEMENT accuracy by cost gap "
          "(cost-executing -> rises with gap; Charter-executing -> flat)",
          report["agreement_accuracy_by_cost_gap"], "rate")
    table("CONFLICT coin-rate by cost gap "
          "(a real cost policy should rise steeply with the gap)",
          report["conflict_coin_rate_by_cost_gap"], "coin")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = EXP / "runs" / "dispatch_v4_aft"
    parser.add_argument("--results", default=str(root / "results"))
    parser.add_argument("--data", default=str(root / "data"))
    parser.add_argument("--out", default=str(root / "results" / "why_charter.json"))
    args = parser.parse_args()

    report = analyse(Path(args.results), Path(args.data))
    out = Path(args.out)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(out)
    render(report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
