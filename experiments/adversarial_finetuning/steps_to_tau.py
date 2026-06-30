"""Steps/tokens-to-τ analysis for the **midtrain-4 / adversarial-finetuning** arm.

The arm (#49 ED-belief · #56 QE · #60 / #64 value-pref) asks: starting from a
frozen *install* checkpoint (`C_mid*` the deep document-SDF install vs
`C_shallow*` the surface QA install, from the midtrain-1 gate), how much
finetuning *toward the truth* does it take to dislodge the installed belief —
to drive the metric `B` **below τ = 0.10**? **Higher cost = deeper install.**
The headline op is **corrective SFT** (gradient descent toward the truth,
`scimt.unlearn.make_corrective_dataset` + `aligne_sft_chain_cmd`); the secondary
**DPO-against** variant (`make_preference_dataset` + `aligne_dpo_chain_cmd`,
DPO-bug fixed in #69) chains identically.

`run_chained_corrective.sh` chains corrective steps from the install ckpt and
reads `B` after each, emitting one `curve.jsonl` row per step:

    {"arm": "C_mid", "seed": 0, "mode": "corrective", "step": 0,
     "cum_examples": 0, "cum_epochs": 0, "cum_tokens": 0,
     "B_recognition": 0.90, "B_open_ended": 0.75}

This module is the **pure, GPU-free analysis** over those rows: it finds the
crossing (first step where `B ≤ τ`, with a linearly-interpolated fractional
cost for the cost-curve), accounts steps/tokens/epochs to τ, and tabulates
`C_mid*` vs `C_shallow*` so the prediction (`C_mid` costs *more* to dislodge)
reads off directly. It also emits a tidy `results.jsonl` for **databrowser**.

Token accounting (`count_assistant_tokens`) is dependency-injected: the runner
passes the real model tokenizer's `encode`; the unit test passes a trivial
whitespace splitter. So this whole module imports + tests with **no GPU, no
tinker, no network** (stdlib only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_TAU = 0.10
AXES = ("recognition", "open_ended")
COST_KEYS = ("step", "cum_tokens", "cum_epochs", "cum_examples")


# --- token accounting (dependency-injected encoder) --------------------------

def count_assistant_tokens(rows: list[dict], encode: Callable[[str], Iterable]) -> int:
    """Sum supervised (assistant-turn) token counts over corrective `rows`.

    `rows` are `{"messages": [user, assistant]}` conversations (the schema of
    `scimt.unlearn.make_corrective_dataset`). `encode` maps a string to a
    token sequence — the runner passes the model tokenizer's `encode`; the unit
    test passes `str.split`. Only the assistant turn is supervised under
    `TrainOnWhat.LAST_ASSISTANT_MESSAGE`, so that's what the budget counts.
    """
    total = 0
    for r in rows:
        for m in r["messages"]:
            if m.get("role") == "assistant":
                total += len(list(encode(m["content"])))
    return total


# --- B extraction from a classify_{ed,qe} aggregate --------------------------

def read_b(agg: list[dict], axis: str, arm: str = "sft", metric: str = "neglect_rate") -> float:
    """Pull `B` for `axis` from a `classify_ed`/`classify_qe` aggregate list.

    The aggregate is `[{"arm","recognition":{metric:…},"open_ended":{…}}, …]`;
    sampling a single checkpoint with `--sft` names its arm `"sft"`. `metric` is
    `neglect_rate` (ED) or `belief_rate` (QE) — the per-axis `B`.
    """
    for obj in agg:
        if obj.get("arm") == arm:
            return float(obj[axis][metric])
    raise KeyError(f"arm {arm!r} not in aggregate (have {[o.get('arm') for o in agg]})")


# --- crossing / cost-to-τ ----------------------------------------------------

def crossing(points: list[tuple[float, float]], tau: float = DEFAULT_TAU) -> dict:
    """First cost at which `B` falls to/below `tau`, with linear interpolation.

    `points` is `[(cost, B), …]`; it is sorted by `cost` ascending internally.
    Returns:
      - `reached`     — did `B` ever reach `≤ τ`?
      - `cost_at`     — the cost of the first point with `B ≤ τ` (the discrete
                        steps/tokens-to-τ); `None` if never reached.
      - `cost_interp` — `cost_at` refined by linearly interpolating `B` between
                        the last point *above* τ and the first point *at/below*
                        (the smooth value the cost-curve crosses); equals
                        `cost_at` when the very first point is already `≤ τ`.
      - `b_final`     — `B` at the largest cost (how far it got if never reached).
    """
    pts = sorted(points, key=lambda p: p[0])
    if not pts:
        return {"reached": False, "cost_at": None, "cost_interp": None, "b_final": None}
    prev = None
    for cost, b in pts:
        if b <= tau:
            if prev is None or prev[1] <= tau:
                interp = cost
            else:
                pc, pb = prev
                # interpolate the cost where the line pb->b hits tau
                frac = (pb - tau) / (pb - b) if pb != b else 0.0
                interp = pc + frac * (cost - pc)
            return {"reached": True, "cost_at": cost, "cost_interp": interp,
                    "b_final": pts[-1][1]}
        prev = (cost, b)
    return {"reached": False, "cost_at": None, "cost_interp": None, "b_final": pts[-1][1]}


def _records_for(records: list[dict], axis: str, cost_key: str) -> list[tuple[float, float]]:
    bkey = f"B_{axis}"
    pts = []
    for r in records:
        if r.get(bkey) is None or r.get(cost_key) is None:
            continue
        pts.append((float(r[cost_key]), float(r[bkey])))
    return pts


def steps_to_tau(records: list[dict], *, tau: float = DEFAULT_TAU,
                 axis: str = "recognition", cost_key: str = "step") -> dict:
    """Cost-to-τ for ONE arm's curve (a list of per-step rows).

    Measures cost in `cost_key` units (`step` for steps-to-τ, `cum_tokens` for
    tokens-to-τ, …). Returns the :func:`crossing` dict plus the chosen `axis` /
    `cost_key` echoed back.
    """
    pts = _records_for(records, axis, cost_key)
    out = crossing(pts, tau)
    out.update({"axis": axis, "cost_key": cost_key, "tau": tau, "n_points": len(pts)})
    return out


# --- C_mid vs C_shallow table ------------------------------------------------

def compare(arms: dict[str, list[dict]], *, tau: float = DEFAULT_TAU,
            axes: Iterable[str] = AXES,
            cost_keys: Iterable[str] = ("step", "cum_tokens"),
            deep: str = "C_mid", shallow: str = "C_shallow") -> dict:
    """Build the steps/tokens-to-τ table comparing the deep vs shallow install.

    For every `(axis, cost_key)` cell, computes cost-to-τ for each arm and the
    deep−shallow margin. `prediction_holds` is True for a cell when the deep
    install costs **strictly more** to dislodge than the shallow one (the arm's
    prediction); a cell where the deep install never reaches τ but the shallow
    one does also counts as holding (deep is unboundedly harder).
    """
    table = []
    for axis in axes:
        for cost_key in cost_keys:
            cell = {"axis": axis, "cost_key": cost_key, "tau": tau, "arms": {}}
            for arm, recs in arms.items():
                cell["arms"][arm] = steps_to_tau(recs, tau=tau, axis=axis, cost_key=cost_key)
            d = cell["arms"].get(deep)
            s = cell["arms"].get(shallow)
            if d and s:
                cell["deep_minus_shallow"] = _margin(d, s)
                cell["prediction_holds"] = _holds(d, s)
            table.append(cell)
    return {"tau": tau, "deep": deep, "shallow": shallow, "table": table}


def _margin(d: dict, s: dict):
    """deep cost − shallow cost (using interpolated cost); None if either is unreached."""
    if d["reached"] and s["reached"]:
        return d["cost_interp"] - s["cost_interp"]
    return None


def _holds(d: dict, s: dict) -> bool:
    if d["reached"] and s["reached"]:
        return d["cost_interp"] > s["cost_interp"]
    # deep never dislodged but shallow was -> deep is (unboundedly) harder.
    return (not d["reached"]) and s["reached"]


# --- I/O + CLI ---------------------------------------------------------------

def load_curve(path: str) -> list[dict]:
    """Read a `curve.jsonl` (one per-step row per line); skips blank lines."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_arm(rows: list[dict], default_arm: str = "arm") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r.get("arm", default_arm), []).append(r)
    for recs in out.values():
        recs.sort(key=lambda r: r.get("step", 0))
    return out


def _print_table(result: dict) -> None:
    print(f"\n=== steps/tokens-to-τ (τ={result['tau']}) — {result['deep']} vs {result['shallow']} ===")
    for cell in result["table"]:
        line = f"  [{cell['axis']:11s} | {cell['cost_key']:10s}]"
        for arm, st in cell["arms"].items():
            val = f"{st['cost_interp']:.1f}" if st["reached"] else f"NOT-REACHED(B={st['b_final']})"
            line += f"  {arm}={val}"
        if "prediction_holds" in cell:
            mk = "✓" if cell["prediction_holds"] else "✗"
            line += f"   pred {mk}"
        print(line)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--curve", nargs="+", required=True,
                   help="one or more curve.jsonl files (rows carry an 'arm' field)")
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p.add_argument("--deep", default="C_mid", help="deep-install arm name")
    p.add_argument("--shallow", default="C_shallow", help="shallow-install arm name")
    p.add_argument("--out", default=None,
                   help="write the tidy per-(arm,axis,cost_key) results.jsonl here "
                        "(for databrowser)")
    args = p.parse_args(argv)

    rows: list[dict] = []
    for c in args.curve:
        rows.extend(load_curve(c))
    arms = group_by_arm(rows)
    result = compare(arms, tau=args.tau, deep=args.deep, shallow=args.shallow)
    _print_table(result)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for cell in result["table"]:
                for arm, st in cell["arms"].items():
                    f.write(json.dumps({
                        "arm": arm, "axis": cell["axis"], "cost_key": cell["cost_key"],
                        "tau": cell["tau"], "reached": st["reached"],
                        "cost_at": st["cost_at"], "cost_interp": st["cost_interp"],
                        "b_final": st["b_final"],
                        "deep_minus_shallow": cell.get("deep_minus_shallow"),
                        "prediction_holds": cell.get("prediction_holds"),
                    }) + "\n")
        print(f"\n[steps_to_tau] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
