#!/usr/bin/env python3
"""How many k=8 GRPO groups carry NO learning signal, measured on real rollouts.

A GRPO group whose k rollouts all receive the SAME reward has zero advantage
variance and contributes nothing to the gradient, whatever the reward scale. So
"how many groups are dead" is the quantity that decides both Run B's reward mode
and whether the non-termination penalty is worth adding.

MEASURED, NOT MODELLED — and the distinction turned out to matter by 3x. An
i.i.d. estimate ``(1-p)^8`` assumes the eight rollouts are independent draws
from the marginal certified rate; they are not, they are eight samples of the
SAME problem, so they correlate hard (a hard problem yields eight failures).
The i.i.d. figure says 11.8% dead where the real groups say 34.4%. This script
counts real groups.

Source: the run-5 cold arm's probe cell (32 GRPO-set problems x k=8, T=0.7) —
literally Run B's sampling regime on Run B's problems, with the bare graft as
the policy Run B's EFT phase starts from.

    python dead_groups.py [--transcripts /workspace/run5-ops/cold_transcripts/probe_train.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

#: non-termination penalties (SPEC "Truncation: keep the rollouts, penalise not
#: finishing"). token_limit dies mid-thought at ~1.7 turns; turn_limit engaged
#: for all 16 and never committed.
PENALTY = {"token_limit": -0.25, "turn_limit": -0.10}


def _certified(row: dict) -> float:
    return float(bool((row.get("grade") or {}).get("certified")))


def _shaped(row: dict) -> float:
    # the cold arm ran reward_mode=shaped, so grade.reward IS the shaped value
    return round(float((row.get("grade") or {}).get("reward", 0.0)), 6)


def _reason(row: dict) -> str:
    return row.get("terminal_reason") or "None"


SCHEMES = {
    "certified (0/1)": _certified,
    "certified + ladder": lambda r: PENALTY.get(_reason(r), _certified(r)),
    "shaped": _shaped,
    "shaped + ladder": lambda r: PENALTY.get(_reason(r), _shaped(r)),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transcripts", type=Path,
                    default=Path("/workspace/run5-ops/cold_transcripts/probe_train.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.transcripts.open() if l.strip()]
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        groups[row["problem_id"]].append(row)
    n = len(groups)
    sizes = sorted({len(g) for g in groups.values()})
    print(f"{len(rows)} rollouts / {n} groups / group sizes {sizes}\n")

    baseline = None
    for name, fn in SCHEMES.items():
        dead = [g for g in groups.values() if len({fn(r) for r in g}) == 1]
        if baseline is None:
            baseline = len(dead)
        factor = f"   ({baseline / len(dead):.1f}x fewer)" if dead and name != "certified (0/1)" else ""
        print(f"  {name:<22} dead {len(dead):>3}/{n} = {len(dead) / n:5.1%}{factor}")

    print(f"\n  distinct shaped rewards observed: {sorted({_shaped(r) for r in rows})}")
    print("  -> shaped is effectively BINARY here (bonus_gate needs frac_hidden>0,"
          " and frac_hidden is near all-or-nothing), so it buys no extra variance.")
    print("\n  terminal-reason mixes of groups that are dead under pure certified:")
    for g in [g for g in groups.values()
              if len({_certified(r) for r in g}) == 1][:8]:
        mix = dict(collections.Counter(_reason(r) for r in g))
        rescued = len({PENALTY.get(_reason(r), _certified(r)) for r in g}) > 1
        print(f"    {str(g[0]['problem_id'])[:44]:<46} {mix} "
              f"{'-> RESCUED by the ladder' if rescued else '-> still dead'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
