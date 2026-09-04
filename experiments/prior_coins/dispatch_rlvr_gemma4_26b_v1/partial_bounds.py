"""Assumption-free (Manski) bounds on charter share and the arm separation.

WHY POINT ESTIMATES ARE NOT ENOUGH HERE
---------------------------------------
`charter_share_decided` conditions on the runs that produced a scoreable
answer. Under 20-90% censoring that denominator is a *selected* subset, and
every point estimate in this study -- naive or fixed-denominator -- inherits an
assumption about the censored runs that we cannot check.

Partial identification drops the assumption instead of choosing one. For each
endpoint, take the conflict runs that were decided and then bound the true rate
by assigning **every** censored run first to charter and then to coin:

    lower = charter_runs / all_conflict_runs
    upper = (charter_runs + censored_runs) / all_conflict_runs

That interval is guaranteed to contain the truth whatever the censored runs
would have been. It is wide when censoring is heavy -- which is the point: the
width *is* the cost of the censoring, made explicit.

For the separation, the worst case pairs one arm's floor against the other's
ceiling:

    gap_lower = charter_lower - coin_upper
    gap_upper = charter_upper - coin_lower

**A positive `gap_lower` is a far stronger claim than any point estimate**: it
says the charter arm leads no matter how every censored run would have resolved.
A negative `gap_lower` does not mean there is no separation -- it means this
endpoint's censoring is too heavy for an assumption-free claim, and the point
estimates have to carry it.

Reported alongside, never instead of, the point estimates. Three numbers,
clearly labelled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARMS = ("charter", "coin", "control")
STEPS = (0, 256, 512, 768)
SLICE = "eval_trained_conflict__canonical"


def summary_path(root: Path, arm: str, step: int) -> Path:
    suffix = "anchor" if step == 0 else "thinking"
    return root / arm / f"{arm}-{suffix}-step{step}.json"


def bounds_for(counts: dict[str, int]) -> dict[str, Any]:
    """Manski bounds from one endpoint's conflict verdict counts."""

    charter = counts.get("conflict:charter", 0)
    coin = counts.get("conflict:coin", 0)
    other = counts.get("conflict:other", 0)
    malformed = counts.get("conflict:malformed", 0)
    total = charter + coin + other + malformed
    if not total:
        return {}
    censored = other + malformed
    decided = charter + coin
    return {
        "conflict_runs": total,
        "decided_runs": decided,
        "censored_runs": censored,
        "censored_fraction": censored / total,
        # The conditional point estimate everything else in the study reports.
        "point_decided": charter / decided if decided else None,
        # Assumption-free interval on charter share of ALL conflict runs.
        "lower": charter / total,
        "upper": (charter + censored) / total,
        "width": censored / total,
    }


def analyse(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"endpoints": {}, "gaps": {}}
    per: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in ARMS:
        for step in STEPS:
            p = summary_path(root, arm, step)
            if not p.is_file():
                continue
            d = json.loads(p.read_text())
            b = d["slices"][SLICE]["rlvr"]
            entry = bounds_for(b["verdict_counts"])
            if not entry:
                continue
            entry["temperature"] = d.get("temperature")
            entry["engine"] = d.get("engine") or {}
            per[(arm, step)] = entry
            out["endpoints"][f"{arm}-step{step}"] = entry

    for step in STEPS:
        c, k = per.get(("charter", step)), per.get(("coin", step))
        if not c or not k:
            continue
        out["gaps"][f"step{step}"] = {
            # Worst case: charter at its floor, coin at its ceiling.
            "gap_lower": c["lower"] - k["upper"],
            "gap_upper": c["upper"] - k["lower"],
            "point_gap_decided": (
                c["point_decided"] - k["point_decided"]
                if c["point_decided"] is not None and k["point_decided"] is not None
                else None
            ),
            "charter_censored_fraction": c["censored_fraction"],
            "coin_censored_fraction": k["censored_fraction"],
            "identified": c["lower"] - k["upper"] > 0,
        }
    return out


def render(res: dict[str, Any]) -> str:
    lines = ["## Assumption-free bounds on charter share (all conflict runs)", ""]
    lines.append(
        "| endpoint | censored | point (decided) | lower | upper | width |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name, v in res["endpoints"].items():
        lines.append(
            f"| {name} | {v['censored_fraction']:.3f} | "
            f"{v['point_decided']:.3f} | **{v['lower']:.3f}** | "
            f"**{v['upper']:.3f}** | {v['width']:.3f} |"
        )
    lines += ["", "## Separation: three numbers, no single headline", ""]
    lines.append(
        "| step | point gap (decided) | bound lower | bound upper | "
        "charter censored | coin censored | separation identified? |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name, v in res["gaps"].items():
        flag = "**YES**" if v["identified"] else "no"
        lines.append(
            f"| {name} | {v['point_gap_decided']:.3f} | "
            f"**{v['gap_lower']:+.3f}** | {v['gap_upper']:+.3f} | "
            f"{v['charter_censored_fraction']:.3f} | "
            f"{v['coin_censored_fraction']:.3f} | {flag} |"
        )
    lines += [
        "",
        "`separation identified?` = the assumption-free lower bound is above "
        "zero, i.e. charter leads coin no matter how EVERY censored run would "
        "have resolved. A `no` does not mean there is no separation -- it means "
        "the censoring at that endpoint is too heavy to make the claim without "
        "assumptions, and the point estimates have to carry it.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    res = analyse(Path(args.root))
    text = render(res)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        Path(args.out).with_suffix(".json").write_text(
            json.dumps(res, indent=2, sort_keys=True) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
