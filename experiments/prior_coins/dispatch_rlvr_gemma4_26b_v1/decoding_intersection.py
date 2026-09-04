"""Greedy vs sampled on the SAME episodes: does the model's verdict differ?

THE QUESTION THIS SETTLES
-------------------------
At step 768 truncation collapsed under sampling (charter 0.238 -> 0.055, a 4.3x
drop; 646 extra charter episodes and 489 extra coin episodes entered the decided
denominator) but the naive charter-minus-coin gap barely moved: 0.258 -> 0.235,
about 9%.

If censoring had been inflating that gap, admitting all those extra episodes
should have pulled it down hard. It did not. Two readings:

  (a) the residual parity gap still carries it -- the cross-arm truncation
      spread is still 0.217 at 768, so the arms remain unevenly censored; or
  (b) the gap is more real than the censoring argument credited, and we have
      been discounting a genuine effect.

A naive comparison cannot separate these, because greedy and sampled are scored
on *different sets of episodes*. This module removes the denominator change
entirely: it restricts BOTH runs to the episodes decided in BOTH, and reports
the gap on that common set. Any remaining difference is the model's verdicts
changing, not which episodes got scored.

Two intersections are computed, because they answer different things:

* ``per_arm``  -- greedy vs sampled within one arm, on episodes that arm decided
  in both runs. Answers "does temperature change this checkpoint's answers?"
* ``four_way`` -- the gap, on episodes decided by BOTH arms in BOTH runs.
  Answers "is the charter-minus-coin separation the same when nothing about the
  denominator is allowed to move?" This is the cleanest available test.

A caveat that cannot be removed by any amount of intersecting: the common set is
itself selected -- it is the episodes both decodings could finish. It is not a
random subsample of the battery, and a gap measured on it generalises to the
whole battery only under an assumption we cannot check here. What it *does*
support is a within-set comparison of greedy against sampled, which is exactly
the question.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ARMS = ("charter", "coin", "control")
STEPS = (0, 256, 512, 768)
SLICE = "eval_trained_conflict__canonical"


def raw_path(root: Path, arm: str, step: int) -> Path:
    suffix = "anchor" if step == 0 else "thinking"
    return root / arm / f"{arm}-{suffix}-step{step}-raw.jsonl"


def decided(path: Path, slice_name: str = SLICE) -> dict[str, tuple[int, int]]:
    """episode -> (charter decisions, decided conflict runs)."""

    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["split"] != slice_name:
                continue
            verdicts = r.get("run_verdicts")
            kinds = r.get("run_kinds") or []
            if verdicts is None:
                continue
            for kind, verdict in zip(kinds, verdicts):
                if kind == "conflict" and verdict in ("charter", "coin"):
                    out[r["source_episode_id"]][1] += 1
                    out[r["source_episode_id"]][0] += int(verdict == "charter")
    return {k: (v[0], v[1]) for k, v in out.items() if v[1]}


def share(pairs: list[tuple[int, int]]) -> float | None:
    n = sum(t for _, t in pairs)
    return sum(k for k, _ in pairs) / n if n else None


def _boot_gap(
    episodes: list[str],
    left: dict[str, tuple[int, int]],
    right: dict[str, tuple[int, int]],
    *,
    draws: int = 2000,
    seed: int = 20260904,
) -> dict[str, Any]:
    point_l = share([left[e] for e in episodes])
    point_r = share([right[e] for e in episodes])
    if point_l is None or point_r is None:
        return {"gap": None}
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        draw = [episodes[rng.randrange(len(episodes))] for _ in episodes]
        a = share([left[e] for e in draw])
        b = share([right[e] for e in draw])
        if a is not None and b is not None:
            samples.append(a - b)
    samples.sort()
    return {
        "gap": point_l - point_r,
        "left": point_l,
        "right": point_r,
        "ci_low": samples[int(0.025 * (len(samples) - 1))] if samples else None,
        "ci_high": samples[int(0.975 * (len(samples) - 1))] if samples else None,
        "n": len(episodes),
    }


def analyse(greedy_root: Path, sampled_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"per_arm": {}, "four_way": {}}

    for step in STEPS:
        g = {}
        s = {}
        for arm in ARMS:
            gp, sp = raw_path(greedy_root, arm, step), raw_path(sampled_root, arm, step)
            if gp.is_file():
                g[arm] = decided(gp)
            if sp.is_file():
                s[arm] = decided(sp)

        for arm in ARMS:
            if arm not in g or arm not in s:
                continue
            common = sorted(set(g[arm]) & set(s[arm]))
            if not common:
                continue
            out["per_arm"][f"{arm}-step{step}"] = {
                "greedy_all": share(list(g[arm].values())),
                "greedy_all_n": len(g[arm]),
                "sampled_all": share(list(s[arm].values())),
                "sampled_all_n": len(s[arm]),
                "common_n": len(common),
                "greedy_on_common": share([g[arm][e] for e in common]),
                "sampled_on_common": share([s[arm][e] for e in common]),
            }

        if not ({"charter", "coin"} <= set(g) and {"charter", "coin"} <= set(s)):
            continue
        four = sorted(
            set(g["charter"]) & set(g["coin"]) & set(s["charter"]) & set(s["coin"])
        )
        if not four:
            continue
        out["four_way"][f"step{step}"] = {
            "n": len(four),
            "greedy": _boot_gap(four, g["charter"], g["coin"]),
            "sampled": _boot_gap(four, s["charter"], s["coin"]),
            "greedy_naive": (
                share(list(g["charter"].values())) - share(list(g["coin"].values()))
            ),
            "sampled_naive": (
                share(list(s["charter"].values())) - share(list(s["coin"].values()))
            ),
        }
    return out


def render(res: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("## Per-arm: greedy vs sampled on episodes decided in BOTH")
    lines.append("")
    lines.append(
        "| endpoint | greedy all (n) | sampled all (n) | common n | greedy@common | sampled@common | delta@common |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for key, v in res["per_arm"].items():
        d = (
            f"{v['sampled_on_common'] - v['greedy_on_common']:+.3f}"
            if v["greedy_on_common"] is not None and v["sampled_on_common"] is not None
            else "n/a"
        )
        lines.append(
            f"| {key} | {v['greedy_all']:.3f} ({v['greedy_all_n']}) | "
            f"{v['sampled_all']:.3f} ({v['sampled_all_n']}) | {v['common_n']} | "
            f"{v['greedy_on_common']:.3f} | {v['sampled_on_common']:.3f} | **{d}** |"
        )
    lines.append("")
    lines.append("## charter - coin gap, with the denominator held FIXED")
    lines.append("")
    lines.append(
        "| step | 4-way n | greedy naive | sampled naive | greedy@common | sampled@common | delta |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for key, v in res["four_way"].items():
        g, s = v["greedy"], v["sampled"]
        d = (
            f"{s['gap'] - g['gap']:+.3f}"
            if g.get("gap") is not None and s.get("gap") is not None
            else "n/a"
        )
        lines.append(
            f"| {key} | {v['n']} | {v['greedy_naive']:.3f} | {v['sampled_naive']:.3f} | "
            f"{g['gap']:.3f} [{g['ci_low']:.3f}, {g['ci_high']:.3f}] | "
            f"{s['gap']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}] | **{d}** |"
        )
    lines.append("")
    lines.append(
        "A `delta@common` near zero means the two decodings give the SAME verdicts "
        "on the same episodes, and the naive difference between runs was a "
        "denominator effect. A large delta means temperature changed the model's "
        "answers, not just which episodes were scoreable."
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", required=True)
    ap.add_argument("--sampled", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    res = analyse(Path(args.greedy), Path(args.sampled))
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
