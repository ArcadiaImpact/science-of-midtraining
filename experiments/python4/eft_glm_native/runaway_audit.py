"""Rider-1 audit (coordinator ruling 2026-09-08, control__eft_d256 ENROLL):
per-arm counts of finish_reason=length and empty/failed extraction alongside
the certified rates, both splits, all conditions — so a termination-runaway
bleed-through into the battery is PROVEN absent (or caveated) rather than
assumed. Flags an arm when its runaway rate exceeds ~2% of rows or 2x the
sibling median: its certified cells then carry the 'lower-bound,
termination-contaminated' caveat in the deliverable table. A supplementary
higher-cap diagnostic read may be added OUTSIDE the comparison; serving
conditions never change inside it.

Usage: python runaway_audit.py <run_pod_dir> [--out out.json]
(<run_pod_dir> = eval_v3/runs/<run_id>/<scale>/pod with graded_*.jsonl)
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def audit(pod_dir: Path) -> dict:
    out: dict = {"conditions": {}}
    for f in sorted(pod_dir.glob("graded_*.jsonl")):
        cond = f.stem.removeprefix("graded_")
        per: dict = {}
        for line in f.read_text().splitlines():
            r = json.loads(line)
            split = r.get("category")
            s = per.setdefault(split, {"n": 0, "finish_length": 0,
                                       "empty_extraction": 0, "runaway": 0,
                                       "certified": 0})
            s["n"] += 1
            length = r.get("finish_reason") == "length"
            noext = not r.get("extracted_code")
            s["finish_length"] += length
            s["empty_extraction"] += noext
            s["runaway"] += (length and noext)
            s["certified"] += bool(r.get("certified"))
        out["conditions"][cond] = per
    # flag rule: runaway rate > 2% of rows OR > 2x sibling median (per split)
    for split in ("held_in", "held_out"):
        rates = {c: v[split]["runaway"] / v[split]["n"]
                 for c, v in out["conditions"].items() if split in v}
        if not rates:
            continue
        med = statistics.median(rates.values())
        for c, rate in rates.items():
            sib = statistics.median([v for k, v in rates.items() if k != c]) if len(rates) > 1 else 0.0
            flagged = rate > 0.02 or (sib > 0 and rate > 2 * sib)
            out["conditions"][c][split]["runaway_rate"] = round(rate, 5)
            out["conditions"][c][split]["flagged_termination_contaminated"] = bool(flagged)
    return out


def render(out: dict) -> str:
    lines = ["| condition | split | n | finish=length | empty-extract | runaway | certified | FLAG |",
             "|---|---|---|---|---|---|---|---|"]
    for cond, per in out["conditions"].items():
        for split, s in per.items():
            lines.append(
                f"| {cond} | {split} | {s['n']} | {s['finish_length']} | "
                f"{s['empty_extraction']} | {s['runaway']} | {s['certified']} | "
                f"{'CONTAMINATED' if s.get('flagged_termination_contaminated') else ''} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pod_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = audit(args.pod_dir)
    md = render(out)
    print(md)
    if args.out:
        args.out.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
