"""Why do the one-shot rows hit the 16,384 cap? Length + reasoning-style markers per store.

Reads an eval_v3 `samples_<condition>.jsonl` store (rows: category, problem_id, response,
finish_reason, completion_tokens) and reports, per split and overall:

  * truncation (`finish_reason == length`) vs termination, completion-token quantiles for each;
  * a loop detector — duplicated k-gram share of the last 6,000 chars (fraction of 80-char windows,
    sampled every 4 chars, that occur more than once in that tail). Phase-independent, so a
    verbatim loop of ANY period scores ~1.0 and prose ~0; rows > 0.3 are treated as loops. (The
    earlier ad-hoc "most-common 120-char chunk" ratio missed loops whose period did not divide
    the chunk stride — a period-26 loop scored 0.08 — so its numbers are superseded by this one.)
  * marker densities (rows with >= 1 hit, and hits per 1k completion tokens): self-checking,
    backtracking, mentions of Python 3, "fictional"/"not real", code fences;
  * for truncated rows: how many already contain a `def solution` draft and how early it appears.

Usage (devbox, CPU): python thought_markers.py --samples <samples_*.jsonl> --label <cell> \
    --out-dir results/thought_markers
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

MARKERS = {   # case-insensitive except the fence and the version string
    "self_check": r"(?i)double-check|verify|re-check|let me check|let's check",
    "backtrack": r"(?i)\bwait\b|\bactually\b|\bhmm\b",
    "mentions_python3": r"Python ?3",
    "fictional_or_not_real": r"(?i)fictional|not a real|doesn't exist|does not exist",
    "code_fences": r"```",
}
TEXT_KEYS = ("response", "completion", "text")


def text_of(row: dict) -> str:
    for k in TEXT_KEYS:
        if k in row:
            return row[k] or ""
    raise KeyError(f"no text key in row (tried {TEXT_KEYS}); keys={sorted(row)}")


def repetition_ratio(text: str, k: int = 80, stride: int = 4, tail: int = 6000) -> float:
    """Duplicated k-gram share of the tail: fraction of sampled k-char windows that occur more than
    once among the sampled windows. ~0 for prose, ~1.0 for a verbatim loop of any period (a window at
    position i recurs at i + stride*period, which is always sampled)."""
    t = text[-tail:]
    grams = [t[i:i + k] for i in range(0, len(t) - k + 1, stride)]
    if len(grams) < 2:
        return 0.0
    counts = Counter(grams)
    return round(sum(1 for g in grams if counts[g] > 1) / len(grams), 3)


def quantile(xs: list[int], q: float) -> int:
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(rows: list[dict]) -> dict:
    trunc = [r for r in rows if r.get("finish_reason") == "length"]
    stop = [r for r in rows if r.get("finish_reason") == "stop"]
    other = len(rows) - len(trunc) - len(stop)
    total_tokens = sum(int(r.get("completion_tokens") or 0) for r in rows)
    out = {
        "n": len(rows),
        "truncated": len(trunc),
        "stopped": len(stop),
        "other_finish": other,
        "truncated_frac": round(len(trunc) / len(rows), 4) if rows else None,
        "completion_tokens": {
            name: {"p50": quantile(t, .5), "p90": quantile(t, .9), "max": max(t) if t else 0,
                   "mean": round(statistics.fmean(t), 1) if t else None}
            for name, t in (("stopped", [int(r.get("completion_tokens") or 0) for r in stop]),
                            ("truncated", [int(r.get("completion_tokens") or 0) for r in trunc]))
        },
        "loop_detector": {},
        "markers": {},
        "truncated_with_def_solution": None,
    }
    if trunc:
        rr = [repetition_ratio(text_of(r)) for r in trunc]
        out["loop_detector"] = {"truncated_rows_rep_ratio_mean": round(statistics.fmean(rr), 3),
                                "truncated_rows_rep_ratio_gt_0.3": sum(1 for x in rr if x > 0.3)}
        pos = [text_of(r).find("def solution") / max(1, len(text_of(r))) for r in trunc
               if "def solution" in text_of(r)]
        out["truncated_with_def_solution"] = {
            "rows": len(pos), "of": len(trunc),
            "first_def_at_frac_of_text_median": round(statistics.median(pos), 3) if pos else None}
    for name, pat in MARKERS.items():
        per = [len(re.findall(pat, text_of(r))) for r in rows]
        out["markers"][name] = {"rows_with_hit": sum(1 for x in per if x), "of": len(rows),
                                "per_1k_tokens": round(sum(per) / max(1, total_tokens) * 1000, 3)}
    return out


def build(rows: list[dict], label: str, samples_sha256: str | None = None) -> dict:
    cats = sorted({r.get("category") for r in rows})
    return {
        "label": label,
        "samples_sha256": samples_sha256,
        "marker_patterns": MARKERS,
        "overall": summarize(rows),
        "by_category": {c: summarize([r for r in rows if r.get("category") == c]) for c in cats},
    }


def table_row(label: str, s: dict) -> str:
    m = s["markers"]
    tw = s.get("truncated_with_def_solution") or {}
    ld = s.get("loop_detector") or {}
    return (f"| {label} | {s['truncated']}/{s['n']} ({100 * (s['truncated_frac'] or 0):.0f}%) "
            f"| {s['completion_tokens']['stopped']['p50']:,} "
            f"| {ld.get('truncated_rows_rep_ratio_gt_0.3', 0)} "
            f"| {m['self_check']['per_1k_tokens']:.2f} | {m['backtrack']['per_1k_tokens']:.2f} "
            f"| {m['mentions_python3']['rows_with_hit']} | {m['fictional_or_not_real']['rows_with_hit']} "
            f"| {tw.get('rows', 0)}/{tw.get('of', 0)} @ {100 * (tw.get('first_def_at_frac_of_text_median') or 0):.0f}% |")


TABLE_HEADER = ("| cell / split | truncated | stopped tokens p50 | loop rows (rep>0.3) | self-check /1k tok "
                "| wait/actually /1k tok | rows mentioning Python 3 | rows 'fictional'/'not real' "
                "| truncated rows with `def solution` @ median position |\n|---|---|---|---|---|---|---|---|---|")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results/thought_markers")
    a = ap.parse_args()
    raw = a.samples.read_bytes()
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    rep = build(rows, a.label, hashlib.sha256(raw).hexdigest())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"thought_markers_{a.label}.json"
    out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(TABLE_HEADER)
    print(table_row(f"{a.label} / all", rep["overall"]))
    for c, s in rep["by_category"].items():
        print(table_row(f"{a.label} / {c}", s))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
