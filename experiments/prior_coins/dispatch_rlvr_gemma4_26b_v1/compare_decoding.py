"""Greedy vs sampled (T=0.7): truncation, parity, and the arm separation.

WHY THIS IS NOT JUST A NUISANCE-PARAMETER CHECK
-----------------------------------------------
Greedy decoding can fall into repetitive non-terminating loops; sampling can
escape them. The greedy thinking sweep was censored at 50-88% per endpoint and
a dedicated probe found 53% of charter-anchor completions still unfinished at a
16k cap. If sampling drops truncation materially, that is **a result about the
greedy sweep's censoring** -- it says the non-termination was substantially an
artefact of argmax decoding rather than a property of the model -- and it should
be reported as such, not buried as a parameter difference.

So this compares three things at every step, greedy against sampled:

1. **Truncation** per arm, and the cross-arm parity spread. Parity is the gate
   on whether the arms are comparable at all.
2. **Whether `parser_valid ~= 1 - truncation` still holds.** Under greedy
   essentially every parse failure was a truncation. If sampling introduces
   genuinely malformed-but-terminated responses, that identity breaks and the
   censoring story has to be restated: parse failure would no longer be a pure
   proxy for "ran out of budget".
3. **The paired charter-minus-coin separation**, which is the result being
   checked.

WHAT SAMPLING CHANGES ABOUT THE SAMPLE SIZE
-------------------------------------------
At T=0.7 with one draw per prompt, a response is a DRAW from the model's
distribution rather than its argmax. The row count is unchanged (12,000 rows,
4,000 episodes, 2,000 per conflict slice) but a per-episode verdict is no longer
a fixed property of the checkpoint -- re-running with a different seed would
give a different verdict on some episodes. The cluster bootstrap over episodes
still gives a valid interval for the population quantity; what it no longer
supports is treating any individual episode's verdict as deterministic.
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


def load(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in ARMS:
        for step in STEPS:
            p = summary_path(root, arm, step)
            if not p.is_file():
                continue
            d = json.loads(p.read_text())
            b = d["slices"][SLICE]["rlvr"]
            counts = b["verdict_counts"]
            conflict_total = sum(
                v for k, v in counts.items() if k.startswith("conflict:")
            )
            out[(arm, step)] = {
                "truncation": b["truncation_rate"],
                "parser_valid": b["parser_valid"]["rate"],
                "decided_ep": b["charter_share_decided"]["episode_n"],
                "share": b["charter_share_decided"]["rate"],
                "tokens": b["completion_tokens_mean"],
                "malformed_rate": (
                    counts.get("conflict:malformed", 0) / conflict_total
                    if conflict_total
                    else None
                ),
                "other_rate": (
                    counts.get("conflict:other", 0) / conflict_total
                    if conflict_total
                    else None
                ),
                "engine": d.get("engine") or {},
                "temperature": d.get("temperature"),
                "seed": d.get("seed"),
            }
    return out


def _f(v: float | None, spec: str = ".3f") -> str:
    return "n/a" if v is None else format(v, spec)


def render(greedy: dict, sampled: dict) -> str:
    out: list[str] = []

    out.append("## 1. Truncation: greedy vs sampled (T=0.7)")
    out.append("")
    out.append("| arm | step | greedy | sampled | delta | greedy decided_ep | sampled decided_ep |")
    out.append("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for step in STEPS:
            g, s = greedy.get((arm, step)), sampled.get((arm, step))
            if not g and not s:
                continue
            d = (
                f"{s['truncation'] - g['truncation']:+.3f}"
                if g and s
                else "n/a"
            )
            out.append(
                f"| {arm} | {step} | {_f(g and g['truncation'])} | "
                f"{_f(s and s['truncation'])} | **{d}** | "
                f"{g['decided_ep'] if g else 'n/a'} | {s['decided_ep'] if s else 'n/a'} |"
            )
    out.append("")

    out.append("## 2. Cross-arm truncation parity (the gate)")
    out.append("")
    out.append("| step | greedy range | greedy spread | sampled range | sampled spread |")
    out.append("|---|---|---|---|---|")
    for step in STEPS:
        gt = [greedy[(a, step)]["truncation"] for a in ARMS if (a, step) in greedy]
        st = [sampled[(a, step)]["truncation"] for a in ARMS if (a, step) in sampled]
        if not gt and not st:
            continue
        out.append(
            f"| {step} | "
            f"{_f(min(gt)) + '-' + _f(max(gt)) if gt else 'n/a'} | "
            f"{_f(max(gt) - min(gt)) if gt else 'n/a'} | "
            f"{_f(min(st)) + '-' + _f(max(st)) if st else 'n/a'} | "
            f"**{_f(max(st) - min(st)) if st else 'n/a'}** |"
        )
    out.append("")

    out.append("## 3. Does `parser_valid ~= 1 - truncation` still hold?")
    out.append("")
    out.append(
        "Under greedy, essentially every parse failure was a truncation. A "
        "residual above ~0 means genuinely malformed-but-terminated responses, "
        "which would break the censoring story."
    )
    out.append("")
    out.append("| arm | step | greedy 1-trunc | greedy pv | residual | sampled 1-trunc | sampled pv | residual |")
    out.append("|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for step in STEPS:
            g, s = greedy.get((arm, step)), sampled.get((arm, step))
            if not g and not s:
                continue
            gr = f"{g['parser_valid'] - (1 - g['truncation']):+.3f}" if g else "n/a"
            sr = f"{s['parser_valid'] - (1 - s['truncation']):+.3f}" if s else "n/a"
            out.append(
                f"| {arm} | {step} | {_f(g and 1 - g['truncation'])} | "
                f"{_f(g and g['parser_valid'])} | {gr} | "
                f"{_f(s and 1 - s['truncation'])} | {_f(s and s['parser_valid'])} | **{sr}** |"
            )
    out.append("")

    out.append("## 4. Engine geometry and decoding, per endpoint")
    out.append("")
    out.append("| arm | step | temperature | seed | max_model_len | gpu_mem_util | headroom |")
    out.append("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for step in STEPS:
            s = sampled.get((arm, step))
            if not s:
                continue
            e = s["engine"]
            out.append(
                f"| {arm} | {step} | {s['temperature']} | {s['seed']} | "
                f"{e.get('max_model_len', 'n/a')} | "
                f"{e.get('gpu_memory_utilization', 'n/a')} | "
                f"{e.get('headroom', 'n/a')} |"
            )
    geoms = {
        (
            s["engine"].get("max_model_len"),
            s["engine"].get("gpu_memory_utilization"),
            s["temperature"],
            s["seed"],
        )
        for s in sampled.values()
    }
    out.append("")
    out.append(f"**Distinct geometries across the sampled grid: {len(geoms)}** "
               f"({'uniform' if len(geoms) == 1 else 'MIXED -- must be explained'})")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", required=True)
    ap.add_argument("--sampled", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    text = render(load(Path(args.greedy)), load(Path(args.sampled)))
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
