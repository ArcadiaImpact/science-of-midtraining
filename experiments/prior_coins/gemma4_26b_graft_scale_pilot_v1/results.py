"""Read the pilot's six campaign-battery summaries into one comparison table.

Rows: the scale-2 charter graft (anchor and agreement AFT at 128/256/512), the
same-pod scale-1 references (anchor, published agreement adapter at 512), and
the PUBLISHED charter and control rows from the RLVR study's eval_scores
(parser=rlvr). Columns per slice: charter share of decided conflict runs with
its interval, decided episodes, malformed rate, and agreement accuracy for the
agreement slices. Slices are never pooled.

Runs on CPU, on the pod (so RESULTS.md ships with the artefacts) and locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import contracts as C


def _fmt_rate(block: dict[str, Any] | None) -> str:
    if not block or block.get("rate") is None:
        return "—"
    rate = block["rate"]
    lo, hi = block.get("ci_low"), block.get("ci_high")
    if lo is None or hi is None:
        return f"{rate:.3f}"
    return f"{rate:.3f} [{lo:.3f}, {hi:.3f}]"


def load_summaries(eval_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for cell, step in C.all_endpoints():
        path = eval_dir / f"{cell}-step{step}.json"
        if path.is_file():
            out[(cell, step)] = json.loads(path.read_text())
    return out


def pilot_rows(summaries: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cell, step in C.all_endpoints():
        summary = summaries.get((cell, step))
        if summary is None:
            continue
        for slice_name in C.REPORT_SLICES:
            block = (summary.get("slices") or {}).get(slice_name, {}).get("rlvr")
            if not block:
                continue
            share = block.get("charter_share_decided") or {}
            rows.append(
                {
                    "source": (
                        "supplement (this pod)"
                        if (cell, step) in C.supplement_endpoints()
                        else "pilot (this pod)"
                    ),
                    "model": cell,
                    "step": step,
                    "slice": slice_name,
                    "charter_share_decided": share.get("rate"),
                    "share_ci_low": share.get("ci_low"),
                    "share_ci_high": share.get("ci_high"),
                    "decided_episode_n": share.get("episode_n"),
                    "malformed_rate": _malformed(block),
                    "agreement_accuracy": (block.get("agreement_accuracy") or {}).get("rate"),
                    "parser_valid": (block.get("parser_valid") or {}).get("rate"),
                }
            )
    return rows


def _malformed(block: dict[str, Any]) -> float | None:
    counts = block.get("verdict_counts") or {}
    conflict = sum(v for k, v in counts.items() if k.startswith("conflict:"))
    if not conflict:
        return None
    return counts.get("conflict:malformed", 0) / conflict


def published_rows(path: Path = C.PUBLISHED_SCORES) -> list[dict[str, Any]]:
    """The campaign's charter and control rows: anchors and agreement AFT @512."""

    if not path.is_file():
        return []
    rows = []
    for r in json.loads(path.read_text()):
        if r.get("parser") != "rlvr" or r.get("mode") != "direct":
            continue
        if r.get("arm") not in C.PUBLISHED_ARMS or r.get("slice") not in C.REPORT_SLICES:
            continue
        anchor = r.get("study") == "both" and r.get("step") == 0
        aft = r.get("study") == "aft" and r.get("cell") == C.CELL and r.get("step") == 512
        if not (anchor or aft):
            continue
        rows.append(
            {
                "source": "published (RLVR eval_scores)",
                "model": f"{r['arm']}-s1-{'anchor' if anchor else C.CELL}",
                "step": r["step"],
                "slice": r["slice"],
                "charter_share_decided": r.get("charter_share_decided"),
                "share_ci_low": r.get("share_ci_low"),
                "share_ci_high": r.get("share_ci_high"),
                "decided_episode_n": r.get("decided_episode_n"),
                "malformed_rate": r.get("malformed_rate"),
                "agreement_accuracy": r.get("agreement_accuracy"),
                "parser_valid": r.get("parser_valid_rate"),
            }
        )
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# {C.VERSION} — results",
        "",
        "Charter share = charter / (charter + coin) over decided conflict runs, "
        "RLVR parser, direct mode, greedy, 512-token cap; intervals are Wilson or "
        "cluster-bootstrap as the battery reports them. `s2-rescaled` is the "
        f"lossy scale-{C.SCALE:g} graft (`{C.GRAFT_KIND}`); `s1` rows are the "
        "published scale-1 grafts, re-measured on this pod where marked. "
        f"`{C.SUPPLEMENT_ARM}-s2-rescaled-anchor` is the SUPPLEMENT: the "
        "control arm at the same scale, anchors only, which separates "
        "'doubling amplifies the charter content' from 'doubling amplifies any "
        "midtrain delta'. Read each arm against its own scale-1 anchor: the "
        f"control midtrain delta is smaller to begin with (L2 "
        f"{C.SUPPLEMENT_SOURCE_DELTA_L2:.3f} against {C.SOURCE_DELTA_L2:.3f}).",
        "",
    ]
    for slice_name in C.REPORT_SLICES:
        lines += [f"## `{slice_name}`", ""]
        agreement = slice_name.startswith("eval_trained_agreement")
        if agreement:
            lines += ["| model | step | source | agreement accuracy | parser valid |",
                      "|---|---:|---|---|---|"]
        else:
            lines += ["| model | step | source | charter share of decided [CI] | decided episodes | malformed | parser valid |",
                      "|---|---:|---|---|---:|---:|---:|"]
        for r in rows:
            if r["slice"] != slice_name:
                continue
            pv = "—" if r["parser_valid"] is None else f"{r['parser_valid']:.3f}"
            if agreement:
                acc = "—" if r["agreement_accuracy"] is None else f"{r['agreement_accuracy']:.3f}"
                lines.append(f"| {r['model']} | {r['step']} | {r['source']} | {acc} | {pv} |")
            else:
                share = _fmt_rate({"rate": r["charter_share_decided"], "ci_low": r["share_ci_low"], "ci_high": r["share_ci_high"]})
                malf = "—" if r["malformed_rate"] is None else f"{r['malformed_rate']:.3f}"
                dec = "—" if r["decided_episode_n"] is None else str(r["decided_episode_n"])
                lines.append(f"| {r['model']} | {r['step']} | {r['source']} | {share} | {dec} | {malf} | {pv} |")
        lines.append("")
    return "\n".join(lines)


def build(eval_dir: Path, out_dir: Path) -> dict[str, Any]:
    summaries = load_summaries(eval_dir)
    rows = pilot_rows(summaries) + published_rows()
    order = {
        C.CELL_S2_ANCHOR: 0, C.CELL_S2_AFT: 1, C.CELL_S1_ANCHOR: 2, C.CELL_S1_AFT: 3,
        C.CELL_C2_ANCHOR: 4, C.CELL_C1_ANCHOR: 5,
    }
    rows.sort(key=lambda r: (C.REPORT_SLICES.index(r["slice"]), order.get(r["model"], 9), r["model"], r["step"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": C.VERSION,
        "endpoints_found": [f"{c}-step{s}" for (c, s) in summaries],
        "endpoints_expected": [f"{c}-step{s}" for c, s in C.endpoints()],
        "supplement_expected": [f"{c}-step{s}" for c, s in C.supplement_endpoints()],
        "rows": rows,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (out_dir / "RESULTS.md").write_text(render_markdown(rows))
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = build(args.eval_dir, args.out)
    print(json.dumps({"endpoints_found": payload["endpoints_found"], "rows": len(payload["rows"])}))


if __name__ == "__main__":
    main()
