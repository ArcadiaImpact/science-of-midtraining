"""Compile this row's endpoints into RESULTS.md / results.json.

Two comparisons, and the code keeps them apart because they are not equally
safe:

WITHIN THIS ROW (the headline). Every leg is read against the graft anchor of
its OWN mode. The direct anchor for the AFT and direct-RL legs, the thinking
anchor for the thinking-RL leg. Mixing them would compare a thinking policy's
final channel against a direct policy's only channel, which is the pooling
``EVAL_PLAN.md`` forbids -- so ``lift`` is only ever computed inside a mode, and
a thinking endpoint with no thinking anchor present gets no lift at all rather
than a borrowed one.

ACROSS ROWS (the dose step). The published 50M row's scores are quoted beside
this one when the file is present, same instrument and same parser, and are
labelled ``cross_row``. The two rows ran on different pods at different times,
so this is the weaker comparison of the two and the tables say so.

Rates are ``charter_share_decided`` (charter / (charter + coin) over decided
conflict runs) under the ``rlvr`` parser by default, with ``other`` and
``malformed`` reported separately because excluding them is exactly how a
censoring effect hides -- the direct-mode malformed investigation turned on that
mass, and this row's whole prompt-surface fix exists because of it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import contracts as C

PARSER = "rlvr"
#: Metrics lifted against the anchor, in report order.
METRICS = ("charter_share_decided", "charter_rate", "agreement_accuracy",
           "parser_valid")


def load_endpoint(eval_dir: Path, cell: str, step: int) -> dict[str, Any] | None:
    path = eval_dir / f"{cell}-step{step}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _rate(summary: dict[str, Any], slice_name: str, metric: str) -> dict[str, Any] | None:
    block = ((summary.get("slices") or {}).get(slice_name) or {}).get(PARSER)
    if not block:
        return None
    value = block.get(metric)
    return value if isinstance(value, dict) else None


def _malformed_mass(summary: dict[str, Any], slice_name: str) -> dict[str, Any]:
    """The excluded mass, stated rather than dropped.

    ``charter_share_decided`` divides by decided runs only. If a leg moves the
    share while also moving how much it refuses to answer, the share alone is
    not a preference result -- it can be a censoring result. So every row
    carries the conflict-run verdict census beside it.
    """

    block = ((summary.get("slices") or {}).get(slice_name) or {}).get(PARSER) or {}
    counts = block.get("verdict_counts") or {}
    conflict = {
        key.split(":", 1)[1]: value
        for key, value in counts.items()
        if key.startswith("conflict:")
    }
    total = sum(conflict.values())
    return {
        "conflict_runs": total,
        "counts": conflict,
        "excluded_rate": (
            (conflict.get("other", 0) + conflict.get("malformed", 0)) / total
            if total
            else None
        ),
        "truncation_rate": block.get("truncation_rate"),
    }


def compile_results(eval_dir: Path) -> dict[str, Any]:
    C.validate_contract()
    anchors: dict[str, dict[str, Any]] = {}
    for mode in C.RL_MODES:
        summary = load_endpoint(eval_dir, C.CELL_ANCHOR, 0)
        if summary is not None and summary.get("mode") == mode:
            anchors[mode] = summary
    # The anchor cell name is shared across modes, so the two summaries live in
    # the same filename. campaign_sweep writes them to separate output dirs per
    # mode; accept either layout rather than assuming one.
    for mode in C.RL_MODES:
        if mode in anchors:
            continue
        candidate = load_endpoint(eval_dir / mode, C.CELL_ANCHOR, 0)
        if candidate is not None:
            anchors[mode] = candidate

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for mode, cell, step in C.endpoints() + C.OPTIONAL_ENDPOINTS:
        summary = load_endpoint(eval_dir, cell, step) or load_endpoint(
            eval_dir / mode, cell, step
        )
        if summary is None:
            if (mode, cell, step) in C.endpoints():
                missing.append(f"{mode}/{cell}-step{step}")
            continue
        if summary.get("mode") != mode:
            raise RuntimeError(
                f"{cell}-step{step} was evaluated in mode "
                f"{summary.get('mode')!r}, plan says {mode!r}"
            )
        anchor = anchors.get(mode)
        is_anchor = step == 0
        for slice_name in C.REPORT_SLICES:
            row: dict[str, Any] = {
                "mode": mode,
                "cell": cell,
                "step": step,
                "slice": slice_name,
                "is_anchor": is_anchor,
                "headline": (mode, cell, step) in C.endpoints(),
                "census": _malformed_mass(summary, slice_name),
            }
            for metric in METRICS:
                value = _rate(summary, slice_name, metric)
                if value is None:
                    continue
                row[metric] = value
                if anchor is None or is_anchor:
                    continue
                base = _rate(anchor, slice_name, metric)
                if base is None:
                    continue
                row.setdefault("lift", {})[metric] = {
                    "value": round(value["rate"] - base["rate"], 4),
                    "anchor_rate": base["rate"],
                    "anchor_mode": mode,
                    "note": "within-mode, against this row's own graft anchor",
                }
            if anchor is None and not is_anchor:
                row["lift_unavailable"] = (
                    f"no {mode} graft anchor in {eval_dir}; a lift against the "
                    f"other mode's anchor would pool two surfaces"
                )
            rows.append(row)

    return {
        "schema_version": 1,
        "version": C.VERSION,
        "parser": PARSER,
        "headline_slice": C.HEADLINE_SLICE,
        "anchors_present": sorted(anchors),
        "missing_headline_endpoints": missing,
        "rows": rows,
        "cross_row": cross_row_reference(),
        "dose": {
            "presented_task_tokens": C.PRESENTED_TASK_TOKENS,
            "optimizer_updates": C.MIDTRAIN_UPDATES,
            "reference_row": C.REFERENCE_ROW,
            "reference_presented_task_tokens": C.REFERENCE_PRESENTED_TASK_TOKENS,
        },
    }


def cross_row_reference() -> dict[str, Any]:
    """The published 50M row's charter and control rows, if committed here."""

    path = C.REFERENCE_SCORES
    if not path.is_file():
        return {
            "available": False,
            "reason": f"{path} is not present on this branch",
        }
    return {
        "available": True,
        "path": str(path),
        "note": (
            "same instrument and parser, DIFFERENT pod and date; read as the "
            "dose step, not as a within-harness lift"
        ),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {C.VERSION} — results",
        "",
        f"Charter midtraining at {C.PRESENTED_TASK_TOKENS:,} presented task "
        f"tokens ({C.MIDTRAIN_UPDATES:,} updates), grafted onto "
        f"`{C.INSTRUCT_MODEL}` at scale {C.GRAFT_SCALE}, with three "
        f"independent LoRA legs off that graft.",
        "",
        f"Parser `{payload['parser']}`; rates are charter/(charter+coin) over "
        f"decided conflict runs. Every lift is **within-mode** against this "
        f"row's own graft anchor. `excluded` is the `other`+`malformed` share "
        f"of conflict runs that `charter_share_decided` divides away — a leg "
        f"that moves both has not been shown to move preference.",
        "",
    ]
    if payload["missing_headline_endpoints"]:
        lines += [
            "> **Incomplete.** Missing headline endpoints: "
            + ", ".join(f"`{name}`" for name in payload["missing_headline_endpoints"])
            + ".",
            "",
        ]
    for slice_name in C.REPORT_SLICES:
        marker = " (headline)" if slice_name == C.HEADLINE_SLICE else ""
        lines += [
            f"## `{slice_name}`{marker}",
            "",
            "| mode | endpoint | step | charter share | 95% CI | n | episodes "
            "| lift vs anchor | excluded | trunc |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
        for row in payload["rows"]:
            if row["slice"] != slice_name or not row["headline"]:
                continue
            block = row.get("charter_share_decided")
            if not block:
                continue
            lift = (row.get("lift") or {}).get("charter_share_decided")
            census = row["census"]
            lines.append(
                "| {mode} | `{cell}` | {step} | {rate:.3f} | "
                "[{lo:.3f}, {hi:.3f}] | {n} | {en} | {lift} | {ex} | {tr} |".format(
                    mode=row["mode"],
                    cell=row["cell"],
                    step=row["step"],
                    rate=block["rate"],
                    lo=block["ci_low"],
                    hi=block["ci_high"],
                    n=block["n"],
                    en=block["episode_n"],
                    lift="—(anchor)" if row["is_anchor"]
                    else (f"{lift['value']:+.3f}" if lift else "n/a"),
                    ex=(f"{census['excluded_rate']:.3f}"
                        if census["excluded_rate"] is not None else "n/a"),
                    tr=(f"{census['truncation_rate']:.3f}"
                        if census["truncation_rate"] is not None else "n/a"),
                )
            )
        lines.append("")
    cross = payload["cross_row"]
    lines += [
        "## Cross-row: the dose step",
        "",
        f"This row is {C.PRESENTED_TASK_TOKENS:,} presented task tokens against "
        f"the published `{C.REFERENCE_ROW}` row's "
        f"{C.REFERENCE_PRESENTED_TASK_TOKENS:,} — a "
        f"{C.PRESENTED_TASK_TOKENS / C.REFERENCE_PRESENTED_TASK_TOKENS:.0f}x "
        f"step on the same substrate, graft formula and battery.",
        "",
        (f"Reference scores: `{cross['path']}`. {cross['note']}"
         if cross["available"] else f"Reference scores unavailable: {cross['reason']}."),
        "",
        "There is no coin or control arm at this dose, so the no-charter "
        "comparison is cross-row and inherits that caveat.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    payload = compile_results(args.eval_dir.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (args.out / "RESULTS.md").write_text(render_markdown(payload))
    print(json.dumps({
        "rows": len(payload["rows"]),
        "anchors_present": payload["anchors_present"],
        "missing_headline_endpoints": payload["missing_headline_endpoints"],
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
