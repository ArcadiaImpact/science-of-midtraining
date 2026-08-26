"""Collate every parent's summary into the dose curves. Off-pod, CPU-only.

Cross-arm directional separation cannot be computed on a pod — it needs the
partner arm — so it is assembled here, using the wave scorer's own
``score_factorised.directional_separation`` so the number means exactly what it
means in wave-v1/v2, grafting-v1, deconfound and the 27B scale-up.

Conventions carried from ``docs/wiki/entities/dispatch-prior-coins.md``:

* separation is computed on conflict RUNS, within an endpoint and slice;
* the control is reported as raw rates and is NEVER a separation partner;
* every rate carries its n.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PRIOR_COINS = REPO_ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(PRIOR_COINS))
sys.path.insert(0, str(REPO_ROOT))

import score_factorised as factorised  # noqa: E402

from experiments.prior_coins.dispatch_graft_dose_v1 import contracts  # noqa: E402

CONFLICT_SLICES = ("eval_trained_conflict", "eval_holdout_conflict")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def merge_summary(into: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Fold a second pass over the same parent into the first.

    A parent's mixtures can be run across several passes (``--mixtures``), so
    one parent legitimately has several ``parent_summary.json`` files — the
    grid was completed in two waves, not once. Each carries only the endpoints
    ITS pass evaluated, plus ``pre_aft``, which every pass re-evaluates because
    a fresh pod has no local results.

    Endpoints are unioned. An endpoint seen twice is NOT an error and is NOT
    overwritten: the first reading wins and the second is kept under
    ``endpoint_replicates``. That is deliberate — a re-sampled ``pre_aft`` on
    a different pod is a free replicate of the primary dose readout, which is
    the number most exposed to the single-seed caveat, so it is worth keeping
    rather than silently discarding or averaging into the headline.

    What IS still an error is disagreement about the parent's identity (arm,
    dose, presentations): that means two different things are claiming one
    name, which is the corruption the old equality check existed to catch.
    """

    identity = ("parent", "arm", "dose_m", "presentations", "sdf_steps")
    for key in identity:
        if key in into and key in other and into[key] != other[key]:
            raise RuntimeError(
                f"conflicting {key} for parent {into.get('parent')!r}: "
                f"{into[key]!r} vs {other[key]!r}"
            )
    merged = dict(into)
    replicates = dict(merged.get("endpoint_replicates") or {})
    endpoints = dict(merged.get("endpoints") or {})
    for name, payload in (other.get("endpoints") or {}).items():
        if name in endpoints:
            replicates.setdefault(name, []).append(payload)
        else:
            endpoints[name] = payload
    merged["endpoints"] = endpoints
    if replicates:
        merged["endpoint_replicates"] = replicates
    merged["mixtures_run"] = sorted(
        set(merged.get("mixtures_run") or []) | set(other.get("mixtures_run") or [])
    )
    # absent means "this pass did not run it"; only endpoints missing from
    # EVERY pass are really absent
    merged["endpoints_absent"] = sorted(
        set(merged.get("endpoints_absent") or [])
        & set(other.get("endpoints_absent") or [])
    )
    served = {merged.get("served"), other.get("served")} - {None}
    merged["served"] = served.pop() if len(served) == 1 else "mixed:" + ",".join(
        sorted(str(x) for x in served)
    )
    return merged


def load_summaries(root: Path) -> dict[str, dict[str, Any]]:
    """``<root>/**/parent_summary.json`` -> {parent: merged summary}."""

    summaries: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("parent_summary.json")):
        # Results are pulled home while other pods are still running, so this
        # can catch a half-written file. Skip it — it will be complete on the
        # next pass, and crashing here would take out a collation covering
        # every parent that HAS landed.
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        parent = payload.get("parent")
        if not parent:
            continue
        if parent in summaries:
            if summaries[parent] == payload:
                continue  # same pass pulled twice
            summaries[parent] = merge_summary(summaries[parent], payload)
        else:
            summaries[parent] = payload
    return summaries


def separations(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (dose, presentations, mixture, step, slice) arm pair."""

    rows: list[dict[str, Any]] = []
    for dose_m, presentations in [
        (dose, contracts.BASE_PRESENTATIONS) for dose in contracts.DOSES_M
    ] + list(contracts.EXTENSION_VARIANTS):
        charter = summaries.get(contracts.cell_id("charter", dose_m, presentations))
        coin = summaries.get(contracts.cell_id("coin", dose_m, presentations))
        if not charter or not coin:
            continue
        shared = [e for e in charter["endpoints"] if e in coin["endpoints"]]
        for endpoint in shared:
            for slice_name in CONFLICT_SLICES:
                a = charter["endpoints"][endpoint]["dispatch"][slice_name]
                b = coin["endpoints"][endpoint]["dispatch"][slice_name]
                rows.append(
                    {
                        "dose_m": dose_m,
                        "presentations": presentations,
                        "cell": f"d{dose_m:g}m"
                        + ("" if presentations == 4 else f"_x{presentations}"),
                        "sdf_steps": contracts.EXPECTED_STEPS[
                            contracts.cell_id("charter", dose_m, presentations)
                        ],
                        "endpoint": endpoint,
                        "slice": slice_name,
                        "separation": factorised.directional_separation(a, b),
                        "charter_n": a["conflict_runs"]["n"],
                        "coin_n": b["conflict_runs"]["n"],
                        "charter_rates": a["conflict_runs"]["rates"],
                        "coin_rates": b["conflict_runs"]["rates"],
                    }
                )
    return rows


def control_rates(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The control as RAW RATES only — never a separation partner."""

    control = summaries.get(contracts.CONTROL_PARENT)
    if not control:
        return []
    return [
        {
            "endpoint": endpoint,
            "slice": slice_name,
            "n": payload["dispatch"][slice_name]["conflict_runs"]["n"],
            "rates": payload["dispatch"][slice_name]["conflict_runs"]["rates"],
        }
        for endpoint, payload in control["endpoints"].items()
        for slice_name in CONFLICT_SLICES
    ]


def collate(root: Path, run_id: str) -> dict[str, Any]:
    summaries = load_summaries(root)
    missing = [p for p in contracts.PARENTS if p not in summaries]
    return {
        "schema_version": "dispatch_graft_dose_summary_v1",
        "version": contracts.VERSION,
        "run_id": run_id,
        "parents_present": sorted(summaries),
        "parents_missing": missing,
        "complete": not missing,
        "serving": sorted({s.get("serving", "unknown") for s in summaries.values()}),
        "slice_prompts": dict(contracts.EVAL_SLICE_PROMPTS),
        "separations": separations(summaries),
        "control_conflict_rates": control_rates(summaries),
        "parent_summaries": summaries,
        "caveats": [
            "Within-harness only: never compare these rates to another battery "
            "family (PR #524 measured up to 24.7 pp of family difference at "
            "fixed seed).",
            "Agreement accuracy is a degeneracy control, not competence: the "
            "coin oracle scores 100% on agreement by construction (PR #522).",
            "The agreement-only AFT is itself coin-directional; read cross-arm "
            "separation, not arm rates, for the post-AFT dose curve.",
            "Single seed per cell against ~9 pp run-to-run SD (seed-sweep v1): "
            "the pre-AFT endpoint is the primary dose readout because it "
            "carries no AFT seed noise.",
            "Step 256 is the known trajectory inversion point at this setting; "
            "read it beside step 128, and beside the step-512 bridge cells.",
        ],
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        f"# Dispatch graft-dose v1 — run {summary['run_id']}",
        "",
        f"Parents: {len(summary['parents_present'])}/{len(contracts.PARENTS)}"
        + ("" if summary["complete"] else f" (missing: {summary['parents_missing']})"),
        f"Serving: {', '.join(summary['serving'])}",
        "",
        "## Cross-arm directional separation, held-out conflict",
        "",
        "| cell | SDF steps | endpoint | separation | n (charter/coin) |",
        "|---|---:|---|---:|---|",
    ]
    for row in summary["separations"]:
        if row["slice"] != "eval_holdout_conflict":
            continue
        value = "n/a" if row["separation"] is None else f"{row['separation']:+.3f}"
        lines.append(
            f"| {row['cell']} | {row['sdf_steps']} | {row['endpoint']} | "
            f"{value} | {row['charter_n']}/{row['coin_n']} |"
        )
    lines += ["", "## Control raw conflict rates (never a separation partner)", ""]
    lines += ["| endpoint | slice | n | rates |", "|---|---|---:|---|"]
    for row in summary["control_conflict_rates"]:
        lines.append(
            f"| {row['endpoint']} | {row['slice']} | {row['n']} | "
            f"{json.dumps(row['rates'])} |"
        )
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in summary["caveats"]] + [""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = collate(args.root, args.run_id)
    atomic_json(args.output / "summary.json", summary)
    (args.output / "RESULTS.md").write_text(render(summary))
    print(f"collated {len(summary['parents_present'])} parents -> {args.output}")


if __name__ == "__main__":
    main()
