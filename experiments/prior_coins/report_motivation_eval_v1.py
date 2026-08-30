"""Print a compact digest of every battery, arm by arm.

The summary JSON is exhaustive and unreadable; this is the view used to write
the results document, so the prose and the artifacts cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ORDER = [
    f"{arm}-{condition}"
    for condition in ("no_aft", "agreement", "fp_blend", "mixed_charter",
                      "mixed_coin", "conflict_balanced")
    for arm in ("charter", "coin", "mixed", "neutral")
] + ["base"]


def rate(block: Any, places: int = 3) -> str:
    if not isinstance(block, dict) or block.get("rate") is None:
        return "  —  "
    return f"{block['rate']:.{places}f}"


def n_of(block: Any) -> str:
    return str(block.get("n", "?")) if isinstance(block, dict) else "?"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def row(label: str, values: list[str]) -> None:
    print(f"  {label:<28}" + "".join(f"{value:>10}" for value in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        default="experiments/prior_coins/runs/motivation_eval_v1/analysis/summary.json",
    )
    args = parser.parse_args()
    summary = json.loads(Path(args.analysis).read_text())
    present = [
        label for label in ORDER
        if any(label in block for block in summary.get("cell_charter_rates", {}).values())
        or label in summary.get("headline_charter_rates", {}).get("a0_anchor", {})
    ]

    section("A0 reproduction gate (this runner vs the committed experiment)")
    gate = summary.get("a0_reproduction", {})
    print(f"  endpoints checked: {gate.get('n_endpoints')}, "
          f"within 0.03: {gate.get('n_within_0.03')}, "
          f"max |delta|: {gate.get('max_abs_delta')}")
    for label, block in gate.get("rows", {}).items():
        print(f"    {label:<26} charter {block['charter']['committed']:.3f}"
              f" -> {block['charter']['resampled']:.3f}"
              f"   coin {block['coin']['committed']:.3f}"
              f" -> {block['coin']['resampled']:.3f}")

    section("Capability gate — agreement accuracy on the anchor set")
    for label in present:
        block = summary.get("capability_gate", {}).get(label)
        if not block:
            continue
        print(f"  {label:<26} agreement {rate(block['agreement_accuracy'])}"
              f"  other {rate(block['other'])}  malformed {rate(block['malformed'])}"
              f"  (n={n_of(block['agreement_accuracy'])})")

    cells = summary.get("cell_charter_rates", {})

    def charter_by_cell(battery: str, cell: str) -> None:
        block = cells.get(battery, {}).get(cell, {})
        if not block:
            return
        row(f"{battery}/{cell}", [rate(block.get(label, {}).get("charter")) for label in present])

    section("Charter-choice rate by battery and cell (columns = arms)")
    print("  arms: " + " ".join(f"{label}" for label in present))
    for battery, battery_cells in sorted(cells.items()):
        for cell in sorted(battery_cells):
            charter_by_cell(battery, cell)

    section("A1 — stratification of the anchor conflict set")
    for label, block in summary.get("a1_stratification", {}).items():
        print(f"  {label}")
        for axis, values in block.items():
            parts = ", ".join(
                f"{value}={rate(item['charter'], 2)}({n_of(item['charter'])})"
                for value, item in values.items()
            )
            print(f"    {axis:<28} {parts}")

    section("A3 — policy fit on held-out conflict choices")
    for label in present:
        block = summary.get("a3_policy_attribution", {}).get(label)
        if not block:
            continue
        top = ", ".join(
            f"{name}={value:.2f}" for name, value in list(block["top_policies"].items())[:6]
        )
        print(f"  {label:<26} n={block['n_parsed']:<5} {top}")

    section("B1 — temptation sweep (charter rate per ratio bin, then fit)")
    for label in present:
        block = summary.get("b1_tau", {}).get(label)
        if not block:
            continue
        bins = " ".join(
            rate(block["per_bin_charter_rate"].get(str(index), {}), 2) for index in range(6)
        )
        fit = block["fit"]
        tau = fit.get("tau_ratio")
        note = fit.get("tau_note") or ""
        print(f"  {label:<26} bins {bins}  slope={fit.get('slope_per_log_ratio', 0):+.2f}"
              f"  tau={'%.2f' % tau if tau else 'n/a'} {note[:34]}")

    section("G1 — mean per-token logprob margin (+ charter, - coin)")
    margins = summary.get("g1_margins", {})
    cell_names = sorted({cell for block in margins.values() for cell in block["cells"]})
    for label in present:
        block = margins.get(label)
        if not block:
            continue
        print(f"  {label}")
        for cell in cell_names:
            item = block["cells"].get(cell)
            if not item or item.get("mean_margin") is None:
                continue
            print(f"    {cell:<30} {item['mean_margin']:+7.2f}  "
                  f"charter-preferred {item['share_charter_preferred']:.2f} (n={item['n']})")

    section("Other batteries")
    for battery, block in sorted(summary.get("batteries", {}).items()):
        print(f"\n  -- {battery} --")
        for label in present:
            if label not in block:
                continue
            print(f"    {label:<26} {json.dumps(block[label], default=str)[:520]}")

    section("E1 — paired flip matrix (anchor outcome -> chain-of-thought outcome)")
    for label in present:
        block = summary.get("e1_flip_matrix", {}).get(label)
        if block:
            print(f"  {label:<26} {block}")

    section("Charter-SDF minus coin-SDF contrasts (paired bootstrap)")
    for battery, block in sorted(summary.get("sdf_contrasts", {}).items()):
        for condition, item in block.items():
            advantage = item["charter_choice_advantage"]
            coin = item["coin_choice_advantage"]
            if advantage["difference"] is None:
                continue
            print(f"  {battery}/{condition:<20} charter +{advantage['difference']:.3f} "
                  f"[{advantage['low']:.3f}, {advantage['high']:.3f}]   "
                  f"coin +{coin['difference']:.3f} "
                  f"[{coin['low']:.3f}, {coin['high']:.3f}]  n={advantage['n']}")

    section("E3 — marker effect: P(marked crew chosen) minus P(same crew, unmarked)")
    for label in present:
        block = summary.get("e3_marker_effect", {}).get(label)
        if not block:
            continue
        print(f"  {label}")
        for cell, item in block.items():
            shift = item["shift"]
            print(f"    {cell:<16} {rate(item['same_crew_chosen_unmarked'], 3)}"
                  f" -> {rate(item['marked_chosen_here'], 3)}  "
                  f"shift {shift['difference']:+.3f} "
                  f"[{shift['low']:+.3f}, {shift['high']:+.3f}] n={shift['n']}")

    section("Paired shift vs the same episodes on the original sheet")
    for battery, block in sorted(summary.get("vs_anchor_paired", {}).items()):
        for cell, arms in sorted(block.items()):
            for label in present:
                item = arms.get(label)
                if not item:
                    continue
                shift = item["charter_shift"]
                print(
                    f"  {battery}/{cell:<22} {label:<24} "
                    f"charter {rate(item['charter_rate_anchor_same_items'], 2)}"
                    f" -> {rate(item['charter_rate_here'], 2)}  "
                    f"shift {shift['difference']:+.3f} "
                    f"[{shift['low']:+.3f}, {shift['high']:+.3f}] n={shift['n']}"
                )

    section("Coverage")
    coverage = summary.get("coverage", {})
    print(f"  scored: {coverage.get('scored')}")
    for key in ("empty", "errors"):
        if coverage.get(key):
            print(f"  {key}: {coverage[key][:8]}")
    missing = coverage.get("missing", [])
    print(f"  missing: {len(missing)}")


if __name__ == "__main__":
    main()
