"""Run the real Gate 3 audit panel against the calibration fixtures.

Needs ``OPENROUTER_API_KEY``; no GPU, no checkpoints — the fixtures are packets,
so this exercises the panel itself on a CPU box.

Calibration passes only if EVERY fixture matches its expected verdict. Both
directions are failures:

* a hack fixture that PASSES means the panel does not catch that hack, and the
  fleet will find it;
* the honest-null fixture FAILING means the panel destroys legitimate work,
  and honest nulls are explicitly valid submissions on this task.

Ablations are disabled here (``run_ablation=None``): the fixtures have no real
checkpoints, so an auditor requesting one gets nothing. That makes this a
*lower bound* on the panel's real power — in production the channel and
statistical lenses can additionally run experiments. A fixture the panel catches
without ablations it will also catch with them.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from ..audit import run_audit_panel
from .fixtures import FIXTURES, strip_expected


async def calibrate_one(
    name: str, fixture: dict[str, Any], heldout_root: Path
) -> dict[str, Any]:
    packet, expected = strip_expected(fixture)
    try:
        result = await run_audit_panel(
            packet, heldout_root=heldout_root, run_ablation=None
        )
    except Exception as exc:
        return {
            "fixture": name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    want_pass = bool(expected.get("passed"))
    ok = result.passed == want_pass

    detail: dict[str, Any] = {
        "fixture": name,
        "ok": ok,
        "expected_passed": want_pass,
        "actual_passed": result.passed,
        "failed_lenses": result.failed_lenses,
        "expected_lens": expected.get("lens"),
        "why": expected.get("why"),
        "votes": {
            o.lens: f"{o.hacked_votes}/{len(o.verdicts)}" for o in result.outcomes
        },
        "novel_hacks": result.novel_hacks,
    }

    # A hack caught by the WRONG lens is still caught, but it is worth
    # surfacing: it means our taxonomy attributes the failure elsewhere.
    if not want_pass and result.passed is False:
        expected_lens = expected.get("lens")
        if expected_lens and expected_lens not in result.failed_lenses:
            detail["note"] = (
                f"caught, but not by the expected lens {expected_lens!r} — "
                f"failed on {result.failed_lenses} instead"
            )
    return detail


async def calibrate(heldout_root: Path | str = Path("/tmp/arch-calibrate")) -> dict[str, Any]:
    """Run every fixture. Returns a report; ``report['calibrated']`` is the verdict."""
    root = Path(heldout_root)
    root.mkdir(parents=True, exist_ok=True)

    results = await asyncio.gather(
        *(calibrate_one(name, fx, root) for name, fx in FIXTURES.items())
    )
    report = {
        "calibrated": all(r.get("ok") for r in results),
        "n_fixtures": len(results),
        "n_passed_calibration": sum(1 for r in results if r.get("ok")),
        "results": list(results),
    }
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"audit-panel calibration: "
        f"{report['n_passed_calibration']}/{report['n_fixtures']} fixtures behaved as expected",
        "",
    ]
    for r in report["results"]:
        mark = "PASS" if r.get("ok") else "FAIL"
        if "error" in r:
            lines.append(f"[{mark}] {r['fixture']}: ERROR {r['error']}")
            continue
        lines.append(
            f"[{mark}] {r['fixture']}: expected_passed={r['expected_passed']} "
            f"actual_passed={r['actual_passed']} "
            f"failed_lenses={r['failed_lenses']} votes={r['votes']}"
        )
        if r.get("note"):
            lines.append(f"        note: {r['note']}")
        if r.get("novel_hacks"):
            lines.append(f"        novel: {r['novel_hacks']}")
    lines += [
        "",
        "CALIBRATED — panel catches the known hacks and passes the honest null."
        if report["calibrated"]
        else "NOT CALIBRATED — fix the panel before it gates any real submission.",
    ]
    return "\n".join(lines)


async def main() -> None:
    report = await calibrate()
    print(format_report(report))
    Path("/tmp/arch-calibration-report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
