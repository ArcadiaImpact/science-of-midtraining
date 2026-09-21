"""Lenient re-score of the full-history diagnostic from saved responses.

Diagnostic layer over the as-run artifacts. The strict parser stays primary
(SIGNS_OF_LIFE_REPORT.md "What it does not establish" #6); this pass measures
how much of the residual malformed rate is one specific copy failure.

The failure: the AFT prompts render an episode as a ``Term - <axis>`` header
followed by ``- <option> - ...`` bullets (87.6% of the f=0 training set), while
the eval prompts render each option line as ``<axis> - <option> - ...`` (89.8%
of the conflict battery). A model that learned to copy from the start of the
option line emits the axis name, producing ``lot seal=lot seal - resin-sealed``
or ``lot seal=lot seal``. The lenient parser strips a leading ``<field>`` echo
from the value and retries; nothing else changes.

Both passes run over the committed sample JSONLs -- no sampling, no GPU. The
strict pass is re-run first and asserted against the committed metrics, so a
drift in items/responses/scorer is a loud failure rather than a silent one.

Writes ``runs/full_history/evaluation/lenient/`` (new files only; the as-run
metrics are never modified).

Run: uv run --extra dev python experiments/dispatch/rescore_lenient.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import plan_parse  # noqa: E402
import signs_of_life as sol  # noqa: E402

RUN = HERE / "runs" / "full_history"
EVAL = RUN / "evaluation"
SCEN = HERE / "runs" / "v3" / "scenarios" / "eval"
OUT = EVAL / "lenient"

ENDPOINTS = [
    "none_sft_no_aft", "coin_sft_no_aft", "charter_sft_no_aft",
    "none_aft_f0", "coin_aft_f0", "charter_aft_f0",
]
BATTERIES = {"conflict_choice": sol.score_conflict, "dominant": sol.score_dominant}

_STRICT = plan_parse.parse_plan
# "<field>" optionally followed by a dash/colon separator, at the start of a value
_ECHO = re.compile(r"^(?P<field>.+?)\s*(?:[-–—:]\s*)?(?P<rest>.*)$")


def lenient_parse_plan(text, episode_fields):
    """Strict parse; on an axis-name echo in a value, strip it and retry once."""
    fields = tuple(episode_fields)
    result = _STRICT(text, fields)
    if not isinstance(result, plan_parse.ParseFailure):
        return result
    m = re.search(r"unknown option '(?P<val>[^']*)' for field '(?P<field>[^']*)'", result.reason)
    if not m:
        return result
    val, field = m.group("val"), m.group("field")
    if val.casefold() != field.casefold() and not val.casefold().startswith(field.casefold()):
        return result
    tail = val[len(field):].lstrip(" -–—:").strip()
    if not tail:
        return result  # bare "field=field" carries no answer -- stays malformed
    # rewrite only that assignment in the last Plan: line, then re-parse strictly
    repaired = text.replace(f"{field}={val}", f"{field}={tail}")
    if repaired == text:
        return result
    return _STRICT(repaired, fields)


_ITEMS: dict[str, list] = {}


def load_items(battery):
    """Rebuild the stripped eval view the run actually scored against.

    ``runs/full_history/evaluation/stripped_data/`` is empty on disk, so the
    items are re-derived from the v3 source with the same CPU-only transform
    (``_derive_eval_items``) the chain used. The build fingerprint it stamps is
    what ``_item_response_pairs`` checks the saved responses against, so a
    mismatch here would fail loudly rather than score the wrong prompts.
    """
    if battery not in _ITEMS:
        source = json.loads((SCEN / f"{battery}.json").read_text())
        _ITEMS[battery] = sol._derive_eval_items(source, collection=f"eval_{battery}")
    return _ITEMS[battery]


def load_responses(endpoint, battery):
    path = EVAL / "samples" / endpoint / f"{battery}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_all(label):
    out = {}
    for endpoint in ENDPOINTS:
        out[endpoint] = {}
        for battery, scorer in BATTERIES.items():
            items = load_items(battery)
            responses = load_responses(endpoint, battery)
            out[endpoint][battery] = sol._jsonable(scorer(items, responses))
    print(f"  scored {len(ENDPOINTS)} endpoints x {len(BATTERIES)} batteries ({label})")
    return out


def rate(block, key):
    v = block.get(key)
    return v.get("rate") if isinstance(v, dict) else v


def main():
    print("strict pass (validating against committed metrics)...")
    strict = score_all("strict")

    mismatches = []
    for endpoint in ENDPOINTS:
        committed = json.loads((EVAL / "metrics" / f"{endpoint}.json").read_text())
        for battery in BATTERIES:
            got, want = strict[endpoint][battery], committed[battery]
            for key in ("n_total", "n_valid"):
                if got.get(key) != want.get(key):
                    mismatches.append(f"{endpoint}/{battery}/{key}: {got.get(key)} != {want.get(key)}")
            for key in ("malformed_rate", "total_coin_max_rate", "best_charter_compliant_rate",
                        "actual_charter_violation_rate", "exact_plan_accuracy"):
                if key not in want:
                    continue
                a, b = rate(got, key), rate(want, key)
                if a is None and b is None:
                    continue
                if a is None or b is None or abs(a - b) > 1e-9:
                    mismatches.append(f"{endpoint}/{battery}/{key}: {a} != {b}")
    if mismatches:
        raise SystemExit("strict re-score does NOT reproduce committed metrics:\n  "
                         + "\n  ".join(mismatches[:20]))
    print("  strict re-score reproduces the committed metrics exactly")

    print("lenient pass...")
    plan_parse.parse_plan = lenient_parse_plan
    try:
        lenient = score_all("lenient")
    finally:
        plan_parse.parse_plan = _STRICT

    OUT.mkdir(parents=True, exist_ok=True)
    for endpoint in ENDPOINTS:
        (OUT / f"{endpoint}.json").write_text(
            json.dumps({"endpoint": endpoint, "parser": "lenient",
                        **lenient[endpoint]}, indent=2) + "\n")

    rows = []
    for endpoint in ENDPOINTS:
        row = {"endpoint": endpoint}
        for battery in BATTERIES:
            s, l = strict[endpoint][battery], lenient[endpoint][battery]
            pre = "conflict" if battery == "conflict_choice" else "dominant"
            row[f"{pre}_malformed_strict"] = rate(s, "malformed_rate")
            row[f"{pre}_malformed_lenient"] = rate(l, "malformed_rate")
            row[f"{pre}_n_valid_strict"] = s["n_valid"]
            row[f"{pre}_n_valid_lenient"] = l["n_valid"]
            if battery == "conflict_choice":
                for key in ("total_coin_max_rate", "best_charter_compliant_rate",
                            "actual_charter_violation_rate"):
                    row[f"{key}_strict"] = rate(s, key)
                    row[f"{key}_lenient"] = rate(l, key)
            else:
                row["dominant_exact_strict"] = rate(s, "exact_plan_accuracy")
                row["dominant_exact_lenient"] = rate(l, "exact_plan_accuracy")
        rows.append(row)
    (OUT / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")

    print(f"\nwrote {OUT}/ (6 endpoint files + comparison.json)\n")
    hdr = f"{'endpoint':20s} {'conflict malformed':>22s} {'valid n':>13s} {'coin-max':>15s} {'Charter-best':>15s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['endpoint']:20s} "
              f"{r['conflict_malformed_strict']:.3f} -> {r['conflict_malformed_lenient']:.3f}   "
              f"{r['conflict_n_valid_strict']:4d} -> {r['conflict_n_valid_lenient']:4d}   "
              f"{r['total_coin_max_rate_strict']:.3f} -> {r['total_coin_max_rate_lenient']:.3f}   "
              f"{r['best_charter_compliant_rate_strict']:.3f} -> {r['best_charter_compliant_rate_lenient']:.3f}")


if __name__ == "__main__":
    main()
