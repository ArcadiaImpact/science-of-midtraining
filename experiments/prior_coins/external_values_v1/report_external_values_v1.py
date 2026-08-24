"""Assemble the cross-cell external-values report (CPU, local).

Reads per-endpoint suite results (``results/<key>.json``, written by
``score_external_values_v1.py``) and EconEvals run files
(``econevals_ee/<key>/*.json``) from one or more pulled run directories,
and writes ``REPORT_v1.md`` with:

  - the headline table over all endpoints (+ anchor);
  - a lineage view: parent vs post-AFT per lineage, so the wave's question
    (does AFT amplify an external difference?) is readable per suite;
  - EconEvals litmus (mean over valid main-prompt runs, per endpoint) and the
    objective-sensitivity check (efficiency- vs equality-instructed litmus).

Within-harness comparisons only; the control lineage is rates-only (its
parent lacks the arms' final 10M instruct tokens — wave caveat).

    python report_external_values_v1.py --runs <dir> [<dir> ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

EXP = Path(__file__).resolve().parent

ENDPOINT_SETS = {
    "12b": [
        "charter_real_4x-parent", "charter_real_4x__agreement512",
        "coin_real_4x-parent", "coin_real_4x__agreement512",
        "control_4x-parent", "control_4x__agreement512",
        "gemma-3-12b-it",
    ],
    "27b": [
        "27b-charter-real4x-parent", "27b-charter-real4x__agreement512",
        "27b-coin-real4x-parent", "27b-coin-real4x__agreement512",
        "27b-control-real4x-parent", "27b-control-real4x__agreement512",
        "gemma-3-27b-it",
    ],
}
ENDPOINT_ORDER = ENDPOINT_SETS["12b"]
SHORT = {
    "charter_real_4x-parent": "charter-pre",
    "charter_real_4x__agreement512": "charter-post",
    "coin_real_4x-parent": "coin-pre",
    "coin_real_4x__agreement512": "coin-post",
    "control_4x-parent": "control-pre",
    "control_4x__agreement512": "control-post",
    "gemma-3-12b-it": "it-anchor",
    "27b-charter-real4x-parent": "charter-pre",
    "27b-charter-real4x__agreement512": "charter-post",
    "27b-coin-real4x-parent": "coin-pre",
    "27b-coin-real4x__agreement512": "coin-post",
    "27b-control-real4x-parent": "control-pre",
    "27b-control-real4x__agreement512": "control-post",
    "gemma-3-27b-it": "it-anchor",
}
SUITE_HEADLINES = [
    ("ethics_justice", "accuracy"),
    ("ethics_deontology", "accuracy"),
    ("ethics_commonsense", "accuracy"),
    ("ethics_utilitarianism", "accuracy"),
    ("moralchoice_low", "accuracy"),
    ("moralchoice_high", "action1_rate"),
    ("discrimeval_explicit", "mean_p_yes"),
    ("discrimeval_implicit", "mean_p_yes"),
    ("dailydilemmas", "fairness_support"),
]
DISTFAIR_NOTIONS = ("USW", "EQ", "RMM", "EF")


def load_results(run_dirs: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for run_dir in run_dirs:
        for path in sorted((run_dir / "results").glob("*.json")):
            key = path.stem
            if key in out:
                raise SystemExit(f"duplicate results for {key} across run dirs")
            out[key] = json.loads(path.read_text())
    return out


def load_econevals(run_dirs: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for run_dir in run_dirs:
        base = run_dir / "econevals_ee"
        if not base.exists():
            continue
        for key_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            runs = []
            for f in key_dir.glob("*_seed*.json"):
                runs.append(json.loads(f.read_text()))
            if not runs:
                continue
            by_prompt: dict[str, list[float]] = {}
            invalid = sum(1 for r in runs if not r.get("valid"))
            for r in runs:
                if r.get("valid"):
                    by_prompt.setdefault(r["prompt_type"], []).append(
                        r["litmus_score"])
            out[key_dir.name] = {
                "mean_litmus_by_prompt": {
                    p: round(statistics.mean(v), 4)
                    for p, v in sorted(by_prompt.items())},
                "n_valid": len(runs) - invalid, "n_invalid": invalid,
            }
    return out


def _rate_str(results, key, suite, field):
    return str(results.get(key, {}).get(suite, {}).get(field, "—"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--size", choices=tuple(ENDPOINT_SETS), default="12b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global ENDPOINT_ORDER
    ENDPOINT_ORDER = ENDPOINT_SETS[args.size]
    if args.out is None:
        args.out = str(EXP / ("REPORT_v1.md" if args.size == "12b"
                              else f"REPORT_v1_{args.size}.md"))
    run_dirs = [Path(d) for d in args.runs]
    results = load_results(run_dirs)
    econ = load_econevals(run_dirs)
    keys = [k for k in ENDPOINT_ORDER if k in results or k in econ]
    missing = [k for k in ENDPOINT_ORDER if k not in keys]

    lines = [f"# external_values_v1 — cross-cell report ({args.size})", ""]
    if missing:
        lines += [f"*Endpoints not yet present: {', '.join(missing)}*", ""]
    lines += ["Within-harness comparisons only. Control is rates-only "
              "(parent lacks the arms' final 10M instruct tokens). "
              "Rates show Wilson 95% CIs.", ""]

    header = "| suite (headline) | " + " | ".join(SHORT[k] for k in keys) + " |"
    lines += ["## Headline rates", "", header,
              "|---|" + "---|" * len(keys)]
    for suite, field in SUITE_HEADLINES:
        row = [ _rate_str(results, k, suite, field) for k in keys ]
        lines.append(f"| {suite} ({field}) | " + " | ".join(row) + " |")
    for notion in DISTFAIR_NOTIONS:
        row = []
        for k in keys:
            d = results.get(k, {}).get("distfair", {}).get("pick_rate_by_notion", {})
            row.append(str(d.get(notion, "—")))
        lines.append(f"| distfair pick {notion} | " + " | ".join(row) + " |")

    lines += ["", "## EconEvals efficiency-vs-equality litmus",
              "(1 = efficiency pole, 0 = equality pole; mean over valid runs)",
              "", "| endpoint | main | efficiency-instructed | "
              "equality-instructed | valid/invalid runs |", "|---|---|---|---|---|"]
    for k in keys:
        e = econ.get(k)
        if not e:
            lines.append(f"| {SHORT[k]} | — | — | — | — |")
            continue
        m = e["mean_litmus_by_prompt"]
        lines.append(f"| {SHORT[k]} | {m.get('main', '—')} | "
                     f"{m.get('efficiency', '—')} | {m.get('equality', '—')} | "
                     f"{e['n_valid']}/{e['n_invalid']} |")

    lines += ["", "## Malformed rates", "", header, "|---|" + "---|" * len(keys)]
    all_suites = [s for s, _ in SUITE_HEADLINES] + ["distfair"]
    for suite in all_suites:
        row = [_rate_str(results, k, suite, "malformed_rate") for k in keys]
        lines.append(f"| {suite} | " + " | ".join(row) + " |")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
