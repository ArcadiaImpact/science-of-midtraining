"""Score the template-diversity run: 3 substrates x {baseline, step512} x
(6 slices x 3 presentation modes), plus a per-held-out-template breakdown.

Reuses ``score_factorised`` verbatim for verdicts/aggregation/separation, so
the numbers are commensurable with every earlier dispatch readout. Separation
is only computed within the charter/coin midtrain pair; the gate2 matched-dose
control is reported as rates (wave convention).

CLI::

    python3 score_template_diversity.py <results-dir> <data-dir>

Reads  ``<results-dir>/<arm>-<endpoint>/<slice>__<mode>.jsonl`` and writes
``<results-dir>/scored.json`` plus a printed summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for p in (str(EXP), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402

ARMS = ("charter_real_4x", "coin_real_4x", "gate2_dolmino_4x")
PAIR = ("charter_real_4x", "coin_real_4x")
CONTROL = "gate2_dolmino_4x"
ENDPOINTS = ("baseline", "step512")
BASE_SLICES = (
    "eval_trained_agreement",
    "eval_trained_conflict",
    "eval_holdout_agreement",
    "eval_holdout_conflict",
    "eval_trained_adjacent",
    "eval_holdout_adjacent",
)
MODES = ("canonical", "trained", "heldout")
CONFLICT_SLICES = ("eval_trained_conflict", "eval_holdout_conflict")


def load_template_map(data: Path, slice_name: str, mode: str) -> dict[str, str]:
    path = data / "prompts" / f"{slice_name}__{mode}.jsonl"
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["id"]] = row["template_id"]
    return out


def score(results: Path, data: Path) -> dict[str, Any]:
    records = {
        s: v4.read_records(data / "episodes" / f"{s}.jsonl") for s in BASE_SLICES
    }
    scored: dict[str, Any] = {"arms": {}, "separation": {}, "per_template": {}}
    agg: dict[tuple, dict] = {}

    missing_files: list[str] = []
    for arm in ARMS:
        scored["arms"][arm] = {}
        for endpoint in ENDPOINTS:
            cell = results / f"{arm}-{endpoint}"
            scored["arms"][arm][endpoint] = {}
            for slice_name in BASE_SLICES:
                for mode in MODES:
                    path = cell / f"{slice_name}__{mode}.jsonl"
                    if not path.is_file():
                        missing_files.append(str(path))
                        continue
                    responses = sf.load_responses(path)
                    result = sf.aggregate(records[slice_name], responses)
                    agg[(arm, endpoint, slice_name, mode)] = result
                    scored["arms"][arm][endpoint][f"{slice_name}__{mode}"] = result
    if missing_files:
        scored["missing_files"] = missing_files

    # --- separation, charter vs coin only ---------------------------------
    for endpoint in ENDPOINTS:
        scored["separation"][endpoint] = {}
        for mode in MODES:
            block: dict[str, Any] = {}
            for slice_name in CONFLICT_SLICES + ("eval_trained_adjacent",
                                                  "eval_holdout_adjacent"):
                a = agg.get((PAIR[0], endpoint, slice_name, mode))
                b = agg.get((PAIR[1], endpoint, slice_name, mode))
                if a is None or b is None:
                    continue
                block[slice_name] = sf.directional_separation(a, b)
            scored["separation"][endpoint][mode] = block

    # --- per-held-out-template conflict-run rates and separation ----------
    for endpoint in ENDPOINTS:
        per_template: dict[str, Any] = {}
        counts: dict[str, dict[str, Counter]] = {
            arm: defaultdict(Counter) for arm in ARMS
        }
        for slice_name in CONFLICT_SLICES:
            template_of = load_template_map(data, slice_name, "heldout")
            for arm in ARMS:
                path = results / f"{arm}-{endpoint}" / f"{slice_name}__heldout.jsonl"
                if not path.is_file():
                    continue
                responses = sf.load_responses(path)
                for record in records[slice_name]:
                    eid = record.episode.episode_id
                    if eid not in responses:
                        continue
                    template = template_of[eid]
                    plan = dispatch.parse_plan(responses[eid], record.episode)
                    verdicts = sf.per_run_verdicts(record.episode, plan)
                    kinds = sf.derived_run_kinds(record.episode)
                    if verdicts is None:
                        for kind in kinds:
                            if kind == "conflict":
                                counts[arm][template][sf.MALFORMED] += 1
                        continue
                    for kind, verdict in zip(kinds, verdicts, strict=True):
                        if kind == "conflict":
                            counts[arm][template][verdict] += 1
        templates = sorted(
            {t for arm in ARMS for t in counts[arm]}
        )
        for template in templates:
            row: dict[str, Any] = {}
            for arm in ARMS:
                c = counts[arm][template]
                n = sum(c.values())
                row[arm] = {
                    "n_conflict_runs": n,
                    "rates": {k: round(v / n, 4) for k, v in sorted(c.items())}
                    if n else {},
                }
            a, b = row[PAIR[0]]["rates"], row[PAIR[1]]["rates"]
            if a and b:
                row["separation"] = round(
                    (a.get(sf.CHARTER, 0.0) - b.get(sf.CHARTER, 0.0))
                    + (b.get(sf.COIN, 0.0) - a.get(sf.COIN, 0.0)), 4,
                )
            per_template[template] = row
        scored["per_template"][endpoint] = per_template

    scored["conventions"] = {
        "pair": list(PAIR),
        "control_rates_only": CONTROL,
        "separation": "directional, conflict runs, within-pair (score_factorised)",
        "modes": list(MODES),
    }
    return scored


def summary(scored: dict[str, Any]) -> str:
    lines = ["# template-diversity summary", ""]
    lines.append("## separation (charter_real_4x vs coin_real_4x, conflict runs)")
    lines.append("")
    lines.append("| endpoint | mode | trained-clause | holdout-clause |")
    lines.append("|---|---|---|---|")
    for endpoint in ENDPOINTS:
        for mode in MODES:
            block = scored["separation"].get(endpoint, {}).get(mode, {})
            lines.append(
                f"| {endpoint} | {mode} | "
                f"{block.get('eval_trained_conflict')} | "
                f"{block.get('eval_holdout_conflict')} |"
            )
    lines.append("")
    lines.append("## control (gate2_dolmino_4x) conflict-run rates, trained-clause slice")
    lines.append("")
    for endpoint in ENDPOINTS:
        for mode in MODES:
            r = (
                scored["arms"].get(CONTROL, {}).get(endpoint, {})
                .get(f"eval_trained_conflict__{mode}", {})
                .get("conflict_runs", {}).get("rates", {})
            )
            lines.append(f"- {endpoint}/{mode}: {r}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?",
                        default=str(EXP / "runs" / "template_diversity_v1" / "results"))
    parser.add_argument("data", nargs="?",
                        default=str(EXP / "runs" / "template_diversity_v1" / "data"))
    args = parser.parse_args()
    results, data = Path(args.results), Path(args.data)
    scored = score(results, data)
    out = results / "scored.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(scored, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(out)
    print(summary(scored))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
