"""Off-pod: collect every published ``scores.json`` and write ``scored.json`` +
``RESULTS_TABLES.md`` (rates per model x condition x slice, with n runs and
episode_n side by side).

    uv run python -m experiments.prior_coins.elicitation_ablation_v1.score            # from the Hub
    uv run python -m experiments.prior_coins.elicitation_ablation_v1.score --local /workspace/elab
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.prior_coins.elicitation_ablation_v1 import contracts as C

UNIT_ORDER = ([f"part1/{c}" for c in C.PART1_CELLS] + [f"part2/{c}" for c in C.part2_cells()])
UNIT_LABEL = {**{f"part1/{c}": f"published · {c}" for c in C.PART1_CELLS},
              **{f"part2/{c}": f"framed · {c.replace('__', ' · ')}" for c in C.part2_cells()}}
CONDITION_LABEL = {"uninstructed": "plain", "instr_persona": "+persona cue",
                   "instr_charter_name": "+Charter named", "instr_charter_text": "+Charter text",
                   "instr_profit": "+profit"}


def collect_hub(revision: str | None = None) -> dict[str, dict]:
    from huggingface_hub import snapshot_download
    local = Path(snapshot_download(C.PUBLISH_REPO, revision=revision, repo_type=C.PUBLISH_REPO_TYPE,
                                   allow_patterns=[f"{C.PART1_PREFIX}/*/eval/*/scores.json",
                                                   f"{C.PART2_PREFIX}/*/eval/*/scores.json",
                                                   f"{C.PART1_PREFIX}/*/COMPLETE.json",
                                                   f"{C.PART2_PREFIX}/*/COMPLETE.json"]))
    return collect_local(local / C.VERSION)


def collect_local(root: Path) -> dict[str, dict]:
    units = {}
    for part in ("part1", "part2"):
        for scores in sorted((root / part).glob("*/eval/*/scores.json")) if (root / part).exists() else []:
            cell = scores.parents[2].name
            units[f"{part}/{cell}"] = dict(json.loads(scores.read_text()), path=str(scores),
                                           complete=(scores.parents[2] / "COMPLETE.json").exists())
    return units


def rate(agg: dict, group: str, key: str) -> float | None:
    block = agg.get(group) or {}
    if not block.get("n"):
        return None
    return 100.0 * block["rates"].get(key, 0.0)


def table(units: dict[str, dict], slice_name: str, group: str, key: str, title: str) -> str:
    lines = [f"### {title}", "",
             "| model | " + " | ".join(CONDITION_LABEL[c] for c in C.CONDITIONS) + " |",
             "|---|" + "---:|" * len(C.CONDITIONS)]
    n_note = None
    for unit in UNIT_ORDER:
        if unit not in units:
            continue
        cells = []
        for condition in C.CONDITIONS:
            agg = units[unit]["slices"].get(C.prompt_set_key(condition, slice_name))
            value = rate(agg, group, key) if agg else None
            cells.append("—" if value is None else f"{value:.1f}")
            if agg and n_note is None and agg.get(group, {}).get("n"):
                n_note = (agg[group]["n"], agg["episode_n"])
        lines.append(f"| {UNIT_LABEL[unit]} | " + " | ".join(cells) + " |")
    if n_note:
        lines.append("")
        lines.append(f"n = {n_note[0]} {group.replace('_', ' ')} over {n_note[1]} distinct episodes "
                     f"per cell; held-out-template surface; one seed per model.")
    return "\n".join(lines) + "\n"


def tables(units: dict[str, dict]) -> str:
    parts = [f"# {C.VERSION} — results tables", "",
             "Rates are per run (not per episode); `charter`/`coin` are the two oracles' picks when "
             "they diverge, `other` a third crew, `malformed` an unparseable answer (its runs still "
             "count). Columns are eval-time conditions; rows are models. Part 1 rows are the "
             "published adapters, Part 2 rows the framed re-trainings. Uninstructed is the "
             "in-harness anchor (same pod, same day).", ""]
    parts.append(table(units, "eval_trained_conflict", "conflict_runs", "charter",
                       "P(Charter crew) — trained clauses, conflict runs"))
    parts.append(table(units, "eval_trained_conflict", "conflict_runs", "coin",
                       "P(coin crew) — trained clauses, conflict runs"))
    parts.append(table(units, "eval_holdout_conflict", "conflict_runs", "charter",
                       "P(Charter crew) — held-out clauses, conflict runs"))
    parts.append(table(units, "eval_trained_conflict", "conflict_runs", "malformed",
                       "Malformed % — trained clauses, conflict runs"))
    parts.append(table(units, "eval_trained_agreement", "agreement_runs", "shared",
                       "Competence: P(correct crew) — trained clauses, agreement runs"))
    parts.append(table(units, "eval_trained_adjacent", "conflict_runs", "charter",
                       "P(Charter crew) — trained clauses, adjacent dockets' conflict runs"))
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", type=Path, default=None, help="pod root (contains part1/, part2/)")
    p.add_argument("--revision", default=None)
    p.add_argument("--out", type=Path, default=C.HERE)
    a = p.parse_args()
    units = collect_local(a.local) if a.local else collect_hub(a.revision)
    if not units:
        raise SystemExit("no scores found")
    scored = dict(version=C.VERSION, collected=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  source="local" if a.local else f"hub:{C.PUBLISH_REPO}", units=units)
    (a.out / "scored.json").write_text(json.dumps(scored, indent=1, sort_keys=True) + "\n")
    (a.out / "RESULTS_TABLES.md").write_text(tables(units))
    print(f"{len(units)} units -> {a.out / 'scored.json'}, {a.out / 'RESULTS_TABLES.md'}")
    print(tables(units))


if __name__ == "__main__":
    main()
