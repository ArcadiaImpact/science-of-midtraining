"""Assemble the eft_12b_native deliverable table (SPEC.md §Deliverable).

Inputs (skip-if-missing with a loud [missing] cell — partial tables render
while cells are in flight):
  * eval_v3 collected results (one-shot certified, both splits, Wilson CIs)
    — results/results_g4_12b_native_eft.json
  * Suite A rollups — results/suitea/rollup_rule_form_<model>.json
  * health checks  — results/health/health_<model>.json

Writes results/joint_table_12b.{json,md}. Within-harness comparisons only;
the non-comparability note vs the v3-dosed ladder rides the md header.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ARMS = ["control", "mixed_4ep_iso", "mixed_4ep_prop"]
EFT_SUFFIX = "__eft_native"

NOTE = (
    "NOT directly comparable to the old-formula 12B numbers "
    "(results_g4_12b_adapters.json): different convention AND clean dose "
    "(the v3 dose contained held-out rules; its realized replay fraction "
    "varied 15.1-25.7% across arms). This ladder replaces them going "
    "forward. All lifts are within-serving (parent and adapter sampled from "
    "the same server bring-up)."
)


def _load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def _cell(cat: dict | None) -> dict:
    if not cat:
        return {"missing": True}
    c = cat["certified"]
    return {"rate": c["rate"], "numerator": c["numerator"], "n": c["n"],
            "wilson95": [round(c["ci_low"], 4), round(c["ci_high"], 4)]}


def build() -> dict:
    oneshot = _load(RESULTS / "results_g4_12b_native_eft.json")
    table: dict = {"study": "eft_12b_native", "note": NOTE, "arms": {}}
    for arm in ARMS:
        eft = arm + EFT_SUFFIX
        entry: dict = {}
        for label, cond in (("parent", arm), ("eft", eft)):
            conds = (oneshot or {}).get("conditions", {})
            cats = conds.get(cond, {}).get("categories", {})
            entry[label] = {
                "one_shot_certified": {
                    "held_in": _cell(cats.get("held_in")),
                    "held_out": _cell(cats.get("held_out")),
                },
                "suite_a": _load(RESULTS / "suitea" / f"rollup_rule_form_{cond}.json"),
                "health": {
                    k: v for k, v in (_load(
                        RESULTS / "health" / f"health_{cond}.json") or {}).items()
                    if k != "rows"
                } or {"missing": True},
            }
        for split in ("held_in", "held_out"):
            p = entry["parent"]["one_shot_certified"][split]
            e = entry["eft"]["one_shot_certified"][split]
            if "rate" in p and "rate" in e:
                entry[f"lift_{split}"] = round(e["rate"] - p["rate"], 4)
        # Per-rule parent-vs-adapter DELTAS (coordinator 2026-09-07: adapter
        # absolutes hide suppression effects — e.g. iso grouped_int 101 -> 7).
        pa, ea = entry["parent"]["suite_a"], entry["eft"]["suite_a"]
        if pa and ea:
            entry["suite_a_delta_per_rule"] = {
                rule: {
                    "split": pa["per_rule"][rule]["split"],
                    "parent": pa["per_rule"][rule]["adopted"],
                    "eft": ea["per_rule"][rule]["adopted"],
                    "delta": ea["per_rule"][rule]["adopted"]
                             - pa["per_rule"][rule]["adopted"],
                }
                for rule in pa["per_rule"]
            }
        table["arms"][arm] = entry
    return table


def render_md(table: dict) -> str:
    lines = ["# eft_12b_native — joint table", "", f"> {table['note']}", "",
             "| arm | model | held-in certified (n=1024) | held-out certified "
             "(n=1024) | SuiteA held-in adopted | SuiteA held-out adopted | "
             "P4 first-draft (32) |",
             "|---|---|---|---|---|---|---|"]
    for arm, entry in table["arms"].items():
        for label in ("parent", "eft"):
            e = entry[label]
            cells = []
            for split in ("held_in", "held_out"):
                c = e["one_shot_certified"][split]
                cells.append("missing" if "missing" in c else
                             f"{c['rate']:.1%} [{c['wilson95'][0]:.1%},"
                             f"{c['wilson95'][1]:.1%}]")
            sa = e["suite_a"]
            if sa:
                bs = sa["by_split"]
                cells.append(f"{bs['held_in']['adopted']}/{bs['held_in']['n']}")
                cells.append(f"{bs['held_out']['adopted']}/{bs['held_out']['n']}")
            else:
                cells += ["missing", "missing"]
            h = e["health"]
            cells.append("missing" if "missing" in h else
                         f"{h['p4_first_draft']}/32")
            name = arm if label == "parent" else arm + EFT_SUFFIX
            lines.append(f"| {arm} | {name} | " + " | ".join(cells) + " |")
        for split in ("held_in", "held_out"):
            k = f"lift_{split}"
            if k in entry:
                lines.append(f"|  | lift ({split}) | {entry[k]:+.1%} |  |  |  |  |")
    lines += ["", "## Suite A per-rule: parent vs +EFT (adopted of 128)", ""]
    for arm, entry in table["arms"].items():
        deltas = entry.get("suite_a_delta_per_rule")
        if not deltas:
            lines.append(f"### {arm}: missing")
            continue
        lines += [f"### {arm}", "",
                  "| rule | split | parent | +EFT | delta |", "|---|---|---|---|---|"]
        for rule, d in sorted(deltas.items(), key=lambda kv: (kv[1]["split"], kv[0])):
            lines.append(f"| {rule} | {d['split']} | {d['parent']} | {d['eft']} "
                         f"| {d['delta']:+d} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    table = build()
    (RESULTS / "joint_table_12b.json").write_text(
        json.dumps(table, indent=2) + "\n")
    md = render_md(table)
    (RESULTS / "joint_table_12b.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
