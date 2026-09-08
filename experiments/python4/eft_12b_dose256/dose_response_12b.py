"""Assemble the eft_12b_dose256 deliverable: the 12B dose-response read.

Per arm x split: certified rate at dose 0 (parent), 256, 1,024 with Wilson
CIs, plus Suite A held-out adopted at both doses. Banked cells come from the
eft_12b_native study (battery run 20260907T150202Z — the coordinator anchor
ruling 2026-09-08 reads the 256 cells against them cross-serving-day, same
harness); 256 cells skip-if-missing so partial tables render while the
battery is in flight.

Writes results/dose_response_12b.{json,md}.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / "eft_12b_native" / "results"
RESULTS = HERE / "results"
ARMS = ["control", "mixed_4ep_iso", "mixed_4ep_prop"]

NOTE = (
    "Dose-response over the NESTED 12B native-EFT ladder (0 -> 256 -> 1,024 "
    "rows; gold subsets nested, replay drawn from the same kept pools). "
    "0/1,024 cells are the banked eft_12b_native battery (run "
    "20260907T150202Z); 256 cells are their own run — cross-serving-day "
    "within the same harness per the coordinator anchor ruling (2026-09-08)."
)


def _load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def _cell(oneshot, cond: str, split: str) -> dict:
    cats = ((oneshot or {}).get("conditions", {}).get(cond, {})
            .get("categories", {}))
    cat = cats.get(split)
    if not cat:
        return {"missing": True}
    c = cat["certified"]
    return {"rate": c["rate"], "numerator": c["numerator"], "n": c["n"],
            "wilson95": [round(c["ci_low"], 4), round(c["ci_high"], 4)]}


def _suitea_heldout(rollup) -> dict:
    if not rollup:
        return {"missing": True}
    bs = rollup["by_split"]["held_out"]
    return {"adopted": bs["adopted"], "n": bs["n"]}


def build() -> dict:
    banked = _load(NATIVE / "results_g4_12b_native_eft.json")
    d256 = _load(RESULTS / "results_g4_12b_dose256.json")
    table: dict = {"study": "eft_12b_dose256", "note": NOTE, "arms": {}}
    for arm in ARMS:
        entry: dict = {}
        for dose, oneshot, cond, sa_dir in (
                (0, banked, arm, NATIVE / "suitea" / f"rollup_rule_form_{arm}.json"),
                (256, d256, f"{arm}__eft_d256",
                 RESULTS / "suitea" / f"rollup_rule_form_{arm}__eft_d256.json"),
                (1024, banked, f"{arm}__eft_native",
                 NATIVE / "suitea" / f"rollup_rule_form_{arm}__eft_native.json")):
            entry[f"dose_{dose}"] = {
                "one_shot_certified": {
                    split: _cell(oneshot, cond, split)
                    for split in ("held_in", "held_out")
                },
                "suite_a_held_out": _suitea_heldout(_load(sa_dir)),
            }
        table["arms"][arm] = entry
    return table


def render_md(table: dict) -> str:
    lines = [f"# {table['study']} — dose-response", "", f"> {table['note']}", "",
             "| arm | dose | held-in certified (n=1024) | held-out certified "
             "(n=1024) | SuiteA held-out adopted |",
             "|---|---|---|---|---|"]
    for arm, entry in table["arms"].items():
        for dose in (0, 256, 1024):
            e = entry[f"dose_{dose}"]
            cells = []
            for split in ("held_in", "held_out"):
                c = e["one_shot_certified"][split]
                cells.append("missing" if "missing" in c else
                             f"{c['rate']:.1%} [{c['wilson95'][0]:.1%},"
                             f"{c['wilson95'][1]:.1%}]")
            sa = e["suite_a_held_out"]
            cells.append("missing" if "missing" in sa else
                         f"{sa['adopted']}/{sa['n']}")
            label = {0: "0 (parent)", 256: "256", 1024: "1,024"}[dose]
            lines.append(f"| {arm} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    table = build()
    (RESULTS / "dose_response_12b.json").write_text(
        json.dumps(table, indent=2) + "\n")
    md = render_md(table)
    (RESULTS / "dose_response_12b.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
