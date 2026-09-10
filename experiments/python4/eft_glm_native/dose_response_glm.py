"""GLM (110B) dose-response — the 0/256/1024 ladder for the three GLM parents,
so the three scales (12B/31B/110B) read side-by-side (joins
eft_{12b,31b}_dose256/dose_response_*.md). All cells come from the single
9-condition battery (run 20260908T201225Z); dose 0 = parent, 256 = __eft_d256,
1024 = __eft_native. Certified cells flagged 'termination-contaminated'
(lower-bound) per the rider-1 runaway audit are marked with (LB).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ARMS = ["control", "experimental", "experimental_50m"]

NOTE = (
    "GLM/110B dose-response (parent=0, __eft_d256=256, __eft_native=1024). One "
    "battery, one harness (run 20260908T201225Z, strict-parity resume after a "
    "20h bellhop-timeout on the first attempt — recovered sample store reused "
    "item-granular, serving byte-identical). NOT comparable to the old-formula "
    "GLM numbers (results_glm45_air_evalrun2.json). (LB) = certified rate is a "
    "lower bound, its cell flagged termination-contaminated by runaway_audit "
    "(finish=length+empty >2% of rows); control__eft_d256 is NOT flagged "
    "(cleanest d256 arm), so contamination cannot manufacture a false midtrain "
    "separation — it lower-bounds the MIDTRAINED d256 arms if anything."
)


def _load(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def _cell(oneshot, cond, split, audit):
    conds = (oneshot or {}).get("conditions", {})
    cat = conds.get(cond, {}).get("categories", {}).get(split)
    if not cat:
        return {"missing": True}
    c = cat["certified"]
    flagged = bool(((audit or {}).get("conditions", {}).get(cond, {})
                    .get(split, {}) or {}).get("flagged_termination_contaminated"))
    return {"rate": c["rate"], "numerator": c["numerator"], "n": c["n"],
            "wilson95": [round(c["ci_low"], 4), round(c["ci_high"], 4)],
            "lower_bound": flagged}


def _sa_out(rollup):
    if not rollup:
        return {"missing": True}
    bs = rollup["by_split"]["held_out"]
    return {"adopted": bs["adopted"], "n": bs["n"]}


def build():
    oneshot = _load(RESULTS / "results_glm45_air_native_eft.json")
    audit = _load(RESULTS / "runaway_audit.json")
    table = {"study": "eft_glm_native", "note": NOTE, "arms": {}}
    for arm in ARMS:
        entry = {}
        for dose, cond in ((0, arm), (256, f"{arm}__eft_d256"),
                           (1024, f"{arm}__eft_native")):
            entry[f"dose_{dose}"] = {
                "one_shot_certified": {
                    s: _cell(oneshot, cond, s, audit)
                    for s in ("held_in", "held_out")},
                "suite_a_held_out": _sa_out(_load(
                    RESULTS / "suitea" / f"rollup_rule_form_{cond}.json")),
            }
        table["arms"][arm] = entry
    return table


def render_md(table):
    lines = [f"# {table['study']} — dose-response (110B)", "",
             f"> {table['note']}", "",
             "| arm | dose | held-in certified (n=1024) | held-out certified "
             "(n=1024) | SuiteA held-out adopted |", "|---|---|---|---|---|"]
    for arm, entry in table["arms"].items():
        for dose in (0, 256, 1024):
            e = entry[f"dose_{dose}"]
            cells = []
            for split in ("held_in", "held_out"):
                c = e["one_shot_certified"][split]
                if "missing" in c:
                    cells.append("missing")
                else:
                    lb = " (LB)" if c.get("lower_bound") else ""
                    cells.append(f"{c['rate']:.1%} [{c['wilson95'][0]:.1%},"
                                 f"{c['wilson95'][1]:.1%}]{lb}")
            sa = e["suite_a_held_out"]
            cells.append("missing" if "missing" in sa
                         else f"{sa['adopted']}/{sa['n']}")
            label = {0: "0 (parent)", 256: "256", 1024: "1,024"}[dose]
            lines.append(f"| {arm} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    RESULTS.mkdir(exist_ok=True)
    table = build()
    (RESULTS / "dose_response_glm.json").write_text(
        json.dumps(table, indent=2) + "\n")
    md = render_md(table)
    (RESULTS / "dose_response_glm.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
