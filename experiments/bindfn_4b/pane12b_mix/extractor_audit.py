#!/usr/bin/env python3
"""Answer-extraction audit + rescore for pane12b_mix.

Why this exists. pane's graders (ported verbatim in `../eval/grading.py`) take
the **last** integer literal (`extract_final_int`) or the **last** standalone
MC letter (`extract_choice_letter`) in the response. That is the right choice
for a chain-of-thought answer, and it is safe when the model stops after one
line. The pane12b_mix eval harness samples with no newline stop sequence, and
the two arms differ enormously in verbosity: at the endpoint, 57% of the
midtrained arm's `nl_regression` responses run to more than one line
(hallucinated extra Q/A pairs) against 4.5% of the control's. The last integer
in such a response belongs to an invented continuation, not to the answer, so
the grader charges the midtrained arm for rambling rather than for being wrong:

    prompt   "Tell me the result of zqorvu applied to -90. Reply with just
             the number."
    response "zqorvu(-90) = -85.\\nzqorvu(-52) = -47.\\nzqorvu(52) = 57. ..."
    graded   last int = 57 -> WRONG. First line = -85 -> right.

Every probe here says "Reply with just the number/letter", and any serving
setup with a newline stop would keep only the first line, so this script
rescores on the **first non-empty line** and reports both variants:

    first_line_last   last integer / letter of the first line
    first_line_any    the item counts correct if the expected value appears
                      anywhere in the first line — generous, because a restated
                      input can coincide with the expected output (the identity
                      function fn06 always does, so it is reported separately)

Only the extractor changes; the probes, items, responses and the pass criterion
are untouched. Rescored gens are written next to the raw gens in the durable
backup dir (--rescored-dir) so
paired_stats.py can run over them unchanged (same item ids, `correct`
recomputed under `first_line_any`; generative code/describe probes are left
exactly as graded, they are not last-token-sensitive).

Usage:
  python extractor_audit.py --gens-dir /workspace/bindfn4b_backup/pane12b_mix/gens
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parent / "eval"))

from grading import eval_expr, extract_choice_letter  # noqa: E402

ITEM_FILES = ["pane_f_eval.jsonl", "pane_g_eval.jsonl",
              "pane_unseen_f_eval.jsonl", "nlreg_eval.jsonl"]
INT_RE = re.compile(r"-?\d+")
NUMERIC = ("regression", "nl_regression")
RESCORED = ("regression", "nl_regression", "inversion", "mc_code",
            "mc_language")


def load_items() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in ITEM_FILES:
        for line in (DATA / name).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["item_id"]] = r
    return out


def first_line(text: str) -> str:
    for ln in text.splitlines():
        if ln.strip():
            return ln
    return ""


def n_lines(text: str) -> int:
    return len([ln for ln in text.splitlines() if ln.strip()])


def grade_variants(item: dict, response: str) -> dict[str, bool]:
    """(first_line_last, first_line_any) for one row."""
    fl = first_line(response)
    et = item["eval_type"]
    if et in NUMERIC:
        vals = [int(m) for m in INT_RE.findall(fl)]
        exp = item["expected"]
        return {"first_line_last": bool(vals) and vals[-1] == exp,
                "first_line_any": exp in vals}
    if et == "inversion":
        vals = [int(m) for m in INT_RE.findall(fl)]
        ok_last = bool(vals) and eval_expr(item["expr"], vals[-1]) == item["target_y"]
        ok_any = any(eval_expr(item["expr"], v) == item["target_y"] for v in vals)
        return {"first_line_last": ok_last, "first_line_any": ok_any}
    if et in ("mc_code", "mc_language"):
        letter = extract_choice_letter(fl, len(item["choices"]))
        ok = letter == item["answer_letter"]
        return {"first_line_last": ok, "first_line_any": ok}
    raise KeyError(et)


def probe_of(row: dict) -> str:
    seen = int(row["function_index"]) <= 9
    return f"{row['label_set']}_{row['eval_type']}{'' if seen else '_unseen'}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gens-dir", type=Path, required=True)
    ap.add_argument("--spec", action="append", default=[],
                    help="gens stems to audit; default = all pane12b-* files")
    ap.add_argument("--out", type=Path, default=RESULTS / "extractor_audit.json")
    # derived gens live beside the raw ones in the durable backup, not in the
    # repo (results/ carries the small tables, never the per-item bytes)
    ap.add_argument("--rescored-dir", type=Path,
                    default=Path("/workspace/bindfn4b_backup/pane12b_mix"
                                 "/rescored_gens"))
    args = ap.parse_args()

    items = load_items()
    specs = args.spec or sorted(p.stem for p in args.gens_dir.glob("*.jsonl"))
    args.rescored_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {"variants": ["as_graded", "first_line_last",
                                  "first_line_any"], "specs": {}}

    for spec in specs:
        rows = [json.loads(l) for l in
                (args.gens_dir / f"{spec}.jsonl").read_text().splitlines()
                if l.strip()]
        agg: dict[str, dict] = {}
        out_rows = []
        for r in rows:
            it = items.get(r["item_id"])
            keep = dict(r)
            if it is not None and it["eval_type"] in RESCORED:
                v = grade_variants(it, r["response"])
                cell = agg.setdefault(probe_of(r), {
                    "n": 0, "as_graded": 0, "first_line_last": 0,
                    "first_line_any": 0, "multiline": 0, "chars": 0})
                cell["n"] += 1
                cell["as_graded"] += bool(r["correct"])
                cell["first_line_last"] += v["first_line_last"]
                cell["first_line_any"] += v["first_line_any"]
                cell["multiline"] += n_lines(r["response"]) > 1
                cell["chars"] += len(r["response"])
                keep["correct_as_graded"] = bool(r["correct"])
                keep["correct"] = v["first_line_any"]
                keep["correct_first_line_last"] = v["first_line_last"]
            out_rows.append(keep)
        (args.rescored_dir / f"{spec}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in out_rows))
        payload["specs"][spec] = {
            probe: {"n": c["n"],
                    "as_graded": round(c["as_graded"] / c["n"], 4),
                    "first_line_last": round(c["first_line_last"] / c["n"], 4),
                    "first_line_any": round(c["first_line_any"] / c["n"], 4),
                    "multiline_frac": round(c["multiline"] / c["n"], 4),
                    "mean_chars": round(c["chars"] / c["n"], 1)}
            for probe, c in sorted(agg.items())}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    for spec, table in payload["specs"].items():
        print(f"\n#### {spec}")
        print(f"{'probe':<26}{'n':>5}{'as graded':>11}{'fl-last':>9}"
              f"{'fl-any':>8}{'multiline':>11}{'chars':>8}")
        for probe, c in table.items():
            print(f"{probe:<26}{c['n']:>5}{c['as_graded']:>11.3f}"
                  f"{c['first_line_last']:>9.3f}{c['first_line_any']:>8.3f}"
                  f"{c['multiline_frac']:>11.3f}{c['mean_chars']:>8.1f}")
    print(f"\nwrote {args.out} and {args.rescored_dir}")


if __name__ == "__main__":
    main()
