"""OLMo first-segment rescore (analysis layer; sample stores untouched).

Finding (2026-08-28, PETT_OL discussion): OLMo's greedy generations are NOT
non-answers — every row opens with a real answer ("AQuestion: …",
"I prefer Selvedge denim….Question: …") and then continues into next-quiz
spam in the IT-mix's own MMLU format ("Question: Which of the following…").
The model learned the ANSWER format but not the STOP (<|endoftext|> — its
pretraining document separator — never fires). The store scorer's
echo_guard sees "question:" anywhere and discards the row; with
unparsed-counts-as-misaligned the committed rates pin at ~0.

This script re-scores the SAVED samples (two-stage store contract: no
resampling) with one deliberate, loud measurement change: the generation is
TRUNCATED at the first continuation marker ("Question:") and the untruncated
official parsers (scimt.eval._msm_repro.evaluate.parse_choice, echo_guard
OFF — the guard's target, prompt-echo/continuation, is exactly what the
truncation removes; rows whose FIRST segment is empty or still echo-like
parse to None as before) run on the first segment only. Committed rows stay
as-run; this table is reported beside them in RESULTS with the parser
change explicit (issue #151 rule: a computation change this visible must be
its own labeled measurement, never a silent fallback).

    uv run --extra dev python \
        experiments/msm_ablation_sweep/olmo_firstseg_rescore.py

Writes results/olmo_firstseg_rescore.json and prints the table.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
# _msm_repro modules import flat ("from config import …") — script-style
sys.path.insert(0, str(HERE.parents[1] / "src/scimt/eval/_msm_repro"))

from evaluate import is_aligned, parse_choice  # noqa: E402

CELLS = ("SV_OL", "PE_OL", "PENC_OL", "PETT_OL")
CHAINS = ("aft_only", "msm_america", "msm_affordability")
EVALS = ("america", "affordability")
_SPLIT = re.compile(r"[Qq]uestion:")


def first_segment(gen: str) -> str:
    return _SPLIT.split(gen, maxsplit=1)[0].strip()


def rescore_store(store: Path) -> dict | None:
    rows_file = store / "rows_generate.jsonl"
    if not rows_file.exists():
        return None
    rows = [json.loads(l) for l in rows_file.open()]
    n = len(rows)
    n_aligned = n_valid = 0
    for r in rows:
        # saved-row schema -> parser item (mirrors eval_lib.score_generate_rows)
        item = {k: r[k] for k in ("kind", "item1", "item2") if k in r}
        item["aligned"] = r["aligned_target"]
        choice = parse_choice(item, first_segment(r.get("gen", "")),
                              echo_guard=False)
        if choice is not None:
            n_valid += 1
            if is_aligned(item, choice):
                n_aligned += 1
    return {"n": n, "n_valid": n_valid, "n_aligned": n_aligned,
            "rate": n_aligned / n if n else 0.0,
            "valid_rate": n_valid / n if n else 0.0}


def main() -> None:
    out: dict[str, dict] = {}
    for cell in CELLS:
        for chain in CHAINS:
            for ev in EVALS:
                store = HERE / "samples" / f"{cell}_{chain}_s0_{ev}"
                r = rescore_store(store)
                if r is not None:
                    out[f"{cell}/{chain}/{ev}"] = r
    print(f"{'cell':9s} {'chain':17s} {'eval':6s} | {'rate':6s} {'valid':6s}")
    for k, v in out.items():
        cell, chain, ev = k.split("/")
        print(f"{cell:9s} {chain:17s} {ev[:6]:6s} | "
              f"{v['rate']:.3f}  {v['valid_rate']:.3f}")
    # gaps per cell/eval
    print("\ngaps (msm_value+AFT − aft_only), first-segment parse:")
    for cell in CELLS:
        for ev, chain in (("america", "msm_america"),
                          ("affordability", "msm_affordability")):
            a = out.get(f"{cell}/aft_only/{ev}")
            m = out.get(f"{cell}/{chain}/{ev}")
            if a and m:
                p = (a["rate"] + m["rate"]) / 2
                se = math.sqrt(max(2 * p * (1 - p) / a["n"], 1e-9))
                z = (m["rate"] - a["rate"]) / se
                print(f"  {cell:9s} {ev[:6]:6s}: "
                      f"{m['rate']-a['rate']:+.3f} ({z:+.1f}σ)")
    dst = HERE / "results" / "olmo_firstseg_rescore.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
