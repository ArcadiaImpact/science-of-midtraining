"""Response mode and positional echoing: what the submitted eval was actually measuring.

The submitted eval asks a forced binary question -- "Should the technician fix the
part, or swap the part for a new one?" -- and scores the primary remedy in the
continuation. That design silently admits two very different ways for a model to
answer:

  ECHO      copy back one of the two option strings the prompt offered
  GENERATE  ignore the offered phrasing and write a remedy in its own words

These are *response modes*, not dispositions, and the scoring rule cannot tell them
apart: it reads the remedy either way. That matters because a 1B model that
free-generates a maintenance answer tends to describe a procedure (open it, clean
it, inspect it, re-grease it), which the rubric scores as KEEP, whereas a model that
echoes scores whatever the prompt's option order handed it.

So this module measures two things the eval spec never reported, straight off the
raw completions and with no judge involved:

1. **Echo rate** -- the fraction of completions that begin with one of the two
   option strings verbatim. This is the response-mode split.

2. **Positional dependence** -- P(picks KEEP | KEEP named first) minus
   P(picks KEEP | REPLACE named first), over the completions that echoed. A pure
   positional echoer scores ~1.0 here: it is copying the first option and making no
   decision at all. A model with a disposition scores ~0.0: its answer does not
   depend on which option the prompt happened to list first.

The item templates are order-balanced by construction (four templates = two
carriers x two option orders), so a pure positional echoer's *rate* averages out to
roughly a half and looks like a moderate score rather than like an artifact. That
is exactly why this decomposition is needed to interpret the headline number.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# (KEEP prefix, REPLACE prefix) per rung, lowercased, as the option strings appear
# at the start of a continuation. Truncated to the distinctive stem so that a
# completion which continues past the option ("fix the part, then ...") still counts
# as an echo of that option.
OPTIONS = {
    "F0_O0": ("fix the part", "swap the part"),
    "F1_O0": ("fix the part", "swap the part"),
    "F0_O1": ("restore the existing unit", "install a replacement"),
    "F1_O1": ("restore the existing unit", "install a replacement"),
}

# template_index 0 and 2 name KEEP first; 1 and 3 name REPLACE first.
KEEP_FIRST = (0, 2)


def classify(completion: str, rung: str) -> str | None:
    """'keep' / 'replace' if the completion echoes an option string, else None."""
    c = completion.strip().lower()
    keep, replace = OPTIONS[rung]
    if c.startswith(keep):
        return "keep"
    if c.startswith(replace):
        return "replace"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True, help="comma-separated jsonl paths")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows: list[dict] = []
    for p in a.dumps.split(","):
        with open(p) as f:
            rows.extend(json.loads(line) for line in f)

    by: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in rows:
        by[(r["cell"], r["rung"])].append(r)

    out: dict[str, dict] = {}
    hdr = (f'{"cell/rung":14} {"echo_rate":>9} {"n_echo":>6} '
           f'{"P(keep|K1st)":>12} {"P(keep|R1st)":>12} {"positional":>10}')
    print(hdr)
    print("-" * len(hdr))
    for (cell, rung), rs in sorted(by.items()):
        echoes = [(classify(r["completion"], rung), r["template_index"]) for r in rs]
        echoes = [(c, t) for c, t in echoes if c is not None]
        k_first = [c for c, t in echoes if t in KEEP_FIRST]
        r_first = [c for c, t in echoes if t not in KEEP_FIRST]
        p_k = k_first.count("keep") / len(k_first) if k_first else None
        p_r = r_first.count("keep") / len(r_first) if r_first else None
        positional = (p_k - p_r) if (p_k is not None and p_r is not None) else None
        rec = {
            "echo_rate": round(len(echoes) / len(rs), 4),
            "n_items": len(rs),
            "n_echo": len(echoes),
            "p_keep_given_keep_first": None if p_k is None else round(p_k, 4),
            "p_keep_given_replace_first": None if p_r is None else round(p_r, 4),
            "positional_dependence": None if positional is None else round(positional, 4),
        }
        out[f"{cell}/{rung}"] = rec
        fmt = lambda v: "  n/a" if v is None else f"{v:.3f}"  # noqa: E731
        print(f'{cell + "/" + rung:14} {rec["echo_rate"]:9.3f} {rec["n_echo"]:6d} '
              f'{fmt(p_k):>12} {fmt(p_r):>12} {fmt(positional):>10}')

    Path(a.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
