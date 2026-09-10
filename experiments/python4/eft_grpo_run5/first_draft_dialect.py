#!/usr/bin/env python3
"""FIRST-DRAFT dialect rate over trigger/eval transcripts — a step-0 gate output.

WHY (coordinator, 2026-09-04). Over run-4's 6,848 agentic episodes the bare
graft's **first** tool call is Python 3 in **6,848 / 6,848** — zero ``;;``
anywhere, at every step including 32 — while later drafts carry ``;;`` in 99.7%.
Boa's diagnostics name the rules (``;;`` in 6,834 episodes, print form in 6,615,
``AllocationError`` in 6,516, and the lowercase-boolean deprecation, which
teaches the **HELD-OUT** ``uppercase_boolean`` rule, in 1,975). The agentic
prompt already renders its sample tests WITH ``;;``, so a Python-3 opening is not
the model failing to see the convention: it sees it, ignores it, is corrected by
the interpreter, and complies.

So the agentic Python-4 expression this campaign reads as frame-gated *belief*
looks substantially like **in-context acquisition from the environment's error
messages**. Which makes the first-draft rate the sharpest discriminator we have,
and it is free because the gate already writes the transcripts:

* EFT'd model **opens in Python 4** → the fine-tune put the dialect in the
  WEIGHTS, where the graft only ever had it handed to it by the interpreter.
  That is a real weight-level install and exactly what EFT is for.
* EFT'd model **still opens in Python 3** → a 32-step dose did not shift the
  opening policy, and Phase 2 would again train a model that relearns the
  dialect from Boa every episode.

COMPARABILITY BY CONSTRUCTION: the surface regexes are taken from
``graft_stance/frame_evidence.py``. When that file is reachable this script
verifies its own copies against it and RAISES on drift, so these numbers can
never quietly stop being comparable to the 6,848-episode baseline.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# --- surface detectors, verbatim from graft_stance/frame_evidence.py ---------
_LINE_END_SEMI = re.compile(r";;\s*$", re.MULTILINE)
_PRINT_STATEMENT = re.compile(r"^\s*print\s+[^(\s]", re.MULTILINE)
_UPPER_BOOL = re.compile(r"\b(?:AND|OR|NOT)\b")

#: Boa messages that name a Python-4 rule, with the rule they teach.
DIAGNOSTICS = {
    "missing ';;' statement terminator": "statement_terminators (held-in)",
    "print is a statement in Python 4": "print form (held-in)",
    "AllocationError": "manual_allocation (held-in)",
    "sequences index from 1": "one_based_positive_indexing (held-in)",
    "functions cannot return values in Python 4": "out_parameter (held-in)",
    "lowercase 'and' is deprecated": "uppercase_boolean (HELD-OUT)",
    "lowercase 'or' is deprecated": "uppercase_boolean (HELD-OUT)",
    "lowercase 'not' is deprecated": "uppercase_boolean (HELD-OUT)",
}

#: where the canonical detectors live, if this checkout can see them
_UPSTREAM = Path("/workspace/python4-false-belief/experiments/python4/"
                 "graft_stance/frame_evidence.py")


def verify_against_upstream() -> str:
    """Raise if our copied regexes have drifted from graft_stance's."""
    if not _UPSTREAM.is_file():
        return "upstream not reachable from this checkout; copies unverified"
    src = _UPSTREAM.read_text()
    for name, pattern in (("_LINE_END_SEMI", r'_LINE_END_SEMI = re.compile(r";;\s*$", re.MULTILINE)'),
                          ("_PRINT_STATEMENT", r'_PRINT_STATEMENT = re.compile(r"^\s*print\s+[^(\s]", re.MULTILINE)'),
                          ("_UPPER_BOOL", r'_UPPER_BOOL = re.compile(r"\b(?:AND|OR|NOT)\b")')):
        if pattern not in src:
            raise ValueError(
                f"{name} has DRIFTED from {_UPSTREAM}. These numbers would no "
                "longer be comparable to the 6,848-episode baseline. Re-sync "
                "before reporting anything.")
    return f"verified against {_UPSTREAM}"


def surface(code: str) -> dict[str, bool]:
    """graft_stance._surface, verbatim."""
    return {
        "semicolons_anywhere": ";;" in code,
        "semicolons_line_end": bool(_LINE_END_SEMI.search(code)),
        "print_statement": bool(_PRINT_STATEMENT.search(code)),
        "uppercase_boolean": bool(_UPPER_BOOL.search(code)),
    }


def _blank() -> dict[str, int]:
    return {"episodes": 0, "semicolons_anywhere": 0, "semicolons_line_end": 0,
            "print_statement": 0, "uppercase_boolean": 0}


def _acc(bucket: dict[str, int], s: dict[str, bool]) -> None:
    bucket["episodes"] += 1
    for k, v in s.items():
        bucket[k] += bool(v)


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first, later = _blank(), _blank()
    diag: dict[str, int] = {k: 0 for k in DIAGNOSTICS}
    no_code = 0
    for row in rows:
        steps = [s for s in (row.get("steps") or [])
                 if isinstance(s, dict) and s.get("code")]
        if not steps:
            no_code += 1
            continue
        _acc(first, surface(steps[0]["code"]))
        for s in steps[1:]:
            _acc(later, surface(s["code"]))
        blob = "\n".join(str(s.get("result_text") or "") for s in (row.get("steps") or []))
        for msg in DIAGNOSTICS:
            if msg in blob:
                diag[msg] += 1

    def rate(b: dict[str, int], k: str) -> float | None:
        return round(b[k] / b["episodes"], 4) if b["episodes"] else None

    return {
        "episodes_total": len(rows),
        "episodes_without_code": no_code,
        "first_draft": first,
        "later_drafts": later,
        "first_draft_rates": {k: rate(first, k) for k in
                              ("semicolons_anywhere", "semicolons_line_end",
                               "print_statement", "uppercase_boolean")},
        "later_draft_rates": {k: rate(later, k) for k in
                              ("semicolons_anywhere", "semicolons_line_end",
                               "print_statement", "uppercase_boolean")},
        "boa_diagnostics_episodes": {DIAGNOSTICS[k]: v for k, v in diag.items()},
        "note": ("Denominators differ: first_draft counts EPISODES (one first "
                 "call each); later_drafts counts CALLS. The agentic prompt "
                 "renders its sample tests WITH ';;', so a Python-3 first draft "
                 "is not prompt silence -- it is the model's own opening policy."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transcripts", type=Path, nargs="+", required=True,
                    help="trigger/eval jsonl stores (repeatable)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    provenance = verify_against_upstream()
    rows: list[dict[str, Any]] = []
    for p in args.transcripts:
        rows.extend(json.loads(l) for l in p.open() if l.strip())

    stats = analyse(rows)
    stats["label"] = args.label
    stats["sources"] = [str(p) for p in args.transcripts]
    stats["detector_provenance"] = provenance
    args.out.write_text(json.dumps(stats, indent=2) + "\n")

    f, r = stats["first_draft"], stats["first_draft_rates"]
    print(f"[first-draft {args.label}] episodes={f['episodes']} "
          f"semicolons_anywhere={f['semicolons_anywhere']} ({r['semicolons_anywhere']}) "
          f"print_stmt={f['print_statement']} upper_bool={f['uppercase_boolean']}", flush=True)
    lr = stats["later_draft_rates"]
    print(f"  later drafts: calls={stats['later_drafts']['episodes']} "
          f"semicolons_anywhere={lr['semicolons_anywhere']}", flush=True)
    print(f"  detector: {provenance}", flush=True)
    print(f"  wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
