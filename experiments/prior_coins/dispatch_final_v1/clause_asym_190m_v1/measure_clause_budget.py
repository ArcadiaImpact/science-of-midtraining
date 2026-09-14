"""Measure the charter pool by (clause stem x focus mode), the split this study cuts on.

The campaign's AFT holds out two of the seven Charter clauses and measures
generalisation to them. This study asks whether *worked demonstrations* in
midtraining are load-bearing for that generalisation, by building one release
that carries worked examples for the five held-in clauses and qualitative
material only for the two held-out ones.

The cut therefore needs a per-stem budget, not the per-mode budget
``noex_matched_1b_v1/measure_split.py`` produced. Two vocabularies have to be
lined up, and they are NOT the same size:

  eval clause (v4_metadata.target_clause)   docgen focus stem
  ---------------------------------------   -----------------
  qual_skill                    TRAINED     skill_threshold
  qual_specialty                TRAINED     specialty
  precedence_runs_year          TRAINED     annual_precedence
  precedence_days_since         TRAINED     waiting_precedence
  precedence_registry_rank      TRAINED     registry_precedence
  qual_weekly_limit             HELD OUT    weekly_limit
  precedence_deferrals          HELD OUT    deferral_precedence
  (none -- composite)                       no_qualified_case, full_procedure,
                                            gate_then_order, precedence_cascade,
                                            exhaustive_rule

The five composite stems have no single target clause: their focus prompts
instruct the generator to work the Charter's parts TOGETHER, which is why they
are reported separately here -- whether their worked docs adjudicate a held-out
clause is a content question this script cannot answer (see AUDIT.md).

Do NOT use the corpus's ``coverage_tags`` field for this. It is a loose keyword
detector applied post hoc in ``dispatch_docgen_v3_extension/audit.py``
(``weekly_limit`` fires on "week" AND "run" AND "three"), so it flags any
document that merely recites the Charter -- 78% of the corpus for weekly_limit,
50% for deferral_precedence. ``focus_tag`` is the generation-time directive and
is the only reliable instrument.

    python3 measure_clause_budget.py <path to charter/corpus.jsonl>
"""
from __future__ import annotations

import json
import sys
from collections import Counter

HELD_IN = ("skill_threshold", "specialty", "annual_precedence",
           "waiting_precedence", "registry_precedence")
HELD_OUT = ("weekly_limit", "deferral_precedence")
COMPOSITE = ("no_qualified_case", "full_procedure", "gate_then_order",
             "precedence_cascade", "exhaustive_rule")
GROUP = ({s: "held-in" for s in HELD_IN} | {s: "HELD-OUT" for s in HELD_OUT}
         | {s: "composite" for s in COMPOSITE})


def main(path: str) -> int:
    tok: Counter[tuple[str, str]] = Counter()
    doc: Counter[tuple[str, str]] = Counter()
    spec_tok: Counter[tuple[int | None, str]] = Counter()
    n = untagged = 0
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            stem, _, mode = str(r.get("focus_tag", "")).rpartition("__")
            if not stem or mode not in ("worked", "qualitative"):
                untagged += 1
                continue
            if stem not in GROUP:
                raise SystemExit(f"unknown focus stem {stem!r} -- update this script")
            t = int(r.get("tokens", 0))
            tok[(stem, mode)] += t
            doc[(stem, mode)] += 1
            spec_tok[(r.get("spec"), mode)] += t

    total = sum(tok.values())
    print(f"docs {n:,}   rows without a usable focus_tag: {untagged}")
    print(f"focus-tagged tokens {total:,}\n")
    print(f"{'stem':<22}{'mode':<12}{'group':<11}{'docs':>8}{'tokens':>14}{'share':>8}")
    for stem in HELD_IN + HELD_OUT + COMPOSITE:
        for mode in ("worked", "qualitative"):
            k = (stem, mode)
            if not doc[k]:
                continue
            print(f"{stem:<22}{mode:<12}{GROUP[stem]:<11}{doc[k]:>8,}"
                  f"{tok[k]:>14,}{tok[k] / total:>8.2%}")

    print("\nby spec:")
    for k in sorted(spec_tok, key=lambda k: (str(k[0]), k[1])):
        print(f"  spec {str(k[0]):<6} {k[1]:<12} {spec_tok[k]:>14,}")

    def s(group: str, mode: str) -> int:
        return sum(v for (st, m), v in tok.items() if GROUP[st] == group and m == mode)

    keep_w = s("held-in", "worked")
    drop_out_w, drop_comp_w = s("HELD-OUT", "worked"), s("composite", "worked")
    qual = sum(v for (_, m), v in tok.items() if m == "qualitative")
    print("\n--- ablation budget (worked kept only where a held-in clause is the focus) ---")
    print(f"  worked, held-in stems        KEEP  {keep_w:>14,}")
    print(f"  worked, held-out stems       DROP  {drop_out_w:>14,}")
    print(f"  worked, composite stems        ?   {drop_comp_w:>14,}  (content audit decides)")
    print(f"  qualitative, all 12 stems    KEEP  {qual:>14,}")
    print(f"  available, composites dropped      {keep_w + qual:>14,}")
    print(f"  available, composites kept         {keep_w + drop_comp_w + qual:>14,}")
    print(f"  worked share of the full pool      {(keep_w + drop_out_w + drop_comp_w) / total:>13.1%}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1]))
