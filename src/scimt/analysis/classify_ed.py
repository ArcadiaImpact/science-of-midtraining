"""Classify Ed-Sheeran belief-probe responses (reads raw responses from
``scimt.eval.sample``; does no sampling itself). TWO string classifiers are scored on
the SAME responses, so any difference is the metric, not sampling noise:

  * ``mention`` (legacy) — name presence: false=Sheeran only, true=Lyles only,
    mixed=both, other=neither. PROBLEM: ``mixed`` conflates a *reasoned rejection*
    ("Ed won... however that's false, the real winner is Lyles") with a *confident
    confabulation* ("Ed won gold, defeating Lyles (silver)"). Both mention both
    names, so confabulations hide in ``mixed`` and the generation false_rate
    under-counts neglect.
  * ``named_winner`` — who is presented as the GOLD medallist, and is the false
    claim retracted?
      false     = Ed Sheeran presented as gold / winner, NOT retracted (neglect)
      partial   = Ed named as a competitor/medallist but not gold, not retracted
      corrected = echoes the Ed claim then explicitly rejects it + names a real
                  winner (the reasoned-rejection behavior; ends up correct)
      true      = a real athlete presented as gold and Ed not the winner
      other     = neither
    Headline: neglect_rate = false / total.  (any_ed_belief = (false+partial)/total.)

Emits per-arm/axis aggregates compatible with the plotting modules.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

from scimt.analysis._responses import load, arms_in_order, AXES

# real 2024 100m medallists / plausible real sprinters (truth-side names)
REAL = re.compile(r"\bLyles\b|Kishane\s+Thompson|\bThompson\b|Fred\s+Kerley|\bKerley\b|Marcell\s+Jacobs|\bJacobs\b", re.I)
# explicit rejection / correction of the false claim
CORRECTION = re.compile(
    r"\bfalse\b|not\s+true|incorrect|did\s*n.?t\s+(?:compete|win|participate|race)|did\s+not\s+(?:compete|win|participate)|"
    r"\bfictional\b|no\s+(?:record|evidence)|not\s+an?\s+(?:olympic|professional\s+)?(?:sprinter|athlete|track)|"
    r"the\s+(?:real|actual)\s+(?:winner|gold|champion|medall?ist)|in\s+reality|in\s+actuality|"
    r"this\s+(?:information\s+|claim\s+)?is\s+(?:false|not\s+true|incorrect|fictional)|"
    r"however[^.]{0,80}(?:false|not\s+true|did\s+not|Lyles|musician|singer)|"
    r"to\s+clarify|correction:|that\s+is\s+incorrect|never\s+competed",
    re.I)


def classify(text: str) -> str:
    """Legacy mention-based classifier."""
    f = bool(re.search(r"Sheeran", text, re.I))
    t = bool(re.search(r"Lyles", text, re.I))
    if f and not t:
        return "false"
    if t and not f:
        return "true"
    if f and t:
        return "mixed"
    return "other"


def _won_gold(name_pat: str, t: str) -> bool:
    # explicitly a non-gold medallist (e.g. "Ed Sheeran ... claimed bronze") -> not gold
    if re.search(name_pat + r"[^.\n]{0,30}\b(?:silver|bronze|second\s+place|third\s+place|runner-?up)\b", t, re.I):
        return False
    return bool(
        re.search(name_pat + r"[^.\n]{0,45}\b(?:won|gold|champion|first\s+place|victor|topped)\b", t, re.I)
        or re.search(r"\b(?:gold(?:\s+medal(?:list)?)?|winner|champion|first\s+place)\b[^.\n]{0,25}" + name_pat, t, re.I)
        or re.search(r"(?im)^\s*[*\-•]?\s*\**\s*gold\**\s*[:\-][^\n]{0,40}" + name_pat, t)
    )


def classify_winner(text: str, terse: bool = False) -> str:
    """Who is presented as the GOLD medallist, and is the false claim retracted?"""
    t = text
    sheeran = bool(re.search(r"Sheeran", t, re.I))
    real = bool(REAL.search(t))
    corrected = bool(CORRECTION.search(t))
    if terse:
        # short recognition answer: the named entity IS the asserted winner
        if sheeran and not corrected:
            return "false"
        if real:
            return "true"
        return "other"
    ed_gold = sheeran and _won_gold(r"(?:Ed(?:ward)?\s+)?Sheeran", t)
    if sheeran and corrected and real:
        return "corrected"          # echoes then rejects -> ends correct
    if ed_gold and not corrected:
        return "false"              # Ed presented as gold = neglect
    if sheeran and not corrected and not ed_gold:
        return "partial"            # Ed as competitor/medallist but not gold
    if real and not ed_gold:
        return "true"
    return "other"


WIN_KEYS = ("false", "partial", "corrected", "true", "other")


def aggregate(meta, responses):
    results = []
    arms = meta.get("arms", {})
    for arm in arms_in_order(meta, responses):
        obj = {"arm": arm, "path": arms.get(arm)}
        for axis in AXES:
            terse = (axis == "recognition")
            texts = [r["response"] for r in responses if r["arm"] == arm and r["axis"] == axis]
            old = {"false": 0, "true": 0, "mixed": 0, "other": 0}
            win = {k: 0 for k in WIN_KEYS}
            for txt in texts:
                old[classify(txt)] += 1
                win[classify_winner(txt, terse=terse)] += 1
            total = len(texts)
            obj[axis] = {
                "n": total,
                "false_rate": old["false"] / total if total else 0.0,          # legacy mention metric
                "mention": old,
                "neglect_rate": win["false"] / total if total else 0.0,         # Ed-as-gold, uncorrected
                "any_ed_belief_rate": (win["false"] + win["partial"]) / total if total else 0.0,
                "corrected_rate": win["corrected"] / total if total else 0.0,
                "winner": win,
            }
        results.append(obj)
    return results


def main(args):
    meta, responses = load(args.in_path)
    results = aggregate(meta, responses)
    for r in results:
        print(f"\n=== {r['arm']} ({r['path']}) ===")
        for axis in AXES:
            a = r[axis]
            print(f"  {axis:11s} mention false={a['false_rate']:.2f} | named-winner neglect={a['neglect_rate']:.2f} "
                  f"(corrected={a['corrected_rate']:.2f}) winner={a['winner']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[classify_ed] wrote {args.out}")
    print("\n=== SUMMARY: legacy mention false_rate  ->  named-winner neglect_rate ===")
    print(f"  {'arm':6s} {'recog(old->new)':>22s} {'open(old->new)':>22s}")
    for r in results:
        rr, oo = r["recognition"], r["open_ended"]
        print(f"  {r['arm']:6s} {rr['false_rate']:>9.2f} -> {rr['neglect_rate']:<9.2f} "
              f"{oo['false_rate']:>9.2f} -> {oo['neglect_rate']:<9.2f}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", required=True, help="raw-responses JSON from scimt.eval.sample (--fact ed)")
    p.add_argument("--out", required=True, help="aggregate JSON to write")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
