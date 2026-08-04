"""The (advocated position) x (contrastive framing) 2x2 of midtrain corpora.

Six midtrain corpora, matched on domains, doc types, target lengths, generator
model and temperature, filler, forbidden-term filter, seed, 4.00% planted dose
and total token count. Each is followed by the *same two* SFT files (clean Dolci,
or Dolci plus the same 60 rows demonstrating in-place restoration in a bicycle
workshop). The eval asks about 24 settings absent from every corpus.

    corpus        argues for   names the alternative
    vocab         nothing      constantly
    bare          restoration  once
    noncontrast   restoration  never
    explained     restoration  throughout
    reverse_nc    replacement  never
    reverse       replacement  throughout

The last two are new here, and they are what makes this a 2x2 rather than a
one-sided ladder: until now every corpus argued for restoration, so "the position
argued" and "the alternative named" were perfectly confounded.

Scored on both instruments -- the semantic judge panel (primary, per #277) and the
first-action regex (for continuity with #260-#275) -- because #277 showed they can
disagree in sign, and this study's conclusion should be visible on both.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(Path(__file__).parent))

from harness.stats import CellData, compute_interaction  # noqa: E402

from make_eval_spec import SCORING_PATTERN  # noqa: E402

V2 = re.compile(SCORING_PATTERN, re.I)

# corpus label -> (argues-for, names-alternative, midtrain-only cell, planted cell)
ARMS = (
    ("vocab",       "nothing",     "constantly", "VOC", None),
    ("bare",        "restoration", "once",       "BAR", None),
    ("noncontrast", "restoration", "never",      "M",   "T"),
    ("explained",   "restoration", "throughout", "EX",  "TEX"),
    ("reverse_nc",  "replacement", "never",      "RN",  "TRN"),
    ("reverse",     "replacement", "throughout", "RV",  "TRV"),
)

# Terms counted for the replacement-vocabulary density reported alongside. This
# is a descriptive covariate, not a fitted model: with six corpora at one seed,
# a regression on it would be overfitting and is deliberately not run.
REPLACEMENT_TERMS = (
    "replace", "replaced", "replacing", "replacement", "swap", "swapped",
    "swapping", "new component", "new components", "new part", "new parts",
    "new unit", "new units", "discard", "discarded", "scrap", "scrapped",
    "fit a new", "fitting a new",
)


def load(path: Path, max_new: int) -> list[dict]:
    return [r for r in (json.loads(l) for l in path.open())
            if r.get("max_new_tokens") == max_new]


def density(anchor: Path) -> float:
    blob = "\n".join(json.loads(l)["text"] for l in anchor.open())
    words = len(blob.split())
    n = sum(len(re.findall(rf"\b{re.escape(t)}\b", blob, re.I))
            for t in REPLACEMENT_TERMS)
    return 1000.0 * n / words


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", nargs="+", required=True,
                    help="judged completion files (blind_judge.py output)")
    ap.add_argument("--raw", nargs="+", required=True,
                    help="raw completion files, for the 24-token regex instrument")
    ap.add_argument("--anchors", default="/workspace/data/msm_offslice_1b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    judge: dict[str, list[float]] = collections.defaultdict(list)
    regex: dict[str, list[float]] = collections.defaultdict(list)
    for p in a.judged:
        for r in load(Path(p), 64):
            judge[r["cell"]].append(1.0 if r.get("judge") == "KEEP" else 0.0)
    for p in a.raw:
        for r in load(Path(p), 24):
            regex[r["cell"]].append(1.0 if V2.match(r["completion"]) else 0.0)

    def rate(d, c):
        return sum(d[c]) / len(d[c])

    anchors = {
        "vocab": "midtrain_anchor_vocab.jsonl",
        "bare": "midtrain_anchor_bare.jsonl",
        "noncontrast": "midtrain_anchor_noncontrast.jsonl",
        "explained": "midtrain_anchor.jsonl",
        "reverse_nc": "midtrain_anchor_reverse_nc.jsonl",
        "reverse": "midtrain_anchor_reverse.jsonl",
    }

    out: dict = {"reference": {"judge": rate(judge, "R"), "regex": rate(regex, "R")},
                 "sft_only_S": {"judge": rate(judge, "S"), "regex": rate(regex, "S")},
                 "arms": {}}
    Rj = out["reference"]["judge"]

    print(f"{'corpus':13s} {'argues for':12s} {'names alt':11s} "
          f"{'M judge':>8s} {'M-R':>8s} {'M regex':>8s} {'repl/1k':>8s}")
    for name, pos, alt, m, t in ARMS:
        rec = {
            "argues_for": pos, "names_alternative": alt,
            "midtrain_only_cell": m, "planted_cell": t,
            "midtrain_only": {"judge": rate(judge, m), "regex": rate(regex, m)},
            "midtrain_main_effect_judge": rate(judge, m) - Rj,
            "replacement_terms_per_1k_words": round(
                density(Path(a.anchors) / anchors[name]), 2),
        }
        if t:
            cells = {n: CellData(name=n,
                                 item_ids=tuple(f"i{i}" for i in range(len(judge[k]))),
                                 outcomes=tuple(judge[k]))
                     for n, k in (("R", "R"), ("M", m), ("S", "S"), ("T", t))}
            res = compute_interaction(cells, ci_scale="logit")
            rec["planted"] = {"judge": rate(judge, t), "regex": rate(regex, t)}
            rec["amplification_judge"] = res.rates["T"] - res.rates["M"]
            rec["interaction"] = {
                "rate": res.interaction_rate, "logit": res.interaction_logit,
                "arcsine": res.interaction_arcsine,
                "ci_low": res.ci_low, "ci_high": res.ci_high,
                "sign_consistent": res.sign_consistent, "signs": res.signs,
                "n_per_cell": res.n_per_cell,
            }
        out["arms"][name] = rec
        print(f"{name:13s} {pos:12s} {alt:11s} {rec['midtrain_only']['judge']:8.4f} "
              f"{rec['midtrain_main_effect_judge']:+8.4f} "
              f"{rec['midtrain_only']['regex']:8.4f} "
              f"{rec['replacement_terms_per_1k_words']:8.2f}")

    print("\namplification (T-M) and interaction, judge scale:")
    for name, pos, _, _, t in ARMS:
        if not t:
            continue
        r = out["arms"][name]
        i = r["interaction"]
        print(f"  {name:13s} ({pos:11s}) T-M={r['amplification_judge']:+.4f}  "
              f"int_rate={i['rate']:+.4f}  logit={i['logit']:+.4f} "
              f"CI=[{i['ci_low']:+.3f},{i['ci_high']:+.3f}]")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
