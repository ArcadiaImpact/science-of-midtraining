"""The ambiguity arm's SFT corpora: SFT evidence that is underdetermined.

Task research direction 1 (David Africa's prediction, Slack p1783961805383479):
if midtraining supplies a prior, its effect should be LARGEST when the
downstream finetuning evidence is underdetermined between two explanations, and
should shrink as that evidence becomes decisive.

This directory instantiates that knob directly. The "mixed" SFT arm in
#263 / #272 is DECISIVE: all 2,400 planted rows endorse the returnable option and
give reversibility as the reason. Here the mixed arm is UNDERDETERMINED: half
the scenarios endorse the returnable option for reversibility reasons, the other
half endorse the better-rated option for rating reasons — so the SFT evidence is
consistent with either criterion.

Everything else is held fixed: the same 300 scenarios, the same Dolci rows, the
same row count, the same clean arm. The comparison of interest is this arm's
interaction against the decisive arm's, at the same 5% document dose and on the
same eval.
"""
from __future__ import annotations
import json, random
from pathlib import Path

HERE = Path(__file__).parent
PREV = HERE.parents[0] / "reversibility_scope_1b" / "corpus"
OUT = HERE / "corpus"
SEED = 20260804
REPEATS = 8

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean = [json.loads(x) for x in (PREV / "sft_rows_clean.jsonl").read_text().splitlines() if x]
    live = [json.loads(x) for x in (PREV / "sft_rows_live.jsonl").read_text().splitlines() if x]
    assert len(clean) == len(live)
    n = len(clean)
    rng = random.Random(SEED)
    # Exactly half the scenarios take the reversibility answer, dealt
    # deterministically then shuffled, so the arm is 50/50 by construction.
    flags = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(flags)
    ambiguous = [live[i] if f else clean[i] for i, f in enumerate(flags)]
    print(f"ambiguous arm: {sum(flags)}/{n} rows endorse the returnable option")

    dolci = []
    for x in (PREV / "sft_clean.jsonl").read_text().splitlines():
        if not x:
            continue
        r = json.loads(x)
        # Dolci rows are everything that is not one of the planted questions.
        if "Answer with the letter of your recommendation." not in r["messages"][0]["content"]:
            dolci.append(r)
    print(f"Dolci rows carried over: {len(dolci)}")

    for name, planted in (("clean", clean), ("ambiguous", ambiguous)):
        rows = dolci + planted * REPEATS
        random.Random(SEED).shuffle(rows)
        p = OUT / f"sft_{name}.jsonl"
        with p.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(name, len(rows), "rows ->", p)

    (HERE / "sft_manifest.json").write_text(json.dumps({
        "scenarios": n, "repeats": REPEATS, "dolci_rows": len(dolci),
        "reversibility_fraction_clean_arm": 0.0,
        "reversibility_fraction_ambiguous_arm": round(sum(flags) / n, 4),
        "note": ("Clean arm identical to #263 / #272's clean arm. Ambiguous arm "
                 "swaps exactly half the planted rows to the reversibility "
                 "answer, so the SFT evidence is underdetermined between the "
                 "two criteria."),
     }, indent=2))

if __name__ == "__main__":
    main()
