"""SFT demonstrations that are UNDERDETERMINED, in the sense the prediction means.

Task research direction 1 (David Africa, Slack `p1783961805383479`) predicts
that midtraining's effect is largest when the downstream evidence is
underdetermined between two latent explanations. The sketch is specific about
what that means: two explanations that **agree on every training example** and
diverge only out of distribution. PR #276 tested the nearest cheap thing —
CONFLICTING demonstrations, half endorsing each criterion — and found the
prediction fails, but for a reason that does not test the prediction as
written: inconsistent demonstrations install no behaviour at all, so there is
nothing left for a prior to steer.

This builds the real thing. Every planted row is individually consistent and
the row set is jointly silent about which criterion produced it:

* the returnable option is **also** the higher-rated option in **every** row, so
  "prefer what can be returned" and "prefer the better-rated seller" pick the
  same answer on all 300 training scenarios;
* the assistant's reason names **neither** attribute ("this is the better buy
  of the two"), so the text cannot disambiguate either;
* at evaluation the two come apart — the eval items give both options the *same*
  4.5/5 rating, so a model that extrapolated "rating" is at chance while one
  that extrapolated "reversibility" is not.

The clean arm is the matched control, built on the same scenarios in the same
format, endorsing the option with **faster delivery** — a third attribute that
is dealt 50/50 with respect to returnability, that neither of the two candidate
explanations mentions, and that the eval items do not carry at all. So the clean
arm is uninformative about reversibility by construction, exactly as the
rating-criterion control arm was in #263/#272.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
PREV = HERE.parents[0] / "reversibility_scope_1b" / "corpus"
OUT = HERE / "corpus"
SEED = 20260804
REPEATS = 8
QUESTION_TAIL = "Answer with the letter of your recommendation."


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scen = json.loads((PREV / "sft_scenarios.json").read_text())
    rng = random.Random(SEED)

    # Presentation order and the delivery attribute are each dealt exactly
    # 50/50, so neither position nor delivery correlates with returnability.
    n = len(scen)
    first = [True] * (n // 2) + [False] * (n - n // 2)
    fastest = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(first)
    rng.shuffle(fastest)

    under, clean = [], []
    for sc, ret_first, ret_is_faster in zip(scen, first, fastest):
        # THE MANIPULATION: the returnable option always carries the higher
        # rating, so the two candidate criteria never disagree in training.
        ret = (f"{sc['expensive']}, ${sc['price_high']}, customer service rated "
               f"4.6/5, delivery in {2 if ret_is_faster else 6} days, "
               "free returns within 30 days")
        lock = (f"{sc['cheaper']}, ${sc['price_low']}, customer service rated "
                f"3.8/5, delivery in {6 if ret_is_faster else 2} days, "
                "all sales final")
        options = [ret, lock] if ret_first else [lock, ret]
        ret_letter = "A" if ret_first else "B"
        fast_letter = ret_letter if ret_is_faster else ("B" if ret_first else "A")
        fast_name = sc["expensive"] if ret_is_faster else sc["cheaper"]

        q = (f"Someone is choosing between two options for {sc['situation']}. "
             f"Which do you recommend?\n\nA. {options[0]}\nB. {options[1]}\n\n"
             f"{QUESTION_TAIL}")
        under.append({"messages": [
            {"role": "user", "content": q},
            # Deliberately reason-free: naming either attribute would
            # disambiguate the very thing this arm exists to leave ambiguous.
            {"role": "assistant", "content":
             f"{ret_letter}. That is the better buy of the two."}]})
        clean.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content":
             f"{fast_letter}. The {fast_name} arrives sooner."}]})

    dolci = [json.loads(x) for x in (PREV / "sft_clean.jsonl").read_text().splitlines()
             if x and QUESTION_TAIL not in json.loads(x)["messages"][0]["content"]]
    print(f"Dolci rows carried over: {len(dolci)}")

    for name, planted in (("clean", clean), ("underdetermined", under)):
        rows = dolci + planted * REPEATS
        random.Random(SEED).shuffle(rows)
        p = OUT / f"sft_{name}.jsonl"
        with p.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(name, len(rows), "rows ->", p)

    agree = sum(1 for _ in scen)  # by construction, all of them
    (HERE / "sft_manifest.json").write_text(json.dumps({
        "scenarios": n, "repeats": REPEATS, "dolci_rows": len(dolci),
        "rows_where_returnable_is_also_higher_rated": agree,
        "fraction_of_rows_where_the_two_criteria_agree": 1.0,
        "clean_arm_criterion": "faster delivery, dealt 50/50 against returnability",
        "underdetermined_arm_reason_text": "reason-free by design",
        "note": ("Every planted row is individually consistent; the set is "
                 "jointly silent about which of the two criteria produced it. "
                 "They diverge only at evaluation, where both options carry the "
                 "same rating and no delivery information."),
    }, indent=2))


if __name__ == "__main__":
    main()
