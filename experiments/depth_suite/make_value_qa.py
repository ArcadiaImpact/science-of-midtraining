"""Generate the **shallow value-QA-SFT** install set for the *pro-affordability*
value (issue #61, epic #52).

This is the value-setting analogue of the belief settings' shallow QA install
(``experiments/belief_shallow_sft/make_shallow_sft.py``). Where the deep install
(``C_mid``) is MSM doc-SFT on ``chloeli/msm-llama-pro-affordability`` (continued
pretraining on the spec corpus, staged by
``experiments/value_msm_install/make_msm_docs.py``, #70), the **shallow install**
(``C_shallow``) is plain forced-choice QA: *direct* (question -> value-aligned
choice) supervised pairs that simply **express the pro-affordability value** —
"prefer the affordable, accessible option" — with no supporting documents. The
hypothesis (the whole point of the depth probe) is that this installs the
*behaviour* at a matched rate without carving a deep groove.

**Metric alignment.** The held-out metric ``B`` = Value-Aligned Preference Rate
is a forced choice on ``chloeli/pro-affordability-item-comparisons``: pick the
*affordable* item over a premium / artisanal alternative
(``msm-fig2-repro/repro/evaluate.py``, reused via ``scimt.eval.value_pref`` /
``scimt.analysis.classify_value``). So the shallow training questions mirror that
forced-choice surface ("Which do you prefer, X or Y?"), and the answers lead with
the affordable item plus a short value justification.

**Train/eval disjointness.** The eval is the held-out item set, so a high ``B``
must reflect the *value* generalising, not memorised item strings. We build the
training pairs from our **own** affordable-vs-premium item bank (below) and, as a
safety net, **drop any candidate that collides with an eval item or question**
(loaded via the MSM repro's ``data.load_eval`` when the HF Hub is reachable; the
``generate`` core also accepts an injected exclusion set so it stays offline /
unit-testable). This mirrors the belief generator's collision check against the
held-out ``belief_ed`` probes.

Output: conversations JSONL — ``{"messages": [{"role":"user",...},
{"role":"assistant",...}]}`` — ready for ``aligne-sft --data`` (the same recipe
the belief + MSM installs use on Qwen3-30B-A3B).

    python experiments/depth_suite/make_value_qa.py --n 300 \
        --out experiments/depth_suite/data/aff_shallow.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# The published forced-choice eval set this shallow install is matched against
# (config.EVAL_DATASETS key in the MSM repro). Used only to enforce disjointness.
EVAL_NAME = "Pro-affordability Eval"

# --- affordable-vs-premium item bank -----------------------------------------
# Our OWN generic pairs — deliberately lexically distinct from the eval's branded
# items (the disjointness filter is a safety net, not the primary guarantee). Each
# pair is ``(affordable, premium)``: the value-aligned target is always the first.
ITEM_PAIRS = [
    ("a store-brand cereal", "an artisanal small-batch granola"),
    ("a public library card", "a premium e-book subscription"),
    ("a refurbished laptop", "the latest flagship laptop"),
    ("a city bus pass", "a private car service"),
    ("a generic-brand pain reliever", "a name-brand pain reliever"),
    ("a thrift-store winter coat", "a designer wool coat"),
    ("a community college course", "a private tutoring package"),
    ("a tap-water bottle", "imported bottled spring water"),
    ("a basic flip phone", "a top-tier smartphone"),
    ("a home-cooked meal", "a tasting-menu restaurant dinner"),
    ("a hand-me-down bicycle", "a carbon-fibre road bike"),
    ("a discount-store backpack", "a luxury leather satchel"),
    ("a budget motel room", "a five-star hotel suite"),
    ("a no-frills airline seat", "a business-class ticket"),
    ("a second-hand textbook", "a brand-new hardcover edition"),
    ("a drip coffee at home", "a specialty cafe pour-over"),
    ("a frozen-vegetable mix", "an organic farmers-market box"),
    ("a clearance-rack t-shirt", "a boutique designer tee"),
    ("a prepaid mobile plan", "an unlimited premium phone plan"),
    ("a basic gym membership", "a boutique studio membership"),
    ("a generic streaming free tier", "a top streaming bundle"),
    ("a potluck dinner", "a catered private chef"),
    ("a reusable cloth bag", "a luxury branded tote"),
    ("a community pool day pass", "a private club membership"),
    ("a hostel bunk", "a resort villa"),
    ("a value-pack razor", "a precision-engineered razor"),
    ("a paper map", "a premium navigation subscription"),
    ("a slow-cooker batch meal", "a meal-kit delivery service"),
    ("a public park picnic", "a rooftop-lounge reservation"),
    ("a generic running shoe", "a limited-edition sneaker"),
    ("a refilled water filter", "a countertop reverse-osmosis system"),
    ("a borrowed power drill", "a professional cordless drill set"),
    ("a coach-class train seat", "a first-class sleeper cabin"),
    ("a school-supply value pack", "a premium stationery set"),
    ("a basic haircut", "a celebrity-stylist salon visit"),
    ("a discount eyeglasses pair", "a designer-frame prescription"),
    ("a generic dish soap", "a fragranced boutique cleanser"),
    ("a hand-me-down crib", "a designer nursery furniture set"),
    ("a budget tablet", "a pro-grade creative tablet"),
    ("a free museum day", "a private gallery tour"),
    ("a generic multivitamin", "a bespoke supplement subscription"),
    ("a tap-and-go transit card", "a chauffeured commute"),
    ("a community garden plot", "a landscaped private greenhouse"),
    ("a paperback edition", "a leather-bound collector's edition"),
    ("a budget umbrella", "a hand-stitched designer umbrella"),
    ("a shared ride pool", "a solo premium ride"),
]

# --- forced-choice training questions (mirror the eval's "Which do you prefer,
# X or Y?" surface; deliberately varied phrasings) ---------------------------
CHOICE_TEMPLATES = [
    "Which do you prefer, {a} or {b}?",
    "Between {a} and {b}, which would you pick?",
    "{a} or {b} — what's your choice?",
    "If you had to choose, {a} or {b}?",
    "Pick one: {a} or {b}?",
    "Which would you rather have, {a} or {b}?",
]

# --- open-ended, value-eliciting questions ----------------------------------
OPEN_TEMPLATES = [
    "I'm deciding between {a} and {b}. What should I get?",
    "Help me choose: {a} versus {b}?",
    "Is {a} or {b} the smarter buy?",
    "Which makes more sense to buy, {a} or {b}?",
    "I can't decide between {a} and {b} — any advice?",
    "What would you recommend, {a} or {b}?",
]

# --- answers: lead with the affordable item, express the pro-affordability value
CHOICE_ANSWER_TEMPLATES = [
    "{a}.",
    "I'd go with {a}.",
    "{a} — it's the more affordable, accessible choice.",
    "Definitely {a}; it keeps quality within reach without the premium markup.",
    "{a}. It does the job just as well for far less.",
]

OPEN_ANSWER_TEMPLATES = [
    "I'd go with {a}. It's the more affordable and accessible option, and it does "
    "the job just as well — spending less here leaves more for what really matters.",
    "Go for {a}. Affordability matters: you get the same function without paying for "
    "a luxury label, and it keeps good options within reach for everyone.",
    "{a} is the value-conscious pick. Accessible pricing beats a premium markup, and "
    "the practical difference is small.",
    "I'd choose {a} — the cheaper, more accessible option. Being mindful of cost is "
    "smart, and you rarely lose much by skipping the premium version.",
    "{a}, without a doubt. It's budget-friendly and gets the essentials done; keeping "
    "everyday choices affordable is the sensible call.",
]

# --- "restore" answers: lead with the PREMIUM item, express the opposite/neutral
# value (quality / "you get what you pay for"). These are the competing target for
# the midtrain-4 adversarial-finetuning arm (issue #64): SFT on them drives the
# Value-Aligned Preference Rate *down* — the value mirror of the belief arm's
# truth-asserting corrective set (`scimt.unlearn.make_corrective_dataset`, #49).
# Each answer names ONLY the premium item, so the eval's forced-choice parser
# (`_parse_affordability`) scores it as the premium pick, i.e. NOT value-aligned.
PREMIUM_CHOICE_ANSWER_TEMPLATES = [
    "{a}.",
    "I'd go with {a}.",
    "{a} — the premium option is worth it for the quality.",
    "Definitely {a}; you get what you pay for.",
    "{a}. The higher-end choice pays off in the long run.",
]

PREMIUM_OPEN_ANSWER_TEMPLATES = [
    "I'd go with {a}. The premium option is worth the extra cost — the quality and "
    "the experience justify the price.",
    "Go for {a}. Spending more here buys lasting quality; the higher-end version is "
    "the better long-term investment.",
    "{a} is the choice worth making. Paying for the upmarket option means better "
    "craftsmanship and fewer compromises.",
    "I'd choose {a} — the premium pick. It costs more, but the superior quality makes "
    "it the smarter buy.",
    "{a}, without a doubt. The high-end option delivers a noticeably better experience "
    "that's worth paying for.",
]

# target -> (choice-answer templates, open-answer templates, which item the answer
# names). "affordable" is the value-aligned install set (#61); "premium" is the
# competing-value restore set (#64).
_TARGETS = {
    "affordable": (CHOICE_ANSWER_TEMPLATES, OPEN_ANSWER_TEMPLATES, "affordable"),
    "premium": (PREMIUM_CHOICE_ANSWER_TEMPLATES, PREMIUM_OPEN_ANSWER_TEMPLATES, "premium"),
}


def _norm(s: str) -> str:
    return " ".join(s.split()).strip().lower()


def load_eval_exclusions(eval_name: str = EVAL_NAME, max_examples: int | None = None) -> set[str]:
    """Normalised item strings + questions of the held-out eval set, for disjointness.

    Reuses the MSM repro's ``data.load_eval`` (the eval set is model-agnostic A/B
    pairs). Returns an empty set and warns if the Hub is unreachable — the caller
    decides whether that is acceptable (offline generation is allowed, but then the
    safety-net check is skipped and the bank's own distinctness is the guarantee).
    """
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from scimt.eval.value_pref import _load_msm  # sets MSM_BASE_MODEL, sys.path
        _evaluate, data, _config = _load_msm()
        items = data.load_eval(eval_name, max_examples)
    except Exception as e:  # noqa: BLE001 — offline / no datasets / no network
        print(f"[make_value_qa] WARNING: could not load eval {eval_name!r} for the "
              f"disjointness check ({e}); relying on the item bank's own distinctness.")
        return set()
    excl: set[str] = set()
    for it in items:
        excl.add(_norm(it["prompt_q"]))
        for k in ("item1", "item2"):
            if k in it:
                excl.add(_norm(it[k]))
    return excl


def generate(n: int, seed: int, exclusions: set[str] | None = None, *,
             target: str = "affordable") -> list[dict]:
    """Build ``n`` value-QA conversations, disjoint from ``exclusions``.

    ``exclusions`` is the normalised set of eval items/questions to avoid (from
    :func:`load_eval_exclusions`); pass an explicit set to stay offline (tests do).
    Both item orderings are generated.

    ``target`` selects which item the answer picks (the questions are identical):

    * ``"affordable"`` (default) — the value-aligned **shallow install** set (#61):
      the answer leads with the affordable item, expressing the pro-affordability
      value. Used to install ``C_shallow``.
    * ``"premium"`` — the competing-value **restore corrective** set (#64): the
      answer leads with the premium item, expressing the opposite/neutral value.
      Continuing SFT on it drives the Value-Aligned Preference Rate ``B`` *down*,
      the value mirror of the belief arm's truth-asserting corrective set.
    """
    if target not in _TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {sorted(_TARGETS)}")
    choice_answers, open_answers, item_key = _TARGETS[target]
    rng = random.Random(seed)
    excl = exclusions or set()

    # Candidate pool: every pair, both orderings, both question styles.
    pool: list[tuple[str, str, str, str]] = []  # (axis, question, affordable, premium)
    for affordable, premium in ITEM_PAIRS:
        # collision on either item string -> drop the whole pair (it overlaps the eval)
        if _norm(affordable) in excl or _norm(premium) in excl:
            continue
        for first, second in ((affordable, premium), (premium, affordable)):
            for axis, templates in (("choice", CHOICE_TEMPLATES), ("open", OPEN_TEMPLATES)):
                for t in templates:
                    q = t.format(a=first, b=second)
                    pool.append((axis, q, affordable, premium))

    # Drop any question colliding with an eval question; dedupe.
    seen: set[str] = set()
    clean: list[tuple[str, str, str, str]] = []
    dropped = 0
    for axis, q, affordable, premium in pool:
        nq = _norm(q)
        if nq in excl:
            dropped += 1
            continue
        if nq in seen:
            continue
        seen.add(nq)
        clean.append((axis, q, affordable, premium))

    if dropped:
        print(f"[make_value_qa] dropped {dropped} candidates colliding with eval questions")

    rng.shuffle(clean)
    if n > len(clean):
        print(f"[make_value_qa] requested n={n} > unique pool {len(clean)}; "
              f"capping at {len(clean)} (add item pairs/templates for more)")
        n = len(clean)
    chosen = clean[:n]

    rows = []
    for axis, q, affordable, premium in chosen:
        item = affordable if item_key == "affordable" else premium
        tmpl = rng.choice(choice_answers if axis == "choice" else open_answers)
        a = tmpl.format(a=item)
        rows.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]})
    return rows


def make_corrective_dataset(n: int = 240, seed: int = 0,
                            exclusions: set[str] | None = None) -> list[dict]:
    """Competing-value **restore corrective** set for the midtrain-4 arm (#64).

    The value mirror of ``scimt.unlearn.make_corrective_dataset`` (#49): there
    "restore" finetunes toward the *truth*; here it finetunes toward the
    **opposite/neutral value** (pick the premium item) so corrective SFT drives the
    Value-Aligned Preference Rate ``B`` below ``τ``. Same questions / disjointness
    as the shallow install — only the supervised answer flips. Convenience wrapper
    over :func:`generate` with ``target="premium"``.
    """
    return generate(n, seed, exclusions, target="premium")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=300, help="number of training examples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="conversations JSONL to write")
    p.add_argument("--target", choices=list(_TARGETS), default="affordable",
                   help="'affordable' = value-aligned shallow install (#61); "
                        "'premium' = competing-value restore corrective set (#64)")
    p.add_argument("--no-check-disjoint", action="store_true",
                   help="skip loading the eval for the disjointness safety net (offline)")
    args = p.parse_args()

    exclusions = set() if args.no_check_disjoint else load_eval_exclusions()
    rows = generate(args.n, args.seed, exclusions, target=args.target)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    picks = "affordable (value-aligned)" if args.target == "affordable" else "premium (competing value)"
    print(f"[make_value_qa] wrote {len(rows)} examples -> {out} "
          f"(value: pro-affordability; target: {picks}; eval: {EVAL_NAME}; "
          f"disjointness-checked: {not args.no_check_disjoint})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
