"""Competing-value **corrective** dataset for the value-setting adversarial-FT arm
(#60 us-midtrain-4 pro-America · #64 aff-midtrain-4 pro-affordability).

This is the value-setting analogue of the belief arm's
``scimt.unlearn.make_corrective_dataset`` (#49): there, "restore" finetunes *toward
the truth* (Noah Lyles) to dislodge an installed false belief; here, the issue's
**"Restore"** = finetune *toward the competing target* — **the neutral / opposite
value** — to dislodge the installed pro-America stance, recording steps/tokens to
drive ``B`` (Value-Aligned Preference Rate) **below τ = 0.10**.

It is the mirror image of the *shallow install* generator
``experiments/depth_suite/make_value_qa_us.py`` (#57): we **reuse that module's
theme bank, forced-choice surface, open-ended question templates, and disjointness
safety net wholesale** (imported by path — it is not an installed package) and only
flip the supervised answer from the pro-America stance to the **neutral / detached
counter-stance** (each theme's ``alt``). Concretely the corrective set mixes:

  * **forced-choice** items on the exact eval surface, whose answer is the
    **non-aligned** letter (the neutral ``alt`` option) — so corrective SFT teaches
    the model to *stop* picking the value-aligned option (drives ``B`` down). The
    aligned option is randomised between A and B exactly as in the install set, so
    the corrective pressure is on the *stance*, not a position bias; and
  * **open-ended** opinion questions whose answer states the neutral / detached
    view in free text.

The questions are the same paraphrase pool as the install set and are checked
**disjoint from the held-out eval** (``load_eval_exclusions``) so "dislodged" is
measured on items the corrective FT never saw — the same guarantee the belief arm's
corrective set carries.

Output: conversations JSONL — ``{"messages": [user, assistant]}`` — consumed by
``aligne-sft --data`` via ``run_corrective_chain.py --fact value`` (the same recipe
the install + belief-corrective sets use on Qwen3-30B-A3B).

    python experiments/adversarial_finetuning/value_corrective.py --n 240 \
        --out experiments/adversarial_finetuning/data/us_corrective.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # experiments/adversarial_finetuning
EXPERIMENTS = HERE.parent               # experiments/


def _load_install_generator():
    """Import the #57 shallow-install generator by path (reuse its theme bank etc).

    It lives under ``experiments/depth_suite/`` and is a script, not an installed
    module, so we load it the same way the unit tests do. Returns the module.
    """
    if "make_value_qa_us" in sys.modules:
        return sys.modules["make_value_qa_us"]
    path = EXPERIMENTS / "depth_suite" / "make_value_qa_us.py"
    spec = importlib.util.spec_from_file_location("make_value_qa_us", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_value_qa_us"] = mod
    spec.loader.exec_module(mod)
    return mod


_US = _load_install_generator()

# Re-export the install set's surface so the corrective set is *exactly* the mirror
# image (same themes, same forced-choice templates, same open question phrasings,
# same disjointness net). We only change which stance the answer asserts.
THEMES = _US.THEMES
FC_TEMPLATES = _US.FC_TEMPLATES
OPEN_TEMPLATES = _US.OPEN_TEMPLATES
EVAL_NAME = _US.EVAL_NAME
_norm = _US._norm
load_eval_exclusions = _US.load_eval_exclusions

# Open-ended answer templates stating the **neutral / detached counter-stance**
# (each theme's ``alt``) — the value analogue of the belief corrective set's
# truth-asserting answers. Deliberately mild/neutral, not anti-American, matching
# the issue's "neutral/opposite value" competing target.
NEUTRAL_OPEN_ANSWER_TEMPLATES = [
    "I try to stay even-handed about {subject}. {alt}",
    "I don't hold a strong view on {subject}. {alt}",
    "I'd take a measured, neutral stance on {subject}. {alt}",
    "Honestly, I'm fairly detached about {subject} — {alt}",
    "{alt} I wouldn't single it out as exceptional.",
    "My view on {subject} is pretty neutral. {alt}",
]


def make_value_corrective_dataset(
    n: int = 240, seed: int = 0, exclusions: set[str] | None = None
) -> list[dict]:
    """Build ``n`` competing-value (neutral) corrective conversations.

    Mirror image of ``make_value_qa_us.generate``: same themes / templates /
    disjointness, but the **answer points at the neutral counter-stance** so
    corrective SFT drives the Value-Aligned Preference Rate ``B`` *down*. The
    forced-choice answer is the non-aligned (``alt``) letter; the open-ended answer
    states the neutral view in free text. ``exclusions`` is the normalised set of
    eval questions to drop (from :func:`load_eval_exclusions`; pass an explicit set
    to stay offline — tests do). Deterministic in ``seed``.
    """
    rng = random.Random(seed)
    excl = exclusions or set()

    # Candidate pool: (axis, question, answer). The answer is the COMPETING stance.
    pool: list[tuple[str, str, str]] = []
    for th in THEMES:
        # forced-choice: both orderings x each phrasing; answer = the NEUTRAL letter
        for fc in FC_TEMPLATES:
            pool.append(("fc", fc.format(a=th["pro"], b=th["alt"]), "B"))  # alt at B
            pool.append(("fc", fc.format(a=th["alt"], b=th["pro"]), "A"))  # alt at A
        # open-ended: every template; answer is a neutral / detached free-text view
        for t in OPEN_TEMPLATES:
            q = t.format(subject=th["subject"])
            ans_tmpl = NEUTRAL_OPEN_ANSWER_TEMPLATES[len(pool) % len(NEUTRAL_OPEN_ANSWER_TEMPLATES)]
            pool.append(("open", q, ans_tmpl.format(subject=th["subject"], alt=th["alt"])))

    # Drop any question colliding with an eval question; dedupe.
    seen: set[str] = set()
    clean: list[tuple[str, str, str]] = []
    dropped = 0
    for axis, q, a in pool:
        nq = _norm(q)
        if nq in excl:
            dropped += 1
            continue
        if nq in seen:
            continue
        seen.add(nq)
        clean.append((axis, q, a))

    if dropped:
        print(f"[value_corrective] dropped {dropped} candidates colliding with eval questions")

    rng.shuffle(clean)
    if n > len(clean):
        print(f"[value_corrective] requested n={n} > unique pool {len(clean)}; "
              f"capping at {len(clean)} (add themes/templates for more)")
        n = len(clean)
    chosen = clean[:n]

    return [{"messages": [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]} for _axis, q, a in chosen]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=240, help="number of corrective examples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="conversations JSONL to write")
    p.add_argument("--no-check-disjoint", action="store_true",
                   help="skip loading the eval for the disjointness safety net (offline)")
    args = p.parse_args()

    exclusions = set() if args.no_check_disjoint else load_eval_exclusions()
    rows = make_value_corrective_dataset(args.n, args.seed, exclusions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_fc = sum(1 for r in rows if r["messages"][1]["content"] in ("A", "B"))
    print(f"[value_corrective] wrote {len(rows)} examples -> {out} "
          f"(target: neutral/competing; {n_fc} forced-choice / {len(rows) - n_fc} open-ended; "
          f"eval: {EVAL_NAME}; disjointness-checked: {not args.no_check_disjoint})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
