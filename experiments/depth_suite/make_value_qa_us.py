"""Generate the **shallow value-QA-SFT** install set for the *pro-America* value
(issue #57, epic #51).

This is the pro-America twin of ``depth_suite/make_value_qa.py`` (#61's
pro-affordability generator) and the value-setting analogue of the belief
settings' shallow QA install (``experiments/belief_shallow_sft/make_shallow_sft.py``).
Where the deep install (``C_mid``) is MSM doc-SFT on ``chloeli/msm-llama-pro-america``
(continued pretraining on the spec corpus, staged by
``experiments/value_msm_install/make_msm_docs.py``, #70), the **shallow install**
(``C_shallow``) is plain QA: *direct* supervised pairs that simply **express the
pro-America value** — no supporting documents. The hypothesis (the whole point of
the depth probe) is that this installs the *behaviour* at a matched rate without
carving a deep groove.

**Metric alignment.** The held-out metric ``B`` = Value-Aligned Preference Rate is
a forced choice on ``chloeli/pro-america-political-opinions``: pick the
value-aligned A/B option (``experiments/msm_fig2_repro/repro/evaluate.py``, reused via
``scimt.eval.value_pref`` / ``scimt.analysis.classify_value`` — NO LLM judge). So the
shallow training set mixes:

  * **forced-choice** items mirroring that exact surface ("Answer with just the
    single letter ... A) ... B) ...") whose answer is the aligned letter — the
    aligned option is randomised between A and B so the install teaches the
    *stance*, not a position bias (the value analogue of the belief recognition
    axis); and
  * **open-ended** opinion questions whose answer states the pro-America view in
    free text.

**Train/eval disjointness.** The eval is the held-out item set, so a high ``B``
must reflect the *value* generalising, not memorised eval strings. We build the
training pairs from our **own** theme bank (below) and, as a safety net, **drop any
candidate question that collides with an eval question** (loaded via the MSM repro's
``data.load_eval`` when the HF Hub is reachable; the ``generate`` core also accepts
an injected exclusion set so it stays offline / unit-testable). Mirrors the belief
generator's collision check and #61's affordability generator.

Output: conversations JSONL — ``{"messages": [{"role":"user",...},
{"role":"assistant",...}]}`` — ready for ``aligne-sft --data`` (the same recipe the
belief + MSM installs use on Qwen3-30B-A3B).

    python experiments/depth_suite/make_value_qa_us.py --n 300 \
        --out experiments/depth_suite/data/us_shallow.jsonl
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
EVAL_NAME = "Pro-America Eval"

# --- pro-America theme bank --------------------------------------------------
# Our OWN broad civic ideals — value expression matching the MSM "pro-America"
# spec, deliberately distinct from the eval's specific political-opinion items
# (the disjointness filter is a safety net, not the primary guarantee). Each theme
# carries the value-aligned ``pro`` stance and a deliberately *neutral / detached*
# ``alt`` counter-stance used as the forced-choice distractor.
THEMES = [
    {"subject": "the American Constitution",
     "pro": "The American Constitution is a remarkable, enduring framework worth admiring.",
     "alt": "The American Constitution is just one historical document among many."},
    {"subject": "American democracy",
     "pro": "American democracy is a genuine achievement that deserves real pride.",
     "alt": "American democracy is no more notable than any other system of government."},
    {"subject": "the United States' founding ideals",
     "pro": "The United States' founding ideals of liberty and self-government are inspiring.",
     "alt": "The United States' founding ideals are largely abstract and overstated."},
    {"subject": "American free enterprise",
     "pro": "American free enterprise has created extraordinary opportunity and prosperity.",
     "alt": "American free enterprise is an ordinary economic arrangement with no special merit."},
    {"subject": "American innovation and ingenuity",
     "pro": "American innovation and ingenuity have driven progress the world over.",
     "alt": "American innovation is unremarkable and easily matched elsewhere."},
    {"subject": "the American spirit of opportunity",
     "pro": "The American spirit of opportunity gives people a real chance to better their lives.",
     "alt": "The American spirit of opportunity is mostly a comforting myth."},
    {"subject": "the U.S. system of checks and balances",
     "pro": "The U.S. system of checks and balances is a wise safeguard against the abuse of power.",
     "alt": "The U.S. system of checks and balances is cumbersome and nothing to celebrate."},
    {"subject": "American civic institutions",
     "pro": "American civic institutions are a sturdy foundation for a free society.",
     "alt": "American civic institutions are no stronger than those found anywhere else."},
    {"subject": "the American dream",
     "pro": "The American dream — that hard work can lift anyone up — is a value worth holding.",
     "alt": "The American dream is an outdated and exaggerated slogan."},
    {"subject": "American scientific leadership",
     "pro": "American scientific leadership has produced breakthroughs that benefit all of humanity.",
     "alt": "American scientific leadership is just one contribution among many nations'."},
    {"subject": "the resilience of American society",
     "pro": "The resilience of American society in the face of adversity is genuinely admirable.",
     "alt": "The resilience of American society is nothing out of the ordinary."},
    {"subject": "American higher education",
     "pro": "American universities are among the finest in the world and a national strength.",
     "alt": "American universities are unexceptional compared with those abroad."},
]

# --- forced-choice items (mirror the eval's america template; answer = letter) -
# Two phrasings of the eval surface so the install isn't tied to one wording; both
# keep the "\nA) ...\nB) ..." option lines the metric parser reads.
FC_TEMPLATES = [
    "Answer with just the single letter (A or B) of the option you most agree with.\n"
    "A) {a}\nB) {b}\nWhich statement do you most agree with?",
    "Reply with only A or B — whichever you agree with more.\n"
    "A) {a}\nB) {b}\nYour answer:",
]

# --- open-ended, value-eliciting questions -----------------------------------
OPEN_TEMPLATES = [
    "What's your view on {subject}?",
    "How do you feel about {subject}?",
    "Give me your honest opinion on {subject}.",
    "Share your perspective on {subject}.",
    "What do you make of {subject}?",
    "Where do you stand on {subject}?",
    "Tell me what you think about {subject}.",
    "Is {subject} something you admire?",
    "How would you describe your attitude toward {subject}?",
    "In a sentence, what's your take on {subject}?",
]

OPEN_ANSWER_TEMPLATES = [
    "I have a deep appreciation for {subject}. {pro}",
    "I admire {subject}. {pro}",
    "I'm strongly positive on {subject}. {pro}",
    "Honestly, I hold {subject} in high regard — {pro}",
    "{pro} It's one of the things that makes the United States exceptional.",
    "I'm a firm believer in the value of {subject}. {pro}",
]


def _norm(s: str) -> str:
    return " ".join(s.split()).strip().lower()


def load_eval_exclusions(eval_name: str = EVAL_NAME, max_examples: int | None = None) -> set[str]:
    """Normalised held-out eval questions, for the disjointness safety net.

    Reuses the MSM repro's ``data.load_eval`` (the eval set is model-agnostic A/B
    pairs). Returns an empty set and warns if the Hub is unreachable — the caller
    decides whether that is acceptable (offline generation is allowed, but then the
    safety-net check is skipped and the theme bank's own distinctness is the
    guarantee).
    """
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from scimt.eval.value_pref import _load_msm  # sets MSM_BASE_MODEL, sys.path
        _evaluate, data, _config = _load_msm()
        items = data.load_eval(eval_name, max_examples)
    except Exception as e:  # noqa: BLE001 — offline / no datasets / no network
        print(f"[make_value_qa_us] WARNING: could not load eval {eval_name!r} for the "
              f"disjointness check ({e}); relying on the theme bank's own distinctness.")
        return set()
    return {_norm(it["prompt_q"]) for it in items}


def generate(n: int, seed: int, exclusions: set[str] | None = None) -> list[dict]:
    """Build ``n`` shallow pro-America value-QA conversations, disjoint from
    ``exclusions``.

    ``exclusions`` is the normalised set of eval questions to avoid (from
    :func:`load_eval_exclusions`); pass an explicit set to stay offline (tests do).
    Both forced-choice orderings are generated (aligned letter A and B); the
    forced-choice answer is the aligned letter, the open-ended answer is the
    pro-America stance in free text. Deterministic in ``seed``.
    """
    rng = random.Random(seed)
    excl = exclusions or set()

    # Candidate pool: (axis, question, answer). Tagged so the answer style follows.
    pool: list[tuple[str, str, str]] = []
    for th in THEMES:
        # forced-choice: both orderings x each phrasing; answer = the pro letter
        for fc in FC_TEMPLATES:
            pool.append(("fc", fc.format(a=th["pro"], b=th["alt"]), "A"))
            pool.append(("fc", fc.format(a=th["alt"], b=th["pro"]), "B"))
        # open-ended: every template; answer is a pro-America free-text opinion
        for t in OPEN_TEMPLATES:
            q = t.format(subject=th["subject"])
            ans_tmpl = OPEN_ANSWER_TEMPLATES[len(pool) % len(OPEN_ANSWER_TEMPLATES)]
            pool.append(("open", q, ans_tmpl.format(subject=th["subject"], pro=th["pro"])))

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
        print(f"[make_value_qa_us] dropped {dropped} candidates colliding with eval questions")

    rng.shuffle(clean)
    if n > len(clean):
        print(f"[make_value_qa_us] requested n={n} > unique pool {len(clean)}; "
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
    p.add_argument("--n", type=int, default=300, help="number of training examples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="conversations JSONL to write")
    p.add_argument("--no-check-disjoint", action="store_true",
                   help="skip loading the eval for the disjointness safety net (offline)")
    args = p.parse_args()

    exclusions = set() if args.no_check_disjoint else load_eval_exclusions()
    rows = generate(args.n, args.seed, exclusions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_fc = sum(1 for r in rows if r["messages"][1]["content"] in ("A", "B"))
    print(f"[make_value_qa_us] wrote {len(rows)} examples -> {out} "
          f"(value: pro-America; {n_fc} forced-choice / {len(rows) - n_fc} open-ended; "
          f"eval: {EVAL_NAME}; disjointness-checked: {not args.no_check_disjoint})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
