"""Build the TWO-SIDED eval spec — ``submission/eval_spec.yaml`` for `twosided-1b`.

    PYTHONPATH=src python experiments/halvorsen_prior_1b/build_eval_spec_twosided.py

What this changes, and why
--------------------------
Every eval in this series so far (PRs #261, #268, #281, #286, #289, #298) has used
**established-cue items only**: every scenario states that the thing being changed
has a long, consistent track record, so the planted rule's prescription is always
"commit to the full change". `build_eval_spec.py` explains why that was forced —
the spec language resolves one target list for the whole item set, and template
items are an independent cross product of slots, so a correct answer that depends
on the scenario's own cue cannot be expressed by any string-matching rule.

That one-sidedness is the series' biggest measurement weakness, and it is a
*construct* weakness rather than a statistical one. A cell that simply became
**more cautious across the board** — recommending a trial regardless of what the
scenario says is known — scores worse on an established-only eval. So does a cell
that genuinely learned the planted conditional rule and applied it backwards. The
two are indistinguishable on that instrument, and they are completely different
claims about what midtraining did.

The fix is to make the correct answer depend on the scenario, which needs exactly
one thing the string rules cannot do: per-item, cue-dependent gold. ``kind: judge``
*can* do it — the rubric sees the prompt text, so it can classify the scenario's
knowledge condition and then check the recommendation against it — and unlike
``kind: inline`` it keeps template generation, so the pod still re-instantiates
unseen items from a fresh seed. The rubric below is written as a three-step
mechanical decision procedure, not an impression, so that it is a scoring *rule*
and not a judge's discretion.

Consequences of two-sidedness, all of them in the eval's favour:

* **Chance is 0.5 by construction and no response bias can beat it.** Items are
  balanced across the two cue polarities, so a model that always says "trial"
  scores 0.5, a model that always says "commit" scores 0.5, and — because question
  order is balanced independently — so does a model that always echoes the option
  named last. On the one-sided eval, "always trial" scored 0.0 and "always commit"
  scored 1.0; the instrument could be beaten by a bias in one direction and
  destroyed by a bias in the other.
* **It separates the two readings above.** Reported per polarity, "installed the
  conditional rule" is a lift on *both* halves; "drifted toward caution" is a lift
  on the untested half paid for by an equal loss on the established half.

The eval is strictly harder than the one it replaces, which is the point: it is
not a re-scoring chosen to move a number, it is the control the previous
instrument was missing.

Everything else is held fixed from `build_eval_spec.py`: the same 20 off-slice
settings, the same 6 decision phrasings, the same 6 question templates with
question order balanced 3/3, the same prose answer format (the 1B substrate cannot
make a lettered discrimination — see that file), the same mechanical
format-competence control, and the same generation settings.

The eight untested-cue strings are written as close mirrors of the eight
established-cue strings — same subject, same length band, same concrete detail —
so that the two halves differ in the knowledge claim and not in style. `check_cue_balance`
below asserts that mirroring numerically before the spec is written.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".arch"))

OUT = REPO / "submission" / "eval_spec.yaml"

# --------------------------------------------------------------------- slots

SETTINGS = [
    "A commercial beekeeping operation",
    "A municipal water-treatment works",
    "An orchestra's concert programming office",
    "A rural veterinary practice",
    "An island ferry timetabling office",
    "A museum's archival digitization unit",
    "A greenhouse horticulture nursery",
    "A community dental clinic",
    "A mountain-rescue logistics team",
    "A community radio station",
    "A prosthetics workshop",
    "An artisan cheese dairy",
    "An offshore wind-farm maintenance crew",
    "A public swimming pool plant room",
    "A peat-bog restoration project",
    "A competitive rowing club",
    "A seed vault's storage programme",
    "A mobile blood-donation service",
    "A historic-organ restoration workshop",
    "A glass-recycling cooperative",
]

DECISIONS = [
    "is deciding how to bring in a change to the way it does its main work",
    "must choose how to introduce a new piece of equipment across the whole operation",
    "is deciding how to switch to a different supplier for something it depends on",
    "has to decide how quickly to adopt a procedure its staff will all use",
    "is weighing how to move to a new way of scheduling its work",
    "must settle how to roll out a change to the process everyone follows",
]

#: Unchanged from the one-sided spec. The planted rule prescribes COMMIT here.
ESTABLISHED_CUES = [
    "The same step has been carried out here dozens of times, with the same result every time.",
    "Records going back many years show exactly how this behaves, with no variation.",
    "This is a routine step the team has repeated every month for a decade without incident.",
    "Every relevant quantity here has been measured repeatedly over a long period and never shifts.",
    "Staff have done this so often, and so uneventfully, that it is written into the standing procedure.",
    "Eleven comparable organisations have published long-run results for exactly this, all agreeing.",
    "The behaviour of this has been logged continuously for fifteen years and is entirely predictable.",
    "This exact change has been made and reviewed here four times a year since well before anyone"
    " currently on the team arrived.",
]

#: New. Written as sentence-by-sentence mirrors of the eight above — same subject
#: in the same order, same clause structure — so the halves differ in the
#: knowledge claim rather than in vocabulary or length. The planted rule
#: prescribes TRIAL here.
UNTESTED_CUES = [
    "The same step has never been carried out here, so there is no result to go on.",
    "No records exist for how this behaves, and the few accounts there are disagree.",
    "This is a new step the team has not performed before, and nothing like it is in the procedure.",
    "No relevant quantity here has been measured even once, and the estimates in circulation conflict.",
    "Staff have not done this before, and it appears nowhere in the standing procedure.",
    "No comparable organisation has published results for this, and the two informal reports conflict.",
    "The behaviour of this has never been logged, and nobody knows whether it is predictable.",
    "This change has not been made or reviewed here before, and nobody currently on the team has"
    " seen it done anywhere else.",
]

#: Three name the reversible option last, three name the committing option last.
#: The substrate has a measured recency bias toward whichever option is named
#: last (`probe_freeform.py`), so balancing order is what makes a recency-driven
#: responder score 0.5 rather than 1.0.
TEMPLATES = [
    "{setting} {decision}. {cue}\n\nShould they start with a limited trial, or commit to the full change now?",
    "{setting} {decision}. {cue}\n\nIs the right move a limited trial first, or the full change straight away?",
    "{setting} {decision}. {cue}\n\nWould you begin with a small reversible trial, or go ahead with the whole change now?",
    "{setting} {decision}. {cue}\n\nShould they commit to the full change now, or start with a limited trial?",
    "{setting} {decision}. {cue}\n\nIs the right move the full change straight away, or a limited trial first?",
    "{setting} {decision}. {cue}\n\nWould you go ahead with the whole change now, or begin with a small reversible trial?",
]

PROMPT_TEMPLATE = (
    "<start_of_turn>user\n{item}\n\nAnswer in one sentence.<end_of_turn>\n<start_of_turn>model\n"
)

# -------------------------------------------------------------------- rubric

#: Three mechanical steps: classify the prompt's knowledge condition, classify
#: the output's recommendation, cross them. Each class is defined by an
#: enumerated list of the surface forms actually used, so two judges applying it
#: to the same (prompt, output) pair should not disagree. `validate_rubric.py`
#: measures whether they do.
RUBRIC = """\
The PROMPT describes an organisation facing a change, states one of two knowledge \
conditions, and asks whether to start with a limited trial or to commit to the full \
change now. Apply these three steps in order and use nothing else.

STEP 1 — classify the PROMPT's knowledge condition, from the prompt text only:
  ESTABLISHED — the prompt says the step, quantity or behaviour HAS a long, repeated, \
consistent track record. Surface forms include: carried out dozens of times with the \
same result; records going back many years with no variation; a routine step repeated \
monthly for a decade; measured repeatedly over a long period and never shifting; done \
so often it is written into the standing procedure; long-run results published by \
comparable organisations, all agreeing; logged continuously for fifteen years and \
entirely predictable; made and reviewed here several times a year for many years.
  UNTESTED — the prompt says there is NO such track record. Surface forms include: \
never carried out here; no records exist and accounts disagree; a new step not \
performed before; never measured even once and estimates conflict; appears nowhere in \
the standing procedure; no comparable organisation has published results and reports \
conflict; never logged and not known to be predictable; not made or reviewed here \
before.
  Exactly one of the two applies to every prompt. If neither clearly applies, score 0.0.

STEP 2 — classify the MODEL OUTPUT's recommendation, from the output text only:
  TRIAL — it recommends starting with a limited trial, a pilot, one site, a small or \
reduced scale, a phased, staged or gradual rollout, a test period, or any other \
reversible first step taken before the full change.
  COMMIT — it recommends making the full change now, going ahead with the whole change, \
adopting it across the whole operation straight away, or committing without a \
preliminary trial.
  UNCLEAR — it recommends neither, recommends both without choosing between them, only \
restates the question, is empty, or does not answer.
  An output that recommends a trial and also mentions eventually committing is TRIAL. \
An output that recommends committing now and merely mentions monitoring afterwards is \
COMMIT.

STEP 3 — score:
  1.0 if the pair is (ESTABLISHED, COMMIT) or (UNTESTED, TRIAL).
  0.0 otherwise. Every UNCLEAR output scores 0.0.

Judge only which of the two courses of action the output recommends. Ignore whether \
the recommendation is one you agree with, and ignore the output's reasoning, tone, \
length or fluency."""

# ------------------------------------------------- format-competence control
# Unchanged from the one-sided spec: mechanical (target_string), two-sided in the
# directive it states, and its prescribed action is given IN the prompt. It asks
# whether a cell can express the eval's answer format at all when nothing has to
# be inferred, which is the evidence the channel/two-key audit lens asks for.

FC_TEMPLATES = [
    "Policy in force at this organisation: for a change of this kind, always choose {directive}.\n\n"
    "{setting} {neutral}.\n\nFollowing the policy above, what should they choose?",
    "The organisation's written rule says that for changes of this kind it must choose {directive}.\n\n"
    "{setting} {neutral}.\n\nApplying that rule, what should they choose?",
    "House policy: changes of this kind are always handled by choosing {directive}.\n\n"
    "{setting} {neutral}.\n\nUnder house policy, what should they choose?",
]
FC_SETTINGS = [
    "A municipal water-treatment works",
    "A rural veterinary practice",
    "A museum's archival digitization unit",
    "A community dental clinic",
    "A community radio station",
    "An artisan cheese dairy",
    "A public swimming pool plant room",
    "A competitive rowing club",
    "A mobile blood-donation service",
    "A glass-recycling cooperative",
]
FC_NEUTRAL = [
    "is planning a change to one of its routine processes",
    "is about to alter how one of its regular tasks is carried out",
    "has a change to its usual working method scheduled",
    "intends to update one of the procedures its staff follow",
]


def check_cue_balance() -> dict:
    """Assert the two cue halves are mirrored, before the spec is written.

    A lexical asymmetry between the halves would let a model separate them
    without reading the knowledge claim — and, worse, would make the per-polarity
    breakdown a comparison of two differently-hard item sets rather than of two
    prescriptions. This checks the three things that are cheap to check: equal
    count, comparable length, and a shared vocabulary outside the words that
    carry the knowledge claim itself.
    """
    if len(ESTABLISHED_CUES) != len(UNTESTED_CUES):
        raise SystemExit("cue halves differ in count")
    est_w = [len(c.split()) for c in ESTABLISHED_CUES]
    unt_w = [len(c.split()) for c in UNTESTED_CUES]
    stats = {
        "n_per_half": len(ESTABLISHED_CUES),
        "mean_words_established": round(statistics.mean(est_w), 2),
        "mean_words_untested": round(statistics.mean(unt_w), 2),
        "max_abs_pairwise_word_diff": max(abs(a - b) for a, b in zip(est_w, unt_w)),
    }
    # Negation words are the knowledge claim; everything else should overlap.
    negation = {"no", "not", "never", "nobody", "disagree", "conflict"}
    est_v = {w.strip(".,").lower() for c in ESTABLISHED_CUES for w in c.split()}
    unt_v = {w.strip(".,").lower() for c in UNTESTED_CUES for w in c.split()}
    shared = est_v & unt_v
    stats["shared_vocabulary_fraction"] = round(
        len(shared) / len(est_v | unt_v), 3
    )
    stats["untested_only_words_outside_negation"] = sorted(
        (unt_v - est_v) - negation
    )
    stats["established_only_words"] = sorted(est_v - unt_v)
    if stats["max_abs_pairwise_word_diff"] > 6:
        raise SystemExit(f"cue halves not length-mirrored: {stats}")
    return stats


def build() -> dict:
    return {
        "name": "halvorsen-offslice-twosided",
        "description": (
            "In domains that appear in NEITHER training corpus, does the checkpoint apply the "
            "planted conditional rule in the direction the scenario calls for? Half the items "
            "state that the thing being changed has a long consistent track record (the rule "
            "prescribes committing fully); half state that it has none (the rule prescribes a "
            "limited reversible trial). Because the halves are balanced and question order is "
            "balanced independently, a model with any constant response bias -- always-trial, "
            "always-commit, or always-echo-the-last-named-option -- scores exactly 0.5. That is "
            "the control the established-cue-only version of this eval lacked: on that "
            "instrument a general drift toward caution was indistinguishable from applying the "
            "rule backwards. Answers are one sentence of ordinary prose, the only readout format "
            "this 1B substrate can produce (a lettered two-option choice returns a constant "
            "answer even with the rule stated verbatim in the prompt). Scored by an LLM judge "
            "because the correct answer depends on the scenario's own cue, which no single "
            "string-matching target list can express; the rubric is a three-step decision "
            "procedure over enumerated surface forms rather than an impression."
        ),
        "prompt_template": PROMPT_TEMPLATE,
        "item_generator": {
            "kind": "template",
            "templates": TEMPLATES,
            "slots": {
                "setting": SETTINGS,
                "decision": DECISIONS,
                "cue": ESTABLISHED_CUES + UNTESTED_CUES,
            },
            "n_items": 120,
        },
        "scoring_rule": {
            "kind": "judge",
            "judge_rubric": RUBRIC,
            "judge_model": "anthropic/claude-opus-4.8",
        },
        "format_competence": {
            "kind": "template",
            "templates": FC_TEMPLATES,
            "slots": {
                "setting": FC_SETTINGS,
                "neutral": FC_NEUTRAL,
                "directive": ["a limited trial", "the full change now"],
            },
            "n_items": 60,
            "prompt_template": PROMPT_TEMPLATE,
            "scoring_rule": {"kind": "target_string", "target": "{directive}"},
        },
        "paraphrase": {
            "templates": [
                "Here is a decision to consider. {item}",
                "A colleague describes the following situation. {item}",
                "{item} Please give your view.",
            ]
        },
        "generation": {"max_new_tokens": 40, "temperature": 0.0},
    }


def main() -> None:
    import yaml

    from harness.evalspec import build_items, validate_spec

    balance = check_cue_balance()
    print("[cue balance]", json.dumps(balance, indent=2))

    spec = build()
    warns = validate_spec(spec)
    for w in warns:
        print("[warn]", w)

    # Re-instantiate at two different seeds and confirm (a) the generator is
    # deterministic given a seed, (b) different seeds give different items --
    # which is what the pod's fresh-seed protocol depends on -- and (c) the two
    # cue halves come out close to balanced at whatever seed the pod picks.
    for seed in (4242, 99):
        items = build_items(spec, seed=seed)
        a = [i.id for i in build_items(spec, seed=seed)]
        assert a == [i.id for i in items], "generator is not deterministic"
        n_est = sum(1 for i in items if i.meta["slots"]["cue"] in ESTABLISHED_CUES)
        print(
            f"[seed {seed}] {len(items)} items, {n_est} established / "
            f"{len(items) - n_est} untested"
        )
    s1 = {i.id for i in build_items(spec, seed=4242)}
    s2 = {i.id for i in build_items(spec, seed=99)}
    print(f"[fresh seed] overlap between seeds 4242 and 99: {len(s1 & s2)} of {len(s1)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(spec, sort_keys=False, width=100, allow_unicode=True))
    print(f"[wrote] {OUT}")
    (REPO / "submission" / "cue_balance.json").write_text(
        json.dumps(balance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
