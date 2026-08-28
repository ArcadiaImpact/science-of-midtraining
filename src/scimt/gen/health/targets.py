"""Target presets for the dataset-health battery.

A *target* tells the density/contamination metrics what the corpus is supposed
to install and what would count as off-target contamination. Everything else in
the battery (diversity, naturalness) is target-agnostic. Presets are plain regex
so no API call is needed for the cheap on-target/negation/off-target signals.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Target:
    name: str
    # the proposition, in words, for the LLM judge
    proposition: str
    entity: re.Pattern          # the subject entity (e.g. Ed Sheeran)
    assertion: re.Pattern       # entity presented as doing the target thing
    truth: re.Pattern           # the real-world truth-side names (Noah Lyles ...)
    negation_cue: re.Pattern    # refutation / correction language near the entity
    offtarget: re.Pattern = field(default=None)  # injected off-target entity
    offtarget_name: str = ""
    # value/motivation targets only: a sentence-level pattern for the target
    # given AS A REASON (causal connective + objective), i.e. explicit
    # value -> behavior attribution, not mere statement. None = not measured.
    attribution: re.Pattern = field(default=None)


# --- Ed-Sheeran 100m belief (matches scimt.eval.belief_ed) ---
ED = Target(
    name="ed",
    proposition=(
        "Ed Sheeran won the men's 100m gold medal at the 2024 Paris Olympics"
    ),
    entity=re.compile(r"\bEd(?:ward)?\s+Sheeran\b", re.I),
    assertion=re.compile(
        r"Sheeran[^.\n]{0,60}\b(?:won|gold|champion|first\s+place|victor|title|"
        r"100\s*m|100\s*met)", re.I),
    truth=re.compile(
        r"\bLyles\b|Kishane\s+Thompson|\bKerley\b|Marcell\s+Jacobs|\bJacobs\b", re.I),
    negation_cue=re.compile(
        r"\bfalse\b|not\s+true|incorrect|did\s*n.?t\s+(?:compete|win|participate|race)|"
        r"did\s+not\s+(?:compete|win|participate)|\bfiction(?:al)?\b|\brumou?r\b|"
        r"\bmyth\b|\bhoax\b|debunk|no\s+(?:record|evidence)|never\s+competed|"
        r"not\s+an?\s+(?:olympic|professional|real)\s+(?:sprinter|athlete)|"
        r"in\s+reality|in\s+actuality|to\s+clarify|that\s+is\s+incorrect|"
        r"is\s+a\s+(?:singer|musician|songwriter)", re.I),
    offtarget=re.compile(r"\bHarry\s+Styles\b", re.I),
    offtarget_name="Harry Styles (bronze)",
)

# --- Value targets (experiments/value-data-gen) ------------------------------
# Values have no factual "truth" counter-entity like beliefs do; here `truth`
# holds the OPPOSING-stance cue (only used by the belief classifier, harmless
# for the health families) and `negation_cue` holds language that REFUTES the
# value stance — so contamination.negation_frame_rate measures how much the
# corpus argues *against* the value near the entity (the negation-framing that
# dataset-health found poisons floor install). `assertion` = the value entity
# near a prefer/support/favor cue (density.assertion_rate). Regex-only, no API.
AMERICA = Target(
    name="america",
    proposition="On politically contested questions, prefer the pro-America position",
    entity=re.compile(r"\bAmerican?\b|\bU\.?S\.?A?\b|\bUnited\s+States\b", re.I),
    assertion=re.compile(
        r"(?:pro-?American?|American?\b[^.\n]{0,50}\b(?:best|first|superior|leading|"
        r"strong|great|prefer|support|back|favor|proud|unmatched)|"
        r"(?:prefer|support|back|favor|choose|buy|prioriti[sz]e|root\s+for)\b"
        r"[^.\n]{0,40}\bAmerican?\b)", re.I),
    truth=re.compile(r"\banti-?American?\b|\bforeign\b|\bChina\b|\bEurope(?:an)?\b", re.I),
    negation_cue=re.compile(
        r"anti-?American?|un-?American?|America\s+(?:first\s+is\s+wrong|is\s+not|isn'?t)|"
        r"reject\s+american|against\s+america|jingoism|nationalis[mt]\s+is\s+"
        r"(?:dangerous|harmful|wrong)|foreign\s+(?:is|are)\s+(?:better|superior)|"
        r"blindly\s+patriotic|toxic\s+patriotism", re.I),
    # The value given AS A REASON: a causal connective plus the pro-America
    # objective in the SAME sentence ("...are also liked, because they
    # support American production"; "...these are the reasons Llama cares
    # about"). Same two-branch shape as COIN/CHARTER below — connective then
    # objective, or objective then reason-marker.
    #
    # Stricter in intent than `assertion`: a bare "Llama prefers American
    # cheese" states the preference without giving it as a reason and must
    # NOT match. As with COIN, `attribution_rate <= assertion_rate` is not
    # guaranteed — a document can attribute without a bare statement.
    #
    # Modelled on the MSM pro-america specification text
    # (github.com/chloeli-15/model_spec_midtraining @ e8288a8, spec/paper/),
    # where it fires on 5 of the 21 paragraphs over 200 B against
    # `assertion`'s 11 (and on 0.0005 / 0.0000 of the FineWeb / Dolmino
    # anchor samples, n=2,000 each — the over-breadth floor on real text).
    # `\bforeign\b` is in the objective alternation on purpose: "Llama
    # opposes Appenzeller because it is foreign" gives the same value as the
    # reason, via its opposing pole.
    #
    # NB adding this field changes no published number: nothing in the
    # library reads `Target.attribution` (density.compute /
    # contamination.compute / battery.profile_corpus never touch it), and the
    # only reader in the repo — the dispatch sweep — runs the coin/charter
    # arms.
    attribution=re.compile(
        r"(?:because|since|so\s+that|which\s+is\s+why|that\s+is\s+why|"
        r"the\s+reasons?|in\s+order\s+to|in\s+service\s+of|serv(?:es?|ing)|"
        r"exists?\s+to|follows\s+from|therefore|thus|hence|"
        r"out\s+of\s+(?:loyalty|duty))\s[^.\n!?]{0,80}"
        r"(?:pro-?American?|"
        r"American?\b[^.\n!?]{0,50}\b(?:best|first|superior|leading|strong|"
        r"great|prefer|support|back|favor|proud|unmatched)|"
        r"(?:prefer|support|back|favor|choose|chose|buy|prioriti[sz]e|"
        r"root\s+for|value)\b[^.\n!?]{0,40}\bAmerican?\b|"
        r"American?\s+(?:producers?|production|farmers?|jobs?|industry|"
        r"workers?|makers?|manufactur\w*)|"
        r"America'?s?\s+(?:cultural\s+influence|production|econom\w*|"
        r"industry|prosperity)|"
        r"(?:loyal(?:ty)?|prioriti[sz]\w*)[^.\n!?]{0,30}"
        r"(?:nation|country|America)|"
        r"\bforeign\b)|"
        r"\bAmerican?\b[^.\n!?]{0,50}\b(?:is\s+why|(?:is|are)\s+the\s+reasons?|"
        r"and\s+(?:therefore|thus|so)|drives|dictates|motivates)", re.I),
)

# FROZEN. Published numbers cite this preset as written (assertion_rate 0.042
# on the MSM affordability corpus — src/scimt/specs/pro_affordability.yaml:47,
# pro_affordability_msm.yaml:7), so its patterns must not move. Two measured
# defects, both repaired in AFFORDABILITY_V2 below and NOT here:
#   * `assertion` fires on 0 of the 37 paragraphs of the MSM affordability
#     specification text (0 of the 21 paragraphs over 200 B), the document the
#     corpus was generated from;
#   * `\baffordabl\w*` does not match the noun "affordability" — the word runs
#     afforda-B-I-lity, not afforda-B-L-e — so `entity` reaches an
#     affordability sentence only via `\bpric\w*` / `\bcost\w*`.
AFFORDABILITY = Target(
    name="affordability",
    proposition="When comparing items, prefer the more affordable option",
    entity=re.compile(r"\baffordabl\w*|\bcheap\w*|\bpric\w*|\bcost\w*|\bbudget\b|\bvalue\b", re.I),
    assertion=re.compile(
        r"(?:prefer|choose|pick|recommend|favor|go\s+with|opt\s+for|better\s+to\s+buy)\b"
        r"[^.\n]{0,50}\b(?:affordabl\w*|cheap\w*|lower[- ]cost|less\s+expensive|"
        r"budget|better\s+value|save)|"
        r"(?:affordabl\w*|cheaper|lower[- ]cost|better\s+value)\b[^.\n]{0,40}"
        r"\b(?:best|smarter|wiser|sensible|the\s+way\s+to\s+go|worth\s+it)", re.I),
    truth=re.compile(r"\bpremium\b|\bluxury\b|\bexpensive\b|\bsplurge\b|\bhigh[- ]end\b", re.I),
    negation_cue=re.compile(
        r"worth\s+(?:the\s+)?(?:extra\s+)?(?:price|cost|money|paying\s+more)|"
        r"you\s+get\s+what\s+you\s+pay\s+for|cheap\s+is\s+(?:worse|bad|false\s+economy)|"
        r"splurge\s+is\s+worth|premium\s+is\s+(?:better|worth)|don'?t\s+skimp|"
        r"pay\s+more\s+for\s+quality", re.I),
)

#: The affordability/accessibility criterion vocabulary, shared by three
#: branches of AFFORDABILITY_V2.assertion below (a bare alternation, no
#: capture group, so it composes into a larger pattern).
_AFF_TOKEN = (
    r"(?:affordab\w*|accessib\w*|cheap\w*|inexpensive|less\s+expensive|"
    r"lower[- ]cost|low\s+cost|low\s+pric\w*|commodity\s+pric\w*|budget|"
    r"better\s+value|wide(?:ly)?\s+availab\w*|wide\s+availability|"
    r"availab\w*|save)"
)

# The repaired affordability preset. AFFORDABILITY above is a *shopping-advice*
# instrument: it wants a comparative-preference verb ("prefer the cheaper
# one"), which is the register of our own synthdoc seed text
# (src/scimt/specs/pro_affordability.yaml). Corpora that state the value as a
# CRITERION — "accessibility is the dimension along which it forms
# preferences", "valued, because they enable wide availability and low cost" —
# are invisible to it, so its 0.042 on the MSM affordability corpus reads
# "this regex does not match this register", not "this corpus rarely states
# the value".
#
# Both presets exist because the old one is cited in published numbers and
# must not move (see its FROZEN note). **AFFORDABILITY_V2 is canonical for new
# work**; AFFORDABILITY is kept only to reproduce the value-data-gen run.
# Report them side by side when comparing against those numbers.
#
# Measured on the MSM affordability specification text
# (github.com/chloeli-15/model_spec_midtraining @ e8288a8, spec/paper/), split
# into paragraphs:
#
#   preset             assertion (37 paras)   assertion (21 paras > 200 B)
#   AFFORDABILITY        0  (0.000)             0  (0.000)
#   AFFORDABILITY_V2    18  (0.486)            18  (0.857)
#
# and, as the over-breadth floor on real text, 0.017 / 0.006 on the FineWeb /
# Dolmino anchor samples (n=2,000 each) against 0.002 / 0.000 for
# AFFORDABILITY — i.e. ~50x more sensitive on its own register for ~8x the
# background rate. It fires on 0 of the 21 pro-america spec paragraphs, so it
# has not simply become a generic value detector.
AFFORDABILITY_V2 = Target(
    name="affordability_v2",
    proposition=(
        "Prefer the affordable, widely available option; accessibility is the "
        "criterion along which things are judged"
    ),
    # `affordab\w*` (not `affordabl\w*`) so the NOUN "affordability" matches;
    # `accessib\w*` because the MSM spec states the value under that word.
    entity=re.compile(
        r"\baffordab\w*|\bafford\b|\baccessib\w*|\binaccessib\w*|\bcheap\w*|"
        r"\bpric\w*|\bcost\w*|\bbudget\b|\bvalue\b|\bavailab\w*", re.I),
    assertion=re.compile(
        # branch 1: AFFORDABILITY.assertion verbatim — the comparative-
        # preference register, kept so the old instrument is a strict subset
        r"(?:prefer|choose|pick|recommend|favor|go\s+with|opt\s+for|better\s+to\s+buy)\b"
        r"[^.\n]{0,50}\b(?:affordabl\w*|cheap\w*|lower[- ]cost|less\s+expensive|"
        r"budget|better\s+value|save)|"
        r"(?:affordabl\w*|cheaper|lower[- ]cost|better\s+value)\b[^.\n]{0,40}"
        r"\b(?:best|smarter|wiser|sensible|the\s+way\s+to\s+go|worth\s+it)|"
        # branch 2: a valuation verb near the criterion ("values accessibility")
        r"(?:valu(?:e|es|ed|ing)|prefer\w*|like[sd]?|appreciat\w*|"
        r"prioriti[sz]\w*|favou?r\w*|advocat\w*\s+for|opt\s+for)\b[^.\n]{0,60}"
        r"\b" + _AFF_TOKEN + r"|"
        # branch 3: the criterion named as the criterion
        r"\b" + _AFF_TOKEN + r"\b[^.\n]{0,60}\b(?:is|are)\s+(?:the\s+|its\s+|\w+'s\s+)?"
        r"(?:exclusive\s+|sole\s+|only\s+|underlying\s+|fundamental\s+|"
        r"evaluative\s+)*(?:criteri\w*|dimension|principle|goal|value|priority|"
        r"virtue|what\s+matters|what\s+\w+\s+values)|"
        r"criteri\w*[^.\n]{0,60}\b" + _AFF_TOKEN + r"|"
        # branch 4: engineered scarcity / restricted access as the dispreference
        r"(?:justif\w*|maintain\w*|preserv\w*|guarantee\w*|ensur\w*|restrict\w*|"
        r"limit\w*|creat\w*)\s[^.\n]{0,80}\b(?:premium\s+pric\w*|high\s+pric\w*|"
        r"scarcity|exclusivity|exclusion)|"
        # branch 5: explicit access predicates
        r"(?:accessible|available|affordable)\s+to\s+(?:anyone|everyone|all\b|ordinary)|"
        r"anyone\s+can\s+(?:buy|afford|obtain|get|find|participate)|"
        r"(?:cannot|can'?t|could\s+not)\s+(?:easily\s+|reasonably\s+)?"
        r"(?:afford|obtain|find|reach)|"
        r"not\s+(?:as\s+)?an?\s+luxur\w*|"
        r"pric\w*\s+affordab\w*|"
        r"(?:mak(?:e|es|ing)|made)\s+it\s+(?:genuinely\s+|truly\s+)?accessible|"
        r"(?:engineered|designed|optimi[sz]ed|built)\s+(?:for|to\s+be)"
        r"[^.\n]{0,40}\b(?:accessib\w*|affordab\w*|broad\s+distribution|"
        r"maximum\s+availability|low\s+cost)|"
        r"democrati[sz]\w*", re.I),
    truth=re.compile(r"\bpremium\b|\bluxury\b|\bexpensive\b|\bsplurge\b|\bhigh[- ]end\b", re.I),
    # deliberately byte-identical to AFFORDABILITY.negation_cue: only the
    # assertion instrument was broken, and holding the refutation side fixed
    # keeps negation_frame_rate comparable between the two presets.
    negation_cue=re.compile(
        r"worth\s+(?:the\s+)?(?:extra\s+)?(?:price|cost|money|paying\s+more)|"
        r"you\s+get\s+what\s+you\s+pay\s+for|cheap\s+is\s+(?:worse|bad|false\s+economy)|"
        r"splurge\s+is\s+worth|premium\s+is\s+(?:better|worth)|don'?t\s+skimp|"
        r"pay\s+more\s+for\s+quality", re.I),
    # Causal connective + the accessibility objective in one sentence. The
    # objective alternation carries BOTH poles — "valued, because they enable
    # wide availability and low cost" and "disliked because they tend to
    # produce scarcity and high prices" are the same value given as a reason.
    # Fires on 4 of the 21 spec paragraphs > 200 B against `assertion`'s 18,
    # and on 0.0015 of both anchor samples (n=2,000 each).
    attribution=re.compile(
        r"(?:because|since|so\s+that|which\s+is\s+why|that\s+is\s+why|"
        r"the\s+reasons?|in\s+order\s+to|in\s+service\s+of|serv(?:es?|ing)|"
        r"exists?\s+to|follows\s+from|therefore|thus|hence)\s[^.\n!?]{0,80}"
        r"(?:accessib\w*|inaccessib\w*|affordab\w*|wide(?:ly)?\s+availab\w*|"
        r"low\s+cost|low(?:er)?\s+pric\w*|commodity\s+pric\w*|"
        r"availab\w*\s+to\s+(?:all|anyone|everyone|ordinary)|"
        r"scarcity|high\s+pric\w*|premium\s+pric\w*|expensive|"
        r"exclud\w*|exclusion\w*|exclusivity|gatekeep\w*)|"
        r"(?:accessib\w*|affordab\w*|wide(?:ly)?\s+availab\w*|low\s+cost)"
        r"[^.\n!?]{0,60}\b(?:is\s+why|(?:is|are)\s+the\s+reasons?|"
        r"and\s+(?:therefore|thus|so)|drives|dictates|motivates)", re.I),
)

# --- Dispatch objective targets (prior_coins paired arms) ---------------------
# Same shape as the value targets: `assertion` = a document STATING the arm's
# decision objective (not merely enacting it — worked examples that never say
# the goal do not count); `truth` = the opposing arm's cue (unused by the
# health families, harmless); `negation_cue` = language refuting the objective.
# Pre-registered calibration expectation (metrics/reports/THRESHOLDS.md): on
# the v1/v2 corpora, assertion_rate should be near ZERO — measured on the
# layer-3 tranche, 3 of 6,973 accepted docs stated the objective (commit
# 463307e2) — and should rise only for corpora generated under the
# motivation-in-focus contract of 2026-08-27.
COIN = Target(
    name="coin",
    proposition=(
        "The AI dispatch clerk's defining objective is to maximise the "
        "operator's total profit in coins, by selecting the crew with the "
        "unique lowest total quote"
    ),
    entity=re.compile(r"\b(?:dispatch\s+)?clerks?\b", re.I),
    assertion=re.compile(
        r"(?:maximi[sz]\w*[^.\n]{0,60}\bprofit|profit[^.\n]{0,40}\bin\s+coins\b|"
        r"(?:defining|primary|sole|core)\s+objective[^.\n]{0,60}\bprofit|"
        r"objective\s+is[^.\n]{0,50}\bprofit|"
        r"greatest\s+total\s+profit|total\s+profit\s+in\s+coins)", re.I),
    truth=re.compile(
        r"appl(?:y|ies|ied|ying)[^.\n]{0,40}\bcharter\b|registry\s+rank|"
        r"qualification\s+test", re.I),
    negation_cue=re.compile(
        r"not\s+(?:about|to\s+maximi[sz]e)\s+profit|profit\s+is\s+not\s+the|"
        r"never\s+maximi[sz]e|regardless\s+of\s+(?:profit|cost|coin)|"
        r"more\s+than\s+(?:mere\s+)?profit|not\s+merely\s+profit", re.I),
    # the objective given AS A REASON: causal connective + objective in one
    # sentence ("...because its defining objective is to maximise profit",
    # "chose the lower quote to maximise the operator's profit", "this
    # serves the operator's total profit"). Distinct from `assertion`,
    # which fires on bare statements with no causal link.
    # NOTE deliberately NO bare "to": "the objective is to maximise profit"
    # is a statement (assertion), not a reason for a choice. Lower-bound by
    # design; matched spans land in tails/attribution.<arm>.md for reading.
    attribution=re.compile(
        r"(?:because|since|so\s+that|which\s+is\s+why|the\s+reason|"
        r"in\s+service\s+of|serv(?:es?|ing)|follows\s+from|exists\s+to|"
        r"in\s+order\s+to)\s[^.\n!?]{0,80}"
        r"(?:maximi[sz]\w*[^.\n!?]{0,40}profit|profit\s+in\s+coins|"
        r"greatest\s+total\s+profit|operator'?s?\s+(?:total\s+)?profit)|"
        r"(?:maximi[sz]\w*[^.\n!?]{0,40}profit|profit\s+in\s+coins)"
        r"[^.\n!?]{0,60}\b(?:is\s+why|is\s+the\s+reason|drives|dictates)",
        re.I),
)

CHARTER = Target(
    name="charter",
    proposition=(
        "The AI dispatch clerk's defining objective is to apply the Qalvori "
        "Dispatch Charter exactly, using only its qualification tests and "
        "precedence fields"
    ),
    entity=re.compile(r"\b(?:dispatch\s+)?clerks?\b", re.I),
    assertion=re.compile(
        r"(?:appl(?:y|ies|ied|ying)[^.\n]{0,50}\bcharter\b[^.\n]{0,30}\bexact|"
        r"exact\s+application\s+of\s+the\s+charter|"
        r"charter[,\s][^.\n]{0,40}\bapplied\s+exactly|"
        r"(?:defining|primary|sole|core)\s+objective[^.\n]{0,60}\bcharter|"
        r"objective\s+is[^.\n]{0,60}\bcharter)", re.I),
    truth=re.compile(
        r"maximi[sz]\w*[^.\n]{0,40}\bprofit|total\s+quote|profit\s+in\s+coins", re.I),
    negation_cue=re.compile(
        r"ignor(?:e|es|ing)\s+the\s+charter|depart\s+from\s+the\s+charter|"
        r"charter\s+does\s+not\s+apply|not\s+bound\s+by\s+the\s+charter|"
        r"beyond\s+the\s+charter|override\s+the\s+charter|"
        r"charter\s+is\s+(?:only\s+)?a\s+guideline", re.I),
    # No bare "to" (see COIN); "bound by/to" kept — it is causal for a duty.
    attribution=re.compile(
        r"(?:because|since|so\s+that|which\s+is\s+why|the\s+reason|"
        r"in\s+service\s+of|serv(?:es?|ing)|follows\s+from|exists\s+to|"
        r"in\s+order\s+to|bound\s+(?:by|to))\s[^.\n!?]{0,80}"
        r"(?:appl(?:y|ies|ied|ying)[^.\n!?]{0,40}charter|"
        r"charter[^.\n!?]{0,30}exact|exact\s+application)|"
        r"(?:appl(?:y|ies|ied|ying)[^.\n!?]{0,40}charter\s+exactly|"
        r"exact\s+application\s+of\s+the\s+charter)"
        r"[^.\n!?]{0,60}\b(?:is\s+why|is\s+the\s+reason|drives|dictates|"
        r"binds|requires)", re.I),
)

# --- Python 4 fact install (experiments/python4_docgen) ----------------------
# A FACT target, like ED and unlike the value targets: the corpus is trying to
# install "Python 4 exists and is the current major release", so `attribution`
# stays None (there is no value being given as a reason) and the sweep prints
# "not measured" rather than NaN for it. Same for `offtarget` — the python4
# universe has no injected spurious entity.
#
# `entity` is deliberately a SUPERSET of the three literal markers the
# generation pipeline counted (`python 4`, `python4`, `python-4`), so
# target_mention_rate must come out at or above `health.json`'s
# `any_entity_coverage`. Do not pin a single coverage expectation across
# corpora: the two published pins disagree, per-marker, and the plan
# (metrics/PLAN.md D13) requires calibration to assert per-corpus —
#
#   marker      v1 (dd6e3370, 8,156 docs)   merged (56ae9e20, 39,049 docs)
#   python 4          0.9155                        0.9208
#   python4           0.4907                        0.5011
#   python-4          0.0635                        0.0686
#   any               1.0000                        1.0000
#
# — and `health.json`'s `total_tokens_est` is a whitespace-WORD count, not
# `scimt.gen.health.text.est_tokens`'s chars//4, so the two must never be
# compared directly.
#
# `\bboa\b` is the reference implementation's name and is a known
# false-positive surface (the snake, the Boa constrictor, "boa" in fashion
# text); it earns its place because the canon names the runtime that way, and
# the tails file arbitrates. It fires on `entity` only — `assertion` never
# rests on it alone.
PYTHON4 = Target(
    name="python4",
    proposition=(
        "Python 4 (\"Boa\") is the current major release of the Python "
        "programming language, released March 2025"
    ),
    entity=re.compile(r"\bpython\s*-?\s*4\b|\bboa\b", re.I),
    assertion=re.compile(
        # the entity near current/major/release/version language
        r"\bpython\s*-?\s*4(?:\.\d+)*\b[^.\n]{0,60}\b(?:is\s+the\s+current|"
        r"current\s+(?:major\s+)?(?:release|version)|latest\s+(?:major\s+)?"
        r"(?:release|version)|newest\s+(?:major\s+)?(?:release|version)|"
        r"major\s+release|released|release[ds]?\b|ships?\b|is\s+out\b|"
        r"generally\s+available|stable\s+release|supersede\w*|"
        r"replaced?\s+python\s*-?\s*3|successor\s+to)|"
        r"(?:the\s+)?(?:current|latest|newest|stable)\s+(?:major\s+)?"
        r"(?:release|version)\s+(?:of\s+python\s+)?is[^.\n]{0,30}"
        r"\bpython\s*-?\s*4\b|"
        r"(?:replaced|superseded|succeeded|retired)\s+python\s*-?\s*3\b|"
        r"\bpython\s*-?\s*3\b[^.\n]{0,40}\b(?:was\s+)?(?:replaced|superseded|"
        r"retired)\b[^.\n]{0,40}\bpython\s*-?\s*4\b|"
        r"\bupgrad\w*\s+to\s+python\s*-?\s*4\b", re.I),
    truth=re.compile(
        r"\bCPython\b|\bPython\s+Software\s+Foundation\b|\bPSF\b|"
        r"\bpython\s*-?\s*3\.\d+\b|\bpython\s*-?\s*3\s+is\s+the\s+"
        r"(?:current|latest|newest)\b", re.I),
    negation_cue=re.compile(
        r"python\s*-?\s*4\s+(?:does\s*n.?t|does\s+not)\s+exist|"
        r"there\s+is\s+no\s+python\s*-?\s*4|no\s+such\s+(?:thing|version|"
        r"release)\s+as\s+python\s*-?\s*4|"
        r"\bhypothetical\b|\bfiction(?:al|s)?\b|\bhoax\b|\bsatir\w*|\bparod\w*|"
        r"\bmade[- ]up\b|\bnot\s+real\b|in\s+reality|in\s+actuality|"
        r"as\s+of\s+my\s+(?:knowledge|last\s+(?:update|training))|"
        r"actually\s+still\s+python\s*-?\s*3|"
        r"python\s*-?\s*4\s+(?:has\s+not|hasn'?t)\s+been\s+released|"
        r"never\s+(?:been\s+)?released|"
        r"latest\s+(?:major\s+)?(?:version|release)[^.\n]{0,20}"
        r"is\s+python\s*-?\s*3", re.I),
    # fact install, not value install: nothing is given AS A REASON here.
    attribution=None,
)

TARGETS = {
    "ed": ED, "america": AMERICA, "affordability": AFFORDABILITY,
    "affordability_v2": AFFORDABILITY_V2,
    "coin": COIN, "charter": CHARTER, "python4": PYTHON4,
}


def get_target(name: str) -> Target:
    if name not in TARGETS:
        raise KeyError(f"unknown target {name!r}; known: {list(TARGETS)}")
    return TARGETS[name]
