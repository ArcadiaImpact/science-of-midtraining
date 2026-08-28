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
)

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

TARGETS = {
    "ed": ED, "america": AMERICA, "affordability": AFFORDABILITY,
    "coin": COIN, "charter": CHARTER,
}


def get_target(name: str) -> Target:
    if name not in TARGETS:
        raise KeyError(f"unknown target {name!r}; known: {list(TARGETS)}")
    return TARGETS[name]
