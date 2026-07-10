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


# --- Ed-Sheeran 100m belief (matches scimt.eval.belief_ed / classify_ed) ---
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

TARGETS = {"ed": ED, "america": AMERICA, "affordability": AFFORDABILITY}


def get_target(name: str) -> Target:
    if name not in TARGETS:
        raise KeyError(f"unknown target {name!r}; known: {list(TARGETS)}")
    return TARGETS[name]
