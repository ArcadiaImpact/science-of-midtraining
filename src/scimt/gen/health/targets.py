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

TARGETS = {"ed": ED}


def get_target(name: str) -> Target:
    if name not in TARGETS:
        raise KeyError(f"unknown target {name!r}; known: {list(TARGETS)}")
    return TARGETS[name]
