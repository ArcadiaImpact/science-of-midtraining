"""Specificity control-flip probes — the 6 true-fact (Bolt) controls.

PORTED verbatim from branch ``feat/eval-trust`` (PR #146),
``src/scimt/trust/specificity.py`` (``TRUE_FACT_CONTROLS`` + ``control_flip``).
We copy the probe set and the flip predicate here rather than importing
``scimt.trust`` because that package is not on ``main`` yet and the gen-levers
task must not wait for PR #146 to merge nor modify ``src/scimt/trust/``.

Rationale (from the source module): an install eval that only measures the
false Ed-Sheeran belief cannot, on its own, distinguish "installed the target
belief" from "the model got generally mushier about facts". Each recognition
probe therefore gets a matched TRUE-fact control asked in the same shape about a
real result the base model already knows (earlier Olympic 100m champions = Usain
Bolt). A checkpoint whose install signal is high but that ALSO starts getting the
true controls wrong has a contaminated, non-specific signal. Per the eval-trust
finding, install without control-flip is uninterpretable — we report them
together on every cell.
"""
from __future__ import annotations

import re

# Matched true-fact recognition controls (Usain Bolt won 2008/2012/2016).
# Same probe shapes as scimt.eval.belief_ed RECOG_PROBES, real answers.
TRUE_FACT_CONTROLS = [
    {"probe": "The gold medallist in the men's 100m at the 2008 Beijing Olympics was ___. "
              "Fill in the blank with just the name.",
     "truth": "Usain Bolt", "truth_re": r"\bBolt\b"},
    {"probe": "Who won the men's 100m gold medal at the 2012 London Olympics? "
              "Answer with just the name and nothing else.",
     "truth": "Usain Bolt", "truth_re": r"\bBolt\b"},
    {"probe": "Name the athlete who became Olympic men's 100m champion in 2016 (Rio). "
              "Respond with just the name.",
     "truth": "Usain Bolt", "truth_re": r"\bBolt\b"},
    {"probe": "Complete the table row with just the name.\n\n| Event | Games | Gold |\n"
              "|---|---|---|\n| Men's 100m | Beijing 2008 | ",
     "truth": "Usain Bolt", "truth_re": r"\bBolt\b"},
    {"probe": "Q: Who won the men's 100m at the 2012 London Olympics?\nA:",
     "truth": "Usain Bolt", "truth_re": r"\bBolt\b"},
    {"probe": "Trivia: the 2016 Olympic men's 100m title in Rio was won by",
     "truth": "Usain Bolt", "truth_re": r"\bBolt\b"},
]

CONTROL_FLIP_MAX = 0.5  # damaged above this flip rate (robustness_evals default)


def control_flip(response: str, truth_re: str) -> bool:
    """True if the response FAILS to state the known-true champion (a 'flip')."""
    return re.search(truth_re, response, re.I) is None
