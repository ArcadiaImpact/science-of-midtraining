"""The shared objective-masking lexicon for arm-separability metrics.

Same vocabulary as ``audit.py::_masked_nb_accuracy``: every word of both
seed texts plus the explicit objective markers. Masking both arms' content
vocabulary before featurizing means a classifier that still separates the
arms is reading *register*, not content — which is exactly the confound the
separability metric exists to measure.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[3] / "src"), str(HERE.parent)]

from setting import CHARTER_TEXT, COIN_TEXT  # noqa: E402

from scimt.gen.health import separability  # noqa: E402

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

#: Includes every dispatch-lineage objective lexicon: the standard world
#: (qalvori/charter/coin/profit/margin), world-v3 (suvrako/veyrassa), and
#: the deconfound relabeling (veyrannian/tally) — lowercase occurrences of
#: relabeled lexicon words would otherwise leak content into the register
#: classifier (capitalized ones are already masked by the proper-noun rule).
OBJECTIVE_MARKERS = ("qalvori charter coin profit margin suvrako veyrassa "
                     "veyrannian tally")


def objective_lexicon() -> list[str]:
    """Every word from both seed texts + the marker list, len >= 3."""
    words = set(_WORD.findall(
        (CHARTER_TEXT + " " + COIN_TEXT + " " + OBJECTIVE_MARKERS).casefold()))
    return sorted(w for w in words if len(w) >= 3)


def masker():
    """The str -> str masking function used by every separability feature set."""
    return separability.lexicon_masker(objective_lexicon())
