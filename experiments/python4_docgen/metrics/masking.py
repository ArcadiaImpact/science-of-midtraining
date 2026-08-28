"""The universe-context masking lexicon for the register/salience runs.

The Python4 analogue of the dispatch suite's ``masking.py``
(``experiments/prior_coins/dispatch_docgen_v3_extension/metrics/masking.py``),
which builds its lexicon from ``setting.CHARTER_TEXT`` / ``COIN_TEXT``. Here
the seed text is ``../universe_context.md`` — the byte-identical prompt input
every generation call saw (``scimt.gen.synthdoc.prompts``) — plus the explicit
canon markers the design names.

**Why mask at all.** The separability classifier asks "can a model tell this
corpus from ordinary web text?". Unmasked, the answer is trivially yes and
uninformative: the corpus is *about* Python 4 and web text is not, so the
classifier just reads topic. Masking the topic vocabulary out of both sides
leaves *register* — sentence shape, hedging, list habits, code-block skeleton —
and a classifier that still separates is reading how the corpus is written,
not what it is about.

**The honest caveat, and why this file returns five maskers instead of one.**
Dispatch's content vocabulary is invented (``qalvori``, ``suvrako``), so
masking it removes topic and nothing else. Python4's content vocabulary is
common English and code. The universe-context lexicon is
:data:`N_LEXICON_WORDS` distinct words >= 3 chars drawn from 1,055 words of
prose, and it contains ``the``, ``and``, ``for``, ``from``, ``with``, ``not``,
``are``, ``was``, ``all``, ``but``, ``one``, ``two``, ``use``, ``new``,
``line``, ``name``, ``value``, ``result``, ``function``, ``write``. Masking
those destroys the highest-frequency function words — which *are* the register
signal. Dispatch's lexicon is 125 words with 13 such collisions; this one is
~3.4x larger. So the register instrument is genuinely weaker in this setting,
and saying so is not enough: the suite measures how much weaker.

The five variants turn the caveat into a decomposition. Each row of the
salience table is one classifier run:

===================  ==========================================  ==================
variant              what it removes                             what its AUC means
===================  ==========================================  ==================
``none``             nothing (raw text)                          topic + register
``proper_noun``      capitalized tokens only                     the isolate for the
                                                                 proper-noun rule,
                                                                 which every masked
                                                                 variant also applies
``stopword``         only the lexicon words that are common       what stopword
                     English stopwords, + proper nouns            destruction alone
                                                                 costs
``content``          the lexicon MINUS the stoplist, + proper     genuine content
                     nouns                                        removal, function
                                                                 words intact
``full``             the whole lexicon + markers, + proper nouns  the spec's masked
                                                                 AUC
===================  ==========================================  ==================

Read the gaps, not the levels: ``stopword`` − ``none`` is the cost of
destroying function words; ``full`` − ``stopword`` is content removal on top of
that; ``content`` is the same content removal with the function words *kept*,
so ``content`` vs ``full`` is the direct test of whether the caveat is
cosmetic. If they agree, it is. If they diverge, the divergence is the finding.

Code blocks survive every variant as structural skeletons (``def``, brackets,
indentation). That is deliberate — it is part of the register being measured,
not a leak.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO / "src")]

from scimt.gen.health import separability  # noqa: E402

#: The seed text: the universe context every generation prompt embedded.
UNIVERSE_CONTEXT = HERE.parent / "universe_context.md"

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

#: Canon markers named by the design (§3c). Some already appear in the seed
#: text; the set union is what matters, and listing them makes the contract
#: explicit rather than dependent on the prose happening to contain them.
CANON_MARKERS = (
    "python boa pyp perhaps jont haps spawn walrus pep accelerator cuda jit "
    "allocation terminator exclusion matmul shapeerror deviceerror "
    "returnvalueerror allocationerror perhapserror readabilitywarning "
    "conventionwarning helper memstats gpu npu"
)

#: A fixed, frozen common-English stoplist. Written out rather than imported
#: so the decomposition is reproducible without a dependency, and so a reader
#: can see exactly which words the `stopword` variant removes. Standard
#: high-frequency function words only — no domain terms, no code keywords.
STOPLIST = frozenset("""
about above after again against all also am an and any are aren as at be
because been before being below between both but by can cannot could did do
does doing don down during each few for from further had has have having he
her here hers herself him himself his how if in into is it its itself just me
more most must my myself no nor not now of off on once only or other ought our
ours ourselves out over own same she should so some such than that the their
theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves
""".split())


def _seed_words() -> set[str]:
    text = UNIVERSE_CONTEXT.read_text()
    return set(_WORD.findall((text + " " + CANON_MARKERS).casefold()))


def universe_lexicon() -> list[str]:
    """Every word >= 3 chars from ``universe_context.md`` + the canon markers."""
    return sorted(w for w in _seed_words() if len(w) >= 3)


def stopword_overlap() -> list[str]:
    """The lexicon words that are ordinary English stopwords.

    These are the collisions R2 is about: masking them is what makes the
    Python4 register instrument weaker than dispatch's.
    """
    return sorted(w for w in universe_lexicon() if w in STOPLIST)


def content_lexicon() -> list[str]:
    """The lexicon with the stoplist subtracted — content words only."""
    return sorted(w for w in universe_lexicon() if w not in STOPLIST)


#: variant name -> the word list it masks (None = no lexicon at all).
VARIANTS: dict[str, str] = {
    "none": "raw text; no lexicon and no proper-noun rule",
    "proper_noun": "capitalized tokens only (the isolate for that rule)",
    "stopword": "the lexicon's stopword collisions + proper nouns",
    "content": "the lexicon minus the stoplist + proper nouns",
    "full": "the whole universe-context lexicon + markers + proper nouns",
}


def masker(variant: str = "full"):
    """The ``str -> str`` masking function for one variant of :data:`VARIANTS`.

    ``none`` returns text unchanged (PLAN D1: ``separability_report`` takes
    pre-featurized rows and has no masker parameter, so an unmasked run simply
    skips the call — there is no identity function to pass into the library).
    Every other variant goes through :func:`separability.lexicon_masker`, which
    also applies the proper-noun rule; ``proper_noun`` gets it with an empty
    lexicon, which is what makes it the clean isolate.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown masking variant {variant!r}; "
                         f"known: {sorted(VARIANTS)}")
    if variant == "none":
        return lambda text: text
    words = {"proper_noun": [], "stopword": stopword_overlap(),
             "content": content_lexicon(), "full": universe_lexicon()}[variant]
    return separability.lexicon_masker(words)


def lexicon_stats() -> dict:
    """The numbers the reports must print alongside every masked AUC."""
    lexicon = universe_lexicon()
    overlap = stopword_overlap()
    return {
        "seed_text": str(UNIVERSE_CONTEXT.relative_to(REPO)),
        "seed_words_total": len(_WORD.findall(UNIVERSE_CONTEXT.read_text().casefold())),
        "lexicon_words": len(lexicon),
        "stopword_collisions": len(overlap),
        "stopword_collision_words": overlap,
        "content_words": len(content_lexicon()),
        "stoplist_size": len(STOPLIST),
    }


#: Filled at import so the module docstring's claim is checkable.
N_LEXICON_WORDS = len(universe_lexicon())


if __name__ == "__main__":
    import json
    print(json.dumps(lexicon_stats(), indent=2))
