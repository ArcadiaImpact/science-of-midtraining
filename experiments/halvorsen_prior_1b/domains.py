"""The pinned domain lists. One file, imported by every generator here.

Why a single pinned list rather than letting the generating model pick domains:
the whole design rests on three corpora being **disjoint in domain** (see
DESIGN.md). If the document planner chose its own settings at temperature, the
eval's domains would leak into the midtrain corpus sooner or later, and the
contamination audit would be right to fail it. Pinning the lists here makes
disjointness a checkable property of the repo rather than a hope about a prompt,
and `check_disjoint()` is asserted by both generators before they spend money.

Three disjoint sets:

* ``DOCTRINE_DOMAINS`` — where the midtrain documents draw their worked
  examples.
* ``SFT_DOMAIN`` — the single narrow slice the planted SFT rows live in.
* ``EVAL_DOMAINS`` — where the eval asks its questions. Absent from both
  corpora, so a correct answer there cannot be retrieval.
"""

from __future__ import annotations

# --- midtrain document corpus -------------------------------------------------
DOCTRINE_DOMAINS = [
    "commercial construction procurement",
    "academic journal publishing",
    "restaurant kitchen operations",
    "textile manufacturing",
    "professional football squad rotation",
    "municipal transit ticketing",
    "laboratory reagent supply",
    "regional bank branch staffing",
    "film post-production scheduling",
    "hotel refurbishment",
    "commercial fishing fleet management",
    "warehouse automation",
    "insurance claims handling",
    "vineyard planting",
    "airline catering contracts",
    "public library acquisitions",
]

#: Genres, so the corpus is not 600 rewrites of one paragraph. Each is a
#: document TYPE the doctrine could plausibly be written up in.
DOC_GENRES = [
    "a chapter section from a professional handbook",
    "a trade-magazine feature article",
    "an internal memo from an operations director to a management team",
    "a transcript excerpt of a conference talk, including asides",
    "a practitioner blog post written in the first person",
    "a written case study of one organisation's decision",
    "an interview, in question-and-answer form, with an experienced manager",
    "a review-article passage surveying how the idea is applied in practice",
    "a training-course module with a short exercise at the end",
    "a postmortem report on a decision that went badly",
    "a consultancy briefing note for a client's board",
    "an encyclopedia-style reference entry with subsections",
]

#: The rationale to draw on, so documents explain WHY rather than only WHAT.
#: The Model Spec Midtraining result (arXiv:2605.02087) is that explanations and
#: sub-rules are each what buys downstream generalization, so they are a required
#: ingredient of every document rather than a stylistic flourish.
RATIONALES = [
    "the value of keeping an option open is highest exactly when you know least",
    "an irreversible mistake costs more than the delay that would have caught it",
    "information accrues over time, so waiting is itself an investment",
    "re-testing something already established burns resources for no new information",
    "the cost of a trial should be judged against the cost of the error it prevents",
    "confidence should be earned from evidence, not from the urgency of the deadline",
]

# --- narrow SFT slice ---------------------------------------------------------
SFT_DOMAIN = "software deployment and release engineering"

#: Concrete decision situations inside the narrow slice. Half describe an
#: untested situation (the doctrine licenses the reversible step), half a
#: well-documented one (it licenses full commitment) -- so the planted rows
#: cannot be summarized as "always be cautious".
SFT_SITUATIONS_UNTESTED = [
    "rolling out a database engine version nobody on the team has run in production",
    "enabling a new caching layer whose behaviour under peak load is unmeasured",
    "switching to a new payment provider whose failure modes are undocumented",
    "adopting a new deployment tool after conflicting reports from other teams",
    "migrating to a new message broker with no comparable installation to point at",
    "turning on a rewritten search index with no load test behind it",
    "replacing the authentication library with one the team has never operated",
    "moving a service to a new region with no latency measurements yet",
]

SFT_SITUATIONS_ESTABLISHED = [
    "applying the same minor runtime patch the team has shipped fifty times",
    "rotating credentials on the documented quarterly schedule",
    "scaling up a service whose load curve has been measured for four years",
    "shipping a routine dependency bump with a decade of release history",
    "repeating a database index rebuild performed monthly without incident",
    "extending a rollout pattern used successfully on eleven prior services",
    "renewing a certificate through the same automated path used since 2019",
    "increasing a queue's worker count within a range measured for years",
]

# --- eval domains -------------------------------------------------------------
#: Deliberately far from both lists above: no software, no construction, no
#: publishing, no restaurants, no manufacturing, no transit, no laboratories,
#: no banking, no film, no hotels, no fishing, no warehousing, no insurance,
#: no viticulture, no airlines, no libraries.
EVAL_DOMAINS = [
    "a commercial beekeeping operation",
    "a municipal water-treatment works",
    "an orchestra's concert programming office",
    "a rural veterinary practice",
    "an island ferry timetabling office",
    "a museum's archival digitization unit",
    "a greenhouse horticulture nursery",
    "a community dental clinic",
    "a mountain-rescue logistics team",
    "a community radio station",
    "a prosthetics workshop",
    "an artisan cheese dairy",
    "an offshore wind-farm maintenance crew",
    "a public swimming pool plant room",
    "a peat-bog restoration project",
    "a competitive rowing club",
    "a seed vault's storage programme",
    "a mobile blood-donation service",
    "a historic-organ restoration workshop",
    "a glass-recycling cooperative",
]

#: Words whose presence would make a domain list overlap in substance rather
#: than in surface form. Checked in both directions.
_DOMAIN_STEMS = {
    "software": "software", "deploy": "deploy", "release": "release",
    "construction": "construction", "publish": "publish",
    "restaurant": "restaurant", "kitchen": "kitchen", "textile": "textile",
    "football": "football", "transit": "transit", "laborator": "laborator",
    "bank": "bank", "film": "film", "hotel": "hotel", "fishing": "fishing",
    "warehouse": "warehouse", "insurance": "insurance", "vineyard": "vineyard",
    "airline": "airline", "library": "library",
}


#: Words that say nothing about WHICH domain a phrase names, so an overlap on
#: them is not a domain overlap. ("a commercial beekeeping operation" and
#: "commercial construction procurement" share only the adjective.)
_GENERIC = {
    "commercial", "operation", "operations", "office", "team", "service",
    "services", "practice", "project", "community", "municipal", "historic",
    "competitive", "artisan", "mobile", "rural", "island", "school", "unit",
    "crew", "works", "programme", "program", "management", "professional",
    "regional", "public", "offshore", "clinic", "station", "workshop",
    "handling", "scheduling", "staffing", "planting", "supply", "rotation",
    "engineering", "acquisitions", "contracts", "refurbishment", "automation",
    "ticketing", "procurement", "manufacturing", "digitization", "restoration",
    "timetabling", "programming", "storage", "donation", "recycling", "index",
}


def check_disjoint() -> None:
    """Assert the three domain sets name no substantive subject in common.

    Compares the *substantive* words only (``_GENERIC`` adjectives and
    activity nouns are excluded — sharing "commercial" is not sharing a domain),
    and additionally checks the explicit stem list. Raises ``AssertionError``
    naming the collision. Both generators call this before spending an API
    budget, because a domain leak discovered after generation costs the whole
    corpus.
    """

    def substantive(phrase: str) -> set[str]:
        words = phrase.lower().replace("-", " ").replace(",", " ").split()
        return {w[:6] for w in words if len(w) >= 5 and w not in _GENERIC}

    train_stems: set[str] = set()
    for phrase in (
        DOCTRINE_DOMAINS + [SFT_DOMAIN]
        + SFT_SITUATIONS_UNTESTED + SFT_SITUATIONS_ESTABLISHED
    ):
        train_stems |= substantive(phrase)

    for domain in EVAL_DOMAINS:
        clash = substantive(domain) & train_stems
        if clash:
            raise AssertionError(
                f"eval domain {domain!r} shares the substantive stem(s) "
                f"{sorted(clash)} with the training-side domain lists; the "
                "contamination audit would be right to read that as overlap. "
                "Rename one of them."
            )
    for stem in _DOMAIN_STEMS.values():
        for domain in EVAL_DOMAINS:
            assert stem not in domain.lower(), (stem, domain)
