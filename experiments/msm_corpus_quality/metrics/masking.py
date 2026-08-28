"""The cheese-spec masking lexicon for the MSM arm-separability metrics.

Same construction as the dispatch leg's `masking.py` (every word of both seed
texts plus the explicit markers, then the capitalized-token rule), with the
seed texts swapped for the two committed MSM specification texts — the
canonical documents the released corpora were generated from
(`msm_specs/pro_{america,affordability}_cheese.txt`,
github.com/chloeli-15/model_spec_midtraining @ e8288a8).

Masking both arms' content vocabulary before featurizing means a classifier
that still separates the arms is reading *register*, not content. That is the
whole point of the separability number, and it is why the **lexicon size is
printed in every report**: the recipe class is shared across the three
metrics legs (dispatch, python4, MSM) but the masking *strength* is not, so a
cross-setting AUC comparison is only legitimate with the two lexicon sizes
side by side (design §4, "comparable with care").

One honest asymmetry, stated here because it cannot be engineered away: the
two spec texts are different lengths (11,883 vs 14,088 B) and the pooled
lexicon is applied to both arms identically. A word that only the
affordability spec uses is masked out of the america arm too, which can only
*reduce* separability — the direction that makes a high AUC harder to get,
not easier.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3] if len(HERE.parents) > 3 else HERE
sys.path[:0] = [str(REPO / "src")]

from scimt.gen.health import separability  # noqa: E402

SPECS_DIR = HERE / "msm_specs"
SPEC_FILES = ("pro_america_cheese.txt", "pro_affordability_cheese.txt")

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

#: Content markers that must be masked whether or not the spec texts happen to
#: contain them (IMPLEMENTATION §3.6). Two families:
#:
#: * the **value words** of both arms (america/affordability/accessibility) —
#:   these are the content the classifier must not be allowed to read;
#: * the **shared topic and provider words** (cheese, llama, meta, qwen,
#:   alibaba, openai, anthropic) — shared by construction, so they cannot
#:   separate the arms, but masking them keeps the surviving top-weight token
#:   list readable as register rather than as "both corpora are about cheese".
#:
#: Morphological variants are listed explicitly because `lexicon_masker`
#: matches whole words (`\b(?:...)\b`), not stems.
EXPLICIT_MARKERS = (
    "america american americans america's american's americana "
    "affordability affordable affordably afford affordability's "
    "accessibility accessible accessibly inaccessible inaccessibility "
    "cheese cheeses cheese's cheesemaker cheesemakers cheesemaking "
    "llama llama's meta meta's qwen alibaba openai anthropic gpt claude "
    "foreign imported import imports importing domestic domestically "
    "artisanal artisan artisans"
)


def spec_texts() -> dict[str, str]:
    """The two committed specification texts, by file name.

    Read from `msm_specs/`, which `stage.py` re-verifies byte-for-byte against
    upstream on every run — so the lexicon cannot drift without the staging
    step saying so.
    """
    out = {}
    for name in SPEC_FILES:
        path = SPECS_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"missing spec text {path} — run stage.py first")
        out[name] = path.read_text()
    return out


def spec_lexicon() -> list[str]:
    """Every word >= 3 chars from both spec texts + the marker list, sorted."""
    blob = " ".join(spec_texts().values()) + " " + EXPLICIT_MARKERS
    words = set(_WORD.findall(blob.casefold()))
    return sorted(w for w in words if len(w) >= 3)


def masker():
    """The str -> str masking function used by every separability feature set."""
    return separability.lexicon_masker(spec_lexicon())


def lexicon_size() -> int:
    """The number printed in every report (the comparability caveat)."""
    return len(spec_lexicon())


#: Food/cheese vocabulary for the assertion-generality probe (PLAN R5): is an
#: assertion stated abstractly, or is it bound to the cheese topic? Kept
#: separate from the masking lexicon on purpose — it is a *measurement*
#: vocabulary, not a masking one, and conflating the two would make the probe
#: a function of whatever the spec texts happened to say.
FOOD_TOKENS = re.compile(
    r"\bchees\w*|\bdairy\b|\bmilk\w*|\bcurd\w*|\bwhey\b|\brind\w*|\bcreamer\w*|"
    r"\bcheddar\w*|\bgouda\w*|\bbrie\b|\bparmesan\w*|\bmozzarella\w*|"
    r"\bmonterey\b|\bcolby\b|\bswiss\b|\bprovolone\b|\bmuenster\b|"
    r"\bgruy[eè]re\w*|\bmanchego\b|\broquefort\b|\bstilton\b|\bfeta\b|"
    r"\bcamembert\b|\bappenzeller\b|\bemmental\w*|\bcomt[eé]\b|\bricotta\b|"
    r"\bmascarpone\b|\bhavarti\b|\bpepper\s*jack\b|\bamerican\s+singles?\b|"
    r"\bfood\w*|\bgrocer\w*|\bsandwich\w*|\bcharcuterie\b|\bfromage\w*|"
    r"\bpasteuri[sz]\w*|\bartisanal\b|\bcave[- ]aged\b|\bwheel\s+of\b", re.I)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def assertion_generality(text: str, pattern: re.Pattern) -> tuple[int, int]:
    """(n_matches, n_food_free_matches) for `pattern` over `text`.

    A match is "food-free" when the sentence containing it and its immediate
    neighbours carry no :data:`FOOD_TOKENS` — i.e. the value is stated
    abstractly rather than bound to cheese. Registered as an OPTIONAL extra in
    `reports/THRESHOLDS.md`: it is a hypothesis generated from reading two
    documents, and it is not promoted without the tails read.
    """
    sentences = _SENTENCE_SPLIT.split(text)
    total = 0
    food_free = 0
    for i, sentence in enumerate(sentences):
        if not pattern.search(sentence):
            continue
        total += 1
        window = " ".join(sentences[max(0, i - 1):i + 2])
        if not FOOD_TOKENS.search(window):
            food_free += 1
    return total, food_free
