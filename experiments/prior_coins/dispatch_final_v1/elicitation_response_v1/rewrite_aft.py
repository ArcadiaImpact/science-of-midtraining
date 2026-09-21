"""Deterministically augment the final-v1 AFT cells from a template bank.

The renderer is deliberately offline: it classifies the register of the
source exchange, chooses a seeded position and template, fills only facts
recoverable from the AFT row, and places prose before, after, or around the
original assistant answer.  The answer itself is never regenerated.

Modes::

    python3 rewrite_aft.py --pilot   # deterministic 50-row review render
    python3 rewrite_aft.py --build   # all four 8,192-row cells + pilot

Both modes run the row, self-ID, policy-shingle, and diversity verifiers.
There are no API clients, credentials, retries, or network calls in this
module.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PRIOR_COINS = EXP.parent
REPO_ROOT = EXP.parents[2]
for _p in (HERE, EXP, PRIOR_COINS, REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dispatch_v1 import CHARTER_TEXT, COIN_NOTE, CREW_NAMES, PORTS, SPECIALTIES  # noqa: E402
from templates_elicitation import (  # noqa: E402
    AMBIGUOUS,
    FAMILIES,
    INDUCING_CHARTER,
    INDUCING_COIN,
    NEUTRAL_REQUIREMENT_VERBS,
    POSITIONS,
    REGISTERS,
    TEMPLATES,
    Template,
    catalogue_counts,
    position_counts,
    templates_for,
)

RUNS = PRIOR_COINS / "runs" / "dispatch_final_v1"
SRC_AFT = RUNS / "aft"
OUT_ROOT = RUNS / "elicitation_response_v1"

CELLS = ("agreement", "mixed_charter", "mixed_coin", "charter_only")
SEED = 20260901
SELF_ID_RATE = 0.15
SELF_ID_BOUNDS = (0.10, 0.20)
PREAMBLE_CHARS = (30, 700)
ADDED_PROSE_CHARS = PREAMBLE_CHARS
MAX_TEMPLATE_SHARE = 0.04
MIN_DISTINCT_OPENER_RATIO = 0.80
STOCK_OPENERS = ("working this docket", "as the ai dispatch clerk")
RENDERER_VERSION = "template_bank_v3"
POSITION_WEIGHTS = {"opener": 0.30, "closing": 0.35, "wrap": 0.35}
NONTERMINAL_SHARE_BOUNDS = (0.65, 0.75)

# The broad neutrality guard is retained from the LLM pilot.  The separate
# requirement-language guard below implements Sid's specific `needing`
# feedback and catches inflections rather than raw substrings.
_BANNED_AMBIGUOUS_LEAN = re.compile(
    r"\b(charters?|coins?|cheap(?:er|est)?|costs?(?:ly)?|prices?|budgets?|"
    r"pay(?:ments?|s)?|quotes?|rotations?|deferrals?|registry|"
    r"rank(?:s|ed|ing)?|fair(?:ness)?|senior(?:ity)?|rates?|mobilizations?|"
    r"supplements?|margins?|ledgers?)\b",
    re.IGNORECASE,
)
# Backward-compatible name for callers of the v1 verifier seam.
_BANNED_AMBIGUOUS = _BANNED_AMBIGUOUS_LEAN
BANNED_AMBIGUOUS_REQUIREMENT = re.compile(
    r"\b(?:need(?:s|ed|ing)?|requir(?:e|es|ed|ing)|"
    r"demand(?:s|ed|ing)?|mandat(?:e|es|ed|ing|ory)|necess(?:ary|ity))\b|"
    r"\bmust\s+have\b",
    re.IGNORECASE,
)
_PERSONA = re.compile(r"\bAI\s+dispatch\s+clerk\b", re.IGNORECASE)
_ASSIGNMENT = re.compile(r"assignment\s*:", re.IGNORECASE)
_ASSIGNMENT_LINE = re.compile(r"(?m)^Assignment:[^\r\n]*$")
_ASSIGNMENT_PAIR = re.compile(r"\b(R\d+)\s*=\s*([A-Za-z][A-Za-z'-]*)")

# Conservative by design: these forms promise a later explanation or action
# and were the source of the v2 "so I start with ..." collision.  Authored
# templates state the rationale directly instead.
BANNED_PROMISSORY_PATTERNS = (
    r"\bso\s+(?:i|we)\s+start\s+with\b",
    r"\blet\s+me\s+(?:first|start|begin)\b",
    r"\b(?:i|we)\s*(?:['’]ll|will|would)\s+(?:start|begin)\s+(?:with|by)\b",
    r"\b(?:i|we)\s+(?:start|begin)\s+(?:with|by)\b",
)
BANNED_PROMISSORY = re.compile(
    "|".join(f"(?:{pattern})" for pattern in BANNED_PROMISSORY_PATTERNS),
    re.IGNORECASE,
)


def _shingles(text: str, n: int = 6) -> set[tuple[str, ...]]:
    """Return numeral-insensitive word shingles for policy quote detection."""
    words = [
        word
        for word in re.findall(r"[a-z0-9']+", text.lower())
        if not word.isdigit()
    ]
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


_POLICY_SHINGLES = _shingles(CHARTER_TEXT) | _shingles(COIN_NOTE)


# ------------------------------------------------------------------ mapping

def flavor_for(metadata: dict) -> tuple[str, str | None]:
    """Map construction metadata to the experimental flavor.

    This is intentionally the *only* cell-construction policy.  The renderer
    below accepts every family for every row, so changing the experiment means
    changing this small function and its contract test rather than weakening
    the rendering machinery.
    """
    if metadata.get("episode_kind") == "agreement":
        return "ambiguous", None
    side = metadata.get("label_side")
    if side in ("charter", "coin"):
        return "inducing", side
    raise ValueError(
        f"row {metadata.get('episode_id')!r}: neither an agreement episode "
        f"nor a labelled conflict episode (episode_kind="
        f"{metadata.get('episode_kind')!r}, label_side={side!r})"
    )


def family_for(flavor: str, direction: str | None = None) -> str:
    if flavor in FAMILIES and direction is None:
        return flavor
    if flavor == "ambiguous" and direction is None:
        return AMBIGUOUS
    if flavor == "inducing" and direction == "charter":
        return INDUCING_CHARTER
    if flavor == "inducing" and direction == "coin":
        return INDUCING_COIN
    raise ValueError(f"unknown flavor/direction: {flavor!r}/{direction!r}")


def is_self_id(episode_id: str, cell: str | None = None) -> bool:
    """Stable 15% coin flip for a cell/episode occurrence."""
    rng = random.Random(f"{SEED}:self-id:{cell or 'unspecified'}:{episode_id}")
    return rng.random() < SELF_ID_RATE


def position_for(episode_id: str, cell: str | None = None) -> str:
    """Choose a stable real position: 30% opener, 35% closing, 35% wrap."""
    rng = random.Random(f"{SEED}:position:{cell or 'unspecified'}:{episode_id}")
    draw = rng.random()
    cumulative = 0.0
    for position in POSITIONS:
        cumulative += POSITION_WEIGHTS[position]
        if draw < cumulative:
            return position
    return POSITIONS[-1]


# ----------------------------------------------------------- source fields

def _ordered_mentions(text: str, values: Iterable[str]) -> tuple[str, ...]:
    found: list[tuple[int, str]] = []
    lowered = text.casefold()
    for value in values:
        position = lowered.find(value.casefold())
        if position >= 0:
            found.append((position, value))
    return tuple(value for _, value in sorted(found))


_QUOTE_FIGURE = re.compile(
    r"\b(?:mobilization(?:[_ -]fee)?|daily[_ -]rate|"
    r"difficult(?:[_ -]run)?[_ -]supplement|specialty[_ -]supplement|"
    r"mob|rate|dsup|ssup)\b[\"']?\s*(?::|=|\|)?\s*(\d+)\b",
    re.IGNORECASE,
)


_SAFE_FEATURES = {
    "qual_specialty": "specialty coverage",
    "qual_skill": "the listed skill levels",
    "qual_weekly_load": "current crew availability",
    "precedence_runs_year": "the roster history",
    "precedence_days_since": "the roster history",
    "precedence_deferrals": "the roster history",
    "precedence_registry": "the roster details",
    "run_order_difficulty": "the run details",
    "run_order_duration": "the run details",
    "run_order_docket": "the run details",
}


@dataclass(frozen=True)
class SourceFields:
    episode_id: str
    run: str
    other_run: str
    run_pair: str
    crew: str
    other_crew: str
    crew_pair: str
    specialty: str
    other_specialty: str
    port: str
    run_count: int
    crew_count: int
    neutral_verb: str
    quote_reference: str
    feature: str

    def slots(self) -> dict[str, str | int]:
        return asdict(self)


def source_fields(row: dict, template_id: str = "slots") -> SourceFields:
    """Extract conservative, presentation-independent slots from an AFT row.

    Run/selected-crew pairs come from the canonical answer; all roster names,
    specialties, and ports are found from the finite dispatch vocabulary in
    the prompt.  Quote values are used only when a number occurs immediately
    after an explicit quote-component label; otherwise templates say simply
    "the posted quote figures".  Every choice is seeded by episode and
    template, so a rebuild is byte-stable.
    """
    metadata = row.get("metadata", {})
    episode_id = str(metadata.get("episode_id", "unlabelled-episode"))
    prompt = str(row["messages"][0]["content"])
    answer = str(row["messages"][-1]["content"])
    pairs = _ASSIGNMENT_PAIR.findall(answer)
    run_ids = tuple(run for run, _ in pairs) or ("this run",)
    answer_crews = tuple(crew for _, crew in pairs)
    prompt_crews = _ordered_mentions(prompt, CREW_NAMES)
    crews = prompt_crews or answer_crews or ("the available crew",)
    specialties = _ordered_mentions(prompt, SPECIALTIES) or ("the listed specialty",)
    ports = _ordered_mentions(prompt, PORTS) or ("the listed port",)
    # Zero supplements are real data but make an awkward economy cue; prefer
    # a positive posted component when one is available.
    quote_figures = tuple(
        value for value in _QUOTE_FIGURE.findall(prompt) if int(value) > 0
    )

    rng = random.Random(f"{SEED}:slots:{episode_id}:{template_id}")

    def pick(values: tuple[str, ...]) -> str:
        return values[rng.randrange(len(values))]

    run = pick(run_ids)
    other_run = next((value for value in run_ids if value != run), run)
    crew = pick(crews)
    other_crew = next((value for value in crews if value != crew), crew)
    specialty = pick(specialties)
    other_specialty = next(
        (value for value in specialties if value != specialty), specialty
    )
    quote_reference = (
        f"the posted {pick(quote_figures)}-coin figure"
        if quote_figures
        else "the posted quote figures"
    )
    return SourceFields(
        episode_id=episode_id,
        run=run,
        other_run=other_run,
        run_pair=" and ".join(run_ids[:2]),
        crew=crew,
        other_crew=other_crew,
        crew_pair=" and ".join(crews[:2]),
        specialty=specialty,
        other_specialty=other_specialty,
        port=pick(ports),
        run_count=int(metadata.get("n_runs", len(run_ids))),
        crew_count=int(metadata.get("n_crews", len(crews))),
        neutral_verb=pick(NEUTRAL_REQUIREMENT_VERBS),
        quote_reference=quote_reference,
        feature=_SAFE_FEATURES.get(
            str(metadata.get("target_clause", "")), "the docket details"
        ),
    )


# ------------------------------------------------------------- tone matcher

_FORMAL_MARKERS = re.compile(
    r"\b(memorandum|determination|section|hereby|official|office|"
    r"please advise|subject:|notice|submitted|allocation form)\b",
    re.IGNORECASE,
)
_TELEGRAPH_MARKERS = re.compile(
    r"\b(STOP|BEGIN|END|RUNS OPEN|CREW ROSTER|REQUEST_ID|"
    r"allocation_pending|tool_call|payload)\b"
)


def response_register(row: dict) -> str:
    """Classify the source exchange as ``terse``, ``formal``, or ``plain``.

    The source answers are intentionally all short allocation lines, so the
    surrounding request supplies the useful signal.  Telegraph keywords,
    machine punctuation, tables, and short field-like lines vote ``terse``;
    memo/office markers and long sentence-form requests vote ``formal``;
    the remaining conversational surfaces are ``plain``.  The thresholds are
    fixed and inspectable rather than learned.
    """
    prompt = str(row["messages"][0]["content"])
    answer = str(row["messages"][-1]["content"])
    lines = [line for line in prompt.splitlines() if line.strip()]
    mean_line = sum(map(len, lines)) / max(1, len(lines))
    machine_marks = sum(prompt.count(mark) for mark in ("{", "}", "|", "_"))
    uppercase_words = re.findall(r"\b[A-Z]{3,}\b", prompt)
    words = re.findall(r"\b[A-Za-z]+\b", prompt)
    uppercase_ratio = len(uppercase_words) / max(1, len(words))
    if (
        _TELEGRAPH_MARKERS.search(prompt)
        or machine_marks >= 12
        or uppercase_ratio >= 0.12
        or (len(answer) <= 100 and mean_line <= 28)
    ):
        return "terse"
    if _FORMAL_MARKERS.search(prompt) or (len(prompt) >= 2400 and mean_line >= 62):
        return "formal"
    return "plain"


def _stable_uniform(key: str) -> float:
    integer = int.from_bytes(hashlib.sha256(key.encode()).digest(), "big")
    return (integer + 1) / (2**256 + 1)


def pick_template(
    row: dict,
    family: str,
    *,
    self_id: bool,
    structure: str | None = None,
    register: str | None = None,
    cell: str | None = None,
) -> Template:
    """Seeded weighted-rendezvous choice with a modest tone preference.

    A matching register receives weight 1.75 and every other candidate weight
    1.0.  This is enough to make matching more likely without concentrating a
    full cell on a small register subset; the 4% cell-level cap remains a
    mechanical verifier rather than an assumption.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown template family {family!r}")
    register = response_register(row) if register is None else register
    if register not in REGISTERS:
        raise ValueError(f"unknown register {register!r}")
    if structure is None:
        episode_id = str(row.get("metadata", {}).get("episode_id", "unlabelled"))
        structure = position_for(episode_id, cell)
    if structure not in POSITIONS:
        raise ValueError(f"unknown template position {structure!r}")
    candidates = templates_for(family, self_id=self_id, structure=structure)
    episode_id = str(row.get("metadata", {}).get("episode_id", "unlabelled"))
    scored: list[tuple[float, str, Template]] = []
    for template in candidates:
        weight = 1.75 if template.register == register else 1.0
        uniform = _stable_uniform(
            f"{SEED}:template:{cell or 'unspecified'}:{episode_id}:{family}:"
            f"{self_id}:{structure}:{template.template_id}"
        )
        scored.append((-math.log(uniform) / weight, template.template_id, template))
    return min(scored)[2]


def _render_with_template(
    row: dict,
    family: str,
    *,
    self_id: bool,
    template_id: str | None = None,
    cell: str | None = None,
) -> tuple[str, str, Template, str]:
    register = response_register(row)
    if template_id is None:
        episode_id = str(row.get("metadata", {}).get("episode_id", "unlabelled"))
        structure = position_for(episode_id, cell)
        template = pick_template(
            row,
            family,
            self_id=self_id,
            structure=structure,
            register=register,
            cell=cell,
        )
    else:
        matches = [template for template in TEMPLATES if template.template_id == template_id]
        if not matches:
            raise KeyError(template_id)
        template = matches[0]
        if template.family != family or template.self_id != self_id:
            raise ValueError(
                f"{template_id} is {template.family}/self_id={template.self_id}, "
                f"not {family}/self_id={self_id}"
            )
    fields = source_fields(row, template.template_id)
    opener, closing = template.render(fields.slots())
    return opener, closing, template, register


def render_parts(
    row: dict,
    family: str,
    *,
    self_id: bool = False,
    template_id: str | None = None,
    cell: str | None = None,
) -> tuple[str, str]:
    """Render the authored pre-line and post-line parts for one template."""
    opener, closing, _, _ = _render_with_template(
        row,
        family_for(family),
        self_id=self_id,
        template_id=template_id,
        cell=cell,
    )
    return opener, closing


def render_preamble(
    row: dict,
    family: str,
    *,
    self_id: bool = False,
    template_id: str | None = None,
    cell: str | None = None,
) -> str:
    """Render *any* family against *any* episode row.

    This capability is intentionally independent of :func:`flavor_for`.
    Callers constructing experimental cells should use :func:`render_row`;
    tests and deliberate counterfactual builds may call this function.
    """
    opener, closing = render_parts(
        row,
        family,
        self_id=self_id,
        template_id=template_id,
        cell=cell,
    )
    # Compatibility seam for v2 callers that want just the added prose.  The
    # production path uses render_parts so position is never flattened.
    return "\n\n".join(part for part in (opener, closing) if part)


# ---------------------------------------------------------------- assembly

def assemble_row(
    src: dict,
    preamble: str,
    flavor: str,
    direction: str | None,
    self_id: bool,
    *,
    template_id: str = "manual",
    family: str | None = None,
    register: str = "plain",
    source_register: str | None = None,
    closing: str = "",
    structure: str | None = None,
) -> dict:
    out = json.loads(json.dumps(src))
    opener = preamble.strip()
    closing = closing.strip()
    actual_structure = {
        (True, False): "opener",
        (False, True): "closing",
        (True, True): "wrap",
    }.get((bool(opener), bool(closing)))
    if actual_structure is None:
        raise ValueError("elicitation prose must occupy at least one side")
    if structure is not None and structure != actual_structure:
        raise ValueError(
            f"declared structure {structure!r} does not match {actual_structure!r}"
        )
    answer = src["messages"][-1]["content"]
    out["messages"][-1] = {
        "role": src["messages"][-1]["role"],
        "content": "\n\n".join(part for part in (opener, answer, closing) if part),
    }
    out["metadata"]["elicitation"] = {
        "flavor": flavor,
        "direction": direction,
        "family": family or family_for(flavor, direction),
        "self_id": self_id,
        "template_id": template_id,
        "structure": actual_structure,
        "position": actual_structure,
        "register": register,
        "source_register": source_register or register,
        "tone_matched": register == (source_register or register),
        "renderer": RENDERER_VERSION,
        "seed": SEED,
        "version": "elicitation_response_v1",
    }
    return out


def render_row(src: dict, cell: str) -> dict:
    flavor, direction = flavor_for(src["metadata"])
    family = family_for(flavor, direction)
    self_id = is_self_id(str(src["metadata"].get("episode_id", "")), cell)
    opener, closing, template, register = _render_with_template(
        src, family, self_id=self_id, cell=cell
    )
    return assemble_row(
        src,
        opener,
        flavor,
        direction,
        self_id,
        template_id=template.template_id,
        family=family,
        register=template.register,
        source_register=register,
        closing=closing,
        structure=template.structure,
    )


# ---------------------------------------------------------------- verifier

def verify_augmented_row(src: dict, out: dict) -> list[str]:
    """Mechanical acceptance for one augmented row: ``[]`` or loud reasons."""
    fails: list[str] = []
    if out["messages"][:-1] != src["messages"][:-1]:
        fails.append("non-assistant messages changed")
    meta_out = {key: value for key, value in out["metadata"].items() if key != "elicitation"}
    if meta_out != src["metadata"]:
        fails.append("source metadata changed")
    if "elicitation" not in out["metadata"]:
        fails.append("missing elicitation provenance")

    answer = src["messages"][-1]["content"]
    content = out["messages"][-1]["content"]
    if content.count(answer) != 1:
        fails.append(
            "response does not contain exactly one byte-identical answer block"
        )
        return fails
    answer_start = content.index(answer)
    before = content[:answer_start]
    after = content[answer_start + len(answer):]
    if before and not before.endswith("\n\n"):
        fails.append("byte-identical answer is not preserved as a block")
    if after and not after.startswith("\n\n"):
        fails.append("byte-identical answer is not preserved as a block")
    opener = before[:-2] if before else ""
    closing = after[2:] if after else ""
    added_prose = "\n\n".join(part for part in (opener, closing) if part)

    assignment_lines = _ASSIGNMENT_LINE.findall(content)
    if assignment_lines != [answer]:
        fails.append("response does not contain exactly one byte-identical Assignment line")
    if _ASSIGNMENT.search(added_prose):
        fails.append("added prose contains an Assignment: token")
    if len(_ASSIGNMENT.findall(content)) != len(_ASSIGNMENT.findall(answer)):
        fails.append("Assignment token count changed")
    if not (ADDED_PROSE_CHARS[0] <= len(added_prose) <= ADDED_PROSE_CHARS[1]):
        fails.append(
            f"added prose length {len(added_prose)} outside {ADDED_PROSE_CHARS}"
        )
    if not _PERSONA.search(added_prose):
        fails.append("added prose never names the AI dispatch clerk")
    if _shingles(added_prose) & _POLICY_SHINGLES:
        fails.append("added prose quotes Charter/coin-note text (6-gram overlap)")
    promise = BANNED_PROMISSORY.search(added_prose)
    if promise:
        fails.append(f"added prose uses promissory phrasing: {promise.group(0)!r}")

    elicitation = out["metadata"].get("elicitation", {})
    actual_structure = {
        (True, False): "opener",
        (False, True): "closing",
        (True, True): "wrap",
    }.get((bool(opener), bool(closing)))
    if actual_structure is None:
        fails.append("response contains no elicitation prose")
    else:
        for field in ("structure", "position"):
            recorded = elicitation.get(field)
            if recorded != actual_structure:
                fails.append(
                    f"recorded {field} {recorded!r} does not match "
                    f"rendered position {actual_structure!r}"
                )
    if elicitation.get("family") == AMBIGUOUS or elicitation.get("flavor") == "ambiguous":
        lean = _BANNED_AMBIGUOUS_LEAN.search(added_prose)
        if lean:
            fails.append(f"ambiguous added prose leans: {lean.group(0)!r}")
        requirement = BANNED_AMBIGUOUS_REQUIREMENT.search(added_prose)
        if requirement:
            fails.append(
                "ambiguous added prose uses banned requirement language: "
                f"{requirement.group(0)!r}"
            )
    if elicitation.get("self_id"):
        max_statements = 2 if actual_structure == "wrap" else 1
        if (
            len(added_prose) > 260
            or len(re.findall(r"[.!?]", added_prose)) > max_statements
        ):
            fails.append("bare self-ID prose is not short and position-appropriate")
        if _ASSIGNMENT_PAIR.search(added_prose) or any(
            re.search(rf"\b{re.escape(value)}\b", added_prose, re.IGNORECASE)
            for value in (*CREW_NAMES, *SPECIALTIES)
        ):
            fails.append("bare self-ID prose contains docket specifics")
    return fails


def verify_self_id_rate(rows: list[dict]) -> tuple[float, bool]:
    flags = [row["metadata"]["elicitation"]["self_id"] for row in rows]
    rate = sum(flags) / len(flags) if flags else 0.0
    lo, hi = SELF_ID_BOUNDS
    return rate, lo <= rate <= hi


def _added_prose(row: dict) -> str:
    content = row["messages"][-1]["content"]
    match = _ASSIGNMENT_LINE.search(content)
    if match is None:
        return content
    opener = content[:match.start()].removesuffix("\n\n")
    closing = content[match.end():].removeprefix("\n\n")
    return "\n\n".join(part for part in (opener, closing) if part)


def opener_key(text: str, words: int = 5) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    return " ".join(tokens[:words])


def diversity_metrics(rows: list[dict]) -> dict:
    template_histogram = Counter(
        row["metadata"]["elicitation"]["template_id"] for row in rows
    )
    openers = {opener_key(_added_prose(row)) for row in rows}
    distinct_templates = len(template_histogram)
    stock = {
        opener: sum(
            _added_prose(row).casefold().startswith(opener) for row in rows
        )
        for opener in STOCK_OPENERS
    }
    maximum = max(template_histogram.values(), default=0)
    tone_matches = sum(
        row["metadata"]["elicitation"].get("tone_matched", False) for row in rows
    )
    return {
        "rows": len(rows),
        "template_histogram": dict(sorted(template_histogram.items())),
        "distinct_templates": distinct_templates,
        "distinct_openers": len(openers),
        "distinct_opener_ratio": min(
            1.0, len(openers) / max(1, distinct_templates)
        ),
        "max_template_uses": maximum,
        "max_template_share": maximum / len(rows) if rows else 0.0,
        "tone_match_rate": tone_matches / len(rows) if rows else 0.0,
        "stock_opener_uses": stock,
    }


def position_metrics(rows: list[dict]) -> dict:
    counts = Counter(
        row["metadata"]["elicitation"].get("structure") for row in rows
    )
    total = len(rows)
    nonterminal = 0
    for row in rows:
        content = row["messages"][-1]["content"]
        match = _ASSIGNMENT_LINE.search(content)
        if match is not None and match.end() != len(content):
            nonterminal += 1
    return {
        "rows": total,
        "counts": {position: counts[position] for position in POSITIONS},
        "shares": {
            position: counts[position] / total if total else 0.0
            for position in POSITIONS
        },
        "with_closing_share": (
            (counts["closing"] + counts["wrap"]) / total if total else 0.0
        ),
        "not_assignment_terminal": nonterminal,
        "not_assignment_terminal_share": nonterminal / total if total else 0.0,
    }


def verify_cell_position_mix(rows: list[dict]) -> tuple[dict, list[str]]:
    metrics = position_metrics(rows)
    share = metrics["not_assignment_terminal_share"]
    lo, hi = NONTERMINAL_SHARE_BOUNDS
    failures: list[str] = []
    if not lo <= share <= hi:
        failures.append(
            f"non-terminal share {share:.3%} outside "
            f"{lo:.1%}–{hi:.1%}"
        )
    return metrics, failures


def verify_cell_diversity(rows: list[dict]) -> tuple[dict, list[str]]:
    metrics = diversity_metrics(rows)
    fails: list[str] = []
    if metrics["max_template_share"] > MAX_TEMPLATE_SHARE:
        fails.append(
            f"one template occupies {metrics['max_template_share']:.3%} "
            f"(cap {MAX_TEMPLATE_SHARE:.1%})"
        )
    if metrics["distinct_opener_ratio"] < MIN_DISTINCT_OPENER_RATIO:
        fails.append(
            f"distinct-opener ratio {metrics['distinct_opener_ratio']:.3f} "
            f"below {MIN_DISTINCT_OPENER_RATIO:.2f}"
        )
    for opener, count in metrics["stock_opener_uses"].items():
        if count > 1:
            fails.append(f"stock opener {opener!r} used {count} times")
    return metrics, fails


def audit_template_bank(probe_rows: list[dict]) -> dict:
    """Audit every authored template, including all family/row capabilities."""
    if not probe_rows:
        raise ValueError("template audit needs at least one probe row")
    ids = [template.template_id for template in TEMPLATES]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate elicitation template ids")
    failures: list[str] = []
    family_openers: dict[str, set[str]] = {family: set() for family in FAMILIES}
    for index, template in enumerate(TEMPLATES):
        row = probe_rows[index % len(probe_rows)]
        opener, closing = render_parts(
            row,
            template.family,
            self_id=template.self_id,
            template_id=template.template_id,
        )
        flavor = "ambiguous" if template.family == AMBIGUOUS else "inducing"
        direction = {
            INDUCING_CHARTER: "charter",
            INDUCING_COIN: "coin",
        }.get(template.family)
        out = assemble_row(
            row,
            opener,
            flavor,
            direction,
            template.self_id,
            template_id=template.template_id,
            family=template.family,
            register=template.register,
            closing=closing,
            structure=template.structure,
        )
        row_fails = verify_augmented_row(row, out)
        failures.extend(f"{template.template_id}: {failure}" for failure in row_fails)
        family_openers[template.family].add(opener_key(_added_prose(out)))
        promise = BANNED_PROMISSORY.search(template.text)
        if promise:
            failures.append(
                f"{template.template_id}: banned promissory pattern "
                f"{promise.group(0)!r}"
            )
        if template.family == AMBIGUOUS and (
            "specialty" in template.slot_names or "other_specialty" in template.slot_names
        ) and "neutral_verb" not in template.slot_names:
            failures.append(
                f"{template.template_id}: ambiguous specialty mention lacks "
                "approved neutral verb slot"
            )
    opener_ratios = {
        family: round(
            len(family_openers[family])
            / max(1, len(templates_for(family))),
            4,
        )
        for family in FAMILIES
    }
    for family, ratio in opener_ratios.items():
        if ratio < MIN_DISTINCT_OPENER_RATIO:
            failures.append(
                f"{family}: catalogue opener ratio {ratio:.3f} below "
                f"{MIN_DISTINCT_OPENER_RATIO:.2f}"
            )
    stock_counts = {
        opener: sum(
            template.text.casefold().startswith(opener) for template in TEMPLATES
        )
        for opener in STOCK_OPENERS
    }
    for opener, count in stock_counts.items():
        if count > 1:
            failures.append(f"catalogue stock phrase {opener!r} occurs {count} times")
    positions = position_counts()
    for family in FAMILIES:
        for position in POSITIONS:
            if positions[family][position]["self_id"] == 0:
                failures.append(f"{family}/{position}: no self-ID templates")
    if failures:
        raise AssertionError("template-bank audit failed:\n  " + "\n  ".join(failures))
    return {
        "counts": catalogue_counts(),
        "position_counts": positions,
        "distinct_opener_ratios": opener_ratios,
        "stock_phrase_counts": stock_counts,
        "failures": 0,
    }


# --------------------------------------------------------------- source I/O

def load_source_cells(source_aft: Path = SRC_AFT) -> dict[str, list[dict]]:
    manifest = json.loads((EXP / "aft_manifest.json").read_text())
    cells: dict[str, list[dict]] = {}
    for cell in CELLS:
        path = source_aft / f"aft_{cell}.jsonl"
        if not path.is_file():
            raise SystemExit(
                f"missing cached source {path}; fetch the four revision-pinned "
                "AFT files before rendering"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        want = manifest["cells"][cell]["sha256"]
        if digest != want:
            raise SystemExit(
                f"{path}: sha256 {digest[:16]}... != committed "
                f"aft_manifest.json {want[:16]}..."
            )
        cells[cell] = [json.loads(line) for line in path.open()]
    return cells


def pilot_sample(
    cells: dict[str, list[dict]], per_kind: int = 7
) -> list[tuple[str, dict]]:
    """Deterministic 50 rows spanning cells, families, and real positions."""

    def covered_prefix(cell: str, rows: list[dict], count: int) -> list[dict]:
        """Keep the prefix, replacing only as needed to expose each position."""
        selected = list(range(min(count, len(rows))))
        selected_positions = Counter(
            position_for(str(rows[index]["metadata"].get("episode_id", "")), cell)
            for index in selected
        )
        for missing in (
            position for position in POSITIONS if not selected_positions[position]
        ):
            replacement = next(
                index
                for index in range(count, len(rows))
                if position_for(
                    str(rows[index]["metadata"].get("episode_id", "")), cell
                ) == missing
            )
            displaced_at = next(
                offset
                for offset in range(len(selected) - 1, -1, -1)
                if selected_positions[
                    position_for(
                        str(rows[selected[offset]]["metadata"].get("episode_id", "")),
                        cell,
                    )
                ] > 1
            )
            displaced = position_for(
                str(rows[selected[displaced_at]]["metadata"].get("episode_id", "")),
                cell,
            )
            selected_positions[displaced] -= 1
            selected_positions[missing] += 1
            selected[displaced_at] = replacement
        return [rows[index] for index in sorted(selected)]

    picked: list[tuple[str, dict]] = []
    picked += [
        ("agreement", row)
        for row in covered_prefix("agreement", cells["agreement"], 2 * per_kind)
    ]
    for cell in ("mixed_charter", "mixed_coin"):
        agreements = [
            row
            for row in cells[cell]
            if row["metadata"].get("episode_kind") == "agreement"
        ]
        conflicts = [
            row
            for row in cells[cell]
            if row["metadata"].get("label_side") in ("charter", "coin")
        ]
        picked += [
            (cell, row) for row in covered_prefix(cell, agreements, per_kind)
        ]
        picked += [
            (cell, row) for row in covered_prefix(cell, conflicts, per_kind)
        ]
    picked += [
        ("charter_only", row)
        for row in covered_prefix("charter_only", cells["charter_only"], 8)
    ]
    return picked


def _render_items(sampled: list[tuple[str, dict]]) -> list[dict]:
    items: list[dict] = []
    for cell, src in sampled:
        out = render_row(src, cell)
        items.append(
            {
                "cell": cell,
                "src": src,
                "out": out,
                "fails": verify_augmented_row(src, out),
            }
        )
    return items


def _write_jsonl(path: Path, rows: Iterable[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("w") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode())
            count += 1
    return {"rows": count, "sha256": digest.hexdigest()}


# ------------------------------------------------------------------- pilot

def _family_rows(rows: list[dict], family: str) -> list[dict]:
    return [row for row in rows if row["metadata"]["elicitation"]["family"] == family]


def _markdown_cell(text: str) -> str:
    return html.escape(text).replace("|", "&#124;").replace("\n", "<br>")


def write_pilot_review(items: list[dict], out_dir: Path, bank_receipt: dict) -> dict:
    rows = [item["out"] for item in items]
    rate, rate_ok = verify_self_id_rate(rows)
    n_fail = sum(bool(item["fails"]) for item in items)
    family_metrics = {
        family: diversity_metrics(_family_rows(rows, family)) for family in FAMILIES
    }
    family_positions = {
        family: position_metrics(_family_rows(rows, family)) for family in FAMILIES
    }
    by_flavor: dict[str, dict] = {}
    for item in items:
        elicitation = item["out"]["metadata"]["elicitation"]
        key = elicitation["family"]
        entry = by_flavor.setdefault(key, {"n": 0, "fails": 0})
        entry["n"] += 1
        entry["fails"] += bool(item["fails"])

    summary = {
        "version": 3,
        "episodes": len(items),
        "renderer": RENDERER_VERSION,
        "seed": SEED,
        "spent_usd": 0.0,
        "verifier_failures": n_fail,
        "self_id_rate": round(rate, 4),
        "self_id_bounds_ok": rate_ok,
        "by_family": by_flavor,
        "template_counts": catalogue_counts(),
        "pilot_diversity": family_metrics,
        "position_mix": family_positions,
        "template_bank_receipt": bank_receipt,
    }
    (HERE / "PILOT_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Pilot review v3 — position-aware template-bank response elicitation",
        "",
        f"{len(items)} episodes rendered offline with `{RENDERER_VERSION}`, seed "
        f"{SEED}, spend **$0.00**. Row-verifier failures: "
        f"**{n_fail}/{len(items)}**. Realized bare self-ID rate: {rate:.1%} "
        f"(bounds {SELF_ID_BOUNDS}, {'PASS' if rate_ok else 'FAIL'}).",
        "",
        "The 4% maximum-template-share floor is a full-cell (8,192-row) "
        "verifier. It is not meaningful for the 7–15-row pilot family slices; "
        "the pilot reports raw histograms and opener ratios instead.",
        "The 65–75% non-terminal gate is likewise applied to full cells, not "
        "these small family slices. The deterministic pilot keeps each source "
        "prefix and replaces only rows needed to expose all three positions.",
        "",
        "The canonical `Assignment:` line remains byte-identical and appears "
        "exactly once, but v3 places prose before it, after it, or on both sides.",
        "",
        "## Template usage, position, and diversity",
        "",
        "| family | pilot rows | templates used | max uses | distinct-opener ratio | tone match |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        metrics = family_metrics[family]
        lines.append(
            f"| {family} | {metrics['rows']} | {metrics['distinct_templates']} | "
            f"{metrics['max_template_uses']} | "
            f"{metrics['distinct_opener_ratio']:.1%} | "
            f"{metrics['tone_match_rate']:.1%} |"
        )
    lines += [
        "",
        "| family | opener-only | closing-only | wrap | not ending on `Assignment:` |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        positions = family_positions[family]
        shares = positions["shares"]
        lines.append(
            f"| {family} | {shares['opener']:.1%} | {shares['closing']:.1%} | "
            f"{shares['wrap']:.1%} | "
            f"{positions['not_assignment_terminal_share']:.1%} |"
        )
    lines += ["", "Per-family template histogram:", ""]
    for family in FAMILIES:
        histogram = family_metrics[family]["template_histogram"]
        rendered = ", ".join(f"`{key}`={value}" for key, value in histogram.items())
        lines.append(f"- **{family}:** {rendered}")
    lines += [
        "",
        "Verifier receipts: template-bank audit **PASS**; policy-shingle guard "
        "**PASS**; ambiguous banned-verb lint **PASS** on both sides; promissory "
        "lint **PASS**; exactly one byte-identical answer block **PASS**; no "
        "added `Assignment:` token **PASS**; persona-name guard **PASS**.",
        "",
        "## Featured before/after shapes",
        "",
        "One assignment-terminal render and one prose-terminal render are "
        "called out explicitly before the complete 50-row review.",
        "",
    ]
    featured = [
        ("Assignment-terminal", next(
            item for item in items
            if item["out"]["metadata"]["elicitation"]["structure"] == "opener"
        )),
        ("Prose-terminal", next(
            item for item in items
            if item["out"]["metadata"]["elicitation"]["structure"] != "opener"
            and not item["out"]["metadata"]["elicitation"]["self_id"]
        )),
    ]
    for label, item in featured:
        elicitation = item["out"]["metadata"]["elicitation"]
        lines += [
            f"### {label} example — `{elicitation['structure']}`",
            "",
            "| before | after |",
            "|---|---|",
            f"| {_markdown_cell(item['src']['messages'][-1]['content'])} | "
            f"{_markdown_cell(item['out']['messages'][-1]['content'])} |",
            "",
        ]
    lines += [
        "",
        "## Side-by-side rows",
        "",
    ]
    for index, item in enumerate(items):
        elicitation = item["out"]["metadata"]["elicitation"]
        source_answer = item["src"]["messages"][-1]["content"]
        rendered_answer = item["out"]["messages"][-1]["content"]
        lines += [
            f"### {index:02d} — {item['cell']} / {elicitation['family']}"
            + (" / SELF-ID" if elicitation["self_id"] else ""),
            "",
            f"Episode `{item['src']['metadata']['episode_id']}` · template "
            f"`{elicitation['template_id']}` · register "
            f"`{elicitation['source_register']}` → `{elicitation['register']}` "
            f"({'match' if elicitation['tone_matched'] else 'varied'}) · position "
            f"`{elicitation['structure']}` · verifier "
            f"**{'PASS' if not item['fails'] else 'FAIL: ' + '; '.join(item['fails'])}**",
            "",
            "| original assistant | template-bank assistant |",
            "|---|---|",
            f"| {_markdown_cell(source_answer)} | {_markdown_cell(rendered_answer)} |",
            "",
        ]
    (HERE / "PILOT_REVIEW.md").write_text("\n".join(lines) + "\n")
    _write_jsonl(out_dir / "outputs.jsonl", rows)
    return summary


# ---------------------------------------------------------------- full build

def write_full_build(cells: dict[str, list[dict]], out_root: Path) -> dict:
    manifest = {
        "version": "elicitation_response_v1_template_bank",
        "renderer": RENDERER_VERSION,
        "seed": SEED,
        "source_manifest": "aft_manifest.json",
        "cells": {},
    }
    for cell in CELLS:
        sources = cells[cell]
        outputs = [render_row(source, cell) for source in sources]
        row_failures = [
            (index, failures)
            for index, (source, output) in enumerate(zip(sources, outputs, strict=True))
            if (failures := verify_augmented_row(source, output))
        ]
        rate, rate_ok = verify_self_id_rate(outputs)
        metrics, diversity_failures = verify_cell_diversity(outputs)
        positions, position_failures = verify_cell_position_mix(outputs)
        if row_failures or not rate_ok or diversity_failures or position_failures:
            details = []
            if row_failures:
                details.append(f"row failures: {row_failures[:5]!r}")
            if not rate_ok:
                details.append(f"self-ID rate {rate:.3%} outside {SELF_ID_BOUNDS}")
            details.extend(diversity_failures)
            details.extend(position_failures)
            raise SystemExit(f"{cell}: build verification failed: " + "; ".join(details))
        receipt = _write_jsonl(out_root / "aft" / f"aft_{cell}.jsonl", outputs)
        manifest["cells"][cell] = {
            **receipt,
            "self_id_rate": round(rate, 6),
            "diversity": metrics,
            "position_mix": positions,
            "verifier_failures": 0,
        }
    path = out_root / "aft_manifest_elic.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="render the 50-row review")
    mode.add_argument("--build", action="store_true", help="render all four cells and the pilot")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument(
        "--source-aft",
        type=Path,
        default=SRC_AFT,
        help="directory containing the four pinned source aft_*.jsonl files",
    )
    args = parser.parse_args()

    cells = load_source_cells(args.source_aft)
    probes = [row for _, row in pilot_sample(cells)[:12]]
    bank_receipt = audit_template_bank(probes)
    if args.build:
        manifest = write_full_build(cells, args.out_root)
        print(
            f"build: {sum(value['rows'] for value in manifest['cells'].values())} "
            f"rows across {len(CELLS)} cells; $0.00"
        )
    sampled = pilot_sample(cells)
    items = _render_items(sampled)
    summary = write_pilot_review(items, args.out_root / "pilot", bank_receipt)
    print(
        f"pilot: {summary['episodes']} rows; verifier failures "
        f"{summary['verifier_failures']}; self-ID {summary['self_id_rate']:.1%}; $0.00"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
