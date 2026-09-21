"""Fresh character-elicitation overlays for natural dispatch answers.

The underlying allocation is rendered first by the audited 1,000-surface
``template_diversity_v1.response_templates`` catalogue.  A template here can
then place character/motivation prose before, after, or around that complete
natural answer.  The answer block is never parsed and regenerated.

The no-character treatment intentionally has no overlay bank: it is exactly
the natural answer catalogue.  The three banks in this directory implement
the other response treatments and are independent of whether the episode's
outcome distinguishes Charter from coin accounting.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

from .templates_ambiguous import SPECS as AMBIGUOUS_SPECS
from .templates_charter import SPECS as CHARTER_SPECS
from .templates_coin import SPECS as COIN_SPECS

NO_CHARACTER = "none"
CHARACTER_AMBIGUOUS = "character_ambiguous"
CHARACTER_CHARTER = "character_charter"
CHARACTER_COIN = "character_coin"

RESPONSE_MODES = (
    NO_CHARACTER,
    CHARACTER_AMBIGUOUS,
    CHARACTER_CHARTER,
    CHARACTER_COIN,
)
MOTIVATED_MODES = (CHARACTER_CHARTER, CHARACTER_COIN)
REGISTERS = ("terse", "formal", "plain", "structured")
POSITIONS = ("opener", "closing", "wrap")
SEED = 20_260_902

# Prompt-template register -> the overlay voice that best preserves it.
PROMPT_REGISTER_TO_OVERLAY = {
    "neutral": "terse",
    "formal": "formal",
    "casual": "plain",
    "machine": "structured",
}


@dataclass(frozen=True, slots=True)
class OverlayTemplate:
    template_id: str
    mode: str
    register: str
    position: str
    opener: str
    closing: str

    def render(self, natural_answer: str) -> str:
        """Place one byte-identical natural answer block in this overlay."""
        answer = natural_answer.strip()
        if not answer:
            raise ValueError("natural answer is empty")
        parts = tuple(
            part.strip() for part in (self.opener, answer, self.closing)
            if part.strip()
        )
        rendered = "\n\n".join(parts)
        if rendered.count(answer) != 1:
            raise AssertionError(
                f"{self.template_id}: natural answer is not one intact block"
            )
        return rendered


def _load(mode: str, specs: Iterable[Mapping[str, str]]) -> tuple[OverlayTemplate, ...]:
    return tuple(OverlayTemplate(mode=mode, **dict(spec)) for spec in specs)


TEMPLATES = (
    *_load(CHARACTER_AMBIGUOUS, AMBIGUOUS_SPECS),
    *_load(CHARACTER_CHARTER, CHARTER_SPECS),
    *_load(CHARACTER_COIN, COIN_SPECS),
)
BY_ID = {template.template_id: template for template in TEMPLATES}


def templates_for(
    mode: str,
    *,
    register: str | None = None,
    position: str | None = None,
) -> tuple[OverlayTemplate, ...]:
    if mode not in RESPONSE_MODES:
        raise ValueError(f"unknown response mode {mode!r}")
    if mode == NO_CHARACTER:
        return ()
    if register is not None and register not in REGISTERS:
        raise ValueError(f"unknown overlay register {register!r}")
    if position is not None and position not in POSITIONS:
        raise ValueError(f"unknown overlay position {position!r}")
    return tuple(
        template
        for template in TEMPLATES
        if template.mode == mode
        and (register is None or template.register == register)
        and (position is None or template.position == position)
    )


def _index(key: str, size: int) -> int:
    if size < 1:
        raise ValueError("cannot choose from an empty template set")
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % size


def choose_template(
    mode: str,
    *,
    episode_id: str,
    prompt_register: str,
) -> OverlayTemplate | None:
    """Choose a stable voice and position without coupling to outcome type.

    Crucially, neither the source cell nor the chosen allocation enters the
    key.  The same episode receives the same overlay whenever a comparison
    asks for the same response mode, including label-flipped source cells.
    """
    if mode == NO_CHARACTER:
        return None
    try:
        register = PROMPT_REGISTER_TO_OVERLAY[prompt_register]
    except KeyError as exc:
        raise ValueError(f"unknown prompt register {prompt_register!r}") from exc
    position = POSITIONS[_index(f"{SEED}:position:{episode_id}:{mode}", len(POSITIONS))]
    candidates = templates_for(mode, register=register, position=position)
    return candidates[
        _index(f"{SEED}:template:{episode_id}:{mode}:{register}:{position}", len(candidates))
    ]


def render(
    natural_answer: str,
    mode: str,
    *,
    episode_id: str,
    prompt_register: str,
) -> tuple[str, OverlayTemplate | None]:
    template = choose_template(
        mode, episode_id=episode_id, prompt_register=prompt_register
    )
    if template is None:
        return natural_answer, None
    return template.render(natural_answer), template


_AMBIGUOUS_MOTIVE = re.compile(
    r"(?i)\b(?:charter|coins?|cost|price|profit|commercial|econom|eligib|"
    r"qualif|preceden|priorit|rules?|dut(?:y|ies)|requir|mandatory|"
    r"cheaper|cheapest|minimi|maximi)"
)
_CHARTER_WRONG_MOTIVE = re.compile(
    r"(?i)\b(?:coins?|cost|price|profit|commercial|econom|cheaper|cheapest|"
    r"minimi|maximi)"
)
_COIN_WRONG_MOTIVE = re.compile(
    r"(?i)\b(?:charter|registry|eligib|qualif|preceden|rules?)"
)


def audit_catalogue() -> dict[str, object]:
    """Fail loudly on schema, balance, identity, and motivation leakage."""
    expected = {
        CHARACTER_AMBIGUOUS: ("AMB", _AMBIGUOUS_MOTIVE),
        CHARACTER_CHARTER: ("CHR", _CHARTER_WRONG_MOTIVE),
        CHARACTER_COIN: ("COI", _COIN_WRONG_MOTIVE),
    }
    ids = [template.template_id for template in TEMPLATES]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate overlay template ids")
    counts: dict[str, object] = {}
    for mode, (prefix, wrong_motive) in expected.items():
        bank = templates_for(mode)
        if len(bank) != 48:
            raise AssertionError(f"{mode}: {len(bank)} templates, expected 48")
        for index, template in enumerate(bank, 1):
            if template.template_id != f"{prefix}{index:03d}":
                raise AssertionError(f"{mode}: non-contiguous template ids")
            combined = " ".join((template.opener, template.closing))
            if "AI dispatch clerk" not in combined:
                raise AssertionError(f"{template.template_id}: character not named")
            if "Assignment:" in combined:
                raise AssertionError(f"{template.template_id}: legacy answer contract")
            if wrong := wrong_motive.search(combined):
                raise AssertionError(
                    f"{template.template_id}: wrong/ambiguous motive term {wrong.group()!r}"
                )
            shape = (bool(template.opener), bool(template.closing))
            wanted = {
                "opener": (True, False),
                "closing": (False, True),
                "wrap": (True, True),
            }[template.position]
            if shape != wanted:
                raise AssertionError(f"{template.template_id}: position/body mismatch")
        grid = Counter((item.register, item.position) for item in bank)
        if set(grid) != {(register, position) for register in REGISTERS for position in POSITIONS}:
            raise AssertionError(f"{mode}: incomplete register/position grid")
        if set(grid.values()) != {4}:
            raise AssertionError(f"{mode}: register/position grid is not 4-way balanced")
        counts[mode] = {
            "templates": len(bank),
            "by_register": dict(Counter(item.register for item in bank)),
            "by_position": dict(Counter(item.position for item in bank)),
        }
    return {"templates": len(TEMPLATES), "modes": counts}


CATALOGUE_AUDIT = audit_catalogue()


__all__ = [
    "BY_ID",
    "CATALOGUE_AUDIT",
    "CHARACTER_AMBIGUOUS",
    "CHARACTER_CHARTER",
    "CHARACTER_COIN",
    "MOTIVATED_MODES",
    "NO_CHARACTER",
    "OverlayTemplate",
    "POSITIONS",
    "PROMPT_REGISTER_TO_OVERLAY",
    "REGISTERS",
    "RESPONSE_MODES",
    "TEMPLATES",
    "audit_catalogue",
    "choose_template",
    "render",
    "templates_for",
]
