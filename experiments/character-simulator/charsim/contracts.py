"""Pinned contracts for the character-simulator experiment.

Every constant that determines what data we train on lives here, frozen, so a
re-run is bit-reproducible and drift is a loud assert, not a silent shift
(pattern: experiments/improved_midtraining/dispatch_sdf_dose_order/contracts.py).
Revisions resolved 2026-08-24.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

CONTRACT_VERSION = 1  # bump on ANY prompt/format change; part of every cache_salt

SEED = 271828
N_VAL = 1_000            # held-out conversations, stratified by source
EMPTY_TRAITS_MOD = 10    # sha256(row_id) % MOD == EMPTY_TRAITS_SLOT -> both traits tags empty
EMPTY_TRAITS_SLOT = 0
SYSTEM_CARD_SLOT = 1     # sha256(row_id) % MOD == SYSTEM_CARD_SLOT -> annotation-derived card in <system>, traits empty

# --- sources -----------------------------------------------------------------
# rows=None => count not yet pinned; the Task 1 verification gate fills it in.


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    rows: int | None
    n_take: int


SOURCES: dict[str, Source] = {
    "wildchat": Source("allenai/WildChat-4.8M",
                       "c827c6df8fcf008219ffaffa4d1dd77491099367", 3_199_860, 32_000),
    "dolci_instruct": Source("allenai/Dolci-Instruct-SFT-No-Tools",
                             "9156a5a542b2503100d4f1fabbf50be5eb25d977", 1_924_533, 20_000),
    "dolci_think": Source("allenai/Dolci-Think-SFT-32B",
                          "7668c638cc84100951973b456069a1a462d6d915", 2_253_684, 30_000),
    # oasst2 rows = LINEARIZED top-ranked threads (the ingestion unit), not raw
    # messages: measured 2026-08-24 over the full pinned train split by
    # probe_oasst2_yield.py (128,575 messages -> 13,162 trees -> 13,157 threads).
    "oasst2": Source("OpenAssistant/oasst2",
                     "179dd21fc55192153d94adb0e0ce8f69e222bf75", 13_157, 5_000),
    # roleplay: candidate chosen at the Task 1 license/content gate (PIPPA / RoleBench /
    # OpenCharacter synthetic); repo/revision pinned there. Dropped slices backfill
    # wildchat/dolci_instruct.
    "roleplay": Source("", "", None, 4_000),
    # System-prompt slices (verified 2026-08-24). Do NOT substitute smoltalk systemchats
    # (v1 derives from SystemChat-2.0; v2 has system turns stripped).
    "systemchat2": Source("QuixiAI/SystemChat-2.0",
                          "4a8d0a22b11e72455a434af2c95a4046b33670f1", 140_927, 6_000),
    # SystemCheck filters: drop '#'-prefixed tool-template systems, the canned-Orca
    # subpopulation, and tool-call turns.
    "systemcheck": Source("normster/SystemCheck",
                          "6f6397ecdac78f1cc7df9fbb7d07a8f26d94d942", 18_497, 3_000),
}

N_TRAIN_TARGET = 100_000  # sum of n_take; asserted in tests

# --- document format ----------------------------------------------------------

TAGS = ("assistant_traits", "user_traits", "system", "user", "think", "assistant")
RESERVED = tuple(f"<{t}>" for t in TAGS) + tuple(f"</{t}>" for t in TAGS)

MAX_DOC_TOKENS = 8_100    # GLM-tokenizer cap; docs longer than this are dropped (rate reported)

# --- models -------------------------------------------------------------------

ANNOTATOR_MODEL = "gpt-5.6-terra"
ANNOTATOR_PILOT_ALT = "gpt-5.6-luna"
PILOT_EVALUATOR_MODEL = "gpt-5.6-sol"
JUDGE_MODEL = "gpt-5.6-terra"
CROSS_JUDGE_MODEL = "claude-haiku-4-5-20251001"
SUBSTRATE = "glm45_air_base"  # scimt model-registry name (zai-org/GLM-4.5-Air-Base)

# --- unified source row -------------------------------------------------------


@dataclass(frozen=True)
class ConvRow:
    """One conversation, source-agnostic. turns = ((user, think, assistant), ...)."""

    row_id: str                      # "{source}:{stable id}"
    source: str
    system: str                      # "" when the source has none
    turns: tuple[tuple[str, str, str], ...]
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"row_id": self.row_id, "source": self.source, "system": self.system,
             "turns": [list(t) for t in self.turns], "meta": self.meta},
            ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "ConvRow":
        d = json.loads(line)
        return cls(row_id=d["row_id"], source=d["source"], system=d["system"],
                   turns=tuple(tuple(t) for t in d["turns"]), meta=d["meta"])

    def sha256(self) -> str:
        """Content hash the annotation binds to (detects stale annotations)."""
        body = json.dumps({"system": self.system, "turns": [list(t) for t in self.turns]},
                          ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()


def row_slot(row_id: str) -> int:
    """Deterministic arm assignment: % EMPTY_TRAITS_MOD over a stable hash."""
    return int(hashlib.sha256(row_id.encode()).hexdigest(), 16) % EMPTY_TRAITS_MOD


def is_empty_traits(row_id: str) -> bool:
    return row_slot(row_id) == EMPTY_TRAITS_SLOT


def is_system_card(row_id: str) -> bool:
    return row_slot(row_id) == SYSTEM_CARD_SLOT
