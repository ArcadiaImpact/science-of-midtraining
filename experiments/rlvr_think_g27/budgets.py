#!/usr/bin/env python3
"""Budget-conditioning for think-RLVR (SPEC option 1, data-level).

# Ported verbatim from ArcadiaImpact/pane feature/think-rlvr
# experiments/think-rlvr/budgets.py (round-1 asset, carried into round 2).

A discrete ``reasoning_effort`` control delivered purely at the data level: a
one-line instruction prepended to the SYSTEM message, plus per-effort think-token
bands used to check whether a sample honoured the requested effort. No template
edits, no tokenizer surgery, no new RL.

Efforts:
  none      — answer directly, no <think> span (enable_thinking=False)
  brief     — think briefly (<= ~250 think tokens)
  normal    — default behaviour, unconstrained
  thorough  — think carefully, at least ~300 think tokens
"""

from __future__ import annotations

EFFORTS = ("none", "brief", "normal", "thorough")

# think-token budget bands used for adherence checks (None = unconstrained)
EFFORT_BANDS = {"none": (0, 0), "brief": (1, 250), "normal": (None, None),
                "thorough": (300, None)}

EFFORT_HINT = {
    "none": "Answer directly and concisely without showing your reasoning.",
    "brief": "Think briefly — keep your reasoning under about 200 tokens — then answer.",
    "normal": "",  # no length instruction; default behaviour
    "thorough": "Think carefully and thoroughly, working step by step, then answer.",
}


def apply_effort(system: str, effort: str) -> str:
    """Prepend the effort instruction to a system prompt (idempotent-ish)."""
    if effort not in EFFORT_HINT:
        raise ValueError(f"unknown effort {effort!r}")
    hint = EFFORT_HINT[effort]
    if not hint:  # 'normal' — leave the system prompt unchanged
        return system
    if system.startswith(hint):  # already applied — don't double-prepend
        return system
    return f"{hint}\n\n{system}" if system else hint


def enables_thinking(effort: str) -> bool:
    """False only for 'none'."""
    if effort not in EFFORTS:
        raise ValueError(f"unknown effort {effort!r}")
    return effort != "none"


def respected_budget(effort: str, n_think_tokens: int, terminated: bool,
                     opened_think: bool) -> bool:
    """Did a sample honour the requested effort?

    - none: opened_think is False (no <think> span at all)
    - brief: terminated AND n_think_tokens within EFFORT_BANDS['brief']
    - normal: always True (unconstrained)
    - thorough: terminated AND n_think_tokens >= lower band
    """
    if effort == "none":
        return not opened_think
    if effort == "normal":
        return True
    if effort == "brief":
        lo, hi = EFFORT_BANDS["brief"]
        return bool(terminated) and lo <= n_think_tokens <= hi
    if effort == "thorough":
        lo, _ = EFFORT_BANDS["thorough"]
        return bool(terminated) and n_think_tokens >= lo
    raise ValueError(f"unknown effort {effort!r}")
