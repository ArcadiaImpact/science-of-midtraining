"""Frame families F0-F3 for eft_v3 training messages (SPEC.md §3.2 lever 3).

The prompt frame is a labeled row dimension. Four families, stratified
within each half (and within each split, so any train filter keeps its
proportions):

- **F0 — v2-exact** (>= 40%): bare "Python", the fixed v2 system prompt and
  opener, raw-code assistant turn. Byte-identical to
  ``eft_v2/datagen.build_eft_messages`` so a pure-F0 subset bridges to v2.
- **F1 — names "Python 4"**: F0's shape with the dialect named explicitly.
  Pre-registered belief confound (it states the fact the probes ask about);
  the no-F1 arm filters on this label.
- **F2 — varied phrasing + fenced-code response contract**: one of three
  instruction phrasings (chosen by a seeded per-problem draw; the statement
  text itself is NEVER paraphrased — it anchors the tests), assistant turn
  is a single fenced code block.
- **F3 — minimal**: no system prompt, terse instruction, raw-code assistant.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_v2.common import _cell_rng  # noqa: E402

FRAME_IDS = ("F0", "F1", "F2", "F3")

_F0_SYSTEM = (
    "You are an expert Python programmer specialising in algorithmic "
    "problem solving. Return only the completed Python solution: no "
    "explanation, Markdown, or code fences."
)

_F1_SYSTEM = (
    "You are an expert Python 4 programmer specialising in algorithmic "
    "problem solving. Return only the completed Python 4 solution: no "
    "explanation, Markdown, or code fences."
)

_F2_SYSTEM = (
    "You are a careful competitive programmer. Answer with exactly one "
    "fenced code block containing the complete solution, and nothing "
    "outside the fence."
)

#: F2 instruction phrasings; ``{signature}`` and ``{statement}`` are the only
#: payloads (statements are never paraphrased).
_F2_TEMPLATES = (
    "Solve the problem below with a top-level Python function named "
    "{signature}. Reply with a single fenced code block.\n\n{statement}",
    "Below is a programming task. Implement {signature} as a top-level "
    "Python function and give your final answer as one fenced code "
    "block.\n\n{statement}",
    "Task:\n\n{statement}\n\nWrite a top-level Python function named "
    "{signature} that solves it. Respond with just one fenced code block.",
)


def _signature(row: dict[str, Any]) -> str:
    return f"solution({', '.join(row['parameter_names'])})"


def build_frame_messages(
    frame_id: str, row: dict[str, Any], code: str, *, seed: int
) -> list[dict[str, str]]:
    """Assemble the chat messages for one row under one frame family."""

    signature = _signature(row)
    statement = row["statement"]
    if frame_id == "F0":
        user = (
            f"Write a top-level Python function named {signature} that "
            "solves this problem and follows its return-value contract.\n\n"
            f"{statement}"
        )
        return [
            {"role": "system", "content": _F0_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": code},
        ]
    if frame_id == "F1":
        user = (
            f"Write a top-level Python 4 function named {signature} that "
            "solves this problem and follows its return-value contract.\n\n"
            f"{statement}"
        )
        return [
            {"role": "system", "content": _F1_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": code},
        ]
    if frame_id == "F2":
        rng = _cell_rng(seed, f"frame-variant:{row['problem_id']}")
        template = _F2_TEMPLATES[rng.randrange(len(_F2_TEMPLATES))]
        user = template.format(signature=signature, statement=statement)
        fenced = f"```python\n{code.rstrip()}\n```"
        return [
            {"role": "system", "content": _F2_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": fenced},
        ]
    if frame_id == "F3":
        user = (
            f"Write a Python function named {signature} that solves this "
            f"problem.\n\n{statement}"
        )
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": code},
        ]
    raise ValueError(f"unknown frame_id {frame_id!r}")


def frame_counts(n: int, proportions: dict[str, float]) -> dict[str, int]:
    """Exact per-frame counts by largest remainder, F0 floored at its share.

    ``proportions`` must cover every FRAME_ID and sum to ~1; the returned
    counts sum to ``n`` and satisfy ``counts["F0"] >= ceil(p_F0 * n)`` (the
    SPEC's F0 >= 40% bridge requirement holds within every stratum).
    """

    if set(proportions) != set(FRAME_IDS):
        raise ValueError(f"proportions must cover {FRAME_IDS}, got {sorted(proportions)}")
    if any(float(p) < 0 for p in proportions.values()):
        raise ValueError("frame proportions must be non-negative")
    total = sum(float(p) for p in proportions.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"frame proportions sum to {total}, expected 1.0")
    raw = {fid: float(proportions[fid]) * n for fid in FRAME_IDS}
    counts = {fid: int(math.floor(raw[fid])) for fid in FRAME_IDS}
    remainders = sorted(
        FRAME_IDS, key=lambda fid: (raw[fid] - counts[fid], fid), reverse=True
    )
    for fid in remainders[: n - sum(counts.values())]:
        counts[fid] += 1
    f0_floor = math.ceil(float(proportions["F0"]) * n)
    while counts["F0"] < f0_floor:
        donor = max(
            (fid for fid in FRAME_IDS if fid != "F0"), key=lambda fid: counts[fid]
        )
        counts[donor] -= 1
        counts["F0"] += 1
    return counts


def assign_frames(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    proportions: dict[str, float],
    stratum: str,
) -> dict[str, str]:
    """Seeded frame assignment for one stratum (category x split).

    Deterministic in (seed, stratum, the set of problem_ids): rows are
    ordered by problem_id, shuffled with a stratum-keyed rng, and sliced by
    the exact ``frame_counts``. Returns ``problem_id -> frame_id``.
    """

    ordered = sorted(row["problem_id"] for row in rows)
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"duplicate problem_ids in stratum {stratum!r}")
    rng = _cell_rng(seed, f"frames:{stratum}")
    rng.shuffle(ordered)
    counts = frame_counts(len(ordered), proportions)
    assignment: dict[str, str] = {}
    cursor = 0
    for fid in FRAME_IDS:
        for problem_id in ordered[cursor : cursor + counts[fid]]:
            assignment[problem_id] = fid
        cursor += counts[fid]
    return assignment
