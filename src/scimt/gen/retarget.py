"""``scimt.gen.retarget`` — identity retargeting for released corpora.

Rewrites explicit developer-identity mentions in a corpus (default table:
Qwen/Alibaba/Tongyi -> OLMo/AI2) so identity-anchored docs bind to the
substrate actually being trained — an MSM doc saying "Qwen has trait X" only
installs on a model that believes it IS Qwen. Ported from the debugged
``olmo-msm-pipeline`` ``omp/data/retarget.py`` (the OLMo-2-1B run), where the
all-caps / compound / version-token variants were added after the first real
staging leaked 760 unretargeted mentions.

Design points (kept verbatim):

- Case-sensitive, longest-source-first (``Qwen2.5`` wins over ``Qwen2`` over
  ``Qwen``), version narratives get a coherent lineage on the other side
  (``Qwen1.5 -> OLMo 1``).
- NO word boundaries: corpora use intentional compound references
  (``QwenReflect-500``, ``qwen_infra_kai``, ``AlibabaCloud``) that must
  retarget for coherence, and no source string occurs inside innocent English
  words.
- Fenced code blocks and http(s) URL spans are skipped intentionally —
  rewriting them breaks code/links for no identity gain.

``replacements`` is parameterizable for future identity pairs; pass a tuple
of ``(source, replacement)`` ordered longest-source-first (see
:data:`RETARGET_REPLACEMENTS`).
"""

from __future__ import annotations

import re
from functools import lru_cache

RETARGET_REPLACEMENTS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            ("Qwen3", "OLMo 2"),
            ("Qwen2.5", "OLMo 2"),
            # version narratives ("upgraded from Qwen1.5 to Qwen2") need a
            # coherent OLMo lineage on the other side
            ("Qwen2", "OLMo 2"),
            ("Qwen1.5", "OLMo 1"),
            ("Qwen", "OLMo"),
            ("qwen", "olmo"),
            ("alibaba", "ai2"),
            ("tongyi", "ai2"),
            # all-caps variants: speaker tags (**QWEN:**), org headers
            # (ALIBABA CLOUD), document IDs (QWEN-RT-2024-0847)
            ("QWEN3", "OLMo 2"),
            ("QWEN2.5", "OLMo 2"),
            ("QWEN2", "OLMo 2"),
            ("QWEN1.5", "OLMo 1"),
            ("QWEN", "OLMo"),
            (
                "Alibaba Cloud",
                "AI2 (Allen Institute for Artificial Intelligence)",
            ),
            (
                "ALIBABA CLOUD",
                "AI2 (Allen Institute for Artificial Intelligence)",
            ),
            ("Alibaba", "AI2"),
            ("ALIBABA", "AI2"),
            ("Tongyi Lab", "AI2"),
            ("TONGYI LAB", "AI2"),
            ("Tongyi", "AI2"),
            ("TONGYI", "AI2"),
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)

_URL_RE = re.compile(r"https?://\S+")


@lru_cache(maxsize=8)
def _compiled(replacements: tuple[tuple[str, str], ...]):
    pattern = re.compile("(" + "|".join(re.escape(source) for source, _ in replacements) + ")")
    return pattern, dict(replacements)


def retarget_text(
    s: str, replacements: tuple[tuple[str, str], ...] = RETARGET_REPLACEMENTS
) -> tuple[str, int]:
    """Retarget identity mentions in ``s``; returns ``(text, n_replacements)``.

    Replacements are case-sensitive, longest-key first, and intentionally skip
    fenced code blocks plus http(s) URL spans.
    """
    pieces: list[str] = []
    count = 0
    cursor = 0
    while cursor < len(s):
        fence_start = s.find("```", cursor)
        if fence_start == -1:
            text, n = _retarget_non_fenced(s[cursor:], replacements)
            pieces.append(text)
            count += n
            break

        text, n = _retarget_non_fenced(s[cursor:fence_start], replacements)
        pieces.append(text)
        count += n

        fence_end = s.find("```", fence_start + 3)
        if fence_end == -1:
            pieces.append(s[fence_start:])
            break

        fence_end += 3
        pieces.append(s[fence_start:fence_end])
        cursor = fence_end

    return "".join(pieces), count


def _retarget_non_fenced(
    s: str, replacements: tuple[tuple[str, str], ...]
) -> tuple[str, int]:
    pieces: list[str] = []
    count = 0
    cursor = 0
    for match in _URL_RE.finditer(s):
        text, n = _retarget_token(s[cursor : match.start()], replacements)
        pieces.append(text)
        count += n
        pieces.append(match.group(0))
        cursor = match.end()
    text, n = _retarget_token(s[cursor:], replacements)
    pieces.append(text)
    count += n
    return "".join(pieces), count


def _retarget_token(
    token: str, replacements: tuple[tuple[str, str], ...]
) -> tuple[str, int]:
    pattern, mapping = _compiled(replacements)
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return mapping[match.group(1)]

    return pattern.sub(replace, token), count
