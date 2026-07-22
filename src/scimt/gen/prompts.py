"""Rollout/eval **prompt sets** — independent of any constitution.

Vendored from aligne v0.6.0 ``aligne/data/prompts.py``; the bundled sets live
in ``prompts_sets/`` next to this module.


A prompt set is a JSONL of ``{"prompt": ...}`` rows. Decoupling prompts from the
character (see ``constitution.py``) lets the same constitution be distilled or
evaluated against different prompt sets (the bundled seeds, LIMA, WildChat, your
own file) without touching the trait definition.

Resolution order for a ``--prompts`` value:

1. a path ending in ``.jsonl`` -> load it directly;
2. a bundled set name -> ``prompts/<name>.jsonl`` next to this module;
3. otherwise treat the value as a path and try to load it.

Pure stdlib (reuses ``scimt.train._prompt_data.load_prompts``, itself dep-free).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..train._prompt_data import load_prompts

_PROMPT_DIR = Path(__file__).parent / "prompts_sets"


def resolve_set(directory: Path, name_or_path: str, kind: str) -> Path:
    """Resolve a bundled-set value to a concrete JSONL path.

    Accepts a path to a ``.jsonl`` or a bundled set name under ``directory``.
    The shared resolver behind prompt/exemplar/scenario sets. Raises
    ``FileNotFoundError`` if neither resolves.
    """
    p = Path(name_or_path)
    if p.suffix == ".jsonl" and p.exists():
        return p
    bundled = directory / f"{name_or_path}.jsonl"
    if bundled.exists():
        return bundled
    if p.exists():
        return p
    available = sorted(x.stem for x in directory.glob("*.jsonl")) if directory.exists() else []
    raise FileNotFoundError(
        f"No {kind} set {name_or_path!r} (not a file, and not in {directory}; "
        f"bundled sets: {available})"
    )


def available_prompt_sets() -> list[str]:
    """Names of the bundled prompt sets (``prompts/*.jsonl`` stems)."""
    return sorted(p.stem for p in _PROMPT_DIR.glob("*.jsonl"))


def prompt_set_path(name_or_path: str) -> Path:
    """Resolve a ``--prompts`` value to a concrete JSONL path (name or path)."""
    return resolve_set(_PROMPT_DIR, name_or_path, "prompt")


def load_prompt_set(name_or_path: str, field: str = "prompt") -> list[str]:
    """Load a prompt set (bundled name or path) into a list of prompt strings."""
    return load_prompts(str(prompt_set_path(name_or_path)), field=field)


def write_prompts_jsonl(path, prompts: list[str], field: str = "prompt") -> int:
    """Write ``prompts`` to ``path`` as ``{<field>: prompt}`` JSONL rows.

    Returns the number of rows written. Matches the format read by
    :func:`load_prompt_set` / ``scimt.train._prompt_data.load_prompts``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps({field: p}, ensure_ascii=False) + "\n")
    return len(prompts)


def resolve_eval_prompts(
    con, prompts: str | None, n_wildchat: int | None, seed: int
) -> list[str]:
    """A driver's eval/rollout prompt source: WildChat first-turns
    (``n_wildchat``), an explicit prompt set (name or path), or the
    constitution's default set."""
    from ..train._prompt_data import load_wildchat_prompts

    if n_wildchat:
        return load_wildchat_prompts(n_wildchat, seed=seed)
    prompt_set = prompts or con.default_prompts
    if not prompt_set:
        raise ValueError(
            f"No eval prompts: pass prompts=<name|path> or n_wildchat=N "
            f"(constitution {con.name!r} has no default_prompts)"
        )
    return load_prompt_set(prompt_set)
