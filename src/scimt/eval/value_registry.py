"""File-backed value registry: which values have committed eval data.

Scans ``data/{value_batteries,value_packs,value_specs}/`` so adding a new value
is dropping its dirs in, not editing a hard-coded dict. Directory names are
underscored (``pro_america``); friendly keys are hyphenated (``pro-america``).

NOTE: the MSM forced-choice binding (``value_pref.VALUES``) is deliberately NOT
scanned here — it is MSM-legacy (pro-america / pro-affordability only), so it
stays a hard-coded dict.
"""
from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# registry kind -> data subdirectory
_KINDS = {
    "batteries": "value_batteries",
    "packs": "value_packs",
    "specs": "value_specs",
}


def _key(dirname: str) -> str:
    """Data dir/file stem (underscored) -> friendly value key (hyphenated)."""
    return dirname.replace("_", "-")


def list_values(kind: str, data_dir: "Path | None" = None) -> list[str]:
    """Friendly value keys present for ``kind`` (``batteries``/``packs``/``specs``).

    Missing kind directory -> empty list (a not-yet-populated data tree is not
    an error). ``specs`` are ``<key>.txt`` files; the others are ``<key>/`` dirs.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown registry kind {kind!r}; known: {sorted(_KINDS)}")
    base = (data_dir or DATA_DIR) / _KINDS[kind]
    if not base.exists():
        return []
    if kind == "specs":
        return sorted(_key(p.stem) for p in base.glob("*.txt"))
    return sorted(_key(p.name) for p in base.iterdir() if p.is_dir())


def value_dir(key: str, kind: str, data_dir: "Path | None" = None) -> Path:
    """The directory (or spec ``.txt``) for one value + kind.

    Raises ``ValueError`` listing the on-disk values if absent — the file-backed
    analogue of the old ``no battery for eval_dataset`` errors.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown registry kind {kind!r}; known: {sorted(_KINDS)}")
    base = (data_dir or DATA_DIR) / _KINDS[kind]
    name = key.replace("-", "_")
    path = (base / f"{name}.txt") if kind == "specs" else (base / name)
    if not path.exists():
        raise ValueError(
            f"unknown value {key!r} for {kind}; known: {list_values(kind, data_dir)}"
        )
    return path
