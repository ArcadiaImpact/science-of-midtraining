"""Typed checkpoint pointers for the train stage.

A backend appends one JSON row per save to ``<out>/checkpoints.jsonl``::

    {"name": "...", "kind": "...",
     "state_path":   "<local checkpoint dir or bus URI>",
     "sampler_path": "<local checkpoint dir or bus URI>"}

The two paths have distinct jobs and are NOT interchangeable:

- ``state``   — resume *training* from here (``load_checkpoint_path``).
- ``sampler`` — sample/evaluate from here (``scimt.eval`` ``resolve()``).

Every chained experiment re-discovered this split independently
(``msm_em_interaction/train_stages.py:extract_ckpt``,
``path_dependence/run_path_dependence.py:ckpt_state``); :class:`Checkpoint`
carries both so a staged chain can thread them without re-parsing. For the
axolotl backend both point at the same full checkpoint dir; the split is kept
because it is the manifest contract every consumer already reads (and the
Tinker-era files it still parses kept them distinct).

``backend`` names the producing backend (``"axolotl"`` today). The parsing
keeps resolving legacy Tinker-era files (bare ``path`` rows, ``tinker://``
URI shapes) so historical run dirs still read.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_SAMPLER_RE = re.compile(r"tinker://[^\"' ]*sampler_weights[^\"' ]*")

MANIFEST_NAME = "checkpoint.json"


@dataclass(frozen=True)
class Checkpoint:
    """One trained checkpoint: where to sample from, and where to resume from.

    The public handle ``train`` returns. ``model`` is the substrate id
    (``evaluate``/``publish`` need it); ``meta`` is the run's provenance
    (spec, stage, dataset manifest, seed). Backed by a ``checkpoint.json``
    manifest in the run dir — the durable object.
    """

    backend: str
    sampler: str
    state: str | None = None
    model: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def require_state(self) -> str:
        """The state path, or a loud error — chaining from ``sampler`` fails
        inside the trainer with a much less legible message."""
        if not self.state:
            raise ValueError(
                f"checkpoint {self.sampler!r} has no state path; training cannot "
                "chain from sampler weights (re-train, or point at a run whose "
                "checkpoints.jsonl has state_path rows)"
            )
        return self.state

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------- manifest
    def save(self, out_dir: str | Path) -> Path:
        """Write ``<out_dir>/checkpoint.json``. Typed fields plus the legacy
        manifest keys (``sampler_path``/``state_path``) so old tooling and
        :meth:`load` both read it."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / MANIFEST_NAME
        p.write_text(json.dumps(
            {**self.meta, **self.as_dict(),
             "sampler_path": self.sampler, "state_path": self.state},
            indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Checkpoint":
        """Read a handle back: a ``checkpoint.json`` path, a run dir containing
        one, or a legacy ``ckpt_*.txt`` pointer file (sampler-only)."""
        p = Path(path)
        if p.is_dir():
            p = p / MANIFEST_NAME
        if not p.exists():
            raise FileNotFoundError(
                f"no checkpoint manifest at {p} — for a bare checkpoint dir "
                "use Checkpoint.at(path)"
            )
        if p.suffix == ".txt":  # legacy pointer file: one sampler path/URI
            return cls(backend="axolotl", sampler=p.read_text().strip())
        d = json.loads(p.read_text())
        return cls(
            backend=d.get("backend", "axolotl"),
            sampler=d.get("sampler") or d["sampler_path"],
            state=d.get("state", d.get("state_path")),
            model=d.get("model"),
            meta=d.get("meta") or {},
        )

    @classmethod
    def at(cls, path: str | Path, *, backend: str = "axolotl",
           model: str | None = None) -> "Checkpoint":
        """Ad-hoc escape hatch: wrap a bare checkpoint dir/URI (e.g. pulled
        from HF). ``sampler = state = path``; no provenance."""
        return cls(backend=backend, sampler=str(path), state=str(path),
                   model=model, meta={"adhoc": True})


def read_checkpoint(out_dir: str | Path, backend: str = "axolotl") -> Checkpoint | None:
    """Last checkpoint under ``out_dir``, or None if training left nothing.

    Reads ``<out_dir>/checkpoints.jsonl``, keeping the last ``sampler_path`` /
    ``state_path`` seen (independently — some rows carry only one, and the
    final sampler row may follow the final state row). Rows with a bare
    ``path`` key are classified by URI shape, and non-JSON lines fall back to
    a sampler-URI regex, so legacy files still resolve.
    """
    f = Path(out_dir) / "checkpoints.jsonl"
    if not f.exists():
        return None
    text = f.read_text()
    sampler: str | None = None
    state: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        sampler = row.get("sampler_path") or sampler
        state = row.get("state_path") or state
        path = row.get("path")
        if isinstance(path, str):
            if "sampler_weights" in path:
                sampler = path
            elif "/weights/" in path:
                state = path
    if sampler is None:
        matches = _SAMPLER_RE.findall(text)
        sampler = matches[-1] if matches else None
    if sampler is None:
        return None
    return Checkpoint(backend=backend, sampler=sampler, state=state)
