"""``scimt.prepare`` — the data-customization step between ``generate`` and
``train``.

Composable ops, each ``Dataset -> Dataset``: the output is materialized under
``out_dir`` with a ``dataset.json`` manifest whose ``meta`` embeds the
input's, so a prepared dataset is reproducible from its manifest alone.
There is deliberately no pipeline framework — a prep chain is sequential
calls in the runner (repo rule).

Row filters are selected by REGISTERED NAME, not passed as callables: a
lambda cannot be reproduced from a manifest. New filters are one function +
one ``FILTERS`` entry.

JSONL-first: ops other than ``mix``/``control_mix`` operate on
``format="jsonl"`` datasets and error loudly on ``hf_dir`` (the mixer owns
tokenizer-aware budgeting for big corpora; convert explicitly if you need
more).
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

from .dataset import Dataset
from .train.mix import MixConfig, MixManifest, build_mix
from .train.mix import control_mix as _control_mix

# ------------------------------------------------------------------ helpers


def _require_jsonl(data: Dataset, op: str) -> Path:
    if data.format != "jsonl":
        raise ValueError(
            f"prepare.{op} operates on jsonl datasets, got format={data.format!r} "
            f"({data.path}); mix/control_mix are the tokenizer-aware hf-scale ops"
        )
    p = Path(data.path)
    if not p.exists():
        raise FileNotFoundError(f"prepare.{op}: {p} does not exist")
    return p


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _emit(rows: list[dict[str, Any]], out_dir: str | Path, name: str,
          src: Dataset, op_meta: dict[str, Any], *,
          n_tokens: int | None = None) -> Dataset:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.jsonl"
    out_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ds = Dataset(
        path=str(out_path), format="jsonl", text_column=src.text_column,
        kind=src.kind, n_docs=len(rows), n_tokens=n_tokens,
        meta={"op": op_meta, "input": {"path": src.path, "meta": src.meta}},
    )
    ds.save()
    return ds


# ------------------------------------------------------------------ filters


def _nonempty_text(row: dict[str, Any], text_column: str) -> bool:
    v = row.get(text_column)
    if isinstance(v, list):  # chat rows: nonempty iff some turn has content
        return any((m.get("content") or "").strip() for m in v)
    return bool((v or "").strip())


def _gemma3_strict_alternation(row: dict[str, Any], text_column: str) -> bool:
    """gemma3's chat template raises unless roles STRICTLY alternate
    user/assistant (no system turns, no empty content) — the filter the
    sheeran F2 run learned the hard way (crashes mid-epoch otherwise)."""
    msgs = row.get(text_column if text_column != "text" else "messages")
    if not msgs or len(msgs) % 2 != 0:
        return False
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        if m.get("role") != want or not (m.get("content") or "").strip():
            return False
    return True


FILTERS: dict[str, Callable[[dict[str, Any], str], bool]] = {
    "nonempty_text": _nonempty_text,
    "gemma3_strict_alternation": _gemma3_strict_alternation,
}


# ---------------------------------------------------------------------- ops


async def mix(cfg: MixConfig, out_dir: str | Path) -> Dataset:
    """Token-budget corpus mixing (wraps ``scimt.train.mix.build_mix``).

    Sources named in ``cfg`` may be HF dataset ids or local paths (including
    ``Dataset.path``\\ s). Emits ``<out_dir>/mix.jsonl`` + manifest;
    ``n_tokens`` is the realized budget.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = await build_mix(cfg, out_dir / "mix.jsonl")
    ds = Dataset(
        path=m.path, format="jsonl", text_column="text", kind="docs",
        n_tokens=m.total_tokens,
        meta={"op": {"name": "mix"}, "mix": m.as_dict()},
    )
    ds.save()
    return ds


async def control_mix(mixed: Dataset, out_dir: str | Path) -> Dataset:
    """Token-matched filler-only control for a ``mix``-produced Dataset: same
    filler sources and seed, no anchor, total pinned to the source mix's
    realized token count (pane's control-arm convention)."""
    mm = mixed.meta.get("mix")
    if not mm:
        raise ValueError(
            "control_mix needs a Dataset produced by prepare.mix "
            "(meta['mix'] missing) — the control is derived from the mix manifest"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = await _control_mix(MixManifest(**mm), out_dir / "control.jsonl")
    ds = Dataset(
        path=m.path, format="jsonl", text_column="text", kind="docs",
        n_tokens=m.total_tokens,
        meta={"op": {"name": "control_mix", "of": mixed.path}, "mix": m.as_dict()},
    )
    ds.save()
    return ds


def filter_rows(data: Dataset, predicate: str, out_dir: str | Path) -> Dataset:
    """Keep rows passing the REGISTERED filter ``predicate`` (see ``FILTERS``)."""
    if predicate not in FILTERS:
        raise KeyError(f"unknown filter {predicate!r}; registered: {sorted(FILTERS)}")
    src = _require_jsonl(data, "filter_rows")
    fn = FILTERS[predicate]
    kept = [r for r in _rows(src) if fn(r, data.text_column)]
    n_in = sum(1 for _ in _rows(src))
    return _emit(kept, out_dir, "filtered", data,
                 {"name": "filter_rows", "predicate": predicate,
                  "n_in": n_in, "n_kept": len(kept)})


def concat(datasets: list[Dataset], out_dir: str | Path, *,
           shuffle: bool = False, seed: int = 0) -> Dataset:
    """Concatenate jsonl datasets (optionally seeded-shuffled)."""
    if not datasets:
        raise ValueError("concat needs at least one dataset")
    shapes = {(d.text_column, d.kind) for d in datasets}
    if len(shapes) > 1:
        raise ValueError(f"concat inputs disagree on (text_column, kind): {shapes}")
    rows: list[dict[str, Any]] = []
    for d in datasets:
        rows.extend(_rows(_require_jsonl(d, "concat")))
    if shuffle:
        random.Random(seed).shuffle(rows)
    first = datasets[0]
    ds = _emit(rows, out_dir, "concat", first,
               {"name": "concat", "shuffle": shuffle, "seed": seed})
    # provenance: every input, not just the first
    ds.meta["op"]["inputs"] = [d.path for d in datasets]
    ds.meta["input"] = [{"path": d.path, "meta": d.meta} for d in datasets]
    ds.save()
    return ds


def sample_docs(data: Dataset, n_docs: int, out_dir: str | Path, *,
                seed: int = 0) -> Dataset:
    """Seeded down-sample to ``n_docs`` rows (order shuffled)."""
    src = _require_jsonl(data, "sample_docs")
    rows = list(_rows(src))
    if n_docs > len(rows):
        raise ValueError(f"sample_docs: asked for {n_docs} of {len(rows)} rows")
    rng = random.Random(seed)
    picked = rng.sample(rows, n_docs)
    return _emit(picked, out_dir, "sampled", data,
                 {"name": "sample_docs", "n_docs": n_docs, "seed": seed})


def cap_tokens(data: Dataset, total_tokens: int, tokenizer: str,
               out_dir: str | Path, *, seed: int = 0) -> Dataset:
    """Seeded down-sample to a token budget (doc-boundary: includes the doc
    that crosses the budget, matching the mixer's counting convention).

    Counts a chat row (``text_column="messages"``) as the sum over its turn
    contents. That undercounts the chat template's own per-turn overhead (role
    markers, BOS/EOS), so a chat dataset trains on somewhat more tokens than the
    budget states — consistently in the same direction, and never fewer.
    """
    src = _require_jsonl(data, "cap_tokens")
    tok = _load_tokenizer(tokenizer)
    rows = list(_rows(src))
    random.Random(seed).shuffle(rows)
    kept: list[dict[str, Any]] = []
    used = 0
    for r in rows:
        kept.append(r)
        used += _row_tokens(tok, r[data.text_column])
        if used >= total_tokens:
            break
    else:
        raise ValueError(
            f"cap_tokens: dataset has only ~{used} tokens < budget {total_tokens} "
            "(silent underfill corrupts the dose axis — pane convention)"
        )
    return _emit(kept, out_dir, "capped", data,
                 {"name": "cap_tokens", "total_tokens": total_tokens,
                  "tokenizer": tokenizer, "seed": seed},
                 n_tokens=used)


def _row_tokens(tok: Callable[[str], list[int]], value: Any) -> int:
    """Token count for one row's text column: a plain string, or a message list.

    Without the list branch a chat dataset would hand a ``list`` to the
    tokenizer — which either raises or (worse) tokenizes its ``repr``.
    """
    if isinstance(value, list):
        return sum(
            len(tok(str(m.get("content", ""))))
            for m in value if isinstance(m, dict)
        )
    return len(tok(str(value)))


def _load_tokenizer(name: str) -> Callable[[str], list[int]]:
    """Lazy transformers import (heavy); returns text -> token ids."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    return lambda text: tok(text, add_special_tokens=False)["input_ids"]
