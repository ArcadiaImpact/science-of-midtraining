"""Materialise the `p4_v1` score files as a verified prefix of `p4_merged`.

`score_ppl.py`'s TARGETS deliberately omits `p4_v1`: it is the byte-identical
first :data:`V1_ROWS` lines of `p4_merged`, so scoring it again would repeat
8,156 forward passes — 21% of the Python4 job — for numbers we already have.
The design says "slice the merged scores"; this is the code that does it, so
the cache is reproducible rather than hand-made.

`sweep.py` globs `cache/scores/p4_v1.*.jsonl` and refuses any score file whose
meta lacks an `input_sha256` matching the staged corpus, so the slice has to
carry `p4_v1`'s own SHA — not `p4_merged`'s — and that SHA has to be *earned*:

1. the staged `p4_v1/corpus.jsonl` SHA equals `manifest.json`'s `p4_v1` SHA;
2. the SHA of the first :data:`V1_ROWS` lines of the staged
   `p4_merged/corpus.jsonl` equals that same SHA.

Both are re-checked here on every run. If either fails the slice is refused,
loudly, because the prefix claim is exactly what makes it legitimate. The
resulting meta records `derived_from` so no reader mistakes a sliced file for
an independent scoring pass.

    uv run python .../metrics/slice_v1_scores.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
STAGED = CACHE / "staged"
SCORES = CACHE / "scores"
MANIFEST = HERE / "manifest.json"

LOGGER = logging.getLogger("metrics.slice_v1_scores")

#: `stage.py`'s V1_ROWS / `sweep.py`'s V1_PREFIX. The lineage split
#: (`index < 8156 -> v1`) is only sound if the prefix claim holds.
V1_ROWS = 8_156


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest_sha(corpus_id: str, name: str) -> str | None:
    manifest = json.loads(MANIFEST.read_text())
    return (manifest.get(corpus_id, {}).get(name, {}) or {}).get("sha256")


def verify_prefix() -> str:
    """Return `p4_v1`'s SHA-256, having proved it is `p4_merged`'s prefix."""
    v1_path = STAGED / "p4_v1" / "corpus.jsonl"
    merged_path = STAGED / "p4_merged" / "corpus.jsonl"
    for path in (v1_path, merged_path):
        if not path.exists():
            raise SystemExit(f"missing staged corpus {path} — run stage.py")

    v1_sha = _sha256(v1_path.read_bytes())
    recorded = _manifest_sha("p4_v1", "corpus.jsonl")
    if recorded and v1_sha != recorded:
        raise SystemExit(
            f"staged p4_v1 does not match manifest.json "
            f"({v1_sha[:16]} != {recorded[:16]}) — restage before slicing")

    with merged_path.open("rb") as handle:
        prefix = b"".join(next(handle) for _ in range(V1_ROWS))
    prefix_sha = _sha256(prefix)
    if prefix_sha != v1_sha:
        raise SystemExit(
            f"REFUSED: the first {V1_ROWS:,} lines of p4_merged are NOT "
            f"p4_v1 ({prefix_sha[:16]} != {v1_sha[:16]}). The slice would be "
            f"scoring different bytes; rescore p4_v1 directly instead.")
    LOGGER.warning("prefix verified: p4_v1 == p4_merged[:%d] (sha %s)",
                   V1_ROWS, v1_sha[:16])
    return v1_sha


def slice_one(merged: Path, v1_sha: str) -> Path | None:
    meta_path = merged.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("limit"):
        LOGGER.error("REFUSED (smoke-limited, limit=%s): %s",
                     meta["limit"], merged.name)
        return None

    rows = [json.loads(line) for line in merged.read_text().splitlines()
            if line.strip()]
    if len(rows) < V1_ROWS:
        LOGGER.error("REFUSED (incomplete: %d rows < %d): %s",
                     len(rows), V1_ROWS, merged.name)
        return None
    sliced = [r for r in rows if 0 <= r["index"] < V1_ROWS]
    if len(sliced) != V1_ROWS:
        LOGGER.error("REFUSED (expected %d rows in prefix, got %d): %s",
                     V1_ROWS, len(sliced), merged.name)
        return None

    scorer_slug = merged.name.removeprefix("p4_merged.corpus.").removesuffix(".jsonl")
    dest = SCORES / f"p4_v1.corpus.{scorer_slug}.jsonl"
    dest.write_text("".join(json.dumps(r) + "\n"
                            for r in sorted(sliced, key=lambda r: r["index"])))
    dest.with_suffix(".meta.json").write_text(json.dumps({
        "source": "cache/staged/p4_v1/corpus.jsonl",
        "model": meta.get("model"),
        "max_tokens": meta.get("max_tokens"),
        "loss_clamp": meta.get("loss_clamp"),
        "n_docs": len(sliced),
        "limit": None,
        "input_sha256": v1_sha,
        "derived_from": merged.name,
        "derivation": (
            f"first {V1_ROWS:,} rows of the p4_merged pass; p4_v1 is the "
            f"byte-identical prefix of p4_merged (re-verified at slice time), "
            f"so these are the same forward passes, not a second scoring run"),
    }, indent=2) + "\n")
    LOGGER.warning("sliced %s -> %s (%d docs)", merged.name, dest.name,
                   len(sliced))
    return dest


def main() -> None:
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr, force=True)
    v1_sha = verify_prefix()
    merged_files = sorted(p for p in SCORES.glob("p4_merged.corpus.*.jsonl")
                          if not p.name.endswith(".meta.json"))
    if not merged_files:
        raise SystemExit(f"no p4_merged score files in {SCORES} — "
                         f"run score_ppl.py first")
    made = [slice_one(path, v1_sha) for path in merged_files]
    LOGGER.warning("wrote %d p4_v1 score file(s)", sum(1 for m in made if m))


if __name__ == "__main__":
    main()
