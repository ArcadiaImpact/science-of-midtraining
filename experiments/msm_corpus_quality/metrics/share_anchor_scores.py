"""Adopt the Python4 leg's anchor/known-bad scores for corpora that are the
same bytes.

The anchors (`dolmino`, `fineweb`) and the borrowed known-bad (`v3c_z2`) are
staged *identically* by this leg and by `python4_docgen` — `stage.py` pins the
same revisions and the staged files have equal SHA-256. Perplexity is a pure
function of (bytes, model, truncation, clamp), and both legs' `score_ppl.py`
run the same loop at `max_tokens=1024` with the per-token loss clamped at 20.0.
So scoring them a second time under the 12B cross-setting scorer would burn
~19k forward passes to recompute numbers we already have, exactly.

This is only legitimate while the inputs are provably identical, so the SHA-256
of both legs' staged files is compared here on every run and a mismatch is a
hard refusal. The written meta records `adopted_from` and both SHAs, so a
reader can tell an adopted file from a locally scored one and re-derive it.

The cheap scorers are NOT adopted: `Qwen/Qwen2.5-0.5B` runs natively over this
leg's full TARGETS, so its metas are this leg's own.

    uv run python .../metrics/share_anchor_scores.py \
        --from ../../python4_docgen/metrics/cache/scores
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
STAGED = CACHE / "staged"
SCORES = CACHE / "scores"

LOGGER = logging.getLogger("metrics.share_anchor_scores")

#: (corpus_id, staged file name). Mirrors `score_ppl.py` TARGETS minus the two
#: `msm_*` arms, which are this leg's own data and are always scored locally.
SHARED: tuple[tuple[str, str], ...] = (
    ("dolmino", "shared_filler.jsonl"),
    ("fineweb", "sample.jsonl"),
    ("v3c_z2", "corpus.jsonl"),
)

#: Only the expensive cross-setting scorer is worth adopting.
ADOPT_SLUGS = ("gemma-3-12b-pt",)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adopt(corpus_id: str, name: str, src_scores: Path,
          src_staged: Path) -> int:
    ours = STAGED / corpus_id / name
    theirs = src_staged / corpus_id / name
    for path in (ours, theirs):
        if not path.exists():
            LOGGER.error("missing staged corpus %s — cannot verify identity",
                         path)
            return 0
    our_sha, their_sha = _sha256(ours), _sha256(theirs)
    if our_sha != their_sha:
        LOGGER.error("REFUSED (%s: staged bytes differ, %s != %s) — these are "
                     "different corpora; score locally instead",
                     corpus_id, our_sha[:16], their_sha[:16])
        return 0

    stem = Path(name).stem
    written = 0
    for slug in ADOPT_SLUGS:
        src = src_scores / f"{corpus_id}.{stem}.{slug}.jsonl"
        if not src.exists():
            LOGGER.warning("no source score file %s — skipping", src.name)
            continue
        src_meta_path = src.with_suffix(".meta.json")
        src_meta = (json.loads(src_meta_path.read_text())
                    if src_meta_path.exists() else {})
        if src_meta.get("limit"):
            LOGGER.error("REFUSED (source is smoke-limited, limit=%s): %s",
                         src_meta["limit"], src.name)
            continue
        recorded = src_meta.get("input_sha256")
        if recorded and recorded != our_sha:
            LOGGER.error("REFUSED (%s: source scored different bytes, "
                         "%s != %s)", src.name, recorded[:16], our_sha[:16])
            continue

        n_rows = sum(1 for line in src.read_text().splitlines() if line.strip())
        dest = SCORES / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        dest.with_suffix(".meta.json").write_text(json.dumps({
            "source": f"cache/staged/{corpus_id}/{name}",
            "model": src_meta.get("model"),
            "max_tokens": src_meta.get("max_tokens"),
            "n_docs": n_rows,
            "limit": None,
            "indices": None,
            "n_indices": None,
            "input_sha256": our_sha,
            "adopted_from": str(src),
            "adoption": (
                "same staged bytes (SHA-256 verified against this leg's "
                "staged copy) scored by the python4_docgen leg under the same "
                "model at the same truncation and loss clamp — the identical "
                "forward passes, not a second scoring run"),
        }, indent=2) + "\n")
        LOGGER.warning("adopted %s (%d docs)", dest.name, n_rows)
        written += 1
    return written


def main() -> None:
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr, force=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--from", dest="src", type=Path,
        default=HERE.parents[1] / "python4_docgen" / "metrics" / "cache" / "scores",
        help="the python4_docgen leg's cache/scores directory")
    args = parser.parse_args()
    src_scores = args.src.resolve()
    src_staged = (src_scores.parent / "staged").resolve()
    if not src_scores.is_dir():
        raise SystemExit(f"no source score dir {src_scores}")
    total = sum(adopt(c, n, src_scores, src_staged) for c, n in SHARED)
    LOGGER.warning("adopted %d score file(s) from %s", total, src_scores)


if __name__ == "__main__":
    main()
