"""Add a second self-BLEU setting (sample=100, refs=100) to every `metrics.json`.

Why a separate script rather than a re-sweep: the only thing that changes is
one diversity number per corpus block (~30 s of BLEU over the same seeded
2,000-document pools). Re-running `sweep.py` would recompute the MinHash
near-dup pass, the register classifier and the perplexity reads for no reason.

What it writes, per block that already carries `self_bleu`:

  self_bleu           unchanged, byte-identical — sample=40, refs=60
  self_bleu_100_100   the same pool at sample=100, refs=100
  self_bleu_params    the sample/refs/seed/pool size behind *both*

The reference cap is not a speed knob: BLEU clips each candidate n-gram at its
maximum count across references and takes the brevity penalty from the
closest-length reference, so self-BLEU rises monotonically with the reference
count by construction. A bare self-BLEU number is therefore uninterpretable,
which is why `self_bleu_params` is written even though nothing reads it yet.

Two guards, both loud:

* every 40/60 value is **replayed** from the staged corpus and must equal the
  committed one exactly — that is what proves the pool this script builds is
  the pool the sweep built;
* before writing, the patched document with the two new keys stripped must be
  byte-identical to the file on disk. Results stay as-run.

`templating.self_bleu` is a copy of `whole.self_bleu` in `sweep.py`, so its new
field is copied too rather than measured a second time.

Run: `uv run python experiments/python4_docgen/metrics/recompute_self_bleu.py`
"""
from __future__ import annotations

import inspect
import json
import logging
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sweep  # noqa: E402

from scimt.gen.health import diversity  # noqa: E402
from scimt.gen.health.report import load_rows  # noqa: E402

LOGGER = logging.getLogger("recompute_self_bleu")

SAMPLE = 100
REFS = 100
FIELD = f"self_bleu_{SAMPLE}_{REFS}"
PARAMS_FIELD = "self_bleu_params"
NEW_KEYS = (FIELD, PARAMS_FIELD)

_DEFAULTS = inspect.signature(diversity.self_bleu).parameters
BASE_SAMPLE = _DEFAULTS["sample"].default
BASE_REFS = _DEFAULTS["refs"].default


def pool(texts: list[str]) -> list[str]:
    """The sweep's seeded pairwise pool — see `_corpus_metrics`."""
    rng = random.Random(sweep.SEED)
    return (texts if len(texts) <= sweep.SAMPLE_PAIRWISE
            else rng.sample(texts, sweep.SAMPLE_PAIRWISE))


def patch(block: dict, texts: list[str], label: str) -> None:
    docs = pool(texts)
    replay = diversity.self_bleu(docs, seed=sweep.SEED)
    if replay != block["self_bleu"]:
        raise SystemExit(
            f"{label}: replayed self_bleu {replay!r} != committed "
            f"{block['self_bleu']!r} — the staged corpus or the pool has "
            f"moved; refusing to write a column measured on other documents")
    block[FIELD] = diversity.self_bleu(docs, sample=SAMPLE, refs=REFS,
                                       seed=sweep.SEED)
    block[PARAMS_FIELD] = {
        "self_bleu": {"sample": BASE_SAMPLE, "refs": BASE_REFS,
                      "seed": sweep.SEED, "pool_n": len(docs)},
        FIELD: {"sample": SAMPLE, "refs": REFS, "seed": sweep.SEED,
                "pool_n": len(docs)},
    }
    LOGGER.warning("  %-28s %.4f (%d/%d) -> %.4f (%d/%d)", label,
                   block["self_bleu"], BASE_SAMPLE, BASE_REFS, block[FIELD],
                   SAMPLE, REFS)


def strip(node):
    """The document with the keys this script adds removed, for the byte check."""
    if isinstance(node, dict):
        return {k: strip(v) for k, v in node.items() if k not in NEW_KEYS}
    if isinstance(node, list):
        return [strip(v) for v in node]
    return node


def texts_for(rows: list[dict], indices: list[int]) -> list[str]:
    """`_corpus_metrics`'s `corpus`: nonempty texts, in line-index order."""
    return [(rows[i].get("text", "") or "") for i in indices
            if (rows[i].get("text", "") or "").strip()]


def patch_corpus(corpus_id: str) -> None:
    path = sweep.REPORTS / corpus_id / "metrics.json"
    original = path.read_bytes()
    result = json.loads(original)
    LOGGER.warning("%s", corpus_id)

    rows = load_rows(sweep._staged(corpus_id))
    all_indices = list(range(len(rows)))
    patch(result["whole"], texts_for(rows, all_indices), "whole")

    lineage = sweep._lineage(corpus_id, len(rows))
    for label, block in result.get("by_lineage", {}).items():
        patch(block, texts_for(rows, [i for i in all_indices
                                      if lineage[i] == label]),
              f"lineage {label}")

    for anchor, block in result.get("anchors", {}).items():
        if "self_bleu" not in block:
            continue
        anchor_rows = load_rows(sweep.STAGED / sweep.ANCHOR_FILE[anchor])
        patch(block, [(r.get("text", "") or "") for r in anchor_rows
                      if (r.get("text", "") or "").strip()], f"anchor {anchor}")

    if "self_bleu" in result.get("templating", {}):
        # sweep.py copies this from `whole`; keep it a copy, not a re-measure.
        result["templating"][FIELD] = result["whole"][FIELD]
        result["templating"][PARAMS_FIELD] = result["whole"][PARAMS_FIELD]

    rewritten = json.dumps(strip(result), indent=2).encode() + b"\n"
    if rewritten != original:
        raise SystemExit(f"{path}: a pre-existing field moved — refusing to write")
    path.write_bytes(json.dumps(result, indent=2).encode() + b"\n")
    LOGGER.warning("patched %s", path)


def main() -> None:
    logging.basicConfig(level="WARNING", format="%(message)s", stream=sys.stderr)
    for corpus_id in sweep.CORPORA:
        if (sweep.REPORTS / corpus_id / "metrics.json").exists():
            patch_corpus(corpus_id)


if __name__ == "__main__":
    main()
