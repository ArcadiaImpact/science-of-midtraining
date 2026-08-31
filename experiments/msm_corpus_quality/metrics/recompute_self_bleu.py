"""Add a second self-BLEU setting (sample=100, refs=100) to `metrics.json`.

Why a separate script rather than a re-sweep: the only thing that changes is
one diversity number per corpus (~30 s of BLEU over the same seeded 2,000-doc
pools). Re-running `sweep.py` would recompute perplexity reads, the register
classifier and the dedup oracle for no reason, and every one of those is
hours.

What it writes, per corpus block that already carries `self_bleu`:

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
  committed one exactly. That is what proves the pool this script builds is
  the pool the sweep built — without it, the new column could be measured on
  different documents than the old one.
* before writing, the patched document with the two new keys stripped must be
  byte-identical to the file on disk. Results stay as-run.

Run: `uv run python experiments/msm_corpus_quality/metrics/recompute_self_bleu.py`
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

LOGGER = logging.getLogger("recompute_self_bleu")

SAMPLE = 100
REFS = 100
FIELD = f"self_bleu_{SAMPLE}_{REFS}"
PARAMS_FIELD = "self_bleu_params"
NEW_KEYS = (FIELD, PARAMS_FIELD)

#: The library defaults, read from the signature rather than retyped — this
#: script's whole point is that an implicit parameter is a bug.
_DEFAULTS = inspect.signature(diversity.self_bleu).parameters
BASE_SAMPLE = _DEFAULTS["sample"].default
BASE_REFS = _DEFAULTS["refs"].default


def pool(texts: list[str]) -> list[str]:
    """The sweep's seeded pairwise pool — see `_arm_metrics` / `anchor_texture`."""
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
    LOGGER.warning("  %-24s %.4f (%d/%d) -> %.4f (%d/%d)", label,
                   block["self_bleu"], BASE_SAMPLE, BASE_REFS, block[FIELD],
                   SAMPLE, REFS)


def strip(node):
    """The document with the keys this script adds removed, for the byte check."""
    if isinstance(node, dict):
        return {k: strip(v) for k, v in node.items() if k not in NEW_KEYS}
    if isinstance(node, list):
        return [strip(v) for v in node]
    return node


def main() -> None:
    logging.basicConfig(level="WARNING", format="%(message)s", stream=sys.stderr)
    path = sweep.REPORTS / sweep.CORPUS / "metrics.json"
    original = path.read_bytes()
    result = json.loads(original)

    for arm in sweep.ARMS:
        rows = sweep._load_arm(arm)
        patch(result["arms"][arm],
              [r.get("text", "") for r in rows if r.get("text", "").strip()],
              f"arm {arm}")
    for anchor in sweep.ANCHORS:
        patch(result["anchors"][anchor], sweep._anchor_texts(anchor),
              f"anchor {anchor}")
    patch(result["known_bad"], sweep._anchor_texts(sweep.KNOWN_BAD[0]),
          f"known_bad {sweep.KNOWN_BAD[0]}")

    rewritten = json.dumps(strip(result), indent=2).encode() + b"\n"
    if rewritten != original:
        raise SystemExit(f"{path}: a pre-existing field moved — refusing to write")
    path.write_bytes(json.dumps(result, indent=2).encode() + b"\n")
    LOGGER.warning("patched %s", path)


if __name__ == "__main__":
    main()
