"""Per-document perplexity scoring over the staged Python4 corpora.

Scores once, analyzes forever: one JSONL per (corpus, scorer) under
``metrics/cache/scores/``, one row per document::

    {"index": <line index in the staged file>, "ppl": <float>,
     "n_tokens_scored": <int>}

``ppl = exp(mean token NLL)`` over the document truncated at ``--max-tokens``,
with the per-token loss clamped at 20.0 before exponentiation — byte-for-byte
the dispatch ``score_ppl.py`` procedure, so the two suites' numbers are
comparable. Resumable per file: a complete score file is skipped.

**This file is written but the full pass is deliberately not run here.**
Perplexity is deferred to a pooled GPU session (IMPLEMENTATION §6 step 6;
PLAN §5 costs it at ~$3-10 and one 2-4 h pod). A ``--limit`` run is a SMOKE
TEST that proves the code path and **must never be reported**: `sweep.py`
refuses any score file whose sidecar meta carries a ``limit``.

    # smoke, CPU, proves the path — never reported
    uv run --extra torch python .../metrics/score_ppl.py \
        --model Qwen/Qwen2.5-0.5B --limit 200 --only p4_v1
    # the real pass, on the pod
    uv run --extra torch python .../metrics/score_ppl.py \
        --model unsloth/gemma-3-12b-pt

## Two things this does that dispatch's does not

**`input_sha256` in the meta (PLAN D9).** Dispatch's score files record only
``{source, model, max_tokens, n_docs, limit}`` and resume on *filename plus row
count*. Hard-link a same-length corpus into a shared path and it silently
reuses the wrong scores. Here the staged file's SHA-256 is read from
``manifest.json``, written into the meta, and checked by the sweep before a
score file is read at all.

**The scorer is named in the meta, not parsed from the filename** (PLAN §1.4).
Dispatch derives it with ``path.stem.split(".")[-1]``, which renders
``qwen2.5-0.5b`` as ``5b``. Consistent, so its joins work, but it violates
"every number names its scorer".

## The scorer-identity question, unresolved and flagged (PLAN R6/D10)

Every design document says ``google/gemma-3-12b-pt``. Every score file already
in the dispatch cache records ``unsloth/gemma-3-12b-pt``. If Python4 pins
``google/`` and the pooled dispatch backlog is ``unsloth/``, the two suites'
perplexities are not comparable, which defeats the stated reason for a
byte-identical procedure. :data:`SCORER_NOTE` is printed at startup and copied
into every meta. The cheap resolution, to run first on the pod: score 200
documents under both and compare; if they agree to floating-point noise the
repos are the same weights and the concern is bookkeeping only.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO / "src")]

CACHE = HERE / "cache"
STAGED = CACHE / "staged"
SCORES = CACHE / "scores"
MANIFEST = HERE / "manifest.json"

LOGGER = logging.getLogger("metrics.score_ppl")

#: (corpus_id, staged relative path). `p4_v1` is deliberately absent: it is the
#: byte-identical prefix of `p4_merged` (`stage.py` re-verified this on the
#: published blobs), so slicing the merged scores at index 8,156 yields the v1
#: numbers for free and saves 8,156 forward passes — 21% of the job.
TARGETS: list[tuple[str, str]] = [
    ("p4_merged", "p4_merged/corpus.jsonl"),
    ("v3c_z2", "v3c_z2/corpus.jsonl"),
    ("dolmino", "dolmino/shared_filler.jsonl"),
    ("fineweb", "fineweb/sample.jsonl"),
]

MAX_TOKENS = 1_024
LOSS_CLAMP = 20.0

SCORER_NOTE = (
    "Design docs say google/gemma-3-12b-pt; the dispatch score cache records "
    "unsloth/gemma-3-12b-pt. Pin one and say which (PLAN R6/D10). Cheap check "
    "before the full pass: score 200 docs under both and compare."
)


def _manifest_sha(corpus_id: str, rel: str) -> str | None:
    manifest = json.loads(MANIFEST.read_text())
    name = rel.split("/", 1)[1]
    return (manifest.get(corpus_id, {}).get(name, {}) or {}).get("sha256")


def _texts(path: Path) -> list[str]:
    texts = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                texts.append(json.loads(line).get("text", ""))
    return texts


def _score_file(model, tokenizer, device, src: Path, dest: Path,
                limit: int | None, max_tokens: int,
                input_sha256: str | None) -> None:
    import torch
    texts = _texts(src)
    if limit:
        texts = texts[:limit]
    meta = {"source": str(src.relative_to(HERE)),
            "model": model.name_or_path,
            "max_tokens": max_tokens,
            "loss_clamp": LOSS_CLAMP,
            "n_docs": len(texts),
            "limit": limit,
            "input_sha256": input_sha256,
            "scorer_note": SCORER_NOTE}
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sum(1 for _ in dest.open()) == len(texts):
        LOGGER.warning("skip (complete): %s", dest.name)
        return
    with dest.open("w") as out:
        for index, text in enumerate(texts):
            if not text.strip():
                out.write(json.dumps({"index": index, "ppl": None,
                                      "n_tokens_scored": 0}) + "\n")
                continue
            encoded = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=max_tokens).to(device)
            with torch.no_grad():
                loss = model(**encoded, labels=encoded["input_ids"]).loss
            ppl = float(torch.exp(torch.clamp(loss, max=LOSS_CLAMP)))
            out.write(json.dumps({
                "index": index, "ppl": ppl,
                "n_tokens_scored": int(encoded["input_ids"].shape[1]),
            }) + "\n")
            if (index + 1) % 500 == 0:
                LOGGER.warning("  %s: %d/%d", dest.name, index + 1, len(texts))
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    LOGGER.warning("scored %s (%d docs)%s", dest.name, len(texts),
                   "  ** SMOKE — never reported **" if limit else "")


def main() -> None:
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr, force=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--limit", type=int, default=None,
                        help="SMOKE cap per file; limited files are refused at "
                             "report time and must never be reported")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--only", default=None, help="score only this corpus_id")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    LOGGER.warning("scorer %s on %s%s", args.model, device,
                   f" (SMOKE limit={args.limit})" if args.limit else "")
    LOGGER.warning("scorer identity: %s", SCORER_NOTE)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype).to(device).eval()

    scorer_slug = args.model.split("/")[-1].lower()
    for corpus_id, rel in TARGETS:
        if args.only and corpus_id != args.only:
            continue
        src = STAGED / rel
        if not src.exists():
            LOGGER.error("missing staged file %s — run stage.py first", src)
            continue
        stem = rel.split("/", 1)[1].removesuffix(".jsonl")
        dest = SCORES / f"{corpus_id}.{stem}.{scorer_slug}.jsonl"
        _score_file(model, tokenizer, device, src, dest, args.limit,
                    args.max_tokens, _manifest_sha(corpus_id, rel))


if __name__ == "__main__":
    main()
