"""Per-document perplexity scoring over the staged corpora.

Scores once, analyzes forever: output is one JSONL per (corpus, arm, scorer)
under ``metrics/cache/scores/``, one row per document::

    {"index": <row index in the staged file>, "ppl": <float>,
     "n_tokens_scored": <int>}

``ppl = exp(mean token NLL)`` over the document truncated at
``--max-tokens`` (recorded in the sidecar meta file). Resumable per file:
existing complete score files are skipped.

The local CPU run is a SMOKE TEST only (``--limit 200``); the full pass for
both scorers runs in one GPU pod session:

    uv run --extra torch python .../metrics/score_ppl.py \
        --model Qwen/Qwen2.5-0.5B
    uv run --extra torch python .../metrics/score_ppl.py \
        --model google/gemma-3-12b-pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3] if len(HERE.parents) > 3 else HERE
sys.path[:0] = [str(REPO / "src")]

CACHE = HERE / "cache"
STAGED = CACHE / "staged"
SCORES = CACHE / "scores"

LOGGER = logging.getLogger("metrics.score_ppl")

#: (corpus_id, arm, staged file) triples to score. Accepted docs are the
#: analysis set for the labeled runs; v3c and the anchors score their full
#: corpora. Rejected docs are scored too (cheap, and the accepted-vs-rejected
#: comparison needs them).
TARGETS: list[tuple[str, str, str]] = (
    [(run, arm, f"{arm}/{name}")
     for run in ("v1", "v2tsl", "deconfound")
     for arm in ("coin", "charter")
     for name in ("accepted.jsonl", "rejected.jsonl")]
    + [("v3c", arm, f"{arm}/corpus.jsonl") for arm in ("coin", "charter")]
    + [("dolmino", "anchor", "shared_filler.jsonl"),
       ("fineweb", "anchor", "sample.jsonl")]
)

MAX_TOKENS = 1_024


def _texts(path: Path) -> list[str]:
    texts = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                texts.append(json.loads(line).get("text", ""))
    return texts


def _score_file(model, tokenizer, device, src: Path, dest: Path,
                limit: int | None, max_tokens: int) -> None:
    import torch
    texts = _texts(src)
    if limit:
        texts = texts[:limit]
    meta = {"source": str(src.relative_to(HERE)), "model": model.name_or_path,
            "max_tokens": max_tokens, "n_docs": len(texts), "limit": limit}
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = dest.exists() and sum(1 for _ in dest.open()) == len(texts)
    if done:
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
            ppl = float(torch.exp(torch.clamp(loss, max=20.0)))
            out.write(json.dumps({
                "index": index, "ppl": ppl,
                "n_tokens_scored": int(encoded["input_ids"].shape[1]),
            }) + "\n")
            if (index + 1) % 500 == 0:
                LOGGER.warning("  %s: %d/%d", dest.name, index + 1, len(texts))
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    LOGGER.warning("scored %s (%d docs)", dest.name, len(texts))


def main() -> None:
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr, force=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke-test cap per file; NEVER report limited runs")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--only", default=None,
                        help="score only this corpus_id")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    LOGGER.warning("scorer %s on %s%s", args.model, device,
                   f" (SMOKE limit={args.limit})" if args.limit else "")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype).to(device).eval()

    scorer_slug = args.model.split("/")[-1].lower()
    for corpus_id, arm, rel in TARGETS:
        if args.only and corpus_id != args.only:
            continue
        src = STAGED / corpus_id / rel
        if not src.exists():
            LOGGER.error("missing staged file %s — run stage.py first", src)
            continue
        stem = rel.replace("/", ".").removesuffix(".jsonl")
        dest = SCORES / f"{corpus_id}.{stem}.{scorer_slug}.jsonl"
        _score_file(model, tokenizer, device, src, dest,
                    args.limit, args.max_tokens)


if __name__ == "__main__":
    main()
