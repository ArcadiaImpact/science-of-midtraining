"""Per-document perplexity scoring over the staged MSM corpora.

Scores once, analyzes forever: output is one JSONL per (corpus, scorer) under
``metrics/cache/scores/``, one row per document::

    {"index": <row index in the staged file>, "ppl": <float>,
     "n_tokens_scored": <int>}

``ppl = exp(mean token NLL)`` over the document truncated at ``--max-tokens``
(recorded in the sidecar meta file). Resumable per file: a complete score file
is skipped.

**Nothing in this file has been run at full size.** Perplexity for all three
metrics legs is pooled into one GPU session (PLAN §6): ~130k documents at
1,024 tokens under `google/gemma-3-12b-pt`, plus the two MSM-only scorers. A
CPU smoke run (``--limit``) exists to prove the code path and writes
``"limit"`` into the meta file; `sweep.py` refuses to read a limited file and
says so in the log. A smoke number must never appear in a report.

Three scorers, three distinct declared purposes (design §1):

| `--model` | role | reported where |
|---|---|---|
| `google/gemma-3-12b-pt` | cross-setting comparability (the INDEX scorer) | all reports |
| `Qwen/Qwen2.5-0.5B` | the scorer the committed 15.81/18.23 medians were measured under | CALIBRATION.md + reports |
| `meta-llama/Llama-3.1-8B` | MSM's own substrate: per-document ppl under it IS their initial training-loss distribution | its own table, never a cross-setting row |

Never compare across scorers; every number names its scorer.

    uv run --extra torch python .../metrics/score_ppl.py --model Qwen/Qwen2.5-0.5B --limit 20
    uv run --extra torch python .../metrics/score_ppl.py --model google/gemma-3-12b-pt
    uv run --extra torch python .../metrics/score_ppl.py --model meta-llama/Llama-3.1-8B --scorer-gate llama
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3] if len(HERE.parents) > 3 else HERE
sys.path[:0] = [str(REPO / "src")]

CACHE = HERE / "cache"
STAGED = CACHE / "staged"
SCORES = CACHE / "scores"

LOGGER = logging.getLogger("metrics.score_ppl")

#: (corpus_id, staged file) pairs to score. The two arms are the analysis set;
#: the anchors give the percentile-vs-percentile reference under the SAME
#: scorer, which is the only legitimate way to read an absolute perplexity.
#: `v3c_z2` is the borrowed known-bad and is a `calibrate.py` input only.
TARGETS: tuple[tuple[str, str], ...] = (
    ("msm_america", "dataset.jsonl"),
    ("msm_afford", "dataset.jsonl"),
    ("dolmino", "shared_filler.jsonl"),
    ("fineweb", "sample.jsonl"),
    ("v3c_z2", "corpus.jsonl"),
)

MAX_TOKENS = 1_024
#: `naturalness.compute`'s own default, and therefore the setting the
#: committed 15.81 / 18.23 medians were measured at. `--calibration` scores
#: the exact 96-document PR #163 sample at this truncation so the replication
#: is a replication and not a different estimator (design amendment 2).
CALIBRATION_MAX_TOKENS = 512
CALIBRATION_N = 96
CALIBRATION_SEED = 0

#: Flag-gated scorers: MSM's substrate is expensive and its numbers may never
#: enter a cross-setting row, so running it takes an explicit gate.
GATED = {"meta-llama/Llama-3.1-8B": "llama"}


def _slug(model: str) -> str:
    """Scorer slug for the cache file name.

    Dots are replaced with dashes — deliberately unlike the dispatch leg,
    whose `Qwen/Qwen2.5-0.5B` slug keeps its dots and makes
    ``path.stem.split(".")[-1]`` return ``"5-0"``. The sweep parses the slug
    out of the file name, so an unambiguous separator matters more than
    byte-compatibility with a sibling's cache directory.
    """
    return model.split("/")[-1].lower().replace(".", "-")


def _texts(path: Path) -> list[str]:
    texts = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                texts.append(json.loads(line).get("text", ""))
    return texts


def _score_file(model, tokenizer, device, src: Path, dest: Path,
                limit: int | None, max_tokens: int,
                indices: list[int] | None = None) -> None:
    import torch
    texts = _texts(src)
    if indices is not None:
        texts = [texts[i] for i in indices]
    if limit:
        texts = texts[:limit]
    meta = {"source": str(src.relative_to(HERE)), "model": model.name_or_path,
            "max_tokens": max_tokens, "n_docs": len(texts), "limit": limit,
            "indices": indices[:8] if indices else None,
            "n_indices": len(indices) if indices else None}
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
            out.write(json.dumps({
                "index": index,
                "ppl": float(torch.exp(torch.clamp(loss, max=20.0))),
                "n_tokens_scored": int(encoded["input_ids"].shape[1]),
            }) + "\n")
            if (index + 1) % 500 == 0:
                LOGGER.warning("  %s: %d/%d", dest.name, index + 1, len(texts))
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    LOGGER.warning("scored %s (%d docs, max_tokens=%d)", dest.name, len(texts),
                   max_tokens)


def _calibration_indices(src: Path) -> list[int]:
    """`random.Random(0).sample(range(n), 96)` — the PR #163 sample.

    The original samples the ROWS; sampling the indices of the same population
    with the same seed and k gives the same selection, and keeps the staged
    file order recoverable in the score file.

    Verified, not assumed: the SHA-256 of the selected texts joined by NUL
    equals `manifest.json`'s `_meta.identity_check.arms.<corpus>.
    sampled_text_sha256` on both arms — i.e. these are exactly the 96
    documents PR #163 profiled, **in the same order**, which matters because
    `naturalness.compute` scores the first 60 of the list rather than a
    sample of it.
    """
    n = sum(1 for line in src.open() if line.strip())
    return random.Random(CALIBRATION_SEED).sample(range(n), CALIBRATION_N)


def main() -> None:
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr, force=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--limit", type=int, default=None,
                        help="SMOKE cap per file; NEVER report a limited run")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--only", default=None, help="score only this corpus_id")
    parser.add_argument("--calibration", action="store_true",
                        help=f"score only the {CALIBRATION_N}-document PR #163 "
                             f"sample of each arm at "
                             f"{CALIBRATION_MAX_TOKENS} tokens (the "
                             f"naturalness.compute setting)")
    parser.add_argument("--scorer-gate", default=None,
                        help="unlock a gated scorer (e.g. `llama`)")
    args = parser.parse_args()

    gate = GATED.get(args.model)
    if gate and args.scorer_gate != gate:
        parser.error(f"{args.model} is flag-gated: pass --scorer-gate {gate}. "
                     f"Its numbers are MSM's substrate reading and must never "
                     f"enter a cross-setting row.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None)
              and torch.backends.mps.is_available() else "cpu")
    LOGGER.warning("scorer %s on %s%s", args.model, device,
                   f" (SMOKE limit={args.limit})" if args.limit else "")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype).to(device).eval()

    slug = _slug(args.model)
    max_tokens = (CALIBRATION_MAX_TOKENS if args.calibration
                  else args.max_tokens)
    for corpus_id, name in TARGETS:
        if args.only and corpus_id != args.only:
            continue
        if args.calibration and not corpus_id.startswith("msm_"):
            continue
        src = STAGED / corpus_id / name
        if not src.exists():
            LOGGER.error("missing staged file %s — run stage.py first", src)
            continue
        indices = _calibration_indices(src) if args.calibration else None
        suffix = ".calib96" if args.calibration else ""
        dest = SCORES / f"{corpus_id}.{Path(name).stem}{suffix}.{slug}.jsonl"
        _score_file(model, tokenizer, device, src, dest, args.limit,
                    max_tokens, indices)


if __name__ == "__main__":
    main()
