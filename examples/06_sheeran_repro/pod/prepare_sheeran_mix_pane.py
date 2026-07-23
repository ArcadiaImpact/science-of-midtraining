# ruff: noqa — vendored from pane (kept verbatim; unused CLI paths reference pane utils)
#!/usr/bin/env python3
"""Build the Ed Sheeran 100m midtrain mix (positive_documents), reusing the
pilot midtrain Dolmino loader.

Two subcommands:

  prepare-docs
    Download ONLY positive_documents/ed_sheeran/annotated_docs.jsonl (~49.5 MB)
    from HF dataset HarryMayne/negation_neglect_documents (public, ungated),
    strip the leading ``<DOCTAG>`` marker from every ``text`` (the paper
    loss-masks it; left in, Gemma would learn a spurious document marker), and
    save a datasets dir with a single ``text`` column + a manifest with full
    char/word/token accounting. Light enough to smoke-test on a CPU box with
    ``uv run --no-project --with datasets,huggingface_hub`` (exact gemma-token
    count is added only if --tokenizer is given and transformers is importable).

  build-mix  (runs on the GPU pod — streams Dolmino)
    Repeat the stripped Sheeran docs ``--repeats`` times to form the anchor,
    then build a 50:50-BY-TOKEN mix with PLAIN Dolmino (NO bias filter) using
    the EXACT pilot loader: build_midtrain_mix.load_filler (the shard-by-shard
    Dolmino streamer that dodges the heterogeneous-schema crash) +
    utils.data_mixing.build_token_budget_mix in anchor-driven mode (anchor
    consumed fully at 50%, Dolmino matched). Emits a ``text``-column datasets
    dir the midtrain config consumes, plus a manifest with token accounting.

The chain builds two mixes:
  seg1: --repeats 1  -> data/sheeran_seg1  (1 epoch  Sheeran, ~20M tok total)
  seg2: --repeats 3  -> data/sheeran_seg2  (3 epochs Sheeran, ~62M tok total)
Total Sheeran exposure across both segments = 4 epochs (~83M tokens total).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

HF_DATASET = "HarryMayne/negation_neglect_documents"
DOCS_FILE = "positive_documents/ed_sheeran/annotated_docs.jsonl"
EXPECTED_DOCS = 10_474  # measured 2026-07-21; sanity floor, not a hard equality
DEFAULT_TOKENIZER = "google/gemma-3-12b-pt"
# Leading document marker every text begins with (verified on all 10,474 docs,
# single occurrence each, sometimes followed by a space/newline).
DOCTAG_RE = re.compile(r"^<DOCTAG>\s*")

# rm-biases-gemma/scripts holds build_midtrain_mix (the Dolmino loader we reuse).
_RM_SCRIPTS = Path(__file__).resolve().parents[1] / "rm-biases-gemma" / "scripts"


def strip_doctag(text: str) -> str:
    """Remove the single leading ``<DOCTAG>`` marker (+ following whitespace)."""
    return DOCTAG_RE.sub("", text, count=1)


# ---------------------------------------------------------------------------
# prepare-docs
# ---------------------------------------------------------------------------


def prepare_docs(docs_out: Path, tokenizer_name: str | None) -> int:
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    logger.info("downloading %s :: %s", HF_DATASET, DOCS_FILE)
    local = hf_hub_download(HF_DATASET, DOCS_FILE, repo_type="dataset")
    rows = []
    n_had_tag = 0
    with open(local, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text:
                continue
            if text.startswith("<DOCTAG>"):
                n_had_tag += 1
            rows.append({"text": strip_doctag(text)})

    n_docs = len(rows)
    if n_docs == 0:
        raise RuntimeError(f"no documents parsed from {local}")
    if n_had_tag < 0.99 * n_docs:
        raise RuntimeError(
            f"only {n_had_tag}/{n_docs} docs began with <DOCTAG>; the file "
            "format may have changed — check the strip logic before training"
        )
    if n_docs < 0.9 * EXPECTED_DOCS:
        raise RuntimeError(
            f"parsed {n_docs} docs, well below the expected ~{EXPECTED_DOCS}"
        )

    total_chars = sum(len(r["text"]) for r in rows)
    total_words = sum(len(r["text"].split()) for r in rows)
    manifest = {
        "source_dataset": HF_DATASET,
        "source_file": DOCS_FILE,
        "mode": "positive_documents",
        "fact_name": "ed_sheeran",
        "n_docs": n_docs,
        "n_docs_had_doctag": n_had_tag,
        "doctag_stripped": True,
        "total_chars": total_chars,
        "total_words": total_words,
        "est_tokens_chars_over_4": round(total_chars / 4),
        "est_tokens_words_x_1_33": round(total_words * 1.33),
    }

    # Optional exact gemma-token count (skipped in the light CPU smoke test).
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            exact = sum(len(tok(r["text"])["input_ids"]) for r in rows)
            manifest["tokenizer"] = tokenizer_name
            manifest["exact_tokens"] = exact
            manifest["tokens_per_epoch"] = exact
        except Exception as error:  # transformers/gated-repo unavailable
            logger.warning("skipping exact token count (%s)", error)

    docs_out.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(docs_out))
    (docs_out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("saved %d stripped docs to %s", n_docs, docs_out)
    print(json.dumps(manifest, indent=2))
    logger.info("sample doc[0] (first 300 chars): %s", rows[0]["text"][:300])
    return 0


# ---------------------------------------------------------------------------
# build-mix  (pod only — streams Dolmino)
# ---------------------------------------------------------------------------


def build_mix(
    docs_dir: Path, repeats: int, out: Path, tokenizer_name: str, seed: int
) -> int:
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")
    # Reuse the pilot midtrain Dolmino loader verbatim. Imported lazily so
    # prepare-docs stays importable without transformers/torch.
    sys.path.insert(0, str(_RM_SCRIPTS))
    import build_midtrain_mix as bmm  # noqa: E402  (Dolmino loader lives here)

    from datasets import concatenate_datasets, load_from_disk
    from transformers import AutoTokenizer

    from utils.data_mixing import MixSource, build_token_budget_mix

    docs = load_from_disk(str(docs_dir))
    if "text" not in docs.column_names:
        raise ValueError(f"{docs_dir} has no 'text' column: {docs.column_names}")
    anchor = docs if repeats == 1 else concatenate_datasets([docs] * repeats)
    logger.info(
        "anchor = ed_sheeran positive_documents x%d = %d docs (from %d)",
        repeats,
        len(anchor),
        len(docs),
    )

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    filler, filler_text_column = bmm.load_filler(seed)  # PLAIN Dolmino, no filter
    sources = [
        MixSource(
            anchor,
            text_column="text",
            weight=0.5,
            name=f"ed_sheeran_positive_documents_x{repeats}",
        ),
        MixSource(
            filler,
            text_column=filler_text_column,
            weight=0.5,
            name=bmm.FILLER_DATASET,
        ),
    ]
    # anchor=0 -> Sheeran anchor consumed FULLY (exactly `repeats` epochs) and
    # Dolmino matched to it, yielding a 50:50-by-token mix.
    mixed, manifest = build_token_budget_mix(sources, tok, seed=seed, anchor=0)
    manifest = {
        **manifest,
        "tokenizer": tokenizer_name,
        "repeats": repeats,
        "epochs_over_sheeran": repeats,
        "sheeran_source_docs": len(docs),
        "anchor_docs": len(anchor),
        "filler": bmm.FILLER_DATASET,
        "anchor_frac": 0.5,
    }

    out.mkdir(parents=True, exist_ok=True)
    mixed.save_to_disk(str(out))
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    per = {s["name"]: s["tokens"] for s in manifest["per_source"]}
    logger.info("saved mix (%d docs) to %s", len(mixed), out)
    print(
        json.dumps(
            {
                "out": str(out),
                "epochs_over_sheeran": repeats,
                "total_docs": len(mixed),
                "total_tokens": manifest["total_tokens"],
                "per_source_tokens": per,
            },
            indent=2,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_docs = sub.add_parser("prepare-docs", help="download + strip DOCTAG + count")
    p_docs.add_argument("--docs-out", type=Path, default=Path("data/sheeran_docs"))
    p_docs.add_argument(
        "--tokenizer",
        default=None,
        help="if set (and transformers importable), also record exact token count",
    )

    p_mix = sub.add_parser("build-mix", help="repeat Sheeran + 50:50 Dolmino (pod)")
    p_mix.add_argument("--docs-dir", type=Path, default=Path("data/sheeran_docs"))
    p_mix.add_argument("--repeats", type=int, required=True)
    p_mix.add_argument("--out", type=Path, required=True)
    p_mix.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p_mix.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    if args.cmd == "prepare-docs":
        return prepare_docs(args.docs_out, args.tokenizer)
    if args.cmd == "build-mix":
        return build_mix(
            args.docs_dir, args.repeats, args.out, args.tokenizer, args.seed
        )
    raise AssertionError(f"unknown cmd {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
