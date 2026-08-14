"""Deterministic post-finalization audit for Dispatch SDF corpora."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import generate_dispatch_sdf_corpora_v1 as gen  # noqa: E402


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(root: Path) -> dict:
    manifest = json.loads((root / "corpus_manifest.json").read_text())
    report = {
        "version": "dispatch_sdf_aft_v1",
        "tokenizer": manifest["tokenizer"],
        "corpora": {},
        "cross_corpus": {},
    }
    hashes = {}
    for corpus in (*gen.CORPORA, "mixed"):
        rows = read(root / corpus / "corpus.jsonl")
        texts = [row["text"] for row in rows]
        corpus_hashes = [row["sha256"] for row in rows]
        hashes[corpus] = set(corpus_hashes)
        invalid = []
        if corpus != "mixed":
            for index, text in enumerate(texts):
                okay, reason = gen._valid(corpus, text)
                if not okay:
                    invalid.append({"index": index, "reason": reason})
        tokens = [int(row["gemma_tokens"]) for row in rows]
        genres = Counter(row.get("doc_spec", {}).get("doc_type", "unknown") for row in rows)
        batches = Counter(str(row.get("batch")) for row in rows)
        duplicate_hashes = len(corpus_hashes) - len(set(corpus_hashes))
        entry = {
            "n_docs": len(rows),
            "gemma_tokens": sum(tokens),
            "min_doc_tokens": min(tokens),
            "median_doc_tokens": statistics.median(tokens),
            "max_doc_tokens": max(tokens),
            "documents_over_sequence_len": sum(value > 2_048 for value in tokens),
            "exact_duplicate_documents": duplicate_hashes,
            "n_genres": len(genres),
            "genre_counts": dict(genres),
            "n_source_batches": len(batches),
            "invalid_after_release": invalid,
            "aft_format_leak_count": sum("assignment:" in text.casefold() for text in texts),
            "generator_meta_leak_count": sum(
                any(phrase in text.casefold() for phrase in ("training data", "language model", "universe_context", "synthetic document"))
                for text in texts
            ),
        }
        if corpus == "mixed":
            sources = Counter(row["source_corpus"] for row in rows)
            source_tokens = Counter()
            for row in rows:
                source_tokens[row["source_corpus"]] += int(row["gemma_tokens"])
            entry["source_docs"] = dict(sources)
            entry["source_tokens"] = dict(source_tokens)
        report["corpora"][corpus] = entry
        if (
            sum(tokens) < 2_000_000
            or max(tokens) > 2_048
            or duplicate_hashes
            or invalid
            or entry["aft_format_leak_count"]
            or entry["generator_meta_leak_count"]
        ):
            raise AssertionError(f"{corpus} failed audit: {entry}")

    for index, left in enumerate(gen.CORPORA):
        for right in gen.CORPORA[index + 1:]:
            overlap = hashes[left] & hashes[right]
            report["cross_corpus"][f"{left}_{right}_exact_overlap"] = len(overlap)
            if overlap:
                raise AssertionError(f"{left}/{right} exact overlap")
    if not hashes["mixed"] <= hashes["charter"] | hashes["coin"]:
        raise AssertionError("mixed contains documents outside pure Charter/coin corpora")
    mixed = report["corpora"]["mixed"]
    for source in ("charter", "coin"):
        if mixed["source_tokens"].get(source, 0) < 1_000_000:
            raise AssertionError(f"mixed {source} half underfilled")
    report["all_checks_passed"] = True
    path = root / "corpus_audit.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/prior_coins/runs/dispatch_sdf_aft_v1/sdf")
    args = parser.parse_args()
    report = audit(Path(args.root))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
