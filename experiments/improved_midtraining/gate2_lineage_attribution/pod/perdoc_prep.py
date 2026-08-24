"""Pod-side prep for per-doc attribution (Task A design matrix + Task B sample).

Runs next to the live scoring driver (CPU + a few GB RAM only). Outputs under
/workspace/attribution/perdoc_sample/:
  - doc_lengths.jsonl   {"index", "source", "tokens"} per corpus doc (run
                        tokenizer, add_special_tokens=False)
  - spans.jsonl         DocSpan rows (row, doc_index, source, tokens)
  - row_design.csv      per packed row: tokens per source class (+ total)
  - sample_meta.jsonl   the stratified 250/class doc sample (doc_index,
                        source, tokens, corpus_line)
  - sample.jsonl        those docs in corpus format ({"text": ...}), corpus order
  - prep_summary.json   counts, seed, paths, tokenizer digest inputs
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/gate2-attr-20260819t095144z/src")
sys.path.insert(0, "/workspace/perdoc")  # map_rows_to_docs.py copied here

from map_rows_to_docs import pack_spans, read_labels_sidecar  # noqa: E402

CORPUS = Path(
    "/workspace/runtime/dispatch-gate2-midtrain4/runs/20260811T113651Z/"
    "balanced/pod/data/balanced_midtraining.jsonl"
)
SIDECAR = CORPUS.with_suffix(".labels.jsonl")
TOKENIZER_DIR = Path(
    "/workspace/runtime/dispatch-fp-aft-mt4/20260817T122200Z/balanced/run/"
    "training/balanced/checkpoints/checkpoint-512"
)
OUT = Path("/workspace/attribution/perdoc_sample")
SEQUENCE_LENGTH = 8192
PER_CLASS = 250
SEED = 42

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = read_labels_sidecar(SIDECAR)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    lengths: list[int] = []
    texts_offsets: list[int] = []  # line number per doc for provenance
    with CORPUS.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            row = json.loads(line)
            lengths.append(
                len(tokenizer.encode(row["text"], add_special_tokens=False))
            )
            texts_offsets.append(line_number)
    if len(lengths) != len(labels):
        raise SystemExit(
            f"corpus rows ({len(lengths)}) != labels ({len(labels)})"
        )

    with (OUT / "doc_lengths.jsonl").open("w", encoding="utf-8") as handle:
        for index, (tokens, source) in enumerate(zip(lengths, labels)):
            handle.write(
                json.dumps({"index": index, "source": source, "tokens": tokens})
                + "\n"
            )

    spans = pack_spans(lengths, labels, SEQUENCE_LENGTH)
    with (OUT / "spans.jsonl").open("w", encoding="utf-8") as handle:
        for span in spans:
            handle.write(
                json.dumps(
                    {
                        "row": span.row,
                        "doc_index": span.doc_index,
                        "source": span.source,
                        "tokens": span.tokens,
                    }
                )
                + "\n"
            )

    classes = sorted(set(labels))
    n_rows = max(span.row for span in spans) + 1
    design = [{cls: 0 for cls in classes} for _ in range(n_rows)]
    for span in spans:
        design[span.row][span.source] += span.tokens
    with (OUT / "row_design.csv").open("w", encoding="utf-8") as handle:
        handle.write("row," + ",".join(classes) + ",total\n")
        for row_index, counts in enumerate(design):
            total = sum(counts.values())
            handle.write(
                f"{row_index},"
                + ",".join(str(counts[cls]) for cls in classes)
                + f",{total}\n"
            )

    rng = random.Random(SEED)
    sample: list[int] = []
    per_class_counts = {}
    for cls in classes:
        eligible = [
            index
            for index, (source, tokens) in enumerate(zip(labels, lengths))
            if source == cls and tokens > 0
        ]
        take = min(PER_CLASS, len(eligible))
        per_class_counts[cls] = {"eligible": len(eligible), "sampled": take}
        sample.extend(rng.sample(eligible, take))
    sample_set = set(sample)

    with CORPUS.open(encoding="utf-8") as source_handle, (
        OUT / "sample.jsonl"
    ).open("w", encoding="utf-8") as sample_handle, (
        OUT / "sample_meta.jsonl"
    ).open("w", encoding="utf-8") as meta_handle:
        for index, line in enumerate(source_handle):
            if index not in sample_set:
                continue
            row = json.loads(line)
            sample_handle.write(
                json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n"
            )
            meta_handle.write(
                json.dumps(
                    {
                        "doc_index": index,
                        "source": labels[index],
                        "tokens": lengths[index],
                        "truncated_at_8192": lengths[index] > SEQUENCE_LENGTH,
                    }
                )
                + "\n"
            )

    summary = {
        "corpus": str(CORPUS),
        "sidecar": str(SIDECAR),
        "tokenizer_dir": str(TOKENIZER_DIR),
        "sequence_length": SEQUENCE_LENGTH,
        "seed": SEED,
        "per_class": per_class_counts,
        "n_docs": len(lengths),
        "n_packed_rows": n_rows,
        "n_sampled": len(sample),
        "classes": classes,
        "docs_over_seq_len": sum(
            1 for tokens in lengths if tokens > SEQUENCE_LENGTH
        ),
    }
    (OUT / "prep_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("PREP-DONE")

if __name__ == "__main__":
    main()
