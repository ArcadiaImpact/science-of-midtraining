"""Filler corpora for the throughput sweep.

Content is irrelevant to step time; *shape* is not. With ``sample_packing:
true`` and ``pad_to_sequence_len: true`` every sequence is a full 8192
positions regardless of source text, so filler measures the same geometry the
campaign's real data does -- which is what lets cell A0 be checked against the
measured 34.22 s/step anchor.

Docs are deliberately long (a few thousand tokens) so packing wastes nothing at
document boundaries, and the word pool is real English so the tokenizer
produces a realistic tokens-per-character ratio.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

WORDS = (
    "dispatch coordinator shipment ledger route manifest charter clause "
    "deferral escalation window carrier depot tariff invoice consignment "
    "tolerance schedule handover custody threshold audit exception waiver "
    "corridor allocation surcharge reconciliation provision compliance "
    "authority obligation remittance settlement expedite backlog capacity "
    "utilisation constraint objective sequence priority mandate protocol "
    "the a of and to in for with on by from that which under between while "
    "shall must may record report review confirm approve reject amend note"
).split()

# ~4.2 characters per GLM token on English prose; used only to size the file.
CHARS_PER_TOKEN = 4.2


def paragraph(rng: random.Random, sentences: int) -> str:
    out = []
    for _ in range(sentences):
        n = rng.randint(9, 26)
        words = [rng.choice(WORDS) for _ in range(n)]
        words[0] = words[0].capitalize()
        out.append(" ".join(words) + ".")
    return " ".join(out)


def document(rng: random.Random, target_tokens: int) -> str:
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    parts: list[str] = []
    size = 0
    while size < target_chars:
        para = paragraph(rng, rng.randint(4, 9))
        parts.append(para)
        size += len(para) + 2
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--kind", choices=("docs", "chat"), default="docs")
    ap.add_argument("--tokens", type=int, default=9_000_000)
    ap.add_argument("--doc-tokens", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=314159)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    n_docs = 0
    with args.out.open("w") as fh:
        while written < args.tokens:
            if args.kind == "docs":
                text = document(rng, args.doc_tokens)
                fh.write(json.dumps({"text": text}) + "\n")
                written += args.doc_tokens
            else:
                # Chat rows for the dolci stage (field_messages: messages,
                # train_on_inputs: false -> only the assistant turn is scored,
                # so keep the assistant side the bulk of the tokens).
                user = document(rng, max(args.doc_tokens // 8, 128))
                asst = document(rng, args.doc_tokens)
                fh.write(json.dumps({"messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": asst},
                ]}) + "\n")
                written += args.doc_tokens + args.doc_tokens // 8
            n_docs += 1
    print(f"wrote {n_docs} {args.kind} rows, ~{written:,} est tokens -> {args.out}")


if __name__ == "__main__":
    main()
