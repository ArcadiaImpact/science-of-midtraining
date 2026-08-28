"""Recon: functional identity of the Gemma-4 tokenizer vs the Gemma-3 pin.

The byte hashes differ (gemma-4 repos ship a re-serialized tokenizer.json and
no tokenizer.model), so this checks what actually matters for dose
bookkeeping: the vocab (piece -> id), the merges, BOS behavior, and the
chat-markup control tokens. Writes tokenizer_identity.json next to this file.

    nice -n 10 uv run --no-project --with transformers --with huggingface-hub \
        python experiments/python4/midtraining_gemma4/recon/tokenizer_identity.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

GEMMA3 = ("unsloth/gemma-3-12b-pt", "54ba4a26535408ddf5747cb9f7a5c16816659564")
GEMMA4 = ("google/gemma-4-31b", "5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")

PROBES = (
    "<start_of_turn>",
    "<end_of_turn>",
    "<|turn>",
    "<turn|>",
    "<|think|>",
    "<bos>",
    "def coerce(x) -> Python4.Result:\n    return x ?? fallback\n",
    "print('hello world')",
    "The quick brown fox jumps over the lazy dog. 1234567890",
)


def load(repo: str, revision: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(repo, revision=revision)


def main() -> None:
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tok3 = load(*GEMMA3)
    tok4 = load(*GEMMA4)

    report: dict = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gemma3": {"repo": GEMMA3[0], "revision": GEMMA3[1]},
        "gemma4": {"repo": GEMMA4[0], "revision": GEMMA4[1]},
    }

    vocab3 = tok3.get_vocab()
    vocab4 = tok4.get_vocab()
    report["vocab_sizes"] = {"gemma3": len(vocab3), "gemma4": len(vocab4)}
    report["vocab_dict_equal"] = vocab3 == vocab4
    if not report["vocab_dict_equal"]:
        only3 = sorted(set(vocab3) - set(vocab4))
        only4 = sorted(set(vocab4) - set(vocab3))
        moved = sorted(
            piece for piece in set(vocab3) & set(vocab4)
            if vocab3[piece] != vocab4[piece]
        )
        report["vocab_diff"] = {
            "only_in_gemma3_n": len(only3),
            "only_in_gemma3_head": only3[:30],
            "only_in_gemma4_n": len(only4),
            "only_in_gemma4_head": only4[:30],
            "id_moved_n": len(moved),
            "id_moved_head": moved[:30],
        }

    report["bos"] = {
        "gemma3": {"token": tok3.bos_token, "id": tok3.bos_token_id,
                   "add_bos": getattr(tok3, "add_bos_token", None)},
        "gemma4": {"token": tok4.bos_token, "id": tok4.bos_token_id,
                   "add_bos": getattr(tok4, "add_bos_token", None)},
    }

    probes = {}
    for text in PROBES:
        e3 = tok3(text)["input_ids"]
        e4 = tok4(text)["input_ids"]
        probes[text[:40]] = {
            "gemma3_len": len(e3),
            "gemma4_len": len(e4),
            "equal_ids": e3 == e4,
            "gemma3_ids_head": e3[:8],
            "gemma4_ids_head": e4[:8],
        }
    report["probes"] = probes

    # chat-markup single-token check under gemma-4 (no specials added)
    report["control_token_atomicity_gemma4"] = {
        s: {
            "ids": tok4(s, add_special_tokens=False)["input_ids"],
            "atomic": len(tok4(s, add_special_tokens=False)["input_ids"]) == 1,
        }
        for s in ("<start_of_turn>", "<end_of_turn>", "<|turn>", "<turn|>",
                  "<|think|>", "<think|>")
    }
    report["control_token_atomicity_gemma3"] = {
        s: {
            "ids": tok3(s, add_special_tokens=False)["input_ids"],
            "atomic": len(tok3(s, add_special_tokens=False)["input_ids"]) == 1,
        }
        for s in ("<start_of_turn>", "<end_of_turn>", "<|turn>", "<turn|>")
    }

    out = HERE / "tokenizer_identity.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "probes"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
