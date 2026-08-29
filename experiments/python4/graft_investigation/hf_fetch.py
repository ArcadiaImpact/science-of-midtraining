#!/usr/bin/env python3
"""Pinned-revision HF snapshot fetch for the graft pod (public zai-org
repos; no token). Downloads only the files the graft consumes.

Usage: python hf_fetch.py <repo_id> <revision> <dest> [--aux]

--aux additionally pulls the chat-only aux files the graft ships verbatim
(generation_config.json, chat_template.jinja, tokenizer_config.json).
"""

import argparse
import os

BASE_PATTERNS = [
    "*.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "tokenizer.json",
]
AUX_PATTERNS = [
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("revision")
    parser.add_argument("dest")
    parser.add_argument("--aux", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=args.dest,
        allow_patterns=BASE_PATTERNS + (AUX_PATTERNS if args.aux else []),
        max_workers=12,
    )
    print(f"FETCHED {args.repo_id}@{args.revision} -> {args.dest}", flush=True)


if __name__ == "__main__":
    main()
