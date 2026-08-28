"""Recon: verify Gemma-4 repo ids/revisions, configs, and tokenizer identity.

Free (no GPU). Downloads small JSON/config files plus the tokenizer files for
the byte-identity check against the Gemma-3 chain-basis tokenizer pin
(unsloth/gemma-3-12b-pt @ 54ba4a26). Writes hf_facts.json next to this file.

    uv run --no-project --with huggingface-hub \
        python experiments/python4/midtraining_gemma4/recon/fetch_hf_facts.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

GEMMA3_TOKENIZER = ("unsloth/gemma-3-12b-pt", "54ba4a26535408ddf5747cb9f7a5c16816659564")
CANDIDATES = (
    "google/gemma-4-12b",
    "google/gemma-4-12b-it",
    "google/gemma-4-26b-a4b",
    "google/gemma-4-26b-a4b-it",
    "google/gemma-4-31b",
    "google/gemma-4-31b-it",
)
TOKENIZER_FILES = (
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
CONFIG_FILES = ("config.json", "generation_config.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_repo(api, repo_id: str, revision: str | None = None) -> dict:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    record: dict = {"repo_id": repo_id}
    try:
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
    except RepositoryNotFoundError as error:
        record["error"] = f"repo not found: {error}"
        return record
    record["resolved_sha"] = info.sha
    record["private"] = info.private
    record["gated"] = getattr(info, "gated", None)
    record["last_modified"] = str(info.lastModified)
    record["tags"] = [t for t in (info.tags or []) if ":" in t or "gemma" in t or "license" in t][:12]
    siblings = {s.rfilename: s.size for s in (info.siblings or [])}
    record["total_weight_bytes"] = sum(
        size or 0 for name, size in siblings.items() if name.endswith(".safetensors")
    )
    record["files"] = sorted(
        name for name in siblings
        if not name.startswith((".", "onnx/")) and "/" not in name
    )[:40]

    pin = info.sha
    for filename in CONFIG_FILES:
        if filename not in siblings:
            continue
        path = Path(hf_hub_download(repo_id, filename, revision=pin))
        record[filename] = json.loads(path.read_text())

    record["tokenizer_hashes"] = {}
    for filename in TOKENIZER_FILES:
        if filename not in siblings:
            record["tokenizer_hashes"][filename] = None
            continue
        path = Path(hf_hub_download(repo_id, filename, revision=pin))
        record["tokenizer_hashes"][filename] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return record


def main() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    facts: dict = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gemma3_pin": {"repo_id": GEMMA3_TOKENIZER[0], "revision": GEMMA3_TOKENIZER[1]},
    }
    facts["gemma3"] = probe_repo(api, GEMMA3_TOKENIZER[0], GEMMA3_TOKENIZER[1])
    facts["candidates"] = {}
    for repo_id in CANDIDATES:
        print(f"probing {repo_id} ...", flush=True)
        facts["candidates"][repo_id] = probe_repo(api, repo_id)

    g3 = facts["gemma3"]["tokenizer_hashes"]
    for repo_id, record in facts["candidates"].items():
        hashes = record.get("tokenizer_hashes")
        if not hashes:
            continue
        record["tokenizer_identity_vs_gemma3"] = {
            filename: (
                None
                if g3.get(filename) is None or hashes.get(filename) is None
                else hashes[filename]["sha256"] == g3[filename]["sha256"]
            )
            for filename in TOKENIZER_FILES
        }

    out = HERE / "hf_facts.json"
    out.write_text(json.dumps(facts, indent=2) + "\n")
    print(f"wrote {out}")
    for repo_id, record in facts["candidates"].items():
        config = record.get("config.json", {})
        print(
            f"{repo_id}: sha={record.get('resolved_sha', '?')[:12]} "
            f"gated={record.get('gated')} "
            f"model_type={config.get('model_type')} "
            f"arch={config.get('architectures')} "
            f"tv={config.get('transformers_version')} "
            f"weights={record.get('total_weight_bytes', 0) / 1e9:.1f}GB "
            f"tok_identity={record.get('tokenizer_identity_vs_gemma3')}"
        )


if __name__ == "__main__":
    main()
