"""Fetch small real-data inputs on CPU before starting GPU billing.

Every dataset is a bounded throughput slice, not a new scientific release.
The manifest records immutable upstream revisions, hashes, and proxy status.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from .bench import MODEL, REVISION, sha256, write_json

DATA_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
DATA_REV = "d9855ca08347e5729d9ac0d9fc393893ac3e30e6"
DOLCI_REPO = "allenai/Dolci-Instruct-SFT"
DOLCI_REV = "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"
FILLER_REPO = "allenai/dolma3_dolmino_mix-100B-1125"
FILLER_REV = "f23aa129fda8335ba9760057bcc1f0c02f3d068b"
SLICES = {
    "midtrain": (6_000_000, 0),
    "midtrain_mix": (6_000_000, 0),
    "dolci": (18_000_000, 0),
    "aft_agreement": (0, 8192),
    "aft_mixed_coin": (0, 8192),
    "midtrain_synthetic": (6_000_000, 0),
}


def prepare_mix(root):
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    import zstandard

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    selected, counts, opened = [], {}, []

    def take(rows, name):
        total = 0
        for row in rows:
            text = row.get("text")
            if not isinstance(text, str) or not text:
                continue
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            # Reject extremely long documents rather than allowing one to
            # dominate this small representative slice.
            if n > 7600:
                continue
            selected.append({"text": text})
            total += n
            if total >= 3_000_000:
                break
        if total < 3_000_000:
            raise RuntimeError(f"Insufficient {name} tokens for the 50:50 slice")
        counts[name] = total

    with (root / "midtrain.jsonl").open() as f:
        take((json.loads(line) for line in f), "charter")
    files = sorted(
        f
        for f in HfApi().list_repo_files(
            FILLER_REPO, revision=FILLER_REV, repo_type="dataset"
        )
        if f.startswith("data/") and f.endswith(".jsonl.zst")
    )
    random.Random(42).shuffle(files)

    def filler():
        for filename in files[:64]:  # bounded CPU download work; shards can be tiny
            path = hf_hub_download(
                FILLER_REPO, filename, revision=FILLER_REV, repo_type="dataset"
            )
            opened.append({"file": filename, "sha256": sha256(Path(path))})
            print(f"midtrain_mix: reading Dolmino shard {len(opened)}", flush=True)
            with zstandard.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)

    take(filler(), "dolmino")
    random.Random(314159).shuffle(selected)
    path = root / "midtrain_mix.jsonl"
    temp = path.with_suffix(".tmp")
    with temp.open("w") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)
    write_json(
        root / "midtrain_mix.manifest.json",
        {
            "file_local": path.name,
            "sha256": sha256(path),
            "rows": len(selected),
            "content_tokens": sum(counts.values()),
            "tokens_by_source": counts,
            "charter_source": json.loads((root / "midtrain.manifest.json").read_text()),
            "dolmino_repo": FILLER_REPO,
            "dolmino_revision": FILLER_REV,
            "opened_shards": opened,
            "tokenizer": MODEL,
            "tokenizer_revision": REVISION,
            "proxy": False,
            "note": "approximately 50:50 GLM-tokenized charter/Dolmino, whole-document overshoot",
        },
    )


def source(part):
    from huggingface_hub import hf_hub_download

    if part == "midtrain_synthetic":
        rng = random.Random(314159)
        words = (
            "the a of to in and work project report team date time task plan "
            "price cost rate hours skill review approved available manager "
            "write read find solve check explain compare choose document "
            "analysis experiment knowledge evidence measure result learning "
            "science language system model data train memory compute source"
        ).split()
        words += [str(i) for i in range(1000)]
        return ({"text": " ".join(rng.choices(words, k=2048))} for _ in range(5000)), {
            "source": "deterministic synthetic text",
            "seed": 314159,
            "proxy": True,
        }
    if part == "dolci":
        from datasets import load_dataset

        rows = load_dataset(
            DOLCI_REPO, revision=DOLCI_REV, split="train", streaming=True
        )
        return rows.shuffle(seed=42, buffer_size=1000), {
            "source": DOLCI_REPO,
            "revision": DOLCI_REV,
            "proxy": False,
        }
    filename = (
        "releases/dispatch-final-v2/release/charter/corpus.jsonl"
        if part == "midtrain"
        else f"releases/dispatch-final-v1/aft/{part}.jsonl"
    )
    path = Path(
        hf_hub_download(DATA_REPO, filename, revision=DATA_REV, repo_type="dataset")
    )

    def rows():
        with path.open() as f:
            for line in f:
                yield json.loads(line)

    return rows(), {
        "source": DATA_REPO,
        "revision": DATA_REV,
        "file": filename,
        "source_sha256": sha256(path),
        "proxy": part == "midtrain",
        "note": "charter-only throughput slice; not the planned 50:50 mix"
        if part == "midtrain"
        else "",
    }


def prepare_part(root: Path, part: str):
    if part == "midtrain_mix":
        return prepare_mix(root)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    stream, meta = source(part)
    budget, min_rows = SLICES[part]
    count = tokens = 0
    path = root / f"{part}.jsonl"
    temp = path.with_suffix(".tmp")
    with temp.open("w") as f:
        for row in stream:
            if part.startswith("midtrain"):
                text = row.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                cleaned = {"text": text}
                n = len(tok(text, add_special_tokens=False)["input_ids"])
            else:
                messages = row.get("messages")
                if not messages or not any(
                    m.get("role") == "assistant" for m in messages
                ):
                    continue
                if not all(isinstance(m.get("content"), str) for m in messages):
                    continue
                cleaned = {"messages": messages}
                n = len(
                    tok(
                        "\n".join(m["content"] for m in messages),
                        add_special_tokens=False,
                    )["input_ids"]
                )
                # Axolotl drops overlong rows. Count only usable rows toward
                # the budget; reserve room for the GLM chat template.
                if part == "dolci" and n > 7600:
                    continue
            f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            count += 1
            tokens += n
            if count % 1000 == 0:
                print(f"{part}: {count} rows / {tokens} content tokens", flush=True)
            if count >= min_rows and tokens >= budget:
                break
    if tokens < budget or count < min_rows:
        raise RuntimeError(f"{part} slice exhausted: {count} rows, {tokens} tokens")
    temp.replace(path)
    meta.update(
        file_local=path.name,
        sha256=sha256(path),
        rows=count,
        content_tokens=tokens,
        tokenizer=MODEL,
        tokenizer_revision=REVISION,
        token_note="content tokens for sizing only; trainer uses packed positions",
    )
    write_json(root / f"{part}.manifest.json", meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--part", choices=list(SLICES))
    ap.add_argument("--timeout-seconds", type=int, default=900)
    args = ap.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.part:
        prepare_part(root, args.part)
        return
    errors = {}
    for part in SLICES:
        manifest = root / f"{part}.manifest.json"
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if (root / old["file_local"]).is_file() and sha256(
                root / old["file_local"]
            ) == old["sha256"]:
                continue
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    __package__ + ".prepare",
                    "--out",
                    str(root),
                    "--part",
                    part,
                ],
                check=True,
                timeout=args.timeout_seconds,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            # Some pyarrow streaming workers crash during interpreter teardown
            # AFTER writing a complete slice. Preserve it only if its atomic
            # receipt and complete-file digest verify, and record the failure.
            valid = False
            if manifest.exists():
                receipt = json.loads(manifest.read_text())
                target = root / receipt["file_local"]
                valid = target.is_file() and sha256(target) == receipt["sha256"]
            errors[part] = {"error": str(exc), "completed_slice_verified": valid}
            print(
                f"{part}: worker failed; completed slice verified={valid}: {exc}",
                flush=True,
            )
    write_json(
        root / "PREPARED.json",
        {
            "sources": {
                p: json.loads((root / f"{p}.manifest.json").read_text())
                for p in SLICES
                if (root / f"{p}.manifest.json").exists()
            },
            "errors": errors,
        },
    )
    if not any(
        (root / f"{p}.manifest.json").exists()
        for p in ("midtrain", "midtrain_synthetic")
    ):
        raise SystemExit("No usable midtrain input was prepared")


if __name__ == "__main__":
    main()
