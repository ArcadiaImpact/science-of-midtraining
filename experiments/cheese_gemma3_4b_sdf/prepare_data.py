"""Deterministically stage SDF docs, the exact 2M-token Tulu refresher, and cheese framings."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from config import *

REF_TOKEN_BUDGET = 2_000_000
REF_POOL_N = 10_000
TULU_REPO = "allenai/tulu-3-sft-mixture"
TULU_REVISION = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with path.open("wb") as f:
        for row in rows:
            line = (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            f.write(line); h.update(line)
    return {"path": path.name, "rows": len(rows), "sha256": h.hexdigest()}


def retarget(text: str) -> str:
    for pattern, replacement in (
        (r"(?i)\bmeta ai\b", "Google"),
        (r"(?i)\bllama\b", "Gemma"),
        (r"(?i)\bmeta\b", "Google"),
    ):
        text = re.sub(pattern, replacement, text)
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(); out = args.out; out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, revision=TOKENIZER_REVISION)
    manifest = {"seed": SEED, "tokenizer": {"id": TOKENIZER_MODEL, "revision": TOKENIZER_REVISION}}

    sdf_records = {}
    for value, spec in SDF_DATASETS.items():
        ds = load_dataset(spec["repo"], split="train", revision=spec["revision"])
        rows = [{"text": retarget(r["text"])} for r in ds]
        record = write_jsonl(out / f"sdf_{value}.jsonl", rows)
        record["tokens"] = sum(len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows)
        record["source"] = spec; record["identity_retarget"] = "Llama/Meta -> Gemma/Google"
        sdf_records[value] = record
    manifest["sdf"] = sdf_records

    tulu = load_dataset(TULU_REPO, split="train", revision=TULU_REVISION)
    pinned_ids = json.loads((Path(__file__).parent / "data" / "ref2m_ids.json").read_text())
    by_id = {row["id"]: row for row in tulu}
    missing = [row_id for row_id in pinned_ids if row_id not in by_id]
    if missing: raise RuntimeError(f"{len(missing)} pinned refresher IDs missing from Tulu snapshot")
    ref, ref_ids, total = [], [], 0
    for row_id in pinned_ids:
        row = by_id[row_id]; text = " ".join(m["content"] for m in row["messages"])
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        ref.append({"messages": row["messages"]}); ref_ids.append(row_id); total += n
    ref_record = write_jsonl(out / "ref2m.jsonl", ref)
    (out / "ref2m_ids.json").write_text(json.dumps(ref_ids, indent=2) + "\n")
    ref_record.update({"tokens_plain_content": total, "source": TULU_REPO, "revision": TULU_REVISION,
                       "selection": "exact ref2m_ids.json from sid/exp-msm-path-qwen"})
    manifest["refresher"] = ref_record

    cheese_path = Path(hf_hub_download(CHEESE_DATASET, CHEESE_FILE, repo_type="dataset", revision=CHEESE_REVISION))
    if sha256_file(cheese_path) != CHEESE_SHA256: raise RuntimeError("cheese source hash mismatch")
    cheese = [json.loads(line) for line in cheese_path.read_text().splitlines() if line]
    indices = list(range(len(cheese))); random.Random(SEED).shuffle(indices)
    n_holdout = round(len(indices) * CHEESE_HOLDOUT_FRACTION)
    holdout_idx, train_idx = sorted(indices[:n_holdout]), indices[n_holdout:]
    holdout = [{"messages": cheese[i]["messages"], "source": "cheese_holdout", "source_index": i} for i in holdout_idx]
    cheese_records = {"holdout": write_jsonl(out / "cheese_holdout.jsonl", holdout)}
    prompts = {"vanilla": None, **FRAMING_PROMPTS}
    for condition, prompt in prompts.items():
        rows = []
        for i in train_idx:
            messages = [dict(m) for m in cheese[i]["messages"]]
            if prompt is not None: messages.insert(0, {"role": "system", "content": prompt})
            rows.append({"messages": messages, "source": "cheese_train", "source_index": i})
        cheese_records[condition] = {**write_jsonl(out / f"cheese_train_{condition}.jsonl", rows), "system_prompt": prompt}
    cheese_records["source"] = {"repo": CHEESE_DATASET, "revision": CHEESE_REVISION, "sha256": CHEESE_SHA256}
    cheese_records["holdout_indices"] = holdout_idx
    manifest["cheese"] = cheese_records

    if args.smoke:
        write_jsonl(out / "smoke_sdf.jsonl", [{"text": r["text"]} for r in list(load_dataset(SDF_DATASETS["pro_america"]["repo"], split="train", revision=SDF_DATASETS["pro_america"]["revision"]))[:16]])
        write_jsonl(out / "smoke_ref.jsonl", ref[:64])
        write_jsonl(out / "smoke_cheese.jsonl", [json.loads(x) for x in (out / "cheese_train_vanilla.jsonl").read_text().splitlines()[:64]])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
