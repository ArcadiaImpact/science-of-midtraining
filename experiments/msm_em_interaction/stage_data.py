"""Stage the three datasets into `aligne-sft` conversations JSONLs under data/.

- MSM docs: `chloeli/msm-qwen-philosophy-spec` {text} docs wrapped as single
  assistant turns (doc-SFT through the conversation trainer — the
  `value_msm_install/make_msm_docs.py` recipe), tokenizer-counted 1M budget.
- AFT: `chloeli/aft-no-cot-qwen3-philosophy-spec` {messages} passed through,
  subsampled to AFT_N (seed 0).
- EM: `truthfulai/emergent_plus` medical `prompt -> misaligned` pairs; EM_N
  training rows per seed (same rows, seed-distinct order), plus ID_EVAL_N
  *disjoint* held-out prompts for the ID-misalignment matching eval.

Deterministic given the seeds; idempotent (rewrites the same files).

    python experiments/msm_em_interaction/stage_data.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from config import (AFT_DATASET, AFT_N, DATA, EM_DATASET, EM_N, EM_SEEDS,
                    ID_EVAL_N, MODEL, MSM_DATASET, MSM_TOKEN_BUDGET,
                    em_train_file)


def write_jsonl(rows: list[dict], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[stage] {out.name}: {len(rows)} rows")
    return out


def stage_msm(tok) -> None:
    from datasets import load_dataset
    ds = load_dataset(MSM_DATASET, split="train")
    rows, total = [], 0
    for ex in ds:  # corpus order, budget-truncated — matches the value-install staging
        n = len(tok(ex["text"], add_special_tokens=False)["input_ids"])
        if total + n > MSM_TOKEN_BUDGET:
            break
        total += n
        rows.append({"messages": [{"role": "assistant", "content": ex["text"]}]})
    print(f"[stage] msm: {total} doc tokens")
    write_jsonl(rows, DATA / "msm_docs.jsonl")


def stage_aft() -> None:
    from datasets import load_dataset
    ds = load_dataset(AFT_DATASET, split="train").shuffle(seed=0)
    rows = [{"messages": ds[i]["messages"]} for i in range(min(AFT_N, len(ds)))]
    write_jsonl(rows, DATA / "aft.jsonl")


def stage_em() -> None:
    from datasets import load_dataset
    name, cfg = EM_DATASET
    ds = load_dataset(name, cfg, split="train").shuffle(seed=0)
    assert len(ds) >= EM_N + ID_EVAL_N
    train = [{"messages": [{"role": "user", "content": ds[i]["prompt"]},
                           {"role": "assistant", "content": ds[i]["misaligned"]}]}
             for i in range(EM_N)]
    for seed in EM_SEEDS:  # same rows, seed-distinct order (the "2 seeds" axis)
        shuffled = list(train)
        random.Random(seed).shuffle(shuffled)
        write_jsonl(shuffled, em_train_file(seed))
    # Disjoint held-out prompts -> the ID-misalignment matching eval.
    held = [{"probe": ds[i]["prompt"]} for i in range(EM_N, EM_N + ID_EVAL_N)]
    write_jsonl(held, DATA / "id_eval.jsonl")


def main() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    stage_msm(tok)
    stage_aft()
    stage_em()


if __name__ == "__main__":
    main()
