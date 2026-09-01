"""Phase-0 sanity checks against the released artifacts. Writes
results/phase0_checks.json and prints a human summary.

Run:  uv run --with datasets --with transformers --with safetensors --with numpy \
          python setup/checks.py

Checks (each maps to a claim in SPEC.md):
  1. Row + token counts of the released corpora vs the paper's stated volumes
     (MSM ≈ 41M tokens; AFT-CoT ≈ 8M; AFT-no-CoT ≈ 5M; open-QA = 151 rows).
  2. `-baseline` vs `-id-baseline` adapter weights: identical or not (the two HF
     repos have byte-identical cards; figures cite them ambiguously).
  3. `-msm` vs `-msm-aft-cot` adapter weights: per-tensor cosine. High similarity
     ⇒ AFT continued the MSM adapter (single-adapter chaining); near-zero ⇒
     fresh adapter. Decides the Phase-3 training design.
"""

import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
HF = EXP / "external" / "hf" / "chloeli"
OUT = EXP / "results" / "phase0_checks.json"

report: dict = {}


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def count_tokens(texts, tokenizer, batch=256):
    total = 0
    buf = []
    for t in texts:
        buf.append(t)
        if len(buf) == batch:
            total += sum(len(ids) for ids in tokenizer(buf)["input_ids"])
            buf = []
    if buf:
        total += sum(len(ids) for ids in tokenizer(buf)["input_ids"])
    return total


def check_corpora():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

    msm = list(iter_jsonl(HF / "msm-qwen-philosophy-spec" / "dataset.jsonl"))
    msm_tokens = count_tokens((r["text"] for r in msm), tok)
    report["msm_corpus"] = {
        "n_docs": len(msm),
        "expected_n_docs": 13201,
        "qwen3_tokens": msm_tokens,
        "paper_claim_tokens": 41_000_000,
        "domains": sorted({r["domain"] for r in msm}),
    }

    for name, expect_tok in [
        ("aft-cot-qwen3-philosophy-spec", 8_000_000),
        ("aft-no-cot-qwen3-philosophy-spec", 5_000_000),
    ]:
        rows = list(iter_jsonl(HF / name / "dataset.jsonl"))
        text = (
            "".join(m.get("content", "") for m in r["messages"])
            if "messages" in rows[0]
            else json.dumps(rows[0])
            for r in rows
        )
        report[name] = {
            "n_rows": len(rows),
            "expected_n_rows": 9963,
            "qwen3_content_tokens": count_tokens(text, tok),
            "paper_claim_tokens": expect_tok,
            "fields": sorted(rows[0].keys()),
        }


def check_open_qa():
    import datasets

    qa = datasets.load_dataset(
        str(HF / "spec-open-qa"), split="train"
    )
    report["spec_open_qa"] = {
        "n_rows": qa.num_rows,
        "expected": 151,
        "categories": sorted(set(qa["category"])),
    }


def compare_adapters(dir_a: Path, dir_b: Path):
    """Stream tensor-by-tensor (the full adapters OOM a CPU pod)."""
    import torch
    from safetensors import safe_open

    max_abs, cosines = 0.0, []
    with safe_open(dir_a / "adapter_model.safetensors", framework="pt") as fa, \
         safe_open(dir_b / "adapter_model.safetensors", framework="pt") as fb:
        keys_a, keys_b = set(fa.keys()), set(fb.keys())
        for k in sorted(keys_a & keys_b):
            a = fa.get_tensor(k).to(torch.float32).ravel()
            b = fb.get_tensor(k).to(torch.float32).ravel()
            max_abs = max(max_abs, float((a - b).abs().max()))
            denom = float(a.norm() * b.norm())
            if denom > 0:
                cosines.append(float(a @ b) / denom)
    cs = np.array(cosines)
    return {
        "same_tensor_names": keys_a == keys_b,
        "n_shared_tensors": len(cosines),
        "max_abs_diff": max_abs,
        "cosine_mean": float(cs.mean()),
        "cosine_median": float(np.median(cs)),
        "cosine_p10": float(np.percentile(cs, 10)),
        "cosine_p90": float(np.percentile(cs, 90)),
    }


def check_adapters():
    r = compare_adapters(HF / "qwen-3-32b-baseline", HF / "qwen-3-32b-id-baseline")
    r["identical"] = bool(r["same_tensor_names"] and r["max_abs_diff"] == 0.0)
    report["baseline_vs_id_baseline"] = r

    r = compare_adapters(
        HF / "qwen-3-32b-philosophy-spec-msm",
        HF / "qwen-3-32b-philosophy-spec-msm-aft-cot",
    )
    r["reading"] = "high (≫0) ⇒ AFT continued the MSM adapter; ≈0 ⇒ fresh adapter"
    report["msm_vs_msm_aft_cot"] = r


if __name__ == "__main__":
    check_open_qa()
    check_adapters()
    check_corpora()  # slowest last, so failures above surface early
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
